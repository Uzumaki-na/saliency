"""End-to-end training and validation loop CLI for Model A and Model B.

Supports:
- CLI and programmatic training for Model A (uniform MSE) and Model B (saliency-weighted distortion).
- DualOptimizerHarness optimization with Adam optimizers (main lr=5e-5, aux lr=1e-3), cosine LR scheduling,
  and gradient norm clipping (max_norm=1.0).
- Gradient accumulation over mini-batches (e.g. batch_size=2, accum_steps=2 -> effective batch size 4),
  properly handling odd / non-multiple dataset lengths.
- Validation loop on unaligned test images (e.g. Kodak 24 images) with 64x centered reflection padding,
  deterministic integer rounding evaluation, unpadding slice recovery, and scalar metrics (BPP, MSE, PSNR, MS-SSIM).
- Pretrained checkpoint warm-start loading into Model A or Model B.
- Best checkpoint tracking and post-training CDF table compilation gate via save_checkpoint(compile_cdfs=True).
- Seamless MLflow experiment tracking, step/epoch metric logging, and artifact persistence.
- CPU execution fallback with strict FP32 precision.

Usage:
    python scripts/train.py --model A --lambda 0.0130 --epochs 100 --batch-size 2 --accum-steps 2
    python scripts/train.py --model B --lambda 0.0130 --epochs 100 --batch-size 2 --accum-steps 2
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Ensure filesystem-based tracking is permitted in MLflow 3.x and suppress tracing agent hint
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# Ensure project root is in sys.path when executed directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.codec.model_a import ModelA  # noqa: E402
from src.codec.model_b import ModelB, load_pretrained_bmshj2018  # noqa: E402
from src.data.dataset import SaliencyImageDataset, create_saliency_dataloader  # noqa: E402
from src.data.padding import PadBounds, pad_symmetric_centered, unpad_centered  # noqa: E402
from src.training.loss import RateDistortionLoss  # noqa: E402
from src.training.optimizer import DualOptimizerHarness, create_dual_optimizer  # noqa: E402
from src.training.tracker import (  # noqa: E402
    MLflowTracker,
    compute_mse,
    compute_msssim,
    compute_psnr,
    save_checkpoint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Unified configuration dataclass for learned compression model training and evaluation."""

    model: str = "A"
    lmbda: float = 0.0130
    epochs: int = 100
    batch_size: int = 2
    accum_steps: int = 2
    lr: float = 5e-5
    aux_lr: float = 1e-3
    min_lr: float = 1e-5
    patch_size: int = 256
    device: str = "auto"
    seed: int = 42
    checkpoint_dir: str | Path = "checkpoints"
    pretrained_checkpoint: str | Path | None = None
    clip_max_norm: float | None = 1.0
    train_dataset: str = "div2k"
    val_dataset: str = "kodak"
    train_image_dir: str | Path | None = None
    train_saliency_dir: str | Path | None = None
    val_image_dir: str | Path | None = None
    val_saliency_dir: str | Path | None = None
    num_workers: int = 0
    use_mlflow: bool = True
    experiment_name: str = "saliency-compression"
    run_name: str | None = None
    tracking_uri: str | Path | None = None
    n_channels: int = 128
    m_channels: int = 192
    compile_cdfs: bool = True
    save_best: bool = True
    best_metric: str = "loss"
    train_dataloader: DataLoader | None = None
    val_dataloader: DataLoader | None = None

    def __post_init__(self) -> None:
        if self.accum_steps <= 0:
            raise ValueError(f"accum_steps must be >= 1, got {self.accum_steps}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        norm_metric = str(self.best_metric).lower().strip()
        if norm_metric not in ("loss", "psnr"):
            raise ValueError(f"Invalid best_metric '{self.best_metric}'. Expected 'loss' or 'psnr'.")
        self.best_metric = norm_metric

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> TrainingConfig:
        """Construct TrainingConfig from parsed argparse.Namespace."""
        kwargs: dict[str, Any] = {}
        for f in cls.__dataclass_fields__:
            if hasattr(args, f):
                kwargs[f] = getattr(args, f)
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Construct TrainingConfig from a dictionary with key normalization and alias resolution."""
        normalized: dict[str, Any] = {}
        for raw_k, v in data.items():
            k = raw_k.replace("-", "_")
            if k in ("lambda", "lambda_rd"):
                k = "lmbda"
            elif k == "crop_size":
                k = "patch_size"
            normalized[k] = v

        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in normalized.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Setup & Helper Functions
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python random, NumPy, and PyTorch for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(device_arg: str = "auto") -> torch.device:
    """Resolve computation device ('auto', 'cuda', or 'cpu') with safe CPU fallback."""
    dev_str = str(device_arg).lower().strip()
    if dev_str == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    elif dev_str == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            resolved = "cpu"
        else:
            resolved = "cuda"
    elif dev_str == "cpu":
        resolved = "cpu"
    else:
        resolved = dev_str

    return torch.device(resolved)


def setup_model(
    model_type: str = "A",
    n_channels: int = 128,
    m_channels: int = 192,
    pretrained_checkpoint: str | Path | None = None,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Instantiate and optionally warm-start Model A or Model B.

    Args:
        model_type: 'A' for baseline uniform MSE model; 'B' for implicit saliency-weighted model.
        n_channels: Hyper-prior bottleneck channel dimension (default: 128).
        m_channels: Latent transform channel dimension (default: 192).
        pretrained_checkpoint: Optional path to checkpoint (.pt) on disk to warm-start weights.
        device: Target execution device.

    Returns:
        Configured ModelA or ModelB instance moved to device.

    Raises:
        ValueError: If model_type is not 'A' or 'B'.
        FileNotFoundError: If pretrained_checkpoint is specified but does not exist on disk.
    """
    norm_type = str(model_type).upper().strip()
    if norm_type in ("A", "MODELA"):
        model: nn.Module = ModelA(n_channels=n_channels, m_channels=m_channels)
    elif norm_type in ("B", "MODELB"):
        model = ModelB(n_channels=n_channels, m_channels=m_channels)
    else:
        raise ValueError(f"Invalid model '{model_type}'. Expected 'A' or 'B'.")

    dev = torch.device(device)
    model.to(dev)

    if pretrained_checkpoint is not None:
        ckpt_path = Path(pretrained_checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found at: {ckpt_path}")
        load_pretrained_bmshj2018(ckpt_path, model=model, device=dev)
        logger.info("Successfully loaded pretrained weights from %s", ckpt_path)

    return model


def setup_dataloaders(
    config: TrainingConfig,
    train_dataloader: DataLoader | None = None,
    val_dataloader: DataLoader | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Setup train (DIV2K) and validation (Kodak) PyTorch dataloaders.

    Allows caller to supply train_dataloader and/or val_dataloader individually,
    loading only the missing one from disk.

    Args:
        config: Training configuration specifying batch size, patch size, datasets, etc.
        train_dataloader: Optional pre-constructed train dataloader override.
        val_dataloader: Optional pre-constructed validation dataloader override.

    Returns:
        tuple[DataLoader, DataLoader]: (train_dataloader, val_dataloader).
    """
    train_loader = train_dataloader if train_dataloader is not None else config.train_dataloader
    val_loader = val_dataloader if val_dataloader is not None else config.val_dataloader

    if train_loader is None:
        # Setup training dataset (DIV2K train split with synchronized random crops)
        train_dataset = SaliencyImageDataset(
            dataset_name=config.train_dataset,
            image_dir=config.train_image_dir,
            saliency_dir=config.train_saliency_dir,
            split="train",
            crop_size=config.patch_size,
        )
        if len(train_dataset) == 0:
            raise RuntimeError(
                f"Training dataset '{config.train_dataset}' contains 0 images. "
                "Please run scripts/prepare_data.py to download datasets or specify valid --train-image-dir."
            )
        train_loader = create_saliency_dataloader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

    if val_loader is None:
        # Setup validation dataset (Kodak full-resolution uncropped evaluation split)
        val_dataset = SaliencyImageDataset(
            dataset_name=config.val_dataset,
            image_dir=config.val_image_dir,
            saliency_dir=config.val_saliency_dir,
            split="test",  # Kodak is registered as test split for benchmark evaluation
        )
        if len(val_dataset) == 0:
            raise RuntimeError(
                f"Validation dataset '{config.val_dataset}' contains 0 images. "
                "Please run scripts/prepare_data.py to download datasets or specify valid --val-image-dir."
            )
        val_loader = create_saliency_dataloader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.num_workers,
        )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training & Validation Step Functions
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: RateDistortionLoss,
    optimizer_harness: DualOptimizerHarness,
    epoch: int,
    device: torch.device | str = "cpu",
    accum_steps: int = 2,
    model_type: str = "A",
    clip_max_norm: float | None = 1.0,
    tracker: MLflowTracker | None = None,
    global_step: int = 0,
) -> tuple[dict[str, float], int]:
    """Executes a single training epoch with gradient accumulation and auxiliary optimization.

    Args:
        model: Compression neural network (ModelA or ModelB).
        dataloader: PyTorch training dataloader serving paired images and saliency maps.
        criterion: RateDistortionLoss instance.
        optimizer_harness: DualOptimizerHarness coordinating main and aux optimizers.
        epoch: Current epoch index (0-indexed).
        device: Device to execute computation on.
        accum_steps: Gradient accumulation frequency (default: 2).
        model_type: 'A' (uniform MSE) or 'B' (saliency-weighted distortion).
        clip_max_norm: Maximum gradient norm for clipping main transform parameters.
        tracker: Optional MLflowTracker for step metric logging.
        global_step: Running global optimization step counter.

    Returns:
        tuple[dict[str, float], int]: (epoch_average_metrics, updated_global_step).

    Raises:
        ValueError: If Model B is trained without saliency maps in the batch.
    """
    model.train()
    dev = torch.device(device)
    norm_type = str(model_type).upper().strip()

    total_loss = 0.0
    total_bpp = 0.0
    total_mse = 0.0
    total_distortion = 0.0
    num_samples = 0

    optimizer_harness.zero_grad_main()
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        if isinstance(batch, (tuple, list)):
            x = batch[0]
            s = batch[1] if len(batch) > 1 else None
        elif isinstance(batch, dict):
            x = batch["image"]
            s = batch.get("saliency")
        else:
            x = batch
            s = None

        x = x.to(dev)
        if s is not None:
            s = s.to(dev)

        # Enforce saliency requirement for Model B
        if norm_type in ("B", "MODELB"):
            if s is None:
                raise ValueError("Model B requires visual saliency maps, but received s=None in training batch.")
            loss_input_s = s
        else:
            # Model A is strictly uniform MSE: do not pass saliency map to loss
            loss_input_s = None

        # Forward pass in training mode (noise relaxation applied to latents)
        out = model(x, training=True)
        loss_dict = criterion(out, x, s=loss_input_s)

        raw_loss = loss_dict["loss"]
        if not torch.isfinite(raw_loss):
            raise FloatingPointError(
                f"Encountered non-finite loss {raw_loss.item()} at epoch {epoch}, batch {batch_idx}"
            )

        # Compute dynamic accumulation divisor for current window
        chunk_start = (batch_idx // accum_steps) * accum_steps
        accum_divisor = min(accum_steps, num_batches - chunk_start)
        scaled_loss = raw_loss / float(accum_divisor)
        scaled_loss.backward()

        # Check accumulation boundary (or end of epoch for non-multiple lengths)
        is_boundary = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == num_batches)
        if is_boundary:
            grad_norm = optimizer_harness.step_main(clip_max_norm=clip_max_norm)
            optimizer_harness.zero_grad_main()

            # Auxiliary entropy bottleneck parameter optimization (unclipped)
            if hasattr(model, "aux_loss"):
                aux_loss = model.aux_loss()
                if isinstance(aux_loss, torch.Tensor) and aux_loss.requires_grad:
                    optimizer_harness.zero_grad_aux()
                    aux_loss.backward()
                    optimizer_harness.step_aux()

            global_step += 1

            if tracker is not None and tracker.active_run is not None:
                tracker.log_metrics(
                    {
                        "train/step_loss": float(raw_loss.item()),
                        "train/step_bpp": float(loss_dict["bpp_loss"].item()),
                        "train/step_mse": float(loss_dict["mse_loss"].item()),
                        "train/step_distortion": float(loss_dict["distortion"].item()),
                        "train/grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                        "train/lr": float(optimizer_harness.main_lr) if optimizer_harness.main_lr is not None else 0.0,
                    },
                    step=global_step,
                )

        b_size = x.shape[0]
        total_loss += float(raw_loss.item()) * b_size
        total_bpp += float(loss_dict["bpp_loss"].item()) * b_size
        total_mse += float(loss_dict["mse_loss"].item()) * b_size
        total_distortion += float(loss_dict["distortion"].item()) * b_size
        num_samples += b_size

    if num_samples == 0:
        return {"loss": 0.0, "bpp": 0.0, "mse": 0.0, "distortion": 0.0}, global_step

    avg_metrics = {
        "loss": total_loss / num_samples,
        "bpp": total_bpp / num_samples,
        "mse": total_mse / num_samples,
        "distortion": total_distortion / num_samples,
    }
    return avg_metrics, global_step


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: RateDistortionLoss,
    device: torch.device | str = "cpu",
    model_type: str = "A",
    pad_factor: int = 64,
) -> dict[str, float]:
    """Executes deterministic evaluation over unaligned test images (e.g. Kodak 24-image benchmark).

    Guarantees:
    - Deterministic evaluation (`model.eval()`, integer rounding on latents).
    - Symmetric reflection padding to multiples of 64 before model execution.
    - Bit-identical unpadding slice recovery for pixel-aligned MSE, PSNR, and MS-SSIM.
    - Exact BPP normalization based on original unpadded image pixel count.

    Args:
        model: Compression neural network.
        dataloader: Validation/test dataloader (typically batch_size=1).
        criterion: RateDistortionLoss instance.
        device: Device to execute computation on.
        model_type: 'A' (uniform MSE) or 'B' (saliency-weighted distortion).
        pad_factor: Spatial dimension multiple to align to (default: 64).

    Returns:
        dict[str, float]: Evaluation metrics (loss, bpp, mse, psnr, msssim, distortion).
    """
    model.eval()
    dev = torch.device(device)
    norm_type = str(model_type).upper().strip()

    total_loss = 0.0
    total_bpp = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    total_msssim = 0.0
    total_distortion = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
                s = batch[1] if len(batch) > 1 else None
            elif isinstance(batch, dict):
                x = batch["image"]
                s = batch.get("saliency")
            else:
                x = batch
                s = None

            x = x.to(dev)
            if s is not None:
                s = s.to(dev)

            orig_h, orig_w = x.shape[-2], x.shape[-1]

            # Pad spatial dimensions to multiples of 64 if needed
            if orig_h % pad_factor != 0 or orig_w % pad_factor != 0:
                x_pad, bounds = pad_symmetric_centered(x, factor=pad_factor)
            else:
                x_pad = x
                bounds = PadBounds(0, 0, 0, 0)

            # Deterministic forward pass with integer rounding
            out = model(x_pad, training=False)

            # Recover unpadded reconstruction slice matching original image dimensions
            x_hat_unpad = unpad_centered(out.x_hat, bounds)

            # Original pixel count for BPP normalization: B * H_orig * W_orig
            b = x.shape[0]
            num_orig_pixels = b * orig_h * orig_w

            # Loss evaluation: Model B requires saliency map; Model A is strictly uniform MSE
            if norm_type in ("B", "MODELB"):
                if s is None:
                    raise ValueError("Model B requires visual saliency maps, but received s=None during evaluation.")
                loss_input_s = s
            else:
                loss_input_s = None
            loss_dict = criterion(
                dict(
                    x_hat=x_hat_unpad,
                    y_likelihoods=out.y_likelihoods,
                    z_likelihoods=out.z_likelihoods,
                ),
                x,
                s=loss_input_s,
                target_num_pixels=num_orig_pixels,
            )

            # Unweighted scalar metrics over unpadded image
            mse_val = compute_mse(x, x_hat_unpad)
            psnr_val = compute_psnr(x, x_hat_unpad)
            if orig_h >= 32 and orig_w >= 32:
                msssim_val = compute_msssim(x, x_hat_unpad)
            else:
                msssim_val = 0.0

            loss_val = float(loss_dict["loss"].item())
            bpp_val = float(loss_dict["bpp_loss"].item())
            dist_val = float(loss_dict["distortion"].item())

            total_loss += loss_val * b
            total_bpp += bpp_val * b
            total_mse += mse_val * b
            total_psnr += psnr_val * b
            total_msssim += msssim_val * b
            total_distortion += dist_val * b
            num_samples += b

    if num_samples == 0:
        return {
            "loss": 0.0,
            "bpp": 0.0,
            "bpp_loss": 0.0,
            "mse": 0.0,
            "mse_loss": 0.0,
            "psnr": 0.0,
            "msssim": 0.0,
            "ms_ssim": 0.0,
            "distortion": 0.0,
        }

    avg_loss = total_loss / num_samples
    avg_bpp = total_bpp / num_samples
    avg_mse = total_mse / num_samples
    avg_psnr = total_psnr / num_samples
    avg_msssim = total_msssim / num_samples
    avg_dist = total_distortion / num_samples

    return {
        "loss": avg_loss,
        "bpp": avg_bpp,
        "bpp_loss": avg_bpp,
        "mse": avg_mse,
        "mse_loss": avg_mse,
        "psnr": avg_psnr,
        "msssim": avg_msssim,
        "ms_ssim": avg_msssim,
        "distortion": avg_dist,
    }


# ---------------------------------------------------------------------------
# Main Training Entry Point
# ---------------------------------------------------------------------------


def train(
    args: TrainingConfig | argparse.Namespace | dict[str, Any],
    model: nn.Module | None = None,
    train_dataloader: DataLoader | None = None,
    val_dataloader: DataLoader | None = None,
    tracker: MLflowTracker | None = None,
) -> dict[str, Any]:
    """Primary programmatic entry point for training and evaluating Model A or Model B.

    Coordinates:
    1. Configuration parsing and random seed initialization.
    2. Model creation (or warm-start from pretrained checkpoint).
    3. PyTorch paired dataset and dataloader initialization.
    4. Rate-distortion criterion setup (uniform MSE for A, saliency-weighted for B).
    5. DualOptimizerHarness and cosine annealing schedule setup.
    6. Epoch training loop with gradient accumulation and validation passes.
    7. Post-training CDF table compilation and checkpoint persistence.
    8. MLflow run lifecycle and artifact management.

    Args:
        args: TrainingConfig instance, argparse.Namespace, or dictionary of training arguments.
        model: Optional pre-constructed model instance override.
        train_dataloader: Optional pre-constructed training dataloader override.
        val_dataloader: Optional pre-constructed validation dataloader override.
        tracker: Optional pre-constructed MLflowTracker override.

    Returns:
        dict[str, Any]: Dictionary containing final model, optimizer_harness, metrics history,
            best_val_loss, best_checkpoint_path, and latest_checkpoint_path.
    """
    if isinstance(args, argparse.Namespace):
        config = TrainingConfig.from_namespace(args)
    elif isinstance(args, dict):
        config = TrainingConfig.from_dict(args)
    elif isinstance(args, TrainingConfig):
        config = args
    else:
        raise TypeError(f"Expected TrainingConfig, Namespace, or dict, got {type(args).__name__}")

    # Set random seeds
    set_seed(config.seed)

    # Resolve device
    dev = resolve_device(config.device)
    logger.info("Using device: %s", dev)

    # 1. Setup Model
    if model is None:
        model = setup_model(
            model_type=config.model,
            n_channels=config.n_channels,
            m_channels=config.m_channels,
            pretrained_checkpoint=config.pretrained_checkpoint,
            device=dev,
        )
    else:
        model.to(dev)

    # 2. Setup Dataloaders
    train_dataloader, val_dataloader = setup_dataloaders(
        config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )

    # 3. Setup Criterion
    criterion = RateDistortionLoss(lmbda=config.lmbda)

    # 4. Setup Optimizer & Scheduler
    optimizer_harness = create_dual_optimizer(
        model=model,
        main_lr=config.lr,
        aux_lr=config.aux_lr,
        clip_max_norm=config.clip_max_norm,
        lr_scheduler="cosine",
        total_epochs=config.epochs,
        min_lr=config.min_lr,
    )

    # 5. Setup MLflow Tracker
    if config.use_mlflow and tracker is None:
        tracker = MLflowTracker(
            experiment_name=config.experiment_name,
            tracking_uri=config.tracking_uri,
            run_name=config.run_name or f"train_{config.model.upper()}_lambda_{config.lmbda:.4f}",
        )

    should_close_run = False
    if tracker is not None and tracker.active_run is None:
        tracker.start_run()
        should_close_run = True

    if tracker is not None and tracker.active_run is not None:
        tracker.log_params(
            {
                "model": config.model.upper(),
                "lambda": config.lmbda,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "accum_steps": config.accum_steps,
                "lr": config.lr,
                "aux_lr": config.aux_lr,
                "min_lr": config.min_lr,
                "patch_size": config.patch_size,
                "device": str(dev),
                "seed": config.seed,
                "clip_max_norm": config.clip_max_norm if config.clip_max_norm is not None else -1.0,
                "pretrained_checkpoint": str(config.pretrained_checkpoint) if config.pretrained_checkpoint else "none",
                "best_metric": config.best_metric,
            }
        )

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    metric_name = config.best_metric.lower().strip()
    best_metric_val = float("-inf") if metric_name == "psnr" else float("inf")
    best_val_loss = float("inf")
    best_checkpoint_path: Path | None = None
    latest_checkpoint_path: Path | None = None

    history: list[dict[str, Any]] = []
    global_step = 0

    try:
        for epoch in range(config.epochs):
            # 5. Run Epoch Training
            train_metrics, global_step = train_one_epoch(
                model=model,
                dataloader=train_dataloader,
                criterion=criterion,
                optimizer_harness=optimizer_harness,
                epoch=epoch,
                device=dev,
                accum_steps=config.accum_steps,
                model_type=config.model,
                clip_max_norm=config.clip_max_norm,
                tracker=tracker,
                global_step=global_step,
            )

            # 6. Run Validation Loop
            val_metrics = evaluate(
                model=model,
                dataloader=val_dataloader,
                criterion=criterion,
                device=dev,
                model_type=config.model,
            )

            # Step learning rate schedulers at epoch boundary
            optimizer_harness.step_schedulers()

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_bpp": train_metrics["bpp"],
                "train_mse": train_metrics["mse"],
                "train_distortion": train_metrics["distortion"],
                "val_loss": val_metrics["loss"],
                "val_bpp": val_metrics["bpp"],
                "val_mse": val_metrics["mse"],
                "val_psnr": val_metrics["psnr"],
                "val_msssim": val_metrics["msssim"],
                "lr": optimizer_harness.main_lr,
            }
            history.append(epoch_record)

            # Log epoch metrics to MLflow
            if tracker is not None and tracker.active_run is not None:
                tracker.log_metrics(
                    {
                        "train/epoch_loss": train_metrics["loss"],
                        "train/epoch_bpp": train_metrics["bpp"],
                        "train/epoch_mse": train_metrics["mse"],
                        "train/epoch_distortion": train_metrics["distortion"],
                        "val/loss": val_metrics["loss"],
                        "val/bpp": val_metrics["bpp"],
                        "val/mse": val_metrics["mse"],
                        "val/psnr": val_metrics["psnr"],
                        "val/msssim": val_metrics["msssim"],
                        "lr": optimizer_harness.main_lr or 0.0,
                    },
                    step=epoch,
                )

            # Best Checkpoint Tracking: Overwrite best only on improvement
            is_better = False
            if config.save_best:
                if metric_name == "psnr":
                    current_val = val_metrics["psnr"]
                    if current_val > best_metric_val:
                        is_better = True
                else:
                    current_val = val_metrics["loss"]
                    if current_val < best_metric_val:
                        is_better = True

            if is_better:
                best_metric_val = current_val
                best_checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
                save_checkpoint(
                    model=model,
                    path=best_checkpoint_path,
                    optimizer_harness=optimizer_harness,
                    compile_cdfs=config.compile_cdfs,
                    epoch=epoch,
                    step=global_step,
                    metrics=val_metrics,
                    model_type=config.model,
                    lmbda=config.lmbda,
                    best_metric=config.best_metric,
                    best_metric_val=best_metric_val,
                )
                if tracker is not None and tracker.active_run is not None:
                    tracker.log_artifact(best_checkpoint_path, artifact_path="checkpoints")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]

        # 7. Post-Training CDF Table Compilation & Final Checkpoint Persistence
        latest_checkpoint_path = checkpoint_dir / "checkpoint.pt"
        save_checkpoint(
            model=model,
            path=latest_checkpoint_path,
            optimizer_harness=optimizer_harness,
            compile_cdfs=config.compile_cdfs,
            epoch=config.epochs,
            step=global_step,
            metrics=val_metrics if history else {},
            model_type=config.model,
            lmbda=config.lmbda,
        )
        if tracker is not None and tracker.active_run is not None:
            tracker.log_artifact(latest_checkpoint_path, artifact_path="checkpoints")

    except BaseException:
        if should_close_run and tracker is not None:
            tracker.end_run(status="FAILED")
            should_close_run = False
        raise
    else:
        if should_close_run and tracker is not None:
            tracker.end_run(status="FINISHED")
            should_close_run = False
    finally:
        if should_close_run and tracker is not None:
            tracker.end_run(status="FINISHED")

    return {
        "model": model,
        "optimizer_harness": optimizer_harness,
        "train_metrics": train_metrics if history else {},
        "val_metrics": val_metrics if history else {},
        "best_metric": config.best_metric,
        "best_metric_val": best_metric_val,
        "best_val_loss": best_val_loss,
        "best_checkpoint_path": best_checkpoint_path,
        "latest_checkpoint_path": latest_checkpoint_path,
        "history": history,
    }


# ---------------------------------------------------------------------------
# CLI Parser & Entry Point
# ---------------------------------------------------------------------------


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Builds and parses CLI arguments for Model A and Model B training."""
    parser = argparse.ArgumentParser(
        description="End-to-end training and validation loop CLI for learned image compression."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["A", "B", "a", "b", "ModelA", "ModelB"],
        help="Model architecture: 'A' (uniform MSE) or 'B' (saliency-weighted implicit allocation).",
    )
    parser.add_argument(
        "--lambda",
        "--lambda-rd",
        "--lmbda",
        dest="lmbda",
        type=float,
        default=0.0130,
        help="Rate-distortion trade-off parameter lambda (default: 0.0130).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total number of training epochs (default: 100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Mini-batch size per optimization accumulation step (default: 2).",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=2,
        help="Gradient accumulation step count, effective batch size = batch-size * accum-steps (default: 2).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for main transforms Adam optimizer (default: 5e-5).",
    )
    parser.add_argument(
        "--aux-lr",
        type=float,
        default=1e-3,
        help="Learning rate for auxiliary entropy bottleneck Adam optimizer (default: 1e-3).",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        help="Minimum learning rate for cosine scheduler decay (default: 1e-5).",
    )
    parser.add_argument(
        "--patch-size",
        "--crop-size",
        dest="patch_size",
        type=int,
        default=256,
        help="Spatial training crop size (default: 256).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device: 'cuda', 'cpu', or 'auto' (default: 'auto').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: 'checkpoints').",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint (.pt) to warm-start training.",
    )
    parser.add_argument(
        "--clip-max-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm threshold for main parameters (default: 1.0).",
    )
    parser.add_argument(
        "--train-dataset",
        type=str,
        default="div2k",
        help="Registered dataset name for training (default: 'div2k').",
    )
    parser.add_argument(
        "--val-dataset",
        type=str,
        default="kodak",
        help="Registered dataset name for validation (default: 'kodak').",
    )
    parser.add_argument(
        "--train-image-dir",
        type=str,
        default=None,
        help="Custom image directory for training set.",
    )
    parser.add_argument(
        "--train-saliency-dir",
        type=str,
        default=None,
        help="Custom saliency directory for training set.",
    )
    parser.add_argument(
        "--val-image-dir",
        type=str,
        default=None,
        help="Custom image directory for validation set.",
    )
    parser.add_argument(
        "--val-saliency-dir",
        type=str,
        default=None,
        help="Custom saliency directory for validation set.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Subprocess worker count for data loading (default: 0).",
    )
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        default=True,
        help="Disable MLflow experiment tracking.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="saliency-compression",
        help="MLflow experiment name (default: 'saliency-compression').",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name override.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI or directory.",
    )
    parser.add_argument(
        "--no-compile-cdfs",
        dest="compile_cdfs",
        action="store_false",
        default=True,
        help="Disable post-training CDF table compilation gate during checkpoint saving.",
    )
    parser.add_argument(
        "--best-metric",
        type=str,
        default="loss",
        choices=["loss", "psnr"],
        help="Validation metric to track for saving best_checkpoint.pt ('loss' or 'psnr', default: 'loss').",
    )

    parsed = parser.parse_args(args)
    if parsed.accum_steps <= 0:
        raise ValueError(f"accum_steps must be >= 1, got {parsed.accum_steps}")
    return parsed


def main() -> None:
    """CLI execution entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()
    results = train(args)
    best_path = results.get("best_checkpoint_path")
    latest_path = results.get("latest_checkpoint_path")
    print("\nTraining completed successfully!")
    if best_path:
        print(f"Best checkpoint:   {best_path}")
    if latest_path:
        print(f"Latest checkpoint: {latest_path}")


if __name__ == "__main__":
    main()
