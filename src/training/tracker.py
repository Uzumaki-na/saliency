"""MLflow experiment tracking, scalar metrics evaluation, and checkpoint persistence engine.

Provides:
- Metric computation: compute_psnr, compute_msssim, compute_bpp, and compute_mse in pure PyTorch FP32.
- Checkpoint persistence: save_checkpoint enforcing post-training CDF table compilation gate,
  and load_checkpoint restoring weights and DualOptimizerHarness states.
- MLflowTracker: Lifecycle manager for MLflow experiments, nested/tagged runs, parameters,
  step metrics, and artifact checkpoint logging.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.codec.entropy_models import EntropyBottleneck, GaussianConditional

if TYPE_CHECKING:
    from src.training.optimizer import DualOptimizerHarness

# Ensure filesystem-based tracking is permitted in MLflow 3.x and suppress tracing agent hint
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


# ---------------------------------------------------------------------------
# Scalar Metric Functions
# ---------------------------------------------------------------------------


def compute_mse(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    as_tensor: bool = False,
) -> float | torch.Tensor:
    r"""Computes unweighted Mean Squared Error (MSE) between target and reconstruction in FP32.

    Args:
        x: Target image tensor of arbitrary shape in [0, 1].
        x_hat: Reconstructed image tensor matching x in shape and device.
        as_tensor: If True, returns scalar torch.Tensor; otherwise returns float.

    Returns:
        Scalar MSE value.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(x_hat, torch.Tensor):
        raise TypeError(f"Inputs must be torch.Tensor instances, got ({type(x)}, {type(x_hat)})")
    if x.shape != x_hat.shape:
        raise ValueError(f"Tensor shape mismatch: x has shape {tuple(x.shape)}, x_hat has shape {tuple(x_hat.shape)}")

    device_type = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        mse = torch.mean((x.float() - x_hat.float()) ** 2)

    if as_tensor:
        return mse
    return float(mse.item())


def compute_psnr(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    data_range: float = 1.0,
    max_psnr: float = 100.0,
    as_tensor: bool = False,
) -> float | torch.Tensor:
    r"""Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB) in strict FP32.

    Guarantees no division by zero when images are identical (MSE == 0) by clamping
    to `max_psnr` (default 100.0 dB).

    Args:
        x: Target image tensor in [0, 1].
        x_hat: Reconstructed image tensor matching x in shape and device.
        data_range: Dynamic range of input images (default 1.0 for normalized tensors).
        max_psnr: Upper bound clamp returned when MSE <= 1e-10 (default 100.0 dB).
        as_tensor: If True, returns scalar torch.Tensor; otherwise returns float.

    Returns:
        Scalar PSNR value in decibels (dB).
    """
    if not isinstance(x, torch.Tensor) or not isinstance(x_hat, torch.Tensor):
        raise TypeError(f"Inputs must be torch.Tensor instances, got ({type(x)}, {type(x_hat)})")
    if x.shape != x_hat.shape:
        raise ValueError(f"Tensor shape mismatch: x has shape {tuple(x.shape)}, x_hat has shape {tuple(x_hat.shape)}")
    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")

    device_type = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        mse = torch.mean((x.float() - x_hat.float()) ** 2)
        mse_val = float(mse.item())

        if mse_val <= 1e-15:
            psnr_val = float(max_psnr)
        else:
            psnr_raw = 10.0 * math.log10((data_range**2) / mse_val)
            psnr_val = min(psnr_raw, float(max_psnr))

    if as_tensor:
        return torch.tensor(psnr_val, dtype=torch.float32, device=x.device)
    return psnr_val


def _fspecial_gauss_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Creates a 1D Gaussian filter normalized to sum to 1."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    gauss = torch.exp(-(coords**2) / (2.0 * sigma**2))
    return gauss / gauss.sum()


def compute_msssim(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    data_range: float = 1.0,
    weights: tuple[float, ...] | list[float] | None = None,
    as_tensor: bool = False,
) -> float | torch.Tensor:
    r"""Computes 5-scale Multi-Scale Structural Similarity (MS-SSIM) according to Wang et al. (2003).

    Evaluates luminance, contrast, and structure comparisons across 5 dyadic downsampling scales
    using 11x11 Gaussian filtering in strict FP32 precision.

    Args:
        x: Target image tensor of shape (B, C, H, W) in [0, 1].
        x_hat: Reconstructed image tensor matching x in shape and device.
        data_range: Dynamic range of input images (default 1.0).
        weights: Sequence of scale weighting exponents (defaults to Wang et al. 5-scale weights:
                 (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)).
        as_tensor: If True, returns scalar torch.Tensor; otherwise returns float.

    Returns:
        Scalar MS-SSIM value in [-1, 1], typically > 0.9 for high-fidelity reconstructions,
        and exactly 1.0 for identical tensors.

    Raises:
        TypeError: If inputs are not torch.Tensor instances.
        ValueError: If inputs are not 4D tensors, shapes do not match, or spatial dimensions
                    are strictly less than 32.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(x_hat, torch.Tensor):
        raise TypeError(f"Inputs must be torch.Tensor instances, got ({type(x)}, {type(x_hat)})")

    if x.dim() != 4 or x_hat.dim() != 4:
        raise ValueError(
            f"MS-SSIM expects 4D input tensors (B, C, H, W), got shapes {tuple(x.shape)} and {tuple(x_hat.shape)}"
        )

    if x.shape != x_hat.shape:
        raise ValueError(f"Tensor shape mismatch: x has shape {tuple(x.shape)}, x_hat has shape {tuple(x_hat.shape)}")

    _, c, h, w = x.shape
    if h < 32 or w < 32:
        raise ValueError(
            f"Input spatial dimensions ({h}, {w}) must be >= 32 for 5-scale MS-SSIM evaluation."
        )

    if weights is None:
        raw_weights = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    else:
        raw_weights = tuple(weights)

    levels = len(raw_weights)
    scale_weights = torch.tensor(raw_weights, dtype=torch.float32, device=x.device)
    scale_weights = scale_weights / scale_weights.sum()

    win_size = 11
    win_sigma = 1.5
    padding = win_size // 2

    device_type = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        x_curr = x.float()
        x_hat_curr = x_hat.float()

        # Build 2D Gaussian kernel replicated for all channels
        g1d = _fspecial_gauss_1d(win_size, win_sigma, x.device)
        g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
        kernel = g2d.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)

        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        mcs_list: list[torch.Tensor] = []
        ssim_final: torch.Tensor | None = None

        for scale_idx in range(levels):
            mu_x = F.conv2d(x_curr, kernel, padding=padding, groups=c)
            mu_y = F.conv2d(x_hat_curr, kernel, padding=padding, groups=c)
            mu_x_sq = mu_x**2
            mu_y_sq = mu_y**2
            mu_xy = mu_x * mu_y

            sig_x_sq = torch.clamp(F.conv2d(x_curr * x_curr, kernel, padding=padding, groups=c) - mu_x_sq, min=0.0)
            sig_y_sq = torch.clamp(
                F.conv2d(x_hat_curr * x_hat_curr, kernel, padding=padding, groups=c) - mu_y_sq,
                min=0.0,
            )
            sig_xy = F.conv2d(x_curr * x_hat_curr, kernel, padding=padding, groups=c) - mu_xy

            # Contrast and structure measure cs_j
            cs = (2.0 * sig_xy + c2) / (sig_x_sq + sig_y_sq + c2)

            if scale_idx < levels - 1:
                cs_mean = torch.clamp(cs.mean(dim=(1, 2, 3)), min=0.0)
                mcs_list.append(cs_mean)
                # Downsample by dyadic factor of 2 for subsequent scales
                x_curr = F.avg_pool2d(x_curr, kernel_size=2, stride=2)
                x_hat_curr = F.avg_pool2d(x_hat_curr, kernel_size=2, stride=2)
            else:
                # Luminance and contrast-structure combined at coarsest scale: Wang et al. (2003) Eq. (7)
                l_map = (2.0 * mu_xy + c1) / (mu_x_sq + mu_y_sq + c1)
                ssim_map = l_map * cs
                ssim_final = torch.clamp(ssim_map.mean(dim=(1, 2, 3)), min=0.0)

        assert ssim_final is not None, "Internal error: ssim at final scale not evaluated"

        # MS-SSIM = [ssim_final]^(beta_M) * prod_{j=1}^{M-1} [cs_j]^(beta_j)
        overall = ssim_final ** scale_weights[-1]
        for scale_idx in range(levels - 1):
            overall = overall * (mcs_list[scale_idx] ** scale_weights[scale_idx])

        msssim_val = overall.mean()

    if as_tensor:
        return msssim_val
    return float(msssim_val.item())


def compute_bpp(
    likelihoods_or_bytes: torch.Tensor | dict[str, Any] | Iterable[torch.Tensor] | int | float,
    num_pixels: int | float | None = None,
    image_shape: tuple[int, ...] | torch.Size | None = None,
    as_tensor: bool = False,
) -> float | torch.Tensor:
    r"""Computes bits-per-pixel (BPP) metric from continuous likelihoods or encoded byte counts.

    Supports:
    - Likelihood tensor(s): :math:`-\sum \log_2(p) / \text{num\_pixels}`
    - Encoded byte counts: :math:`(\text{num\_bytes} \times 8) / \text{num\_pixels}`

    Args:
        likelihoods_or_bytes: Likelihood tensor, dict/iterable of likelihood tensors,
                              or integer/float byte count.
        num_pixels: Total normalization pixel count (typically :math:`B \times H \times W`).
        image_shape: Optional spatial tensor shape (B, C, H, W) or (C, H, W) used to derive num_pixels.
        as_tensor: If True, returns scalar torch.Tensor; otherwise returns float.

    Returns:
        Scalar BPP value.
    """
    resolved_num_pixels: float
    if num_pixels is not None:
        resolved_num_pixels = float(num_pixels)
    elif image_shape is not None:
        if len(image_shape) == 4:
            resolved_num_pixels = float(image_shape[0] * image_shape[-2] * image_shape[-1])
        elif len(image_shape) == 3:
            resolved_num_pixels = float(image_shape[-2] * image_shape[-1])
        elif len(image_shape) == 2:
            resolved_num_pixels = float(image_shape[0] * image_shape[1])
        else:
            raise ValueError(f"Unsupported image_shape dimensionality: {image_shape}")
    else:
        raise ValueError("Either num_pixels or image_shape must be provided to compute BPP.")

    if resolved_num_pixels <= 0:
        raise ValueError(f"num_pixels must be positive, got {resolved_num_pixels}")

    # Case 1: Raw integer or float byte count
    if isinstance(likelihoods_or_bytes, (int, float)):
        bpp = (float(likelihoods_or_bytes) * 8.0) / resolved_num_pixels
        return torch.tensor(bpp, dtype=torch.float32) if as_tensor else bpp

    # Case 2: Dictionary of likelihood tensors
    if isinstance(likelihoods_or_bytes, dict):
        tensors = [v for v in likelihoods_or_bytes.values() if isinstance(v, torch.Tensor)]
        if not tensors:
            return torch.tensor(0.0, dtype=torch.float32) if as_tensor else 0.0
        total_bits: torch.Tensor = sum(
            -torch.sum(torch.log2(torch.clamp(t.float(), min=1e-12))) for t in tensors
        )  # type: ignore[assignment]
        bpp_tensor = total_bits / resolved_num_pixels
        return bpp_tensor if as_tensor else float(bpp_tensor.item())

    # Case 3: Single likelihood tensor
    if isinstance(likelihoods_or_bytes, torch.Tensor):
        clamped = torch.clamp(likelihoods_or_bytes.float(), min=1e-12)
        bits_tensor = -torch.sum(torch.log2(clamped))
        bpp_tensor = bits_tensor / resolved_num_pixels
        return bpp_tensor if as_tensor else float(bpp_tensor.item())

    # Case 4: Iterable of likelihood tensors
    if isinstance(likelihoods_or_bytes, Iterable):
        tensors = [item for item in likelihoods_or_bytes if isinstance(item, torch.Tensor)]
        if not tensors:
            return torch.tensor(0.0, dtype=torch.float32) if as_tensor else 0.0
        total_bits: torch.Tensor = sum(
            -torch.sum(torch.log2(torch.clamp(t.float(), min=1e-12))) for t in tensors
        )  # type: ignore[assignment]
        bpp_tensor = total_bits / resolved_num_pixels
        return bpp_tensor if as_tensor else float(bpp_tensor.item())

    raise TypeError(f"Unsupported input type for compute_bpp: {type(likelihoods_or_bytes)}")


# ---------------------------------------------------------------------------
# Checkpoint Persistence & CDF Gate
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    path: Path | str,
    optimizer_harness: DualOptimizerHarness | None = None,
    compile_cdfs: bool = True,
    epoch: int | None = None,
    step: int | None = None,
    metrics: dict[str, Any] | None = None,
    **metadata: Any,
) -> Path:
    r"""Persists model checkpoint to disk enforcing the post-training CDF serialization gate.

    When `compile_cdfs=True`, compiles cumulative distribution function (CDF) tables
    via `model.update(force=True)` and verifies that non-empty `_quantized_cdf` buffers
    exist before writing the checkpoint.

    Writes atomically via a temporary file to avoid corruption during interruption.

    Args:
        model: Compression neural network (e.g. ModelA, ModelB).
        path: Destination checkpoint path (e.g. `checkpoint_epoch10.pt`).
        optimizer_harness: Optional DualOptimizerHarness whose state will be saved.
        compile_cdfs: Whether to enforce post-training CDF table compilation (default True).
        epoch: Optional training epoch counter.
        step: Optional training step/iteration counter.
        metrics: Optional dictionary of evaluation metrics.
        **metadata: Additional arbitrary metadata stored in checkpoint.

    Returns:
        Path: Resolved absolute path to the saved checkpoint file.

    Raises:
        RuntimeError: If CDF compilation fails or empty `_quantized_cdf` buffers remain.
    """
    raw_model = getattr(model, "module", model)

    if compile_cdfs:
        # Step 1: Trigger update on root model if present
        if hasattr(raw_model, "update") and callable(raw_model.update):
            try:
                raw_model.update(force=True)
            except TypeError:
                raw_model.update()

        if hasattr(raw_model, "update_scale_table") and callable(raw_model.update_scale_table):
            raw_model.update_scale_table()

        # Step 2: Ensure submodules with update or update_scale_table are invoked
        for module in raw_model.modules():
            if module is not raw_model:
                if hasattr(module, "update") and callable(module.update):
                    try:
                        module.update(force=True)
                    except TypeError:
                        module.update()
                if hasattr(module, "update_scale_table") and callable(module.update_scale_table):
                    module.update_scale_table()

        # Step 3: Enforce post-training CDF serialization gate on unwrapped model
        state_dict = raw_model.state_dict()
        cdf_buffers = {k: v for k, v in state_dict.items() if k.endswith("_quantized_cdf")}

        # Detect entropy modules dynamically across all submodules
        has_entropy_modules = any(
            isinstance(m, (EntropyBottleneck, GaussianConditional))
            or m.__class__.__name__ in ("EntropyBottleneck", "GaussianConditional")
            or hasattr(m, "_quantized_cdf")
            for m in raw_model.modules()
        )
        if has_entropy_modules and not cdf_buffers:
            raise RuntimeError(
                "Post-training CDF serialization gate failed: model has entropy modules but no "
                "'_quantized_cdf' buffers were registered in state_dict."
            )

        for buf_name, buf in cdf_buffers.items():
            if buf.numel() == 0:
                raise RuntimeError(
                    f"Post-training CDF serialization gate failed: buffer '{buf_name}' is empty (0 elements). "
                    f"Checkpoints must not be saved with uncompiled CDF tables when compile_cdfs=True."
                )
    else:
        state_dict = raw_model.state_dict()

    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_epoch = epoch if epoch is not None else metadata.pop("epoch", None)
    resolved_step = step if step is not None else metadata.pop("step", None)
    resolved_metrics = metrics if metrics is not None else metadata.pop("metrics", {})

    opt_state = optimizer_harness.state_dict() if optimizer_harness is not None else None

    checkpoint_data: dict[str, Any] = {
        "state_dict": state_dict,
        "model_state_dict": state_dict,
        "optimizer_harness": opt_state,
        "optimizer": opt_state,
        "epoch": resolved_epoch,
        "step": resolved_step,
        "metrics": resolved_metrics,
        "metadata": metadata,
    }

    # Atomic persistence: write to temporary sibling file and atomically replace
    tmp_path = out_path.with_suffix(f"{out_path.suffix}.tmp_{os.getpid()}")
    try:
        torch.save(checkpoint_data, tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    return out_path


def load_checkpoint(
    path: Path | str,
    model: nn.Module | None = None,
    optimizer_harness: DualOptimizerHarness | None = None,
    device: str | torch.device | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    r"""Restores model weights, DualOptimizerHarness states, and training metadata from a checkpoint.

    Args:
        path: Checkpoint file path.
        model: Optional neural network module to restore weights into.
        optimizer_harness: Optional DualOptimizerHarness to restore optimizer/scheduler states into.
        device: Device location to map tensors onto (default 'cpu').
        strict: Whether to enforce exact key matching during `load_state_dict`.

    Returns:
        dict: Full loaded checkpoint dictionary.

    Raises:
        FileNotFoundError: If the specified checkpoint file does not exist.
        ValueError: If state dictionary format is corrupted.
    """
    chk_path = Path(path).resolve()
    if not chk_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {chk_path}")

    map_location = device if device is not None else "cpu"
    checkpoint = torch.load(chk_path, map_location=map_location, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format: expected dictionary, got {type(checkpoint)}")

    if model is not None:
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        if not isinstance(state_dict, dict):
            raise ValueError(f"Corrupted checkpoint state_dict: expected dictionary, got {type(state_dict)}")
        model.load_state_dict(state_dict, strict=strict)

    if optimizer_harness is not None:
        opt_state = checkpoint.get("optimizer_harness", checkpoint.get("optimizer", None))
        if opt_state is not None:
            optimizer_harness.load_state_dict(opt_state)

    return checkpoint


# ---------------------------------------------------------------------------
# MLflow Experiment Tracking Lifecycle Engine
# ---------------------------------------------------------------------------


class MLflowTracker:
    """Encapsulates MLflow experiment tracking, run lifecycle, scalar logging, and artifact management.

    Supports:
    - Experiment resolution and automatic directory creation.
    - Context manager run execution (`with tracker.run(...)`).
    - Nested and tagged runs.
    - Parameter, scalar metric (with step counter), and artifact logging.
    - Checkpoint logging via `log_checkpoint`.
    """

    def __init__(
        self,
        experiment_name: str = "saliency-compression",
        tracking_uri: str | Path | None = None,
        run_name: str | None = None,
        tags: dict[str, Any] | None = None,
        artifact_location: str | Path | None = None,
    ) -> None:
        # Enforce filesystem tracking support in MLflow 3.x
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

        if tracking_uri is not None:
            if isinstance(tracking_uri, Path) or (
                isinstance(tracking_uri, str)
                and not tracking_uri.startswith(("http://", "https://", "sqlite://", "file://"))
            ):
                uri_dir = Path(tracking_uri).resolve()
                uri_dir.mkdir(parents=True, exist_ok=True)
                self.tracking_uri = str(uri_dir)
            else:
                self.tracking_uri = str(tracking_uri)
            mlflow.set_tracking_uri(self.tracking_uri)
        else:
            self.tracking_uri = mlflow.get_tracking_uri()

        self.experiment_name = experiment_name
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            art_loc = str(artifact_location) if artifact_location is not None else None
            self.experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=art_loc,
            )
        else:
            self.experiment_id = exp.experiment_id

        self.default_run_name = run_name
        self.default_tags: dict[str, Any] = dict(tags) if tags is not None else {}
        self._run_stack: list[mlflow.ActiveRun] = []

    @property
    def active_run(self) -> mlflow.ActiveRun | None:
        """Returns the currently active MLflow run, or None if inactive."""
        if self._run_stack:
            return self._run_stack[-1]
        return mlflow.active_run()

    @property
    def run_id(self) -> str | None:
        """Returns active MLflow run ID string, or None if inactive."""
        active = self.active_run
        return active.info.run_id if active is not None else None

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, Any] | None = None,
        nested: bool = False,
        description: str | None = None,
    ) -> mlflow.ActiveRun:
        """Starts an MLflow run under the configured experiment."""
        mlflow.set_tracking_uri(self.tracking_uri)
        merged_tags = {**self.default_tags, **(tags or {})}
        resolved_name = run_name or self.default_run_name

        active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=resolved_name,
            tags=merged_tags if merged_tags else None,
            nested=nested,
            description=description,
        )
        self._run_stack.append(active_run)
        return active_run

    def end_run(self, status: str = "FINISHED") -> None:
        """Closes the current active MLflow run with the designated status."""
        if self._run_stack:
            self._run_stack.pop()
        mlflow.end_run(status=status)

    @contextmanager
    def run(
        self,
        run_name: str | None = None,
        tags: dict[str, Any] | None = None,
        nested: bool = False,
        description: str | None = None,
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """Context manager managing the complete run lifecycle with automatic error termination."""
        mlflow.set_tracking_uri(self.tracking_uri)
        active = self.start_run(
            run_name=run_name,
            tags=tags,
            nested=nested,
            description=description,
        )
        try:
            yield active
        except BaseException:
            self.end_run(status="FAILED")
            raise
        else:
            self.end_run(status="FINISHED")

    def __enter__(self) -> MLflowTracker:
        self.start_run()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.end_run(status=status)

    def log_param(self, key: str, value: Any) -> None:
        """Logs a single parameter to the active MLflow run."""
        mlflow.log_param(key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        """Logs multiple parameters to the active MLflow run."""
        mlflow.log_params(params)

    def log_metric(
        self,
        key: str,
        value: float | int | torch.Tensor,
        step: int | None = None,
    ) -> None:
        """Logs a scalar metric with an optional step counter."""
        if isinstance(value, torch.Tensor):
            scalar_val = float(value.item())
        else:
            scalar_val = float(value)
        mlflow.log_metric(key, scalar_val, step=step)

    def log_metrics(
        self,
        metrics: dict[str, float | int | torch.Tensor],
        step: int | None = None,
    ) -> None:
        """Logs multiple scalar metrics with an optional step counter."""
        converted: dict[str, float] = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                converted[k] = float(v.item())
            else:
                converted[k] = float(v)
        mlflow.log_metrics(converted, step=step)

    def log_artifact(
        self,
        local_path: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """Logs a local file or directory as an artifact to MLflow."""
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_artifacts(
        self,
        local_dir: str | Path,
        artifact_path: str | None = None,
    ) -> None:
        """Logs a directory of artifacts to MLflow."""
        mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)

    def set_tag(self, key: str, value: Any) -> None:
        """Sets a single tag on the active MLflow run."""
        mlflow.set_tag(key, value)

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Sets multiple tags on the active MLflow run."""
        mlflow.set_tags(tags)

    def log_checkpoint(
        self,
        model: nn.Module,
        path: Path | str,
        optimizer_harness: DualOptimizerHarness | None = None,
        compile_cdfs: bool = True,
        epoch: int | None = None,
        step: int | None = None,
        metrics: dict[str, Any] | None = None,
        artifact_path: str = "checkpoints",
        **metadata: Any,
    ) -> Path:
        """Saves a checkpoint enforcing the CDF gate and uploads it as an MLflow artifact."""
        saved_path = save_checkpoint(
            model=model,
            path=path,
            optimizer_harness=optimizer_harness,
            compile_cdfs=compile_cdfs,
            epoch=epoch,
            step=step,
            metrics=metrics,
            **metadata,
        )
        self.log_artifact(saved_path, artifact_path=artifact_path)

        if metrics:
            numeric_metrics = {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1)
            }
            if numeric_metrics:
                self.log_metrics(numeric_metrics, step=step)

        return saved_path
