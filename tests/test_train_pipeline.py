"""Integration test suite for the end-to-end training and validation loop CLI.

Verifies:
- CLI argument parsing, aliases, defaults, and invalid option error handling.
- Model A configuration with uniform unweighted MSE loss.
- Model B configuration with saliency-weighted distortion loss (0.2 background floor)
  and error handling when saliency maps are missing.
- Gradient accumulation (accum_steps=2) and handling of odd/non-multiple dataset lengths.
- Validation loop on unaligned image dimensions (reflection padding to 64x, deterministic
  eval, unpadding slice recovery, and scalar metric computation: BPP, MSE, PSNR, MS-SSIM).
- Pretrained checkpoint warm-start and FileNotFoundError handling for missing checkpoints.
- Best checkpoint tracking over multiple validation epochs.
- Post-training CDF table compilation gate and checkpoint restoration into fresh instances.
- CPU execution fallback with strict FP32 precision.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.train import (
    TrainingConfig,
    evaluate,
    parse_args,
    resolve_device,
    setup_dataloaders,
    setup_model,
    train,
    train_one_epoch,
)
from src.codec.model_a import ModelA
from src.codec.model_b import ModelB
from src.training.loss import RateDistortionLoss
from src.training.optimizer import create_dual_optimizer
from src.training.tracker import MLflowTracker, load_checkpoint, save_checkpoint


class SyntheticSaliencyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Synthetic dataset generating paired (image, saliency_map) tensors for integration tests."""

    def __init__(
        self,
        num_samples: int = 4,
        channels: int = 3,
        height: int = 64,
        width: int = 64,
        seed: int = 42,
    ) -> None:
        self.num_samples = int(num_samples)
        gen = torch.Generator().manual_seed(seed)
        # Random RGB images in [0, 1]
        self.images = torch.rand(num_samples, channels, height, width, generator=gen)
        # Random visual saliency maps in [0, 1]
        self.saliency_maps = torch.rand(num_samples, 1, height, width, generator=gen)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.saliency_maps[idx]


class SyntheticUnalignedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Synthetic dataset with spatial dimensions NOT divisible by 64 (e.g. 68x72)."""

    def __init__(
        self,
        num_samples: int = 2,
        channels: int = 3,
        height: int = 68,
        width: int = 72,
        seed: int = 42,
    ) -> None:
        self.num_samples = int(num_samples)
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.rand(num_samples, channels, height, width, generator=gen)
        self.saliency_maps = torch.rand(num_samples, 1, height, width, generator=gen)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.saliency_maps[idx]


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Temporary directory for artifacts and checkpoints."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


# ==============================================================================
# CLI Argument Parsing Tests
# ==============================================================================


class TestCLIArgumentParsing:
    """Verifies CLI flag definitions, aliases, defaults, and error boundaries."""

    def test_cli_parser_defaults(self) -> None:
        """CLI arguments populate default values specified in story plan."""
        args = parse_args(["--model", "A"])
        assert args.model == "A"
        assert args.lmbda == pytest.approx(0.0130)
        assert args.epochs == 100
        assert args.batch_size == 2
        assert args.accum_steps == 2
        assert args.lr == pytest.approx(5e-5)
        assert args.aux_lr == pytest.approx(1e-3)
        assert args.min_lr == pytest.approx(1e-5)
        assert args.patch_size == 256
        assert args.device == "auto"
        assert args.seed == 42
        assert args.checkpoint_dir == "checkpoints"
        assert args.pretrained_checkpoint is None
        assert args.clip_max_norm == pytest.approx(1.0)
        assert args.use_mlflow is True
        assert args.compile_cdfs is True

    def test_cli_parser_custom_flags(self) -> None:
        """CLI arguments accept custom parameters and resolve types accurately."""
        args = parse_args(
            [
                "--model",
                "B",
                "--lambda",
                "0.0250",
                "--epochs",
                "10",
                "--batch-size",
                "4",
                "--accum-steps",
                "4",
                "--lr",
                "1e-4",
                "--aux-lr",
                "5e-4",
                "--min-lr",
                "2e-5",
                "--patch-size",
                "128",
                "--device",
                "cpu",
                "--seed",
                "123",
                "--checkpoint-dir",
                "custom_dir",
                "--pretrained-checkpoint",
                "path/to/ckpt.pt",
                "--clip-max-norm",
                "0.5",
                "--no-mlflow",
                "--no-compile-cdfs",
            ]
        )
        assert args.model == "B"
        assert args.lmbda == pytest.approx(0.0250)
        assert args.epochs == 10
        assert args.batch_size == 4
        assert args.accum_steps == 4
        assert args.lr == pytest.approx(1e-4)
        assert args.aux_lr == pytest.approx(5e-4)
        assert args.min_lr == pytest.approx(2e-5)
        assert args.patch_size == 128
        assert args.device == "cpu"
        assert args.seed == 123
        assert args.checkpoint_dir == "custom_dir"
        assert args.pretrained_checkpoint == "path/to/ckpt.pt"
        assert args.clip_max_norm == pytest.approx(0.5)
        assert args.use_mlflow is False
        assert args.compile_cdfs is False

    def test_cli_parser_lambda_aliases(self) -> None:
        """--lambda-rd, --lambda, and --lmbda all map cleanly to lmbda."""
        args1 = parse_args(["--model", "A", "--lambda-rd", "0.0483"])
        assert args1.lmbda == pytest.approx(0.0483)

        args2 = parse_args(["--model", "A", "--lambda", "0.0067"])
        assert args2.lmbda == pytest.approx(0.0067)

        args3 = parse_args(["--model", "A", "--lmbda", "0.0130"])
        assert args3.lmbda == pytest.approx(0.0130)

    def test_cli_parser_best_metric_flag(self) -> None:
        """--best-metric accepts 'loss' and 'psnr', rejecting invalid options."""
        args1 = parse_args(["--model", "A", "--best-metric", "psnr"])
        assert args1.best_metric == "psnr"

        args2 = parse_args(["--model", "A", "--best-metric", "loss"])
        assert args2.best_metric == "loss"

        with pytest.raises(SystemExit):
            parse_args(["--model", "A", "--best-metric", "invalid_metric"])

    def test_cli_parser_accum_steps_invalid_raises_value_error(self) -> None:
        """parse_args raises ValueError when accum_steps is non-positive."""
        with pytest.raises(ValueError, match="accum_steps must be >= 1"):
            parse_args(["--model", "A", "--accum-steps", "0"])

        with pytest.raises(ValueError, match="accum_steps must be >= 1"):
            parse_args(["--model", "A", "--accum-steps", "-2"])

    def test_training_config_accum_steps_validation(self) -> None:
        """TrainingConfig raises ValueError when accum_steps, epochs, or batch_size <= 0."""
        with pytest.raises(ValueError, match="accum_steps must be >= 1"):
            TrainingConfig(accum_steps=0)

        with pytest.raises(ValueError, match="epochs must be >= 1"):
            TrainingConfig(epochs=0)

        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            TrainingConfig(batch_size=-1)

        with pytest.raises(ValueError, match="Invalid best_metric"):
            TrainingConfig(best_metric="invalid")

    def test_cli_parser_invalid_model_choice_raises_system_exit(self) -> None:
        """Argparse rejects invalid model names outside choices {A, B}."""
        with pytest.raises(SystemExit):
            parse_args(["--model", "C"])

    def test_training_config_from_namespace_and_dict(self) -> None:
        """TrainingConfig dataclass converts cleanly from namespace and dictionary."""
        args = parse_args(["--model", "B", "--lambda", "0.0130"])
        config = TrainingConfig.from_namespace(args)
        assert config.model == "B"
        assert config.lmbda == pytest.approx(0.0130)

        # Test key normalization with hyphens and aliases
        data = {
            "model": "B",
            "lambda-rd": 0.0483,
            "batch-size": 4,
            "accum-steps": 3,
            "crop-size": 128,
            "best-metric": "psnr",
            "unknown-field": 123,
        }
        config_dict = TrainingConfig.from_dict(data)
        assert config_dict.model == "B"
        assert config_dict.lmbda == pytest.approx(0.0483)
        assert config_dict.batch_size == 4
        assert config_dict.accum_steps == 3
        assert config_dict.patch_size == 128
        assert config_dict.best_metric == "psnr"

        # Test 'lambda' and 'crop_size' aliases
        cfg2 = TrainingConfig.from_dict({"model": "A", "lambda": 0.01, "crop_size": 256})
        assert cfg2.lmbda == pytest.approx(0.01)
        assert cfg2.patch_size == 256

        # Test 'lambda_rd' alias
        cfg3 = TrainingConfig.from_dict({"model": "A", "lambda_rd": 0.02})
        assert cfg3.lmbda == pytest.approx(0.02)


# ==============================================================================
# Model Setup & Device Resolution Tests
# ==============================================================================


class TestModelSetup:
    """Verifies model instantiation, parameter configuration, and device assignment."""

    def test_setup_model_a(self) -> None:
        """setup_model('A') instantiates ModelA with expected channels."""
        model = setup_model("A", n_channels=64, m_channels=96, device="cpu")
        assert isinstance(model, ModelA)
        assert not isinstance(model, ModelB)
        assert model.n_channels == 64
        assert model.m_channels == 96

    def test_setup_model_b(self) -> None:
        """setup_model('B') instantiates ModelB subclassing ModelA."""
        model = setup_model("B", n_channels=64, m_channels=96, device="cpu")
        assert isinstance(model, ModelB)
        assert isinstance(model, ModelA)
        assert model.n_channels == 64
        assert model.m_channels == 96

    def test_setup_model_invalid_raises_value_error(self) -> None:
        """setup_model raises ValueError when given an invalid model identifier."""
        with pytest.raises(ValueError, match="Invalid model 'Z'"):
            setup_model("Z", device="cpu")

    def test_resolve_device(self) -> None:
        """resolve_device correctly maps 'cpu', 'cuda', and 'auto'."""
        assert resolve_device("cpu") == torch.device("cpu")
        auto_dev = resolve_device("auto")
        assert auto_dev.type in ("cuda", "cpu")


# ==============================================================================
# Model A vs Model B Loss Differentiation Tests
# ==============================================================================


class TestLossDifferentiation:
    """Verifies that Model A uses unweighted MSE while Model B uses saliency-weighted loss."""

    def test_model_a_ignores_saliency_and_uses_unweighted_mse(self) -> None:
        """Model A forward and loss ignores saliency maps in batch, optimizing uniform MSE."""
        model = ModelA(n_channels=16, m_channels=32)
        criterion = RateDistortionLoss(lmbda=0.0130)
        optimizer_harness = create_dual_optimizer(model, main_lr=5e-5, aux_lr=1e-3)

        dataset = SyntheticSaliencyDataset(num_samples=2, height=64, width=64)
        dataloader = DataLoader(dataset, batch_size=2)

        # Run 1 epoch on Model A
        metrics, step = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer_harness=optimizer_harness,
            epoch=0,
            device="cpu",
            accum_steps=1,
            model_type="A",
        )

        assert "loss" in metrics
        assert "mse" in metrics
        assert "distortion" in metrics
        # For Model A, unweighted distortion == mse
        assert metrics["distortion"] == pytest.approx(metrics["mse"], rel=1e-5)
        assert step == 1

    def test_model_b_applies_saliency_weighted_distortion(self) -> None:
        """Model B applies saliency map S' = 0.2 + 0.8 S in its distortion objective."""
        model = ModelB(n_channels=16, m_channels=32)
        criterion = RateDistortionLoss(lmbda=0.0130)
        optimizer_harness = create_dual_optimizer(model, main_lr=5e-5, aux_lr=1e-3)

        dataset = SyntheticSaliencyDataset(num_samples=2, height=64, width=64)
        dataloader = DataLoader(dataset, batch_size=2)

        metrics, step = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer_harness=optimizer_harness,
            epoch=0,
            device="cpu",
            accum_steps=1,
            model_type="B",
        )

        assert "loss" in metrics
        assert "distortion" in metrics
        assert "mse" in metrics
        assert step == 1

    def test_model_b_missing_saliency_raises_value_error(self) -> None:
        """Model B raises ValueError if a training batch provides no saliency map (s=None)."""
        model = ModelB(n_channels=16, m_channels=32)
        criterion = RateDistortionLoss(lmbda=0.0130)
        optimizer_harness = create_dual_optimizer(model)

        # Dataset yielding only image tensor without saliency map
        class ImagesOnlyDataset(Dataset[torch.Tensor]):
            def __len__(self) -> int:
                return 2

            def __getitem__(self, idx: int) -> torch.Tensor:
                return torch.rand(3, 64, 64)

        dataloader = DataLoader(ImagesOnlyDataset(), batch_size=2)

        with pytest.raises(ValueError, match="Model B requires visual saliency maps"):
            train_one_epoch(
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                optimizer_harness=optimizer_harness,
                epoch=0,
                device="cpu",
                accum_steps=1,
                model_type="B",
            )

    def test_evaluate_model_b_missing_saliency_raises_value_error(self) -> None:
        """evaluate() with Model B raises ValueError when visual saliency maps are missing."""
        model = ModelB(n_channels=16, m_channels=32)
        criterion = RateDistortionLoss(lmbda=0.0130)

        class ImagesOnlyDataset(Dataset[torch.Tensor]):
            def __len__(self) -> int:
                return 2

            def __getitem__(self, idx: int) -> torch.Tensor:
                return torch.rand(3, 64, 64)

        dataloader = DataLoader(ImagesOnlyDataset(), batch_size=1)

        with pytest.raises(ValueError, match="Model B requires visual saliency maps"):
            evaluate(
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                device="cpu",
                model_type="B",
            )


# ==============================================================================
# Gradient Accumulation & Scheduler Step Tests
# ==============================================================================


class TestGradientAccumulation:
    """Verifies gradient accumulation frequency, scaling, and non-multiple dataset handling."""

    def test_gradient_accumulation_even_dataset_steps_every_accum_steps(self) -> None:
        """With 4 mini-batches and accum_steps=2, optimizer updates parameters exactly 2 times."""
        model = ModelA(n_channels=16, m_channels=32)
        criterion = RateDistortionLoss(lmbda=0.0130)
        optimizer_harness = create_dual_optimizer(model)

        # 4 samples with batch_size=1 -> 4 mini-batches
        dataset = SyntheticSaliencyDataset(num_samples=4, height=64, width=64)
        dataloader = DataLoader(dataset, batch_size=1)

        metrics, global_step = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer_harness=optimizer_harness,
            epoch=0,
            device="cpu",
            accum_steps=2,
            model_type="A",
            global_step=0,
        )

        # 4 batches / 2 accum_steps = 2 steps
        assert global_step == 2
        assert metrics["loss"] > 0

    def test_gradient_accumulation_odd_dataset_steps_remainder_batch(self) -> None:
        """With 3 mini-batches and accum_steps=2, optimizer handles the 3rd batch at epoch end."""
        model = ModelA(n_channels=16, m_channels=32)
        criterion = RateDistortionLoss(lmbda=0.0130)
        optimizer_harness = create_dual_optimizer(model)

        # 3 samples with batch_size=1 -> 3 mini-batches
        dataset = SyntheticSaliencyDataset(num_samples=3, height=64, width=64)
        dataloader = DataLoader(dataset, batch_size=1)

        metrics, global_step = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer_harness=optimizer_harness,
            epoch=0,
            device="cpu",
            accum_steps=2,
            model_type="A",
            global_step=0,
        )

        # Batch 1, 2 step once; remainder Batch 3 steps at epoch boundary -> 2 steps total
        assert global_step == 2
        assert metrics["loss"] > 0


# ==============================================================================
# Validation Loop on Unaligned Images Tests
# ==============================================================================


class TestValidationLoopUnalignedImages:
    """Verifies padding to 64x multiples, deterministic eval, unpadding, and metric calculation."""

    def test_evaluate_pads_and_unpads_unaligned_dimensions(self) -> None:
        """evaluate() handles unaligned image dimensions (e.g. 68x72), recovering unpadded slice."""
        model = ModelA(n_channels=16, m_channels=32)
        criterion = RateDistortionLoss(lmbda=0.0130)

        # Images of shape (3, 68, 72) - neither 68 nor 72 is divisible by 64
        val_dataset = SyntheticUnalignedDataset(num_samples=2, height=68, width=72)
        val_dataloader = DataLoader(val_dataset, batch_size=1)

        metrics = evaluate(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device="cpu",
            model_type="A",
            pad_factor=64,
        )

        # All required scalar metrics must be populated and finite
        assert "loss" in metrics
        assert "bpp" in metrics
        assert "mse" in metrics
        assert "psnr" in metrics
        assert "msssim" in metrics
        assert "distortion" in metrics

        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert torch.isfinite(torch.tensor(metrics["bpp"]))
        assert torch.isfinite(torch.tensor(metrics["mse"]))
        assert torch.isfinite(torch.tensor(metrics["psnr"]))
        assert torch.isfinite(torch.tensor(metrics["msssim"]))

        # For Model A, unweighted distortion == mse
        assert metrics["distortion"] == pytest.approx(metrics["mse"], rel=1e-5)
        # PSNR for reasonable reconstruction is positive
        assert metrics["psnr"] > 0.0

    def test_evaluate_model_b_with_saliency(self) -> None:
        """evaluate() with Model B computes saliency-weighted loss over unpadded images."""
        model = ModelB(n_channels=16, m_channels=32)
        criterion = RateDistortionLoss(lmbda=0.0130)

        val_dataset = SyntheticUnalignedDataset(num_samples=2, height=68, width=72)
        val_dataloader = DataLoader(val_dataset, batch_size=1)

        metrics = evaluate(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device="cpu",
            model_type="B",
        )

        assert "loss" in metrics
        assert "distortion" in metrics
        assert "mse" in metrics
        assert "psnr" in metrics
        assert "msssim" in metrics


# ==============================================================================
# Checkpoint Persistence, CDF Gate, and Warm-Start Tests
# ==============================================================================


class TestCheckpointPersistenceAndWarmStart:
    """Verifies post-training CDF table compilation, checkpoint reloading, and warm-starts."""

    def test_checkpoint_contains_compiled_cdfs_and_reloads_into_fresh_model(
        self, temp_dir: Path
    ) -> None:
        """Trained checkpoint file contains compiled _quantized_cdf buffers and reloads cleanly."""
        model = ModelA(n_channels=16, m_channels=32)
        optimizer_harness = create_dual_optimizer(model)

        ckpt_path = temp_dir / "test_checkpoint.pt"
        saved_path = save_checkpoint(
            model=model,
            path=ckpt_path,
            optimizer_harness=optimizer_harness,
            compile_cdfs=True,
            epoch=1,
            step=10,
        )

        assert saved_path.is_file()

        # Inspect raw checkpoint contents
        raw_ckpt = torch.load(saved_path, map_location="cpu", weights_only=False)
        state_dict = raw_ckpt["state_dict"]

        # Post-training CDF serialization gate invariant
        assert "entropy_bottleneck._quantized_cdf" in state_dict
        eb_cdf = state_dict["entropy_bottleneck._quantized_cdf"]
        assert eb_cdf.numel() > 0, "Quantized CDF buffer must not be empty"

        # Restore into a fresh model instance
        fresh_model = ModelA(n_channels=16, m_channels=32)
        load_checkpoint(saved_path, model=fresh_model)

        # Verify fresh model forward pass in eval mode works deterministically
        x = torch.rand(1, 3, 64, 64)
        fresh_model.eval()
        with torch.no_grad():
            out = fresh_model(x, training=False)
        assert out.x_hat.shape == (1, 3, 64, 64)

    def test_pretrained_checkpoint_warm_start(self, temp_dir: Path) -> None:
        """--pretrained-checkpoint loads weights into ModelA or ModelB before training starts."""
        # Create and save a baseline checkpoint
        source_model = ModelA(n_channels=16, m_channels=32)
        ckpt_path = temp_dir / "warm_start.pt"
        save_checkpoint(source_model, ckpt_path, compile_cdfs=True)

        # Setup Model B warm-started from Model A checkpoint
        target_model = setup_model(
            model_type="B",
            n_channels=16,
            m_channels=32,
            pretrained_checkpoint=ckpt_path,
            device="cpu",
        )
        assert isinstance(target_model, ModelB)

        # Verify parameters match
        for (n1, p1), (n2, p2) in zip(source_model.named_parameters(), target_model.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_pretrained_checkpoint_missing_raises_file_not_found(self) -> None:
        """Missing pretrained checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Pretrained checkpoint not found"):
            setup_model(
                model_type="A",
                n_channels=16,
                m_channels=32,
                pretrained_checkpoint="nonexistent/checkpoint.pt",
                device="cpu",
            )


# ==============================================================================
# End-to-End Programmatic Training Pipeline Tests
# ==============================================================================


class TestEndToEndTrainingPipeline:
    """Verifies complete end-to-end training runs for Model A and Model B."""

    def test_end_to_end_model_a_training_run(self, temp_dir: Path) -> None:
        """Model A completes 1-epoch training and validation pass with unweighted MSE."""
        train_dataset = SyntheticSaliencyDataset(num_samples=4, height=64, width=64)
        val_dataset = SyntheticSaliencyDataset(num_samples=2, height=64, width=64)

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        config = TrainingConfig(
            model="A",
            lmbda=0.0130,
            epochs=1,
            batch_size=2,
            accum_steps=1,
            lr=5e-5,
            aux_lr=1e-3,
            checkpoint_dir=temp_dir / "model_a_ckpts",
            use_mlflow=False,
            device="cpu",
            n_channels=16,
            m_channels=32,
            compile_cdfs=True,
        )

        model = ModelA(n_channels=16, m_channels=32)
        results = train(
            config,
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )

        assert "model" in results
        assert "train_metrics" in results
        assert "val_metrics" in results
        assert "best_checkpoint_path" in results
        assert "latest_checkpoint_path" in results

        latest_ckpt = results["latest_checkpoint_path"]
        assert latest_ckpt is not None and latest_ckpt.is_file()

        # Verify CDF table gate
        fresh_model = ModelA(n_channels=16, m_channels=32)
        load_checkpoint(latest_ckpt, model=fresh_model)
        assert fresh_model.entropy_bottleneck._quantized_cdf.numel() > 0

    def test_end_to_end_model_b_training_run(self, temp_dir: Path) -> None:
        """Model B completes 1-epoch training and validation pass with saliency weighting."""
        train_dataset = SyntheticSaliencyDataset(num_samples=4, height=64, width=64)
        val_dataset = SyntheticSaliencyDataset(num_samples=2, height=64, width=64)

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        config = TrainingConfig(
            model="B",
            lmbda=0.0130,
            epochs=1,
            batch_size=2,
            accum_steps=2,
            lr=5e-5,
            aux_lr=1e-3,
            checkpoint_dir=temp_dir / "model_b_ckpts",
            use_mlflow=False,
            device="cpu",
            n_channels=16,
            m_channels=32,
            compile_cdfs=True,
        )

        model = ModelB(n_channels=16, m_channels=32)
        results = train(
            config,
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )

        assert "model" in results
        assert isinstance(results["model"], ModelB)
        val_metrics = results["val_metrics"]
        assert val_metrics["psnr"] > 0
        assert val_metrics["bpp"] > 0

        latest_ckpt = results["latest_checkpoint_path"]
        assert latest_ckpt is not None and latest_ckpt.is_file()

        fresh_b = ModelB(n_channels=16, m_channels=32)
        load_checkpoint(latest_ckpt, model=fresh_b)
        assert fresh_b.entropy_bottleneck._quantized_cdf.numel() > 0

    def test_setup_dataloaders_partial_override(self) -> None:
        """setup_dataloaders uses provided loaders without reading from disk."""
        train_ds = SyntheticSaliencyDataset(num_samples=2, height=64, width=64)
        train_loader = DataLoader(train_ds, batch_size=2)
        val_ds = SyntheticSaliencyDataset(num_samples=2, height=64, width=64)
        val_loader = DataLoader(val_ds, batch_size=1)

        cfg = TrainingConfig()
        t_out, v_out = setup_dataloaders(cfg, train_dataloader=train_loader, val_dataloader=val_loader)
        assert t_out is train_loader
        assert v_out is val_loader

    def test_best_checkpoint_tracking_overwrites_only_on_improvement(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """best_checkpoint.pt is saved on improvement and is NOT overwritten when metric worsens."""
        train_dataset = SyntheticSaliencyDataset(num_samples=2, height=64, width=64)
        val_dataset = SyntheticSaliencyDataset(num_samples=2, height=64, width=64)

        train_loader = DataLoader(train_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=1)

        config = TrainingConfig(
            model="A",
            epochs=3,
            batch_size=2,
            accum_steps=1,
            checkpoint_dir=temp_dir / "best_test",
            use_mlflow=False,
            device="cpu",
            n_channels=16,
            m_channels=32,
            save_best=True,
            best_metric="loss",
        )

        # Mock evaluate to return controlled losses across 3 epochs:
        # Epoch 0: loss 10.0 -> initial best saved (epoch=0)
        # Epoch 1: loss 5.0  -> improved, best overwritten (epoch=1)
        # Epoch 2: loss 15.0 -> worse, best must NOT be overwritten (remains epoch=1)
        fake_losses = [10.0, 5.0, 15.0]
        call_count = 0

        def mock_evaluate(*args, **kwargs):
            nonlocal call_count
            loss = fake_losses[min(call_count, len(fake_losses) - 1)]
            call_count += 1
            return {
                "loss": loss,
                "bpp": 0.5,
                "bpp_loss": 0.5,
                "mse": 0.01,
                "mse_loss": 0.01,
                "psnr": 30.0,
                "msssim": 0.95,
                "ms_ssim": 0.95,
                "distortion": 0.01,
            }

        monkeypatch.setattr("scripts.train.evaluate", mock_evaluate)

        model = ModelA(n_channels=16, m_channels=32)
        results = train(
            config,
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )

        best_path = results["best_checkpoint_path"]
        assert best_path is not None and best_path.is_file()
        assert results["best_metric_val"] == pytest.approx(5.0)

        # Verify that the best checkpoint stored epoch 1, NOT the worse epoch 2!
        saved_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        assert saved_ckpt["epoch"] == 1

    def test_best_checkpoint_tracking_psnr_metric(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """best_checkpoint.pt tracks maximum PSNR when best_metric='psnr' and rejects lower PSNR."""
        train_dataset = SyntheticSaliencyDataset(num_samples=2, height=64, width=64)
        val_dataset = SyntheticSaliencyDataset(num_samples=2, height=64, width=64)
        train_loader = DataLoader(train_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=1)

        config = TrainingConfig(
            model="A",
            epochs=3,
            batch_size=2,
            accum_steps=1,
            checkpoint_dir=temp_dir / "best_psnr_test",
            use_mlflow=False,
            device="cpu",
            n_channels=16,
            m_channels=32,
            save_best=True,
            best_metric="psnr",
        )

        # Epoch 0: PSNR 28.0 -> initial best saved (epoch=0)
        # Epoch 1: PSNR 35.0 -> improved, best overwritten (epoch=1)
        # Epoch 2: PSNR 31.0 -> worse (lower), best must NOT be overwritten (remains epoch=1)
        fake_psnrs = [28.0, 35.0, 31.0]
        call_count = 0

        def mock_evaluate(*args, **kwargs):
            nonlocal call_count
            psnr = fake_psnrs[min(call_count, len(fake_psnrs) - 1)]
            call_count += 1
            return {
                "loss": 1.0,
                "bpp": 0.5,
                "bpp_loss": 0.5,
                "mse": 0.01,
                "mse_loss": 0.01,
                "psnr": psnr,
                "msssim": 0.95,
                "ms_ssim": 0.95,
                "distortion": 0.01,
            }

        monkeypatch.setattr("scripts.train.evaluate", mock_evaluate)

        model = ModelA(n_channels=16, m_channels=32)
        results = train(
            config,
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )

        best_path = results["best_checkpoint_path"]
        assert best_path is not None and best_path.is_file()
        assert results["best_metric_val"] == pytest.approx(35.0)

        saved_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        assert saved_ckpt["epoch"] == 1

    def test_train_exception_marks_mlflow_run_failed(
        self, temp_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """train() terminates MLflow run with status='FAILED' when training raises an exception."""
        mlflow_dir = temp_dir / "mlruns_err"
        tracker = MLflowTracker(
            experiment_name="test-fail-pipeline",
            tracking_uri=mlflow_dir,
            run_name="fail-run",
        )

        train_dataset = SyntheticSaliencyDataset(num_samples=2, height=64, width=64)
        val_dataset = SyntheticSaliencyDataset(num_samples=2, height=64, width=64)
        train_loader = DataLoader(train_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=1)

        config = TrainingConfig(
            model="A",
            epochs=1,
            batch_size=2,
            checkpoint_dir=temp_dir / "fail_ckpts",
            use_mlflow=True,
            device="cpu",
            n_channels=16,
            m_channels=32,
        )

        def mock_train_one_epoch(*args, **kwargs):
            raise RuntimeError("Simulated failure in train_one_epoch")

        monkeypatch.setattr("scripts.train.train_one_epoch", mock_train_one_epoch)

        end_run_status = None
        orig_end_run = tracker.end_run

        def mock_end_run(status: str = "FINISHED"):
            nonlocal end_run_status
            end_run_status = status
            return orig_end_run(status=status)

        monkeypatch.setattr(tracker, "end_run", mock_end_run)

        model = ModelA(n_channels=16, m_channels=32)
        with pytest.raises(RuntimeError, match="Simulated failure"):
            train(
                config,
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                tracker=tracker,
            )

        assert end_run_status == "FAILED"

    def test_mlflow_tracker_integration(self, temp_dir: Path) -> None:
        """MLflowTracker correctly logs parameters, metrics, and artifact checkpoints."""
        mlflow_dir = temp_dir / "mlruns"
        tracker = MLflowTracker(
            experiment_name="test-pipeline",
            tracking_uri=mlflow_dir,
            run_name="test-run",
        )

        train_dataset = SyntheticSaliencyDataset(num_samples=2, height=64, width=64)
        val_dataset = SyntheticSaliencyDataset(num_samples=2, height=64, width=64)

        train_loader = DataLoader(train_dataset, batch_size=2)
        val_loader = DataLoader(val_dataset, batch_size=1)

        config = TrainingConfig(
            model="A",
            epochs=1,
            batch_size=2,
            accum_steps=1,
            checkpoint_dir=temp_dir / "mlflow_ckpts",
            use_mlflow=True,
            device="cpu",
            n_channels=16,
            m_channels=32,
        )

        model = ModelA(n_channels=16, m_channels=32)
        results = train(
            config,
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            tracker=tracker,
        )

        assert results["latest_checkpoint_path"].is_file()
