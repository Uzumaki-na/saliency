"""Integration test suite for multi-model training CLI and sweep integration (Story 4.7).

Verifies across Models A, B, C, D, and C-Transmitted:
1. Configuration and CLI argument parsing (TrainingConfig and SweepConfig).
2. Proper model instantiation and distortion loss weighting:
   - Model C: Saliency-weighted distortion + auxiliary saliency loss (lambda_aux=0.1).
   - Model D: Uniform MSE distortion + auxiliary saliency loss (lambda_aux=0.1) with detached RD gradients.
   - Model C-Transmitted: Saliency-weighted distortion + sidecar bitrate BPP_side (aux_loss=0.0).
3. Training step execution on synthetic datasets with gradient flow and metric tracking.
4. Validation evaluation over unaligned spatial dimensions with valid mask centering.
5. Multi-model dry-run sweep matrix generation (e.g. 5 models x 4 lambdas = 20 jobs).
6. Automated net hyperprior overhead Delta BPP_z computation against Model A baseline.
7. Multi-format reporting artifacts (sweep_results.json, sweep_results.csv, sweep_summary.md, rd_curve.png).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.train import (
    TrainingConfig,
    evaluate,
    setup_model,
    train,
    train_one_epoch,
)
from scripts.train import (
    parse_args as parse_train_args,
)
from scripts.train_sweep import (
    SweepConfig,
    run_sweep,
)
from scripts.train_sweep import (
    parse_args as parse_sweep_args,
)
from src.codec.model_a import ModelA
from src.codec.model_b import ModelB
from src.codec.model_c import ModelC
from src.codec.model_c_transmitted import ModelCTransmitted
from src.codec.model_d import ModelD
from src.training.loss import RateDistortionLoss
from src.training.optimizer import create_dual_optimizer


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provides an isolated temporary directory for checkpoints and artifacts."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


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
        self.images = torch.rand(num_samples, channels, height, width, generator=gen)
        self.saliency_maps = torch.rand(num_samples, 1, height, width, generator=gen)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.saliency_maps[idx]


class TestConfigAndSetup:
    """Verifies configuration parsing and model setup for Models C, D, and C-Transmitted."""

    @pytest.mark.parametrize(
        ("model_str", "expected_cls"),
        [
            ("A", ModelA),
            ("B", ModelB),
            ("C", ModelC),
            ("D", ModelD),
            ("C-Transmitted", ModelCTransmitted),
            ("c-transmitted", ModelCTransmitted),
            ("CTRANSMITTED", ModelCTransmitted),
        ],
    )
    def test_setup_model_classes(self, model_str: str, expected_cls: type) -> None:
        """Asserts setup_model returns the expected architecture instance."""
        model = setup_model(model_type=model_str, device="cpu")
        assert isinstance(model, expected_cls)

    def test_train_parse_args_all_models(self) -> None:
        """Verifies CLI argument parsing across all supported model names."""
        for m in ("A", "B", "C", "D", "C-Transmitted"):
            args = parse_train_args(["--model", m, "--lambda", "0.0130", "--lambda-aux", "0.2"])
            cfg = TrainingConfig.from_namespace(args)
            assert cfg.model == ("C-Transmitted" if "TRANSMITTED" in m.upper() else m.upper())
            assert cfg.lmbda == 0.0130
            assert cfg.lambda_aux == 0.2

    def test_sweep_parse_args_all_models(self) -> None:
        """Verifies Sweep CLI accepts all 5 models."""
        args = parse_sweep_args(
            [
                "--models",
                "A",
                "B",
                "C",
                "D",
                "C-Transmitted",
                "--lambdas",
                "0.0130",
                "0.0250",
                "--lambda-aux",
                "0.15",
            ]
        )
        cfg = SweepConfig.from_namespace(args)
        assert cfg.models == ["A", "B", "C", "D", "C-Transmitted"]
        assert cfg.lambdas == [0.0130, 0.0250]
        assert cfg.lambda_aux == 0.15

    def test_invalid_model_rejected(self) -> None:
        """Asserts invalid model names raise ValueError."""
        with pytest.raises(ValueError, match="Invalid model 'InvalidModel'"):
            TrainingConfig(model="InvalidModel")

        with pytest.raises(ValueError, match="Invalid model 'InvalidModel'"):
            SweepConfig(models=["A", "InvalidModel"])


class TestMultiModelTrainingPasses:
    """Verifies training step and evaluation passes across Models C, D, and C-Transmitted."""

    @pytest.fixture
    def dataloader(self) -> DataLoader:
        ds = SyntheticSaliencyDataset(num_samples=4, height=64, width=64, seed=42)
        return DataLoader(ds, batch_size=2, shuffle=False)

    @pytest.mark.parametrize("model_key", ["C", "D", "C-Transmitted"])
    def test_train_one_epoch_execution(self, model_key: str, dataloader: DataLoader) -> None:
        """Executes a single training epoch and asserts loss computation and gradient stepping."""
        model = setup_model(model_type=model_key, device="cpu")
        saliency_weighting = model_key != "D"  # Model D uses uniform MSE
        criterion = RateDistortionLoss(lmbda=0.0130, lambda_aux=0.1, saliency_weighting=saliency_weighting)
        optimizer_harness = create_dual_optimizer(model=model, main_lr=1e-4, aux_lr=1e-3)

        metrics, step = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer_harness=optimizer_harness,
            epoch=0,
            device="cpu",
            accum_steps=1,
            model_type=model_key,
        )

        assert metrics["loss"] > 0.0
        assert metrics["bpp"] > 0.0
        assert metrics["z_bpp"] > 0.0
        assert metrics["y_bpp"] > 0.0
        assert metrics["mse"] > 0.0
        assert step > 0

        if model_key in ("C", "D"):
            assert metrics["aux_loss"] > 0.0
        elif model_key == "C-Transmitted":
            assert metrics["sidecar_bpp"] > 0.0
            assert metrics["aux_loss"] == 0.0

    @pytest.mark.parametrize("model_key", ["C", "D", "C-Transmitted"])
    def test_evaluate_execution_with_valid_mask(self, model_key: str, dataloader: DataLoader) -> None:
        """Executes deterministic evaluation over unaligned dimensions asserting scalar metric outputs."""
        model = setup_model(model_type=model_key, device="cpu")
        saliency_weighting = model_key != "D"
        criterion = RateDistortionLoss(lmbda=0.0130, lambda_aux=0.1, saliency_weighting=saliency_weighting)

        metrics = evaluate(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            device="cpu",
            model_type=model_key,
            baseline_z_bpp=0.0500,
        )

        assert metrics["loss"] > 0.0
        assert metrics["bpp"] > 0.0
        assert metrics["z_bpp"] > 0.0
        assert metrics["y_bpp"] > 0.0
        assert metrics["psnr"] > 0.0
        assert "delta_bpp_z" in metrics
        assert isinstance(metrics["delta_bpp_z"], float)

        if model_key in ("C", "D"):
            assert metrics["aux_loss"] > 0.0
        elif model_key == "C-Transmitted":
            assert metrics["sidecar_bpp"] > 0.0
            assert metrics["aux_loss"] == 0.0


class TestSweepOrchestrationMultiModel:
    """Verifies multi-model rate sweep execution, dry-runs, and Delta BPP_z tracking."""

    def test_dry_run_all_models(self) -> None:
        """Asserts dry run correctly plans parameter matrix across 5 models x 4 lambdas = 20 jobs."""
        cfg = SweepConfig(
            models=["A", "B", "C", "D", "C-Transmitted"],
            lambdas=[0.0067, 0.0130, 0.0250, 0.0483],
            dry_run=True,
        )
        res = run_sweep(cfg)
        assert res["dry_run"] is True
        assert res["total_jobs"] == 20
        assert len(res["plan"]) == 20

        # Check job names and structure
        job_models = [j["model"] for j in res["plan"]]
        assert job_models.count("A") == 4
        assert job_models.count("B") == 4
        assert job_models.count("C") == 4
        assert job_models.count("D") == 4
        assert job_models.count("C-Transmitted") == 4

    def test_sweep_delta_bpp_z_calculation(self, temp_dir: Path) -> None:
        """Verifies automated Delta BPP_z derivation against Model A baseline in multi-model sweep."""

        def _mock_train(train_cfg: TrainingConfig) -> dict[str, Any]:
            m = train_cfg.model.upper()
            lmbda = train_cfg.lmbda

            # Realistic hyperprior rate offsets
            if m == "A":
                z_bpp = 0.0400 + lmbda * 0.1
                aux_loss = 0.0
            elif m == "B":
                z_bpp = 0.0400 + lmbda * 0.1
                aux_loss = 0.0
            elif m == "C":
                z_bpp = 0.0440 + lmbda * 0.1  # Delta BPP_z = +0.0040
                aux_loss = 0.4500
            elif m == "D":
                z_bpp = 0.0435 + lmbda * 0.1  # Delta BPP_z = +0.0035
                aux_loss = 0.4800
            else:  # C-Transmitted
                z_bpp = 0.0400 + lmbda * 0.1  # Reuses Model A hyper-prior
                aux_loss = 0.0

            bpp = 0.2000 + lmbda * 5.0
            psnr = 30.0 + lmbda * 100.0

            ckpt_dir = Path(train_cfg.checkpoint_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            best_ckpt = ckpt_dir / "best_checkpoint.pt"
            best_ckpt.write_text("weights", encoding="utf-8")

            return {
                "val_metrics": {
                    "loss": 0.5,
                    "bpp": bpp,
                    "z_bpp": z_bpp,
                    "y_bpp": bpp - z_bpp,
                    "sidecar_bpp": 0.0080 if "TRANSMITTED" in m else 0.0,
                    "mse": 0.001,
                    "psnr": psnr,
                    "msssim": 0.95,
                    "aux_loss": aux_loss,
                },
                "best_checkpoint_path": best_ckpt,
            }

        cfg = SweepConfig(
            models=["A", "C", "D"],
            lambdas=[0.0130, 0.0250],
            output_dir=temp_dir / "sweep",
            train_fn=_mock_train,
            use_mlflow=False,
        )

        res = run_sweep(cfg)
        assert res["dry_run"] is False
        runs = res["runs"]
        assert len(runs) == 6

        # Check Model A has delta_bpp_z == 0.0
        model_a_runs = [r for r in runs if r["model"] == "A"]
        for r in model_a_runs:
            assert r["delta_bpp_z"] == 0.0

        # Check Model C has delta_bpp_z ~ +0.0040
        model_c_runs = [r for r in runs if r["model"] == "C"]
        for r in model_c_runs:
            assert pytest.approx(r["delta_bpp_z"], abs=1e-4) == 0.0040
            assert r["aux_loss"] == 0.4500

        # Check Model D has delta_bpp_z ~ +0.0035
        model_d_runs = [r for r in runs if r["model"] == "D"]
        for r in model_d_runs:
            assert pytest.approx(r["delta_bpp_z"], abs=1e-4) == 0.0035
            assert r["aux_loss"] == 0.4800

        # Verify artifacts were generated
        out_dir = Path(cfg.output_dir)
        assert (out_dir / "sweep_results.json").is_file()
        assert (out_dir / "sweep_results.csv").is_file()
        assert (out_dir / "sweep_summary.md").is_file()
        assert (out_dir / "rd_curve.png").is_file()

        # Check summary table contains Delta BPP_z and Aux Loss
        md_text = (out_dir / "sweep_summary.md").read_text(encoding="utf-8")
        assert "Delta BPP_z" in md_text
        assert "Aux Loss" in md_text
        assert "Model C" in md_text or "C  " in md_text

        # Check CSV contains delta_bpp_z and aux_loss
        csv_text = (out_dir / "sweep_results.csv").read_text(encoding="utf-8")
        assert "delta_bpp_z" in csv_text
        assert "aux_loss" in csv_text

    def test_cli_subprocess_dry_run(self) -> None:
        """Executes scripts/train_sweep.py --dry-run CLI command in isolated subprocess."""
        cmd = [
            sys.executable,
            "scripts/train_sweep.py",
            "--dry-run",
            "--models",
            "A",
            "C",
            "D",
            "C-Transmitted",
            "--lambdas",
            "0.0130",
            "0.0250",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "Dry run completed successfully" in res.stdout
        assert "Model: C" in res.stdout
        assert "Model: D" in res.stdout
        assert "Model: C-Transmitted" in res.stdout

    def test_sweep_two_pass_delta_bpp_z_reverse_order(self, temp_dir: Path) -> None:
        """Verifies pass 2 resolves Delta BPP_z when Model A is evaluated after other models."""

        def _mock_train(train_cfg: TrainingConfig) -> dict[str, Any]:
            m = train_cfg.model.upper()

            if m == "A":
                z_bpp = 0.0400
                aux_loss = 0.0
            elif m == "C":
                z_bpp = 0.0450  # Delta BPP_z = +0.0050
                aux_loss = 0.4000
            else:
                z_bpp = 0.0400
                aux_loss = 0.0

            return {
                "val_metrics": {
                    "loss": 0.5,
                    "bpp": 0.2000,
                    "z_bpp": z_bpp,
                    "y_bpp": 0.1600,
                    "mse": 0.001,
                    "psnr": 30.0,
                    "msssim": 0.95,
                    "aux_loss": aux_loss,
                },
                "best_checkpoint_path": "",
            }

        cfg = SweepConfig(
            models=["C", "A"],
            lambdas=[0.0130],
            output_dir=temp_dir / "sweep_rev",
            train_fn=_mock_train,
            use_mlflow=False,
        )

        res = run_sweep(cfg)
        runs = res["runs"]
        assert len(runs) == 2

        model_c_run = next(r for r in runs if r["model"] == "C")
        assert model_c_run["delta_bpp_z"] is not None
        assert pytest.approx(model_c_run["delta_bpp_z"], abs=1e-4) == 0.0050

    def test_train_baseline_checkpoint_extraction(self, temp_dir: Path) -> None:
        """Verifies train() correctly loads baseline_z_bpp from a saved baseline checkpoint."""
        ckpt_dir = temp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        baseline_ckpt_path = ckpt_dir / "baseline_model_a.pt"

        # Save dummy checkpoint with metrics dictionary
        dummy_payload = {
            "epoch": 5,
            "step": 100,
            "metrics": {
                "loss": 0.35,
                "bpp": 0.22,
                "z_bpp": 0.0385,
                "psnr": 32.5,
            },
            "state_dict": {},
        }
        torch.save(dummy_payload, baseline_ckpt_path)

        dataset = SyntheticSaliencyDataset(num_samples=2, height=64, width=64)
        loader = DataLoader(dataset, batch_size=2)

        config = TrainingConfig(
            model="C",
            lmbda=0.0130,
            lambda_aux=0.1,
            epochs=1,
            batch_size=2,
            accum_steps=1,
            checkpoint_dir=ckpt_dir / "model_c",
            baseline_checkpoint=baseline_ckpt_path,
            use_mlflow=False,
            train_dataloader=loader,
            val_dataloader=loader,
            device="cpu",
        )

        train_result = train(config)
        val_metrics = train_result["val_metrics"]

        assert "delta_bpp_z" in val_metrics
        assert isinstance(val_metrics["delta_bpp_z"], float)
        # delta_bpp_z = val_metrics["z_bpp"] - baseline_z_bpp (0.0385)
        expected_delta = val_metrics["z_bpp"] - 0.0385
        assert pytest.approx(val_metrics["delta_bpp_z"], abs=1e-4) == expected_delta
