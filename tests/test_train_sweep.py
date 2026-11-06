"""Test suite for the multi-rate training sweep orchestrator.

Verifies:
- CLI argument parsing, defaults (6 pre-registered lambdas for models A and B), custom overrides.
- Model and lambda validation (rejection of invalid models, non-positive lambdas, empty lists).
- Dry run parameter matrix generation (12 jobs for A+B, skips training, outputs structured plan).
- Sequential job dispatch without concurrent GPU contention.
- Metric aggregation into sorted JSON (sweep_results.json), CSV (sweep_results.csv), and Markdown (sweep_summary.md).
- Graceful handling of missing run keys during metric collation and table generation.
- Headless matplotlib rate-distortion curve visualization (PSNR vs. BPP and MS-SSIM vs. BPP).
- Isolated checkpoint directory allocation per run.
- Error handling under fail_fast=True and fail_fast=False.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import matplotlib
import pytest

from scripts.train import TrainingConfig
from scripts.train_sweep import (
    DEFAULT_LAMBDAS,
    DEFAULT_MODELS,
    SweepConfig,
    generate_csv_data,
    generate_rd_plots,
    generate_summary_table,
    main,
    parse_args,
    run_sweep,
)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provides an isolated temporary directory for sweep artifacts and checkpoints."""
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def make_mock_train_fn(invocation_log: list[dict[str, Any]] | None = None):
    """Factory creating deterministic mock train function for sweep tests."""

    def _mock_train(config: TrainingConfig) -> dict[str, Any]:
        lmbda = config.lmbda
        model = config.model.upper()

        if invocation_log is not None:
            invocation_log.append(
                {
                    "model": model,
                    "lambda": lmbda,
                    "checkpoint_dir": str(config.checkpoint_dir),
                    "run_name": config.run_name,
                }
            )

        # Realistic rate-distortion progression: higher lambda -> higher BPP, lower MSE, higher PSNR
        offset = 0.05 if model == "B" else 0.0
        bpp = float(0.1 + lmbda * 8.0 + offset)
        mse = float(0.01 / (1.0 + lmbda * 60.0))
        psnr = float(28.0 + 12.0 * (lmbda / 0.05))
        msssim = float(0.88 + 0.10 * (lmbda / 0.05))
        loss = float(lmbda * 100.0)

        ckpt_dir = Path(config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt = ckpt_dir / "best_checkpoint.pt"
        latest_ckpt = ckpt_dir / "checkpoint.pt"
        best_ckpt.write_text("mock_best_weights", encoding="utf-8")
        latest_ckpt.write_text("mock_latest_weights", encoding="utf-8")

        return {
            "model": None,
            "optimizer_harness": None,
            "val_metrics": {
                "loss": loss,
                "bpp": bpp,
                "bpp_loss": bpp,
                "mse": mse,
                "mse_loss": mse,
                "psnr": psnr,
                "msssim": msssim,
                "ms_ssim": msssim,
                "distortion": mse,
            },
            "best_checkpoint_path": best_ckpt,
            "latest_checkpoint_path": latest_ckpt,
            "history": [{"val_loss": loss}],
        }

    return _mock_train


# ==============================================================================
# Configuration & CLI Argument Parsing Tests
# ==============================================================================


class TestSweepConfigAndArgs:
    """Verifies CLI argument parsing, normalization, defaults, and input validation."""

    def test_default_cli_invocation(self) -> None:
        """Default CLI arguments target all 6 standard lambda points for Model A and Model B."""
        args = parse_args([])
        assert set(args.models) == set(DEFAULT_MODELS)
        assert tuple(args.lambdas) == DEFAULT_LAMBDAS
        assert len(args.lambdas) == 6
        assert args.dry_run is False
        assert args.output_dir == "experiments/sweep_results"

        config = SweepConfig.from_namespace(args)
        assert config.models == list(DEFAULT_MODELS)
        assert tuple(config.lambdas) == DEFAULT_LAMBDAS
        assert config.dry_run is False

    def test_custom_models_normalization(self) -> None:
        """Model arguments are normalized to uppercase 'A' or 'B'."""
        args = parse_args(["--models", "a", "modelb"])
        config = SweepConfig.from_namespace(args)
        assert config.models == ["A", "B"]

        single_arg = parse_args(["--models", "A"])
        single_cfg = SweepConfig.from_namespace(single_arg)
        assert single_cfg.models == ["A"]

    def test_custom_lambdas_override(self) -> None:
        """Custom lambda arguments override standard 6-point grid."""
        args = parse_args(["--lambdas", "0.01", "0.025", "0.05"])
        config = SweepConfig.from_namespace(args)
        assert config.lambdas == [0.01, 0.025, 0.05]

    def test_dry_run_flag(self) -> None:
        """--dry-run sets dry_run attribute to True."""
        args = parse_args(["--dry-run"])
        assert args.dry_run is True
        config = SweepConfig.from_namespace(args)
        assert config.dry_run is True

    def test_invalid_model_rejection(self) -> None:
        """Invalid model names raise ValueError."""
        with pytest.raises(ValueError, match="Invalid model 'C'"):
            SweepConfig(models=["C"])

        with pytest.raises(ValueError, match="models list must not be empty"):
            SweepConfig(models=[])

    def test_invalid_lambda_rejection(self) -> None:
        """Non-positive or empty lambda lists raise ValueError."""
        with pytest.raises(ValueError, match="Lambda values must be positive"):
            SweepConfig(lambdas=[-0.01])

        with pytest.raises(ValueError, match="Lambda values must be positive"):
            SweepConfig(lambdas=[0.0])

        with pytest.raises(ValueError, match="lambdas list must not be empty"):
            SweepConfig(lambdas=[])

    def test_invalid_numeric_parameters(self) -> None:
        """Batch size, accum steps, and epochs must be >= 1."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            SweepConfig(batch_size=0)

        with pytest.raises(ValueError, match="accum_steps must be >= 1"):
            SweepConfig(accum_steps=0)

        with pytest.raises(ValueError, match="epochs must be >= 1"):
            SweepConfig(epochs=0)

        with pytest.raises(ValueError, match="Invalid best_metric"):
            SweepConfig(best_metric="invalid_metric")

    def test_from_dict_aliases_and_normalization(self) -> None:
        """from_dict normalizes parameter aliases (e.g. model, lambda, crop_size)."""
        data = {
            "model": "A",
            "lambda": 0.0130,
            "crop_size": 128,
            "output_dir": "custom_output",
        }
        cfg = SweepConfig.from_dict(data)
        assert cfg.models == ["A"]
        assert cfg.lambdas == [0.0130]
        assert cfg.patch_size == 128
        assert cfg.output_dir == Path("custom_output")


# ==============================================================================
# Dry Run Execution Tests
# ==============================================================================


class TestDryRunExecution:
    """Verifies dry run execution plan generation and structured parameter schedule."""

    def test_dry_run_full_schedule_12_jobs(self) -> None:
        """Default dry run generates 12-job matrix for 2 models x 6 lambdas without training."""
        invocations: list[dict[str, Any]] = []
        cfg = SweepConfig(
            models=["A", "B"],
            lambdas=list(DEFAULT_LAMBDAS),
            dry_run=True,
            train_fn=make_mock_train_fn(invocations),
        )
        result = run_sweep(cfg)

        assert result["dry_run"] is True
        assert result["total_jobs"] == 12
        plan = result["plan"]
        assert len(plan) == 12

        # Verify training was skipped
        assert len(invocations) == 0

        # Verify all combinations are planned
        models_in_plan = {job["model"] for job in plan}
        lambdas_in_plan = {job["lambda"] for job in plan}
        assert models_in_plan == {"A", "B"}
        assert lambdas_in_plan == set(DEFAULT_LAMBDAS)

        # Check job structure
        for job in plan:
            assert "model" in job
            assert "lambda" in job
            assert "run_name" in job
            assert "checkpoint_dir" in job

    def test_dry_run_single_model(self) -> None:
        """Dry run for a single model generates 6 jobs."""
        cfg = SweepConfig(models=["A"], dry_run=True)
        result = run_sweep(cfg)
        assert result["dry_run"] is True
        assert result["total_jobs"] == 6
        assert len(result["plan"]) == 6
        for job in result["plan"]:
            assert job["model"] == "A"

    def test_main_cli_dry_run(self, capsys: pytest.CaptureFixture[str]) -> None:
        """CLI main() invocation with --dry-run prints job schedule."""
        main(["--dry-run", "--models", "A", "--lambdas", "0.0130"])
        captured = capsys.readouterr()
        assert "Dry run completed successfully" in captured.out
        assert "Model: A | Lambda: 0.0130" in captured.out


# ==============================================================================
# Sequential Dispatch & Execution Tests
# ==============================================================================


class TestSequentialDispatchAndExecution:
    """Verifies sequential dispatch, isolated checkpoint directories, and error handling."""

    def test_sequential_execution_full_sweep(self, temp_dir: Path) -> None:
        """Full sweep runs sequential training passes, saves checkpoints, outputs summary."""
        invocations: list[dict[str, Any]] = []
        test_lambdas = [0.0018, 0.0035]

        cfg = SweepConfig(
            models=["A", "B"],
            lambdas=test_lambdas,
            output_dir=temp_dir / "sweep_out",
            dry_run=False,
            train_fn=make_mock_train_fn(invocations),
        )

        result = run_sweep(cfg)

        assert result["dry_run"] is False
        assert len(invocations) == 4
        # Verify sequential order: A at 0.0018, A at 0.0035, B at 0.0018, B at 0.0035
        expected_seq = [("A", 0.0018), ("A", 0.0035), ("B", 0.0018), ("B", 0.0035)]
        actual_seq = [(inv["model"], inv["lambda"]) for inv in invocations]
        assert actual_seq == expected_seq

        # Verify distinct checkpoint directories per run to avoid overwrites
        ckpt_dirs = [inv["checkpoint_dir"] for inv in invocations]
        assert len(set(ckpt_dirs)) == 4

        # Verify artifacts exist on disk
        json_path = result["results_json_path"]
        csv_path = result["results_csv_path"]
        md_path = result["summary_md_path"]
        rd_plot_path = result["rd_plot_path"]

        assert json_path.is_file()
        assert csv_path.is_file()
        assert md_path.is_file()
        assert rd_plot_path.is_file()

        # Verify JSON contents
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["models"] == ["A", "B"]
        assert len(payload["runs"]) == 4

        # Verify CSV contents
        csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert csv_lines[0] == "model,lambda,bpp,mse,psnr,ms_ssim,loss"
        assert len(csv_lines) == 5  # header + 4 rows

        # Verify Markdown summary
        md_text = md_path.read_text(encoding="utf-8")
        assert "| Model | Lambda | BPP | MSE | PSNR (dB) | MS-SSIM |" in md_text
        assert "Model A" not in md_text or "A" in md_text

    def test_single_model_sweep(self, temp_dir: Path) -> None:
        """Single model sweep executes only specified model."""
        invocations: list[dict[str, Any]] = []
        cfg = SweepConfig(
            models=["A"],
            lambdas=[0.0067, 0.0130],
            output_dir=temp_dir / "single_model_sweep",
            train_fn=make_mock_train_fn(invocations),
        )
        result = run_sweep(cfg)

        assert len(invocations) == 2
        assert all(inv["model"] == "A" for inv in invocations)
        assert len(result["runs"]) == 2
        for r in result["runs"]:
            assert r["model"] == "A"

    def test_fail_fast_false_resilience(self, temp_dir: Path) -> None:
        """With fail_fast=False, training failure on one run does not halt the sweep."""

        def _faulty_train(config: TrainingConfig) -> dict[str, Any]:
            if config.model == "A" and abs(config.lmbda - 0.0018) < 1e-5:
                raise RuntimeError("Simulated GPU OOM or divergence")
            return make_mock_train_fn()(config)

        cfg = SweepConfig(
            models=["A", "B"],
            lambdas=[0.0018, 0.0035],
            output_dir=temp_dir / "faulty_sweep",
            fail_fast=False,
            train_fn=_faulty_train,
        )

        result = run_sweep(cfg)
        assert len(result["runs"]) == 4

        failed_run = next(r for r in result["runs"] if r["model"] == "A" and r["lambda"] == 0.0018)
        assert failed_run["status"] == "failed"
        assert "Simulated GPU OOM" in failed_run["error"]

        successful_runs = [r for r in result["runs"] if r["status"] == "success"]
        assert len(successful_runs) == 3

    def test_fail_fast_true_raises(self, temp_dir: Path) -> None:
        """With fail_fast=True, an exception is immediately re-raised."""

        def _faulty_train(config: TrainingConfig) -> dict[str, Any]:
            raise RuntimeError("Immediate failure")

        cfg = SweepConfig(
            models=["A"],
            lambdas=[0.0018],
            output_dir=temp_dir / "fail_fast_sweep",
            fail_fast=True,
            train_fn=_faulty_train,
        )

        with pytest.raises(RuntimeError, match="Immediate failure"):
            run_sweep(cfg)


# ==============================================================================
# Metric Aggregation & Table Formatting Tests
# ==============================================================================


class TestMetricAggregationAndTable:
    """Verifies Markdown summary table and CSV data generation with clean formatting."""

    def test_generate_summary_table_formatting(self) -> None:
        """Summary table generates sorted Markdown with expected headers and columns."""
        runs = [
            {"model": "B", "lambda": 0.0130, "bpp": 0.3501, "mse": 0.001234, "psnr": 34.21, "msssim": 0.9450},
            {"model": "A", "lambda": 0.0018, "bpp": 0.1142, "mse": 0.006789, "psnr": 30.12, "msssim": 0.9015},
            {"model": "A", "lambda": 0.0130, "bpp": 0.3204, "mse": 0.001500, "psnr": 33.80, "msssim": 0.9380},
        ]

        md = generate_summary_table(runs)
        lines = md.strip().splitlines()

        assert lines[0] == "| Model | Lambda | BPP | MSE | PSNR (dB) | MS-SSIM |"
        assert lines[1] == "| --- | --- | --- | --- | --- | --- |"

        # Model A at 0.0018 should come first after sorting
        assert "A" in lines[2] and "0.0018" in lines[2] and "0.1142" in lines[2]
        # Model A at 0.0130 second
        assert "A" in lines[3] and "0.0130" in lines[3] and "0.3204" in lines[3]
        # Model B at 0.0130 third
        assert "B" in lines[4] and "0.0130" in lines[4] and "0.3501" in lines[4]

    def test_generate_summary_table_missing_keys_graceful(self) -> None:
        """Missing or NaN metric keys are rendered as 'N/A' without errors."""
        runs = [
            {"model": "A", "lambda": 0.0018, "bpp": 0.12, "mse": None, "psnr": None, "msssim": None},
            {"model": "B", "lambda": 0.0035, "bpp": None, "mse": 0.005, "psnr": float("nan"), "msssim": None},
        ]

        md = generate_summary_table(runs)
        assert "N/A" in md
        assert "| Model | Lambda | BPP | MSE | PSNR (dB) | MS-SSIM |" in md

    def test_generate_csv_data(self) -> None:
        """CSV generator outputs proper columns and sorted rows."""
        runs = [
            {
                "model": "B",
                "lambda": 0.025,
                "bpp": 0.5,
                "mse": 0.001,
                "psnr": 35.0,
                "ms_ssim": 0.96,
                "loss": 2.5,
            },
            {
                "model": "A",
                "lambda": 0.025,
                "bpp": 0.48,
                "mse": 0.0011,
                "psnr": 34.8,
                "msssim": 0.955,
                "loss": 2.6,
            },
        ]
        csv_text = generate_csv_data(runs)
        lines = csv_text.strip().splitlines()
        assert lines[0] == "model,lambda,bpp,mse,psnr,ms_ssim,loss"
        # Sorted: Model A first, then B
        assert lines[1].startswith("A,0.025,0.48")
        assert lines[2].startswith("B,0.025,0.5")


# ==============================================================================
# Rate-Distortion Plot Generation Tests
# ==============================================================================


class TestRateDistortionPlotGeneration:
    """Verifies headless matplotlib rate-distortion curve visualization."""

    def test_generate_rd_plots_creates_valid_png(self, temp_dir: Path) -> None:
        """generate_rd_plots writes a valid PNG plot with PSNR vs. BPP and MS-SSIM vs. BPP."""
        runs = [
            {"model": "A", "lambda": 0.0018, "bpp": 0.12, "mse": 0.0050, "psnr": 30.5, "msssim": 0.91},
            {"model": "A", "lambda": 0.0035, "bpp": 0.20, "mse": 0.0030, "psnr": 32.0, "msssim": 0.93},
            {"model": "A", "lambda": 0.0067, "bpp": 0.30, "mse": 0.0018, "psnr": 34.0, "msssim": 0.95},
            {"model": "B", "lambda": 0.0018, "bpp": 0.14, "mse": 0.0045, "psnr": 31.0, "msssim": 0.92},
            {"model": "B", "lambda": 0.0035, "bpp": 0.22, "mse": 0.0028, "psnr": 32.5, "msssim": 0.94},
            {"model": "B", "lambda": 0.0067, "bpp": 0.32, "mse": 0.0016, "psnr": 34.5, "msssim": 0.96},
        ]

        plot_path = temp_dir / "rd_curves.png"
        saved_path = generate_rd_plots(runs, plot_path)

        assert saved_path == plot_path
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 1000

        # Verify PNG magic number bytes
        with open(plot_path, "rb") as f:
            header = f.read(8)
            assert header == b"\x89PNG\r\n\x1a\n"

    def test_generate_rd_plots_single_model(self, temp_dir: Path) -> None:
        """Plotting runs for a single model succeeds cleanly."""
        runs = [
            {"model": "A", "lambda": 0.0018, "bpp": 0.12, "psnr": 30.5, "msssim": 0.91},
            {"model": "A", "lambda": 0.0035, "bpp": 0.20, "psnr": 32.0, "msssim": 0.93},
        ]
        plot_path = temp_dir / "single_model_rd.png"
        generate_rd_plots(runs, plot_path)
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 1000

    def test_generate_rd_plots_empty_data_handled(self, temp_dir: Path) -> None:
        """Empty run lists produce a valid plot placeholder without raising exceptions."""
        plot_path = temp_dir / "empty_rd.png"
        generate_rd_plots([], plot_path)
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 500

    def test_matplotlib_headless_agg_backend(self) -> None:
        """Verifies matplotlib operates under Agg backend for headless environments."""
        backend = matplotlib.get_backend().lower()
        assert "agg" in backend
