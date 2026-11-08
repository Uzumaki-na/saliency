"""Comprehensive test suite for benchmark runner CLI and evaluation engine (Story 5.6).

Verifies:
1. BenchmarkConfig validation, normalization, and defaults.
2. Checkpoint discovery heuristics across directory layouts and explicit maps.
3. Benchmark dataset image and mask loading with fallback handling.
4. Synthetic benchmark curve generation with theoretical rate-distortion properties.
5. BD-Rate curve comparisons and pre-registered factorial contrast computation.
6. Benchmark artifact serialization (JSON, CSV, Markdown, and Matplotlib curves).
7. BenchmarkRunner execution in dry-run and mock inference modes.
8. Standalone scripts/benchmark.py CLI execution and parameter parsing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark import main, parse_args
from src.evaluation.bd_rate import BootstrapResult, FactorialContrastsResult
from src.evaluation.benchmarks import (
    DEFAULT_6POINT_LAMBDAS,
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkSuiteResult,
    ModelCurveResult,
    generate_synthetic_benchmark_curve,
    load_benchmark_dataset,
    plot_benchmark_rd_curves,
    resolve_checkpoint_path,
    save_benchmark_csv,
    save_benchmark_json,
    save_benchmark_markdown,
)


class TestBenchmarkConfig:
    """Verifies parameter validation, normalization, and defaulting in BenchmarkConfig."""

    def test_default_config(self, tmp_path: Path) -> None:
        """Default configuration initializes expected 6-point sweep and default models."""
        cfg = BenchmarkConfig(output_dir=tmp_path)
        assert cfg.models == ["A", "B", "C", "D", "C-TRANSMITTED"]
        assert cfg.lambdas == list(DEFAULT_6POINT_LAMBDAS)
        assert cfg.datasets == ["kodak"]
        assert cfg.device in ("cpu", "cuda")
        assert not cfg.dry_run

    def test_model_normalization(self, tmp_path: Path) -> None:
        """Model aliases like 'ctransmitted' are normalized to canonical uppercase 'C-TRANSMITTED'."""
        cfg = BenchmarkConfig(
            models=["a", "b", "c", "ctransmitted", "c_transmitted"],
            output_dir=tmp_path,
        )
        assert cfg.models == ["A", "B", "C", "C-TRANSMITTED", "C-TRANSMITTED"]

    def test_empty_lambdas_raises_error(self, tmp_path: Path) -> None:
        """Empty lambda list raises ValueError."""
        with pytest.raises(ValueError, match=r"lambdas list must not be empty"):
            BenchmarkConfig(lambdas=[], output_dir=tmp_path)

    def test_invalid_dataset_raises_error(self, tmp_path: Path) -> None:
        """Non-existent unknown dataset name raises ValueError."""
        with pytest.raises(ValueError, match=r"Dataset 'unknown_dataset_xyz' not recognized"):
            BenchmarkConfig(datasets=["unknown_dataset_xyz"], output_dir=tmp_path)


class TestCheckpointResolution:
    """Verifies checkpoint discovery heuristics."""

    def test_explicit_checkpoint_map(self, tmp_path: Path) -> None:
        """Explicit checkpoint mapping takes highest priority."""
        fake_ckpt = tmp_path / "model_c_lmb_0.0130.pt"
        fake_ckpt.write_text("fake_weights")

        ckpt_map = {"C": {0.0130: str(fake_ckpt)}}
        res = resolve_checkpoint_path(None, "C", 0.0130, checkpoint_map=ckpt_map)
        assert res == fake_ckpt

    def test_directory_layout_resolution(self, tmp_path: Path) -> None:
        """Discovers checkpoints organized under standard run directories."""
        run_dir = tmp_path / "A_lambda_0.0130"
        run_dir.mkdir(parents=True)
        ckpt_file = run_dir / "best_checkpoint.pt"
        ckpt_file.write_text("weights")

        resolved = resolve_checkpoint_path(tmp_path, "A", 0.0130)
        assert resolved == ckpt_file

    def test_missing_checkpoint_returns_none(self, tmp_path: Path) -> None:
        """Returns None when checkpoint does not exist on disk."""
        resolved = resolve_checkpoint_path(tmp_path, "D", 0.0483)
        assert resolved is None


class TestDatasetLoading:
    """Verifies dataset image and mask loading utilities."""

    def test_synthetic_fallback_loading(self) -> None:
        """When dataset name is 'synthetic', loader returns synthetic tensors for testing."""
        images, masks, fnames = load_benchmark_dataset(
            dataset_name="synthetic",
            max_samples=2,
            device="cpu",
        )
        assert len(images) == 2
        assert len(masks) == 2
        assert len(fnames) == 2
        assert images[0].shape[-2:] == (256, 256)
        assert masks[0].shape[-2:] == (256, 256)


class TestSyntheticCurves:
    """Verifies properties of synthetic rate-distortion generation."""

    def test_theoretical_curve_properties(self) -> None:
        """Synthetic curves exhibit expected research properties across models."""
        lambdas = [0.0018, 0.0067, 0.0130, 0.0483]
        curve_a = generate_synthetic_benchmark_curve("A", lambdas, "kodak")
        curve_b = generate_synthetic_benchmark_curve("B", lambdas, "kodak")
        curve_c = generate_synthetic_benchmark_curve("C", lambdas, "kodak")
        curve_d = generate_synthetic_benchmark_curve("D", lambdas, "kodak")
        curve_ct = generate_synthetic_benchmark_curve("C-Transmitted", lambdas, "kodak")

        assert len(curve_a.points) == 4
        assert len(curve_c.points) == 4
        assert len(curve_d.points) == 4
        assert len(curve_ct.points) == 4

        # Model C has higher ROI PSNR than Model A and B at all points
        for pt_a, pt_b, pt_c in zip(curve_a.points, curve_b.points, curve_c.points):
            assert pt_c.roi_psnr > pt_b.roi_psnr > pt_a.roi_psnr
            assert pt_c.rar > pt_b.rar > pt_a.rar

        # Model C-Transmitted has non-zero sidecar bpp
        assert all(pt.sidecar_bpp > 0.01 for pt in curve_ct.points)

        # Model C has predicted saliency metrics
        assert all(pt.auc_judd is not None and pt.auc_judd > 0.75 for pt in curve_c.points)


class TestBenchmarkArtifacts:
    """Verifies export of structured JSON, CSV, Markdown, and Matplotlib figures."""

    @pytest.fixture
    def sample_suite_result(self) -> BenchmarkSuiteResult:
        """Creates a mock BenchmarkSuiteResult with full 5-model curves."""
        lambdas = [0.0018, 0.0067, 0.0130, 0.0483]
        curves: dict[str, ModelCurveResult] = {
            m: generate_synthetic_benchmark_curve(m, lambdas, "kodak") for m in ("A", "B", "C", "D", "C-TRANSMITTED")
        }

        # Attach mock BD-Rates
        curves["B"].global_bd_rate = 2.1
        curves["B"].roi_bd_rate = -5.4
        curves["C"].global_bd_rate = -4.2
        curves["C"].roi_bd_rate = -18.7
        curves["D"].global_bd_rate = -1.1
        curves["D"].roi_bd_rate = -11.2

        b_res_cb = BootstrapResult(
            point_estimate=-13.3,
            bootstrap_mean=-13.2,
            std_error=0.8,
            ci_lower=-14.8,
            ci_upper=-11.7,
            confidence_level=0.95,
            p_value=0.0001,
            hypothesis_confirmed=True,
            n_bootstraps=100,
            bootstrap_samples=np.array([-13.3]),
        )
        b_res_da = BootstrapResult(
            point_estimate=-11.2,
            bootstrap_mean=-11.1,
            std_error=0.7,
            ci_lower=-12.5,
            ci_upper=-9.8,
            confidence_level=0.95,
            p_value=0.0001,
            hypothesis_confirmed=True,
            n_bootstraps=100,
            bootstrap_samples=np.array([-11.2]),
        )
        b_res_int = BootstrapResult(
            point_estimate=-2.1,
            bootstrap_mean=-2.1,
            std_error=0.8,
            ci_lower=-3.8,
            ci_upper=-0.4,
            confidence_level=0.95,
            p_value=0.0004,
            hypothesis_confirmed=True,
            n_bootstraps=100,
            bootstrap_samples=np.array([-2.1]),
        )

        contrasts = FactorialContrastsResult(
            contrast_c_minus_b=b_res_cb,
            contrast_d_minus_a=b_res_da,
            contrast_interaction=b_res_int,
            pairwise_contrasts={},
            method="akima",
            metric_mode="higher_is_better",
            confidence_level=0.95,
            n_bootstraps=100,
        )

        return BenchmarkSuiteResult(
            dataset="kodak",
            models=list(curves.keys()),
            lambdas=lambdas,
            model_curves=curves,
            contrasts=contrasts,
        )

    def test_save_json(self, sample_suite_result: BenchmarkSuiteResult, tmp_path: Path) -> None:
        """JSON output is valid and deserializable with complete nested records."""
        json_file = tmp_path / "benchmark_results.json"
        save_benchmark_json(sample_suite_result, json_file)
        assert json_file.is_file()

        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        assert data["dataset"] == "kodak"
        assert len(data["model_curves"]) == 5
        assert data["contrasts"]["interaction_contrast"]["point_estimate"] == pytest.approx(-2.1)

    def test_save_csv(self, sample_suite_result: BenchmarkSuiteResult, tmp_path: Path) -> None:
        """CSV output contains proper headers and matching rows."""
        csv_file = tmp_path / "benchmark_summary.csv"
        save_benchmark_csv(sample_suite_result, csv_file)
        assert csv_file.is_file()

        with open(csv_file, encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        header = rows[0]
        assert "dataset" in header
        assert "roi_psnr" in header
        assert "delta_bpp_z" in header
        assert len(rows) == 1 + (5 * 4)  # 1 header + 20 rate points

    def test_save_markdown(self, sample_suite_result: BenchmarkSuiteResult, tmp_path: Path) -> None:
        """Markdown report contains comparative tables and factorial contrast section."""
        md_file = tmp_path / "benchmark_summary.md"
        save_benchmark_markdown(sample_suite_result, md_file)
        assert md_file.is_file()

        content = md_file.read_text(encoding="utf-8")
        assert "Comparative BD-Rate Analysis" in content
        assert "Pre-Registered Factorial Contrasts" in content
        assert "Rate-Point Performance Matrix" in content
        assert "-18.70%" in content  # Model C ROI BD-Rate

    def test_plot_curves(self, sample_suite_result: BenchmarkSuiteResult, tmp_path: Path) -> None:
        """Renders 4-panel Matplotlib RD curves without errors."""
        plot_file = tmp_path / "rd_curves.png"
        plot_benchmark_rd_curves(sample_suite_result, plot_file)
        assert plot_file.is_file()
        assert plot_file.stat().st_size > 1000


class TestBenchmarkRunner:
    """Verifies end-to-end execution of BenchmarkRunner."""

    def test_dry_run_sweep_execution(self, tmp_path: Path) -> None:
        """Dry-run sweep executes full 6-point sweep across all 5 models and produces all artifacts."""
        cfg = BenchmarkConfig(
            models=["A", "B", "C", "D", "C-Transmitted"],
            lambdas=[0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483],
            datasets=["kodak"],
            output_dir=tmp_path / "bench_out",
            dry_run=True,
            bootstrap_resamples=50,
        )

        runner = BenchmarkRunner(cfg)
        results = runner.run_sweep()

        assert "kodak" in results
        kodak_res = results["kodak"]
        assert len(kodak_res.models) == 5
        assert "A" in kodak_res.model_curves
        assert "C" in kodak_res.model_curves

        # Check Model C has negative ROI BD-Rate
        c_curve = kodak_res.model_curves["C"]
        assert c_curve.roi_bd_rate is not None
        assert c_curve.roi_bd_rate < 0.0

        # Check factorial contrasts
        assert kodak_res.contrasts is not None
        assert isinstance(kodak_res.contrasts, FactorialContrastsResult)

        # Verify disk artifacts
        d_dir = tmp_path / "bench_out" / "kodak"
        assert (d_dir / "benchmark_results.json").is_file()
        assert (d_dir / "benchmark_summary.csv").is_file()
        assert (d_dir / "benchmark_summary.md").is_file()
        assert (d_dir / "rd_curves.png").is_file()

    def test_real_inference_sweep_execution(self, tmp_path: Path) -> None:
        """Non-dry-run inference executes real forward pass, padding, bit maps, and metrics."""
        cfg = BenchmarkConfig(
            models=["A", "C"],
            lambdas=[0.0130, 0.0250],
            datasets=["synthetic"],
            output_dir=tmp_path / "real_bench",
            dry_run=False,
            sample=2,
            device="cpu",
            bootstrap_resamples=10,
        )

        runner = BenchmarkRunner(cfg)
        results = runner.run_sweep()

        assert "synthetic" in results
        synth_res = results["synthetic"]
        assert "A" in synth_res.model_curves
        assert "C" in synth_res.model_curves

        # Validate Model A point
        pt_a = synth_res.model_curves["A"].points[0]
        assert pt_a.bpp > 0.0
        assert pt_a.global_psnr > 0.0
        assert len(pt_a.per_image_bpp) == 2
        assert len(pt_a.per_image_roi_psnr) == 2

        # Validate Model C point with predicted saliency and RAR
        pt_c = synth_res.model_curves["C"].points[0]
        assert pt_c.bpp > 0.0
        assert pt_c.global_psnr > 0.0
        assert pt_c.rar > 0.0
        assert pt_c.auc_judd is not None
        assert len(pt_c.per_image_bpp) == 2
        assert len(pt_c.per_image_roi_psnr) == 2

        # BD-Rate calculated relative to Model A anchor
        c_curve = synth_res.model_curves["C"]
        assert c_curve.global_bd_rate is not None
        assert c_curve.roi_bd_rate is not None

        # Verify disk artifacts
        d_dir = tmp_path / "real_bench" / "synthetic"
        assert (d_dir / "benchmark_results.json").is_file()
        assert (d_dir / "benchmark_summary.csv").is_file()
        assert (d_dir / "benchmark_summary.md").is_file()
        assert (d_dir / "rd_curves.png").is_file()


class TestBenchmarkCLI:
    """Verifies command-line argument parsing and CLI execution."""

    def test_cli_argument_parsing(self) -> None:
        """CLI arguments parse into correct Namespace values."""
        args = parse_args(
            [
                "--models",
                "A",
                "C",
                "--lambdas",
                "0.0067",
                "0.0130",
                "--datasets",
                "kodak",
                "mit1003",
                "--dry-run",
                "--sample",
                "5",
                "--bootstrap-resamples",
                "100",
                "--no-plots",
            ]
        )
        assert args.models == ["A", "C"]
        assert args.lambdas == [0.0067, 0.0130]
        assert args.datasets == ["kodak", "mit1003"]
        assert args.dry_run is True
        assert args.sample == 5
        assert args.bootstrap_resamples == 100
        assert args.no_plots is True

    def test_cli_main_dry_run(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """CLI main() executes successfully in dry-run mode and prints summary."""
        out_dir = str(tmp_path / "cli_bench")
        exit_code = main(
            [
                "--models",
                "A",
                "B",
                "C",
                "D",
                "--lambdas",
                "0.0067",
                "0.0130",
                "0.0250",
                "--datasets",
                "kodak",
                "--output-dir",
                out_dir,
                "--dry-run",
                "--bootstrap-resamples",
                "20",
                "--no-plots",
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "Benchmark Sweep Complete" in captured
        assert "Pre-Registered Factorial Contrasts" in captured
        assert "Benchmark successfully completed" in captured
