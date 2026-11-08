"""Unit and integration tests for research synthesis and publication artifact compilation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from scripts.compile_research_artifacts import main as cli_main
from scripts.compile_research_artifacts import parse_args
from src.evaluation.bd_rate import evaluate_factorial_contrasts
from src.evaluation.benchmarks import (
    BenchmarkSuiteResult,
    generate_synthetic_benchmark_curve,
)
from src.evaluation.causal import CausalInterventionResult, CausalRegressionResult
from src.evaluation.figures import (
    apply_publication_theme,
    plot_causal_intervention_summary,
    plot_causal_regression_scatter,
    plot_rd_curves_with_ci,
    plot_spatial_bit_difference_heatmap,
)
from src.evaluation.spatial_allocation import SpatialBitMap
from src.evaluation.synthesis import ResearchSynthesizer


@pytest.fixture
def mock_curves() -> Dict[str, Any]:
    """Generates mock ModelCurveResults for Models A, B, C, D."""
    lambdas = [0.0035, 0.0067, 0.0130, 0.0250]
    curves = {}
    for m in ("A", "B", "C", "D"):
        curves[m] = generate_synthetic_benchmark_curve(m, lambdas, "kodak")
    return curves


class TestPublicationFigures:
    """Tests for publication-quality figure generation functions."""

    def test_apply_publication_theme(self) -> None:
        """Verifies publication theme modifies matplotlib parameters."""
        apply_publication_theme()
        import matplotlib.pyplot as plt

        assert plt.rcParams["figure.dpi"] == 300
        assert plt.rcParams["savefig.dpi"] == 300
        assert plt.rcParams["axes.grid"] is True

    def test_plot_rd_curves_with_ci(self, mock_curves: Dict[str, Any], tmp_path: Path) -> None:
        """Verifies multi-metric Rate-Distortion curves are plotted and saved."""
        out_p = tmp_path / "test_rd.png"
        saved = plot_rd_curves_with_ci(
            mock_curves,
            out_p,
            dataset_name="Kodak",
            metrics=("roi_psnr", "global_psnr"),
        )
        assert saved.is_file()
        assert saved.stat().st_size > 5000

    def test_plot_rd_curves_empty_models(self, tmp_path: Path) -> None:
        """Verifies plotting with empty curves does not raise ValueError on legend."""
        out_p = tmp_path / "empty_rd.png"
        saved = plot_rd_curves_with_ci({}, out_p, dataset_name="Empty")
        assert saved.is_file()

    def test_plot_spatial_bit_difference_heatmap(self, tmp_path: Path) -> None:
        """Verifies spatial bit difference heatmap generation with 4 panels."""
        h, w = 24, 32
        s_gt = torch.rand(1, 1, h, w)
        s_pred = torch.rand(1, 1, h, w)
        bm_c = torch.rand(1, 1, h, w)
        bm_a = torch.rand(1, 1, h, w)

        out_p = tmp_path / "test_heatmap.png"
        saved = plot_spatial_bit_difference_heatmap(
            bit_map_c=bm_c,
            bit_map_a=bm_a,
            saliency_gt=s_gt,
            saliency_pred=s_pred,
            output_path=out_p,
            image_name="test_sample",
        )
        assert saved.is_file()
        assert saved.stat().st_size > 5000

    def test_plot_causal_intervention_summary(self, tmp_path: Path) -> None:
        """Verifies causal intervention comparative bar chart generation."""
        res_dict = {
            "rar_normal": 1.28,
            "rar_inverted": 0.74,
            "rar_shuffled_4x4": 1.02,
            "rar_shuffled_8x8": 1.06,
        }
        out_p = tmp_path / "test_interventions.png"
        saved = plot_causal_intervention_summary(res_dict, out_p)
        assert saved.is_file()
        assert saved.stat().st_size > 5000

    def test_plot_causal_regression_scatter(self, tmp_path: Path) -> None:
        """Verifies causal fixed-effects regression scatterplot generation."""
        res = CausalRegressionResult(
            beta_s=0.45,
            beta_base=0.12,
            se_s=0.03,
            se_base=0.01,
            t_s=15.0,
            t_base=12.0,
            p_s=1e-12,
            p_base=1e-10,
            alpha_images=[0.1] * 24,
            r2_within=0.29,
            r2_overall=0.31,
            num_images=24,
            num_cells_total=500,
            degrees_of_freedom=474,
            hypothesis_confirmed=True,
        )
        out_p = tmp_path / "test_regression.png"
        saved = plot_causal_regression_scatter(res, out_p)
        assert saved.is_file()
        assert saved.stat().st_size > 5000


class TestResearchSynthesizer:
    """Tests for automated research report and summary JSON synthesis."""

    def test_synthesizer_end_to_end(self, mock_curves: Dict[str, Any], tmp_path: Path) -> None:
        """Verifies complete synthesis generates figures, report, and summary JSON."""
        lambdas = [0.0035, 0.0067, 0.0130, 0.0250]
        rates_dict = {k: np.vstack([p.per_image_bpp for p in mock_curves[k].points]).T for k in ("A", "B", "C", "D")}
        dist_dict = {
            k: np.vstack([p.per_image_roi_psnr for p in mock_curves[k].points]).T for k in ("A", "B", "C", "D")
        }
        contrasts = evaluate_factorial_contrasts(
            rates_by_model=rates_dict,
            dist_by_model=dist_dict,
            n_bootstraps=50,
            seed=42,
        )

        suite = BenchmarkSuiteResult(
            dataset="kodak",
            models=list(mock_curves.keys()),
            lambdas=lambdas,
            model_curves=mock_curves,
            contrasts=contrasts,
        )

        synthesizer = ResearchSynthesizer(output_dir=tmp_path / "artifacts")
        summary = synthesizer.synthesize({"kodak": suite})

        assert (tmp_path / "artifacts" / "research_report.md").is_file()
        assert (tmp_path / "artifacts" / "research_summary.json").is_file()
        assert len(summary.hypotheses) == 5
        assert len(summary.figure_paths) >= 4

        # Verify JSON contents
        with open(tmp_path / "artifacts" / "research_summary.json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["initiative"] == "Saliency-Aware Learned Neural Compression"
        assert "kodak" in data["datasets_evaluated"]
        assert len(data["hypotheses"]) == 5

        # Verify Markdown contents
        with open(tmp_path / "artifacts" / "research_report.md", encoding="utf-8") as f:
            md_text = f.read()
        assert "# Research Evaluation Report" in md_text
        assert "Pre-Registered Hypotheses Ledger" in md_text
        assert "CONFIRMED" in md_text

    def test_synthesizer_with_real_domain_objects(self, mock_curves: Dict[str, Any], tmp_path: Path) -> None:
        """Verifies synthesizer handles real CausalRegressionResult and CausalInterventionResult."""
        reg_res = CausalRegressionResult(
            beta_s=0.4284,
            beta_base=0.1042,
            se_s=0.0352,
            se_base=0.0121,
            t_s=12.17,
            t_base=8.61,
            p_s=1.2e-8,
            p_base=3.4e-6,
            alpha_images=[0.05] * 24,
            r2_within=0.342,
            r2_overall=0.285,
            num_images=24,
            num_cells_total=1152,
            degrees_of_freedom=1126,
            hypothesis_confirmed=True,
        )

        bm = SpatialBitMap(
            y_bits=torch.zeros(1, 1, 16, 16),
            z_bits=torch.zeros(1, 1, 16, 16),
            total_bits=torch.zeros(1, 1, 16, 16),
        )
        interv_res = CausalInterventionResult(
            model_name="Model_C",
            block_size=4,
            baseline_bit_map=bm,
            intervened_bit_map=bm,
            unshuffled_bit_map=bm,
            original_saliency=torch.zeros(1, 1, 16, 16),
            shuffled_saliency=torch.zeros(1, 1, 16, 16),
            permutation=torch.arange(16),
            corr_interv_vs_shuffled=0.68,
            corr_interv_vs_original=0.42,
            corr_unshuffled_vs_baseline=0.75,
            causal_shift=0.26,
        )

        suite = BenchmarkSuiteResult(
            dataset="kodak",
            models=list(mock_curves.keys()),
            lambdas=[0.0035, 0.0067],
            model_curves=mock_curves,
            contrasts=None,
        )

        synthesizer = ResearchSynthesizer(output_dir=tmp_path / "domain_obj_artifacts")
        summary = synthesizer.synthesize(
            {"kodak": suite},
            causal_regression=reg_res,
            causal_intervention=interv_res,
        )

        assert summary.causal_regression["beta_s"] == pytest.approx(0.4284)
        assert summary.causal_regression["p_value_s"] == pytest.approx(1.2e-8)
        assert summary.causal_intervention["rar_normal"] == pytest.approx(1.26)
        assert summary.factorial_contrasts is None

    def test_synthesizer_with_none_contrasts(self, mock_curves: Dict[str, Any], tmp_path: Path) -> None:
        """Verifies synthesizer behaves gracefully when contrasts are None (H1, H2 not confirmed)."""
        suite = BenchmarkSuiteResult(
            dataset="kodak",
            models=list(mock_curves.keys()),
            lambdas=[0.0035, 0.0067],
            model_curves=mock_curves,
            contrasts=None,
        )
        synthesizer = ResearchSynthesizer(output_dir=tmp_path / "none_contrasts_artifacts")
        summary = synthesizer.synthesize({"kodak": suite})

        assert summary.factorial_contrasts is None
        h1 = next(h for h in summary.hypotheses if h.hypothesis_id == "H1")
        h2 = next(h for h in summary.hypotheses if h.hypothesis_id == "H2")
        assert not h1.confirmed
        assert not h2.confirmed


class TestArtifactCompilationCLI:
    """Tests for compile_research_artifacts CLI parsing and dry-run execution."""

    def test_cli_argument_parsing(self) -> None:
        """Verifies argument parsing handles custom options and defaults."""
        args = parse_args(["--dry-run", "--output-dir", "out_dir", "--device", "cpu"])
        assert args.dry_run is True
        assert args.output_dir == "out_dir"
        assert args.device == "cpu"

    def test_cli_dry_run_main(self, tmp_path: Path) -> None:
        """Verifies CLI main function executes dry-run synthesis cleanly."""
        out_dir = tmp_path / "cli_artifacts"
        code = cli_main(["--dry-run", "--output-dir", str(out_dir), "--bootstrap-resamples", "50"])
        assert code == 0
        assert (out_dir / "research_report.md").is_file()
        assert (out_dir / "research_summary.json").is_file()
        assert (out_dir / "figures" / "rd_curves_kodak.png").is_file()
