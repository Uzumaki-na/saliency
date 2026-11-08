#!/usr/bin/env python3
"""Command-line interface to compile all research evaluation artifacts and figures.

Synthesizes Rate-Distortion curves, spatial bit reallocation heatmaps, causal
intervention comparisons, fixed-effects regressions, and hypothesis ledger
into publication-quality figures and Markdown/JSON research reports.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is in sys.path when executed directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402

from src.evaluation.bd_rate import evaluate_factorial_contrasts  # noqa: E402
from src.evaluation.benchmarks import (  # noqa: E402
    DEFAULT_6POINT_LAMBDAS,
    DEFAULT_BENCHMARK_MODELS,
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkSuiteResult,
    generate_synthetic_benchmark_curve,
)
from src.evaluation.synthesis import ResearchSynthesizer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("compile_research_artifacts")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for artifact compilation."""
    parser = argparse.ArgumentParser(
        description="Compile publication-ready research artifacts, figures, and summary report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/research_artifacts",
        help="Destination directory for all publication figures and reports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Execute synthetic sweep and generate demonstration figures without real models/datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["kodak", "mit1003"],
        help="Datasets to evaluate in real mode ('kodak', 'mit1003', 'synthetic').",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_BENCHMARK_MODELS,
        help="Model architectures to evaluate ('A', 'B', 'C', 'D', 'C-Transmitted').",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=DEFAULT_6POINT_LAMBDAS,
        help="Rate points (Lagrange multipliers lambda) for evaluation sweep.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional maximum number of dataset images to evaluate per benchmark.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for BD-rate confidence intervals.",
    )

    return parser.parse_args(args)


def run_dry_run_synthesis(
    output_dir: Path,
    models: List[str],
    lambdas: List[float],
    bootstrap_resamples: int = 100,
) -> int:
    """Executes rapid synthetic dry-run synthesis generating full publication artifact suite."""
    logger.info("Executing dry-run research artifact compilation...")

    suite_results: Dict[str, BenchmarkSuiteResult] = {}
    datasets = ["kodak", "mit1003"]

    runner = BenchmarkRunner(
        BenchmarkConfig(
            models=models,
            lambdas=lambdas,
            datasets=datasets,
            output_dir=output_dir,
            dry_run=True,
        )
    )

    for ds in datasets:
        model_curves = {}
        for m in models:
            model_curves[m] = generate_synthetic_benchmark_curve(m, lambdas, ds)

        # Compute BD-rates against Model A baseline
        runner._compute_curve_bd_rates(model_curves)

        # Factorial contrasts
        contrasts = None
        if all(k in model_curves for k in ("A", "B", "C", "D")):
            rates_dict = {
                k: np.vstack([p.per_image_bpp for p in model_curves[k].points]).T for k in ("A", "B", "C", "D")
            }
            dist_dict = {
                k: np.vstack([p.per_image_roi_psnr for p in model_curves[k].points]).T for k in ("A", "B", "C", "D")
            }
            contrasts = evaluate_factorial_contrasts(
                rates_by_model=rates_dict,
                dist_by_model=dist_dict,
                n_bootstraps=bootstrap_resamples,
                seed=42,
            )

        suite_results[ds] = BenchmarkSuiteResult(
            dataset=ds,
            models=list(model_curves.keys()),
            lambdas=lambdas,
            model_curves=model_curves,
            contrasts=contrasts,
        )

    from src.evaluation.causal import CausalRegressionResult

    causal_regression = CausalRegressionResult(
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
        num_cells_total=24 * 32 * 48,
        degrees_of_freedom=24 * 32 * 48 - 26,
        hypothesis_confirmed=True,
    )

    synthesizer = ResearchSynthesizer(output_dir=output_dir)
    summary = synthesizer.synthesize(suite_results, causal_regression=causal_regression)

    print("\n" + "=" * 78)
    print(" RESEARCH EVALUATION SYNTHESIS REPORT (DRY-RUN)")
    print("=" * 78)
    print(f"Artifacts Output Directory: {output_dir.resolve()}")
    print(f"Report Markdown:            {output_dir / 'research_report.md'}")
    print(f"Summary JSON:               {output_dir / 'research_summary.json'}")
    print("\nPre-Registered Hypotheses Status:")
    for h in summary.hypotheses:
        verdict = "[PASS] CONFIRMED" if h.confirmed else "[FAIL] NOT CONFIRMED"
        print(f"  {h.hypothesis_id}: {verdict:<16} | {h.target_metric} = {h.observed_value:.4f}")

    print("\nGenerated Publication Figures:")
    for name, p in summary.figure_paths.items():
        print(f"  - {name:<22}: {Path(p).name}")
    print("=" * 78 + "\n")

    return 0


def main(args: Optional[List[str]] = None) -> int:
    """Entry point for compile_research_artifacts CLI."""
    parsed = parse_args(args)
    out_dir = Path(parsed.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if parsed.dry_run:
        return run_dry_run_synthesis(
            output_dir=out_dir,
            models=parsed.models,
            lambdas=parsed.lambdas,
            bootstrap_resamples=min(parsed.bootstrap_resamples, 200),
        )

    # Real benchmark execution
    logger.info("Executing real benchmark sweep on datasets: %s", parsed.datasets)
    cfg = BenchmarkConfig(
        models=parsed.models,
        lambdas=parsed.lambdas,
        datasets=parsed.datasets,
        output_dir=out_dir / "benchmarks",
        dry_run=False,
        sample=parsed.sample,
        device=parsed.device,
        bootstrap_resamples=parsed.bootstrap_resamples,
    )

    runner = BenchmarkRunner(cfg)
    suite_results = runner.run_sweep()

    synthesizer = ResearchSynthesizer(output_dir=out_dir)
    summary = synthesizer.synthesize(suite_results)

    logger.info(
        "Research artifact synthesis completed successfully at %s (%d hypotheses evaluated)",
        out_dir,
        len(summary.hypotheses),
    )
    print(f"Research Report: {out_dir / 'research_report.md'}")
    print(f"Summary JSON:    {out_dir / 'research_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
