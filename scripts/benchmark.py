"""Command-line benchmark orchestrator for learned neural compression models.

Executes comprehensive 6-point rate sweeps across Models A, B, C, D, and C-Transmitted
on standard evaluation datasets (Kodak with DeepGaze IIE independent masks and MIT1003
with human eye-tracking fixation splits). Evaluates global and ROI Bjøntegaard-Delta
rates, Relative Allocation Ratios (RAR), net hyperprior overhead (Delta BPP_z), and
pre-registered factorial contrasts with MLflow experiment tracking.

Usage:
    python scripts/benchmark.py --dry-run
    python scripts/benchmark.py --models A B C D C-Transmitted --datasets kodak --dry-run
    python scripts/benchmark.py --checkpoints-dir checkpoints/ --datasets kodak mit1003
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path when executed directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch  # noqa: E402

from src.evaluation.benchmarks import (  # noqa: E402
    DEFAULT_6POINT_LAMBDAS,
    DEFAULT_BENCHMARK_MODELS,
    BenchmarkConfig,
    BenchmarkRunner,
)

logger = logging.getLogger(__name__)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the benchmark runner CLI."""
    parser = argparse.ArgumentParser(
        description="Comprehensive 6-point rate sweep CLI and benchmark runner for saliency-aware neural compression.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_BENCHMARK_MODELS),
        help="List of model architectures to benchmark (A, B, C, D, C-Transmitted).",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=list(DEFAULT_6POINT_LAMBDAS),
        help="List of rate-distortion trade-off lambda parameters to sweep.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["kodak"],
        help="List of benchmark datasets to evaluate (kodak, mit1003, div2k).",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=None,
        help="Base directory containing trained model checkpoints (e.g. checkpoints/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/benchmark_results",
        help="Directory to store benchmark reports, summary tables, and RD curve figures.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to execute neural network inference on.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without loading models or computing heavy metrics.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit number of evaluation images per dataset (useful for rapid testing).",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="Number of Monte Carlo bootstrap resamples for BD-Rate confidence intervals.",
    )
    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        help="Enable MLflow experiment and artifact tracking.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="saliency_benchmark",
        help="MLflow experiment name for tracking runs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable Matplotlib rate-distortion curve plot generation.",
    )
    parser.add_argument(
        "--lpips-net",
        type=str,
        default="vgg",
        choices=["vgg", "alex"],
        help="LPIPS perceptual network backbone.",
    )

    return parser.parse_args(args)


def main(args_list: list[str] | None = None) -> int:
    """Entry point for the benchmark execution script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args(args_list)

    print("=" * 80)
    print("  Saliency-Aware Neural Compression: Comprehensive Benchmark Suite")
    print("=" * 80)
    print(f"Models:           {', '.join(args.models)}")
    print(f"Lambdas:          {', '.join(f'{x:.4f}' for x in sorted(args.lambdas))}")
    print(f"Datasets:         {', '.join(args.datasets)}")
    print(f"Device:           {args.device}")
    print(f"Checkpoints Dir:  {args.checkpoints_dir or 'None (instantiate/mock)'}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Dry-Run Mode:     {args.dry_run}")
    print(f"MLflow Logging:   {args.use_mlflow}")
    print("=" * 80)

    config = BenchmarkConfig(
        models=args.models,
        lambdas=args.lambdas,
        datasets=args.datasets,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
        device=args.device,
        dry_run=args.dry_run,
        sample=args.sample,
        bootstrap_resamples=args.bootstrap_resamples,
        use_mlflow=args.use_mlflow,
        experiment_name=args.experiment_name,
        save_plots=not args.no_plots,
        lpips_net=args.lpips_net,
    )

    runner = BenchmarkRunner(config)
    t0 = time.perf_counter()
    results = runner.run_sweep()
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 80)
    print(f"  Benchmark Sweep Complete (Elapsed: {elapsed:.2f}s)")
    print("=" * 80)

    for d_name, suite_res in results.items():
        print(f"\n--- Dataset: {d_name.upper()} ---")
        header_str = (
            f"{'Model':<15} | {'Global BD-Rate':<16} | {'ROI BD-Rate':<16} | "
            f"{'Global BD-PSNR':<15} | {'ROI BD-PSNR':<15}"
        )
        print(header_str)
        print("-" * 85)

        for m_id, curve in suite_res.model_curves.items():
            if m_id == "A":
                g_bd = "0.00% (anchor)"
                r_bd = "0.00% (anchor)"
            else:
                g_bd = f"{curve.global_bd_rate:+.2f}%" if curve.global_bd_rate is not None else "N/A"
                r_bd = f"{curve.roi_bd_rate:+.2f}%" if curve.roi_bd_rate is not None else "N/A"
            g_psnr = f"{curve.global_bd_psnr:+.2f} dB" if curve.global_bd_psnr is not None else "0.00 dB"
            r_psnr = f"{curve.roi_bd_psnr:+.2f} dB" if curve.roi_bd_psnr is not None else "0.00 dB"
            print(f"{m_id:<15} | {g_bd:<16} | {r_bd:<16} | {g_psnr:<15} | {r_psnr:<15}")

        if suite_res.contrasts is not None:
            c = suite_res.contrasts
            ci_str = f"[{c.contrast_interaction.ci_lower:+.2f}%, {c.contrast_interaction.ci_upper:+.2f}%]"
            p_str = f"{c.contrast_interaction.p_value:.4f}"
            print("\nPre-Registered Factorial Contrasts:")
            print(f"  Mechanism under saliency loss (C - B):  {c.contrast_c_minus_b.point_estimate:+.2f}%")
            print(f"  Mechanism under uniform loss (D - A):   {c.contrast_d_minus_a.point_estimate:+.2f}%")
            print(
                f"  Interaction ((C - B) - (D - A)):        "
                f"{c.contrast_interaction.point_estimate:+.2f}% (95% CI: {ci_str}, p={p_str})"
            )

        d_out = Path(args.output_dir) / d_name
        print(f"\nArtifacts saved in {d_out}:")
        print(f"  - Results JSON:    {d_out / 'benchmark_results.json'}")
        print(f"  - Summary CSV:     {d_out / 'benchmark_summary.csv'}")
        print(f"  - Summary Report:  {d_out / 'benchmark_summary.md'}")
        if not args.no_plots:
            print(f"  - RD Curve Plot:   {d_out / 'rd_curves.png'}")

    print("\nBenchmark successfully completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
