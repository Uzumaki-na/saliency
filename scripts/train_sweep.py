"""Multi-rate training sweep orchestrator for Model A and Model B.

Orchestrates sequential multi-rate training across standard rate points (lambdas),
aggregates validation metrics, and produces:
- sweep_results.json (structured JSON metrics)
- sweep_results.csv (CSV comparison table)
- sweep_summary.md (Markdown comparison table)
- rd_curve.png (matplotlib rate-distortion curves: PSNR vs. BPP and MS-SSIM vs. BPP)

Usage:
    python scripts/train_sweep.py --dry-run
    python scripts/train_sweep.py --models A B --lambdas 0.0018 0.0035 0.0067 0.0130 0.0250 0.0483
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Headless matplotlib backend configuration must precede pyplot import
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# Ensure project root is in sys.path when executed directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.train import TrainingConfig, train  # noqa: E402

logger = logging.getLogger(__name__)

# Pre-registered standard rate points and models
DEFAULT_LAMBDAS: tuple[float, ...] = (0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483)
DEFAULT_MODELS: tuple[str, ...] = ("A", "B")

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "A": "Model A (Uniform MSE)",
    "B": "Model B (Saliency-Weighted)",
}

MODEL_PLOT_STYLES: dict[str, dict[str, Any]] = {
    "A": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "B": {"color": "#d62728", "marker": "s", "linestyle": "-"},
}


# ---------------------------------------------------------------------------
# Sweep Configuration
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """Unified configuration dataclass for multi-rate training sweep orchestration."""

    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    lambdas: list[float] = field(default_factory=lambda: list(DEFAULT_LAMBDAS))
    output_dir: str | Path = "experiments/sweep_results"
    epochs: int = 100
    batch_size: int = 2
    accum_steps: int = 2
    lr: float = 5e-5
    aux_lr: float = 1e-3
    min_lr: float = 1e-5
    patch_size: int = 256
    device: str = "auto"
    seed: int = 42
    checkpoint_dir: str | Path | None = None
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
    experiment_name: str = "saliency-compression-sweep"
    tracking_uri: str | Path | None = None
    n_channels: int = 128
    m_channels: int = 192
    compile_cdfs: bool = True
    save_best: bool = True
    best_metric: str = "loss"
    dry_run: bool = False
    fail_fast: bool = False
    train_fn: Callable[..., dict[str, Any]] | None = None
    train_dataloader: DataLoader | None = None
    val_dataloader: DataLoader | None = None

    def __post_init__(self) -> None:
        if self.accum_steps <= 0:
            raise ValueError(f"accum_steps must be >= 1, got {self.accum_steps}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

        if not self.models:
            raise ValueError("models list must not be empty.")
        normalized_models: list[str] = []
        for m in self.models:
            norm_m = str(m).upper().strip()
            if norm_m in ("A", "MODELA"):
                normalized_models.append("A")
            elif norm_m in ("B", "MODELB"):
                normalized_models.append("B")
            else:
                raise ValueError(f"Invalid model '{m}'. Expected 'A' or 'B'.")
        self.models = normalized_models

        if not self.lambdas:
            raise ValueError("lambdas list must not be empty.")
        normalized_lambdas: list[float] = []
        for val in self.lambdas:
            flt_val = float(val)
            if flt_val <= 0.0:
                raise ValueError(f"Lambda values must be positive, got {flt_val}")
            normalized_lambdas.append(flt_val)
        self.lambdas = normalized_lambdas

        norm_metric = str(self.best_metric).lower().strip()
        if norm_metric not in ("loss", "psnr"):
            raise ValueError(f"Invalid best_metric '{self.best_metric}'. Expected 'loss' or 'psnr'.")
        self.best_metric = norm_metric

        self.output_dir = Path(self.output_dir)
        if self.checkpoint_dir is not None:
            self.checkpoint_dir = Path(self.checkpoint_dir)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> SweepConfig:
        """Construct SweepConfig from parsed argparse.Namespace."""
        kwargs: dict[str, Any] = {}
        for f in cls.__dataclass_fields__:
            if hasattr(args, f):
                kwargs[f] = getattr(args, f)
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SweepConfig:
        """Construct SweepConfig from a dictionary with key normalization and alias resolution."""
        normalized: dict[str, Any] = {}
        for raw_k, v in data.items():
            k = raw_k.replace("-", "_")
            if k in ("model", "model_type"):
                if isinstance(v, str):
                    normalized["models"] = [v]
                elif isinstance(v, (list, tuple)):
                    normalized["models"] = list(v)
                continue
            elif k in ("lambda", "lambda_rd", "lmbda"):
                if isinstance(v, (int, float)):
                    normalized["lambdas"] = [float(v)]
                elif isinstance(v, (list, tuple)):
                    normalized["lambdas"] = [float(x) for x in v]
                continue
            elif k == "crop_size":
                k = "patch_size"
            normalized[k] = v

        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in normalized.items() if k in valid_fields}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Metric & Run Extraction Helpers
# ---------------------------------------------------------------------------


def _extract_runs(results: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalizes various container formats into a list of run dictionaries."""
    if isinstance(results, list):
        return results
    if isinstance(results, dict):
        if "runs" in results and isinstance(results["runs"], list):
            return results["runs"]
        if "results" in results and isinstance(results["results"], list):
            return results["results"]
        if "plan" in results and isinstance(results["plan"], list):
            return results["plan"]
        if all(isinstance(v, dict) for v in results.values()):
            return list(results.values())
    return []


def _sort_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sorts run records deterministically by model name and lambda value."""

    def _key(r: dict[str, Any]) -> tuple[str, float]:
        model_name = str(r.get("model", ""))
        lmbda_val = r.get("lambda", 0.0)
        try:
            flt_lambda = float(lmbda_val) if lmbda_val is not None else 0.0
        except (ValueError, TypeError):
            flt_lambda = 0.0
        return (model_name, flt_lambda)

    return sorted(runs, key=_key)


# ---------------------------------------------------------------------------
# Reporting & Visualizations
# ---------------------------------------------------------------------------


def generate_summary_table(results: dict[str, Any] | list[dict[str, Any]]) -> str:
    """Generates a GitHub-flavored Markdown comparison table for evaluated runs.

    Args:
        results: Run records dictionary or list containing BPP, MSE, PSNR, MS-SSIM.

    Returns:
        Formatted Markdown table string.
    """
    runs = _extract_runs(results)
    sorted_runs = _sort_runs(runs)

    headers = ["Model", "Lambda", "BPP", "MSE", "PSNR (dB)", "MS-SSIM"]
    header_line = f"| {' | '.join(headers)} |"
    sep_line = f"| {' | '.join(['---'] * len(headers))} |"

    lines = [header_line, sep_line]
    for r in sorted_runs:
        model = str(r.get("model", "N/A"))
        lmbda_val = r.get("lambda")
        try:
            lmbda_str = f"{float(lmbda_val):.4f}" if lmbda_val is not None else "N/A"
        except (ValueError, TypeError):
            lmbda_str = str(lmbda_val)

        bpp_val = r.get("bpp") if r.get("bpp") is not None else r.get("bpp_loss")
        try:
            bpp_str = f"{float(bpp_val):.4f}" if (bpp_val is not None and not np.isnan(float(bpp_val))) else "N/A"
        except (ValueError, TypeError):
            bpp_str = "N/A"

        mse_val = r.get("mse") if r.get("mse") is not None else r.get("mse_loss")
        try:
            mse_str = f"{float(mse_val):.6f}" if (mse_val is not None and not np.isnan(float(mse_val))) else "N/A"
        except (ValueError, TypeError):
            mse_str = "N/A"

        psnr_val = r.get("psnr")
        try:
            psnr_str = f"{float(psnr_val):.2f}" if (psnr_val is not None and not np.isnan(float(psnr_val))) else "N/A"
        except (ValueError, TypeError):
            psnr_str = "N/A"

        msssim_val = r.get("msssim") if r.get("msssim") is not None else r.get("ms_ssim")
        try:
            msssim_str = (
                f"{float(msssim_val):.4f}" if (msssim_val is not None and not np.isnan(float(msssim_val))) else "N/A"
            )
        except (ValueError, TypeError):
            msssim_str = "N/A"

        row = f"| {model:<5} | {lmbda_str:<6} | {bpp_str:<6} | {mse_str:<8} | {psnr_str:<9} | {msssim_str:<7} |"
        lines.append(row)

    return "\n".join(lines)


def generate_csv_data(results: dict[str, Any] | list[dict[str, Any]]) -> str:
    """Generates RFC-4180 compliant CSV string containing aggregated sweep metrics.

    Args:
        results: Run records dictionary or list containing evaluation metrics.

    Returns:
        CSV formatted string.
    """
    runs = _extract_runs(results)
    sorted_runs = _sort_runs(runs)

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["model", "lambda", "bpp", "mse", "psnr", "ms_ssim", "loss"])

    for r in sorted_runs:
        model = str(r.get("model", ""))
        lmbda = r.get("lambda", "")
        bpp = r.get("bpp") if r.get("bpp") is not None else r.get("bpp_loss", "")
        mse = r.get("mse") if r.get("mse") is not None else r.get("mse_loss", "")
        psnr = r.get("psnr", "")
        msssim = r.get("msssim") if r.get("msssim") is not None else r.get("ms_ssim", "")
        loss = r.get("loss", "")
        writer.writerow([model, lmbda, bpp, mse, psnr, msssim, loss])

    return output.getvalue()


def generate_rd_plots(results: dict[str, Any] | list[dict[str, Any]], path: str | Path) -> Path:
    """Renders rate-distortion curves (PSNR vs. BPP and MS-SSIM vs. BPP) using matplotlib.

    Guarantees:
    - Renders headless without display server dependencies using Agg backend.
    - Groups data points by model architecture.
    - Sorts points by bitrate (BPP) for clean curve interpolation.
    - Exports high-resolution PNG or SVG to destination path.

    Args:
        results: Run records dictionary or list containing evaluation metrics.
        path: Destination file path for saved plot.

    Returns:
        Path: Resolved Path instance of the saved plot.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runs = _extract_runs(results)

    # Group runs by model
    model_groups: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        m = str(r.get("model", "Unknown")).upper()
        model_groups.setdefault(m, []).append(r)

    fig, (ax_psnr, ax_msssim) = plt.subplots(1, 2, figsize=(13, 5))

    has_data = False
    for model_name, model_runs in sorted(model_groups.items()):
        # Extract valid points for PSNR vs BPP
        psnr_points: list[tuple[float, float]] = []
        msssim_points: list[tuple[float, float]] = []

        for r in model_runs:
            bpp = r.get("bpp") if r.get("bpp") is not None else r.get("bpp_loss")
            mse = r.get("mse") if r.get("mse") is not None else r.get("mse_loss")
            psnr = r.get("psnr")
            msssim = r.get("msssim") if r.get("msssim") is not None else r.get("ms_ssim")

            # Fallback calculate PSNR from MSE if not directly present
            if psnr is None and mse is not None:
                try:
                    flt_mse = float(mse)
                    if flt_mse > 0.0:
                        psnr = 10.0 * np.log10(1.0 / flt_mse)
                except (ValueError, TypeError):
                    pass

            try:
                if bpp is not None and psnr is not None:
                    flt_bpp = float(bpp)
                    flt_psnr = float(psnr)
                    if not (np.isnan(flt_bpp) or np.isnan(flt_psnr)):
                        psnr_points.append((flt_bpp, flt_psnr))
            except (ValueError, TypeError):
                pass

            try:
                if bpp is not None and msssim is not None:
                    flt_bpp = float(bpp)
                    flt_msssim = float(msssim)
                    if not (np.isnan(flt_bpp) or np.isnan(flt_msssim)):
                        msssim_points.append((flt_bpp, flt_msssim))
            except (ValueError, TypeError):
                pass

        # Sort points by bitrate (BPP) for clean line plotting
        psnr_points.sort(key=lambda pt: pt[0])
        msssim_points.sort(key=lambda pt: pt[0])

        style = MODEL_PLOT_STYLES.get(model_name, {"color": None, "marker": "o", "linestyle": "-"})
        display_name = MODEL_DISPLAY_NAMES.get(model_name, f"Model {model_name}")

        if psnr_points:
            has_data = True
            bpp_vals, psnr_vals = zip(*psnr_points, strict=False)
            ax_psnr.plot(
                bpp_vals,
                psnr_vals,
                label=display_name,
                color=style.get("color"),
                marker=style.get("marker", "o"),
                linestyle=style.get("linestyle", "-"),
                linewidth=2,
                markersize=6,
            )

        if msssim_points:
            has_data = True
            bpp_vals, msssim_vals = zip(*msssim_points, strict=False)
            ax_msssim.plot(
                bpp_vals,
                msssim_vals,
                label=display_name,
                color=style.get("color"),
                marker=style.get("marker", "s"),
                linestyle=style.get("linestyle", "-"),
                linewidth=2,
                markersize=6,
            )

    # Subplot 1: PSNR vs. BPP
    ax_psnr.set_xlabel("Bitrate (bpp)", fontsize=11)
    ax_psnr.set_ylabel("PSNR (dB)", fontsize=11)
    ax_psnr.set_title("Rate-Distortion: PSNR vs. BPP", fontsize=12)
    ax_psnr.grid(True, linestyle="--", alpha=0.5)
    if has_data and ax_psnr.get_legend_handles_labels()[0]:
        ax_psnr.legend(loc="lower right")

    # Subplot 2: MS-SSIM vs. BPP
    ax_msssim.set_xlabel("Bitrate (bpp)", fontsize=11)
    ax_msssim.set_ylabel("MS-SSIM", fontsize=11)
    ax_msssim.set_title("Rate-Distortion: MS-SSIM vs. BPP", fontsize=12)
    ax_msssim.grid(True, linestyle="--", alpha=0.5)
    if has_data and ax_msssim.get_legend_handles_labels()[0]:
        ax_msssim.legend(loc="lower right")

    if not has_data:
        ax_psnr.text(
            0.5,
            0.5,
            "No evaluation data points available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax_psnr.transAxes,
            color="gray",
        )
        ax_msssim.text(
            0.5,
            0.5,
            "No evaluation data points available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax_msssim.transAxes,
            color="gray",
        )

    fig.suptitle("Rate-Distortion Performance Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# Sweep Orchestrator
# ---------------------------------------------------------------------------


def run_sweep(
    config: SweepConfig | argparse.Namespace | dict[str, Any],
) -> dict[str, Any]:
    """Primary programmatic entry point to orchestrate a multi-rate training sweep.

    Workflow:
    1. Parses and validates SweepConfig.
    2. Builds the sequential parameter matrix (models x lambdas).
    3. If dry_run is enabled, prints the planned schedule and returns the execution plan.
    4. Executes training passes sequentially on single GPU/CPU without concurrent contention.
    5. Aggregates evaluation metrics (BPP, MSE, PSNR, MS-SSIM).
    6. Produces multi-format artifacts:
       - sweep_results.json
       - sweep_results.csv
       - sweep_summary.md
       - rd_curve.png

    Args:
        config: SweepConfig, argparse.Namespace, or dictionary of sweep arguments.

    Returns:
        dict[str, Any]: Dictionary containing run metrics, summary tables, artifact paths,
            and execution metadata.
    """
    if isinstance(config, argparse.Namespace):
        sweep_cfg = SweepConfig.from_namespace(config)
    elif isinstance(config, dict):
        sweep_cfg = SweepConfig.from_dict(config)
    elif isinstance(config, SweepConfig):
        sweep_cfg = config
    else:
        raise TypeError(f"Expected SweepConfig, Namespace, or dict, got {type(config).__name__}")

    output_dir = Path(sweep_cfg.output_dir)
    base_ckpt_dir = (
        Path(sweep_cfg.checkpoint_dir) if sweep_cfg.checkpoint_dir is not None else output_dir / "checkpoints"
    )

    # 1. Build sequential job matrix
    plan: list[dict[str, Any]] = []
    for model in sweep_cfg.models:
        for lmbda in sweep_cfg.lambdas:
            run_name = f"{model}_lambda_{lmbda:.4f}"
            run_ckpt_dir = base_ckpt_dir / run_name
            plan.append(
                {
                    "model": model,
                    "lambda": lmbda,
                    "run_name": run_name,
                    "checkpoint_dir": str(run_ckpt_dir),
                    "epochs": sweep_cfg.epochs,
                    "batch_size": sweep_cfg.batch_size,
                    "accum_steps": sweep_cfg.accum_steps,
                    "device": sweep_cfg.device,
                }
            )

    # 2. Dry run gate: return plan without training
    if sweep_cfg.dry_run:
        logger.info(
            "Dry run requested. Planned %d training jobs across models %s and lambdas %s",
            len(plan),
            sweep_cfg.models,
            sweep_cfg.lambdas,
        )
        return {
            "dry_run": True,
            "plan": plan,
            "models": sweep_cfg.models,
            "lambdas": sweep_cfg.lambdas,
            "total_jobs": len(plan),
        }

    # 3. Create output directory for completed sweep
    output_dir.mkdir(parents=True, exist_ok=True)
    train_fn = sweep_cfg.train_fn or train

    runs: list[dict[str, Any]] = []
    total_jobs = len(plan)
    logger.info("Starting sequential execution of %d sweep jobs...", total_jobs)

    for idx, job in enumerate(plan, 1):
        model = job["model"]
        lmbda = job["lambda"]
        run_name = job["run_name"]
        run_ckpt_dir = Path(job["checkpoint_dir"])

        logger.info(
            "[%d/%d] Executing sequential run: Model=%s, Lambda=%.4f (run_name=%s)",
            idx,
            total_jobs,
            model,
            lmbda,
            run_name,
        )

        run_train_cfg = TrainingConfig(
            model=model,
            lmbda=lmbda,
            epochs=sweep_cfg.epochs,
            batch_size=sweep_cfg.batch_size,
            accum_steps=sweep_cfg.accum_steps,
            lr=sweep_cfg.lr,
            aux_lr=sweep_cfg.aux_lr,
            min_lr=sweep_cfg.min_lr,
            patch_size=sweep_cfg.patch_size,
            device=sweep_cfg.device,
            seed=sweep_cfg.seed,
            checkpoint_dir=run_ckpt_dir,
            pretrained_checkpoint=sweep_cfg.pretrained_checkpoint,
            clip_max_norm=sweep_cfg.clip_max_norm,
            train_dataset=sweep_cfg.train_dataset,
            val_dataset=sweep_cfg.val_dataset,
            train_image_dir=sweep_cfg.train_image_dir,
            train_saliency_dir=sweep_cfg.train_saliency_dir,
            val_image_dir=sweep_cfg.val_image_dir,
            val_saliency_dir=sweep_cfg.val_saliency_dir,
            num_workers=sweep_cfg.num_workers,
            use_mlflow=sweep_cfg.use_mlflow,
            experiment_name=sweep_cfg.experiment_name,
            run_name=run_name,
            tracking_uri=sweep_cfg.tracking_uri,
            n_channels=sweep_cfg.n_channels,
            m_channels=sweep_cfg.m_channels,
            compile_cdfs=sweep_cfg.compile_cdfs,
            save_best=sweep_cfg.save_best,
            best_metric=sweep_cfg.best_metric,
            train_dataloader=sweep_cfg.train_dataloader,
            val_dataloader=sweep_cfg.val_dataloader,
        )

        try:
            train_result = train_fn(run_train_cfg)
            val_metrics = train_result.get("val_metrics", train_result)

            bpp = val_metrics.get("bpp") if val_metrics.get("bpp") is not None else val_metrics.get("bpp_loss")
            mse = val_metrics.get("mse") if val_metrics.get("mse") is not None else val_metrics.get("mse_loss")
            psnr = val_metrics.get("psnr")
            msssim = val_metrics.get("msssim") if val_metrics.get("msssim") is not None else val_metrics.get("ms_ssim")
            loss = val_metrics.get("loss")

            run_record: dict[str, Any] = {
                "model": model,
                "lambda": lmbda,
                "bpp": float(bpp) if bpp is not None else None,
                "mse": float(mse) if mse is not None else None,
                "psnr": float(psnr) if psnr is not None else None,
                "msssim": float(msssim) if msssim is not None else None,
                "ms_ssim": float(msssim) if msssim is not None else None,
                "loss": float(loss) if loss is not None else None,
                "best_checkpoint_path": str(train_result.get("best_checkpoint_path") or ""),
                "latest_checkpoint_path": str(train_result.get("latest_checkpoint_path") or ""),
                "status": "success",
            }
            runs.append(run_record)

        except Exception as exc:
            logger.error(
                "Training failed during sweep for model %s at lambda %.4f: %s",
                model,
                lmbda,
                exc,
                exc_info=True,
            )
            if sweep_cfg.fail_fast:
                raise
            run_record = {
                "model": model,
                "lambda": lmbda,
                "bpp": None,
                "mse": None,
                "psnr": None,
                "msssim": None,
                "ms_ssim": None,
                "loss": None,
                "best_checkpoint_path": "",
                "latest_checkpoint_path": "",
                "status": "failed",
                "error": str(exc),
            }
            runs.append(run_record)

    # 4. Collate and sort metrics
    sorted_runs = _sort_runs(runs)

    # 5. Output multi-format reporting artifacts
    summary_md = generate_summary_table(sorted_runs)
    md_path = output_dir / "sweep_summary.md"
    md_path.write_text(summary_md, encoding="utf-8")

    csv_data = generate_csv_data(sorted_runs)
    csv_path = output_dir / "sweep_results.csv"
    csv_path.write_text(csv_data, encoding="utf-8", newline="")

    json_path = output_dir / "sweep_results.json"
    results_payload = {
        "models": sweep_cfg.models,
        "lambdas": sweep_cfg.lambdas,
        "runs": sorted_runs,
    }
    json_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")

    rd_plot_path = output_dir / "rd_curve.png"
    generate_rd_plots(sorted_runs, rd_plot_path)

    logger.info("Sweep execution completed successfully. Artifacts saved to: %s", output_dir)

    return {
        "dry_run": False,
        "runs": sorted_runs,
        "summary_table": summary_md,
        "csv_data": csv_data,
        "results_json_path": json_path,
        "results_csv_path": csv_path,
        "summary_md_path": md_path,
        "rd_plot_path": rd_plot_path,
        "output_dir": output_dir,
    }


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# ---------------------------------------------------------------------------


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Builds and parses CLI arguments for multi-rate training sweep orchestrator."""
    parser = argparse.ArgumentParser(
        description="Sequential multi-rate training sweep orchestrator for Model A and Model B."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=["A", "B"],
        help="List of model architectures to evaluate, e.g. --models A B (default: 'A' 'B').",
    )
    parser.add_argument(
        "--lambdas",
        "--lambda",
        nargs="+",
        type=float,
        default=list(DEFAULT_LAMBDAS),
        help=f"List of rate-distortion lambda values (default: {list(DEFAULT_LAMBDAS)}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Display the execution plan and parameter schedule without training.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/sweep_results",
        help="Directory to save sweep results, tables, and plots (default: 'experiments/sweep_results').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total number of training epochs per run (default: 100).",
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
        help="Gradient accumulation step count (default: 2).",
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
        default=None,
        help="Base directory to save checkpoints (defaults to output_dir / checkpoints).",
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
        default="saliency-compression-sweep",
        help="MLflow experiment name (default: 'saliency-compression-sweep').",
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
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        default=False,
        help="Immediately raise exception if an individual training run fails.",
    )

    parsed = parser.parse_args(args)
    if parsed.accum_steps <= 0:
        raise ValueError(f"accum_steps must be >= 1, got {parsed.accum_steps}")
    return parsed


def main(args: Sequence[str] | None = None) -> None:
    """CLI execution entry point."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parsed = parse_args(args)
    config = SweepConfig.from_namespace(parsed)
    results = run_sweep(config)

    if results.get("dry_run"):
        print("\nDry run completed successfully. Planned jobs:")
        plan = results.get("plan", [])
        for idx, job in enumerate(plan, 1):
            print(
                f"  [{idx:02d}] Model: {job['model']} | "
                f"Lambda: {job['lambda']:.4f} | "
                f"Run: {job['run_name']} | "
                f"Checkpoint: {job['checkpoint_dir']}"
            )
    else:
        print("\nSweep completed successfully!")
        print(f"Results JSON:   {results.get('results_json_path')}")
        print(f"Results CSV:    {results.get('results_csv_path')}")
        print(f"Summary Table:  {results.get('summary_md_path')}")
        print(f"RD Curve Plot:  {results.get('rd_plot_path')}")
        print("\nSummary Table:")
        print(results.get("summary_table"))


if __name__ == "__main__":
    main()
