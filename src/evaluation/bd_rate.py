"""Bjøntegaard-Delta evaluation engine with bootstrap confidence intervals.

Implements Bjøntegaard-Delta rate (BD-Rate) and distortion (BD-PSNR / BD-Metric)
algorithms using both standard cubic polynomial (ITU-T VCEG-M33) and Akima
piecewise cubic interpolation (Akima 1970; scipy.interpolate.Akima1DInterpolator),
along with PCHIP interpolation (Fritsch & Carlson 1980).

Provides:
1. Rate-distortion Pareto monotonicity enforcement and plateau handling.
2. BD-Rate (% rate saving) and BD-Metric (average distortion gain).
3. 1000-resample image-level bootstrap 95% confidence intervals and hypothesis tests.
4. Pre-registered factorial contrasts:
   - Mechanism effect under saliency loss: Contrast (C - B)
   - Mechanism effect under uniform MSE loss: Contrast (D - A)
   - Super-additive interaction effect: ((C - B) - (D - A))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
from scipy.optimize import isotonic_regression

logger = logging.getLogger(__name__)

BDRateMethod = Literal["cubic", "akima", "pchip"]
MetricMode = Literal["higher_is_better", "lower_is_better"]


@dataclass
class BDComparisonResult:
    """Detailed result of a pairwise Bjøntegaard-Delta rate and metric comparison.

    Attributes:
        bd_rate: Percentage rate change (negative indicates rate savings/efficiency gain).
        bd_metric: Average difference in distortion metric units across matched rate range
            (positive indicates gain for higher-is-better metrics such as PSNR;
             negative indicates gain for lower-is-better metrics such as LPIPS).
        overlap_rate_range: Overlapping rate range [r_min, r_max] in native rate units.
        overlap_metric_range: Overlapping metric range [d_min, d_max] in native metric units.
        method: Interpolation method used ("cubic", "akima", or "pchip").
        metric_mode: Optimization direction ("higher_is_better" or "lower_is_better").
        ref_num_points: Number of valid points in reference curve.
        target_num_points: Number of valid points in target curve.
    """

    bd_rate: float
    bd_metric: float
    overlap_rate_range: Tuple[float, float]
    overlap_metric_range: Tuple[float, float]
    method: BDRateMethod
    metric_mode: MetricMode
    ref_num_points: int
    target_num_points: int

    def __repr__(self) -> str:
        rate_str = f"{self.bd_rate:+.4f}%" if not math.isnan(self.bd_rate) else "NaN"
        metric_str = f"{self.bd_metric:+.4f}" if not math.isnan(self.bd_metric) else "NaN"
        return (
            f"BDComparisonResult(bd_rate={rate_str}, bd_metric={metric_str}, "
            f"method='{self.method}', metric_mode='{self.metric_mode}')"
        )


@dataclass
class BootstrapResult:
    """Bootstrap inference statistics for a rate-distortion metric or contrast.

    Attributes:
        point_estimate: Point estimate evaluated on the full aggregate dataset.
        bootstrap_mean: Mean of the bootstrap resample estimates.
        std_error: Standard error derived from bootstrap sample standard deviation.
        ci_lower: Lower bound of the confidence interval (e.g. 2.5th percentile for 95% CI).
        ci_upper: Upper bound of the confidence interval (e.g. 97.5th percentile for 95% CI).
        confidence_level: Confidence level (e.g. 0.95 for 95% CI).
        p_value: Empirical two-tailed p-value testing the null hypothesis H0: effect == 0.
        hypothesis_confirmed: True if null hypothesis is rejected at significance
            alpha = (1 - confidence_level) in the expected beneficial direction.
        n_bootstraps: Number of valid bootstrap resamples conducted.
        bootstrap_samples: 1D NumPy array of bootstrap resample estimates.
    """

    point_estimate: float
    bootstrap_mean: float
    std_error: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    p_value: float
    hypothesis_confirmed: bool
    n_bootstraps: int
    bootstrap_samples: np.ndarray = field(repr=False)


@dataclass
class FactorialContrastsResult:
    """Pre-registered 4-model factorial contrasts and mechanism tests.

    Attributes:
        contrast_c_minus_b: Mechanism effect under saliency loss (Model C vs Model B).
        contrast_d_minus_a: Mechanism effect under uniform MSE loss (Model D vs Model A).
        contrast_interaction: Super-additive interaction effect ((C - B) - (D - A)).
        pairwise_contrasts: Mapping of all pairwise model comparisons (e.g. "B_vs_A", "C_vs_A").
        method: Interpolation method used ("cubic", "akima", or "pchip").
        metric_mode: Optimization direction ("higher_is_better" or "lower_is_better").
        confidence_level: Confidence level for all intervals (e.g. 0.95).
        n_bootstraps: Number of resamples performed.
    """

    contrast_c_minus_b: BootstrapResult
    contrast_d_minus_a: BootstrapResult
    contrast_interaction: BootstrapResult
    pairwise_contrasts: Dict[str, BootstrapResult]
    method: BDRateMethod
    metric_mode: MetricMode
    confidence_level: float
    n_bootstraps: int


def _to_numpy_1d(
    x: Union[Sequence[float], np.ndarray, torch.Tensor],
    name: str = "x",
) -> np.ndarray:
    """Convert input sequence, NumPy array, or PyTorch tensor to a 1D float64 NumPy array."""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x, dtype=np.float64)

    arr = np.atleast_1d(arr).ravel().astype(np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Input '{name}' must be 1-dimensional, got shape {arr.shape}.")
    return arr


def _to_numpy_2d(
    x: Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor],
    name: str = "x",
) -> np.ndarray:
    """Convert 2D input sequence, NumPy array, or PyTorch tensor to a 2D float64 NumPy array."""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x, dtype=np.float64)

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Input '{name}' must be 2-dimensional of shape (n_images, n_lambdas), got shape {arr.shape}.")
    return arr


def enforce_monotonicity(
    rates: Union[Sequence[float], np.ndarray, torch.Tensor],
    distortions: Union[Sequence[float], np.ndarray, torch.Tensor],
    metric_mode: MetricMode = "higher_is_better",
) -> Tuple[np.ndarray, np.ndarray]:
    """Sort rate-distortion points by rate and enforce strict Pareto monotonicity.

    In rate-distortion theory (ITU-T VCEG / JVET / AOM), points with higher rate
    but worse or equal distortion are strictly dominated operating points. Pareto
    filtering eliminates dominated points and duplicate rates, ensuring strictly
    increasing coordinates without introducing microscopic knot gaps that cause
    spline derivative oscillations.

    If Pareto filtering yields fewer than 2 points, isotonic regression is applied
    as a convex fallback, followed by plateau deduplication.

    Args:
        rates: Sequence of bitrate values (must be strictly positive).
        distortions: Sequence of distortion/quality values.
        metric_mode: "higher_is_better" (e.g. PSNR, MS-SSIM) or
                     "lower_is_better" (e.g. LPIPS, MSE).

    Returns:
        Tuple of (sorted_rates, monotonic_distortions) as 1D float64 NumPy arrays.

    Raises:
        ValueError: If array lengths do not match, fewer than 2 points are provided,
            or rates are non-positive.
    """
    r_arr = _to_numpy_1d(rates, name="rates")
    d_arr = _to_numpy_1d(distortions, name="distortions")

    if len(r_arr) != len(d_arr):
        raise ValueError(f"Length mismatch: rates has {len(r_arr)} points, distortions has {len(d_arr)} points.")
    if len(r_arr) < 2:
        raise ValueError(f"At least 2 rate-distortion points are required, got {len(r_arr)}.")
    if np.any(r_arr <= 0.0):
        raise ValueError(f"All rates must be strictly positive (> 0), got minimum {r_arr.min()}.")

    # 1. Sort primarily by rate ascending
    sort_idx = np.argsort(r_arr)
    r_sorted = r_arr[sort_idx]
    d_sorted = d_arr[sort_idx]

    # 2. Pareto frontier filter: prune strictly dominated operating points
    pareto_r: List[float] = [float(r_sorted[0])]
    pareto_d: List[float] = [float(d_sorted[0])]

    for i in range(1, len(r_sorted)):
        cur_r = float(r_sorted[i])
        cur_d = float(d_sorted[i])

        if cur_r == pareto_r[-1]:
            # Same rate: keep the one with better quality
            if metric_mode == "higher_is_better" and cur_d > pareto_d[-1]:
                pareto_d[-1] = cur_d
            elif metric_mode == "lower_is_better" and cur_d < pareto_d[-1]:
                pareto_d[-1] = cur_d
            continue

        if metric_mode == "higher_is_better":
            if cur_d > pareto_d[-1]:
                pareto_r.append(cur_r)
                pareto_d.append(cur_d)
        elif metric_mode == "lower_is_better":
            if cur_d < pareto_d[-1]:
                pareto_r.append(cur_r)
                pareto_d.append(cur_d)
        else:
            raise ValueError(f"Invalid metric_mode '{metric_mode}'. Expected 'higher_is_better' or 'lower_is_better'.")

    if len(pareto_r) >= 2:
        return np.array(pareto_r, dtype=np.float64), np.array(pareto_d, dtype=np.float64)

    # 3. Fallback: isotonic regression (PAVA) + plateau deduplication
    increasing = metric_mode == "higher_is_better"
    iso_res = isotonic_regression(d_sorted, increasing=increasing)
    iso_d = iso_res.x

    unique_d: List[float] = []
    unique_r: List[float] = []
    for r_val, d_val in zip(r_sorted, iso_d, strict=False):
        if not unique_d or d_val != unique_d[-1]:
            unique_d.append(float(d_val))
            unique_r.append(float(r_val))

    if len(unique_d) >= 2:
        return np.array(unique_r, dtype=np.float64), np.array(unique_d, dtype=np.float64)

    # Minimal 2-point fallback
    return r_sorted[:2], d_sorted[:2]


def _integrate_curve(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    method: BDRateMethod,
) -> float:
    """Fit curve y = f(x) and evaluate the definite integral from x_min to x_max.

    Args:
        x: Strictly sorted, strictly increasing independent variable coordinates.
        y: Dependent variable coordinates.
        x_min: Lower integration limit.
        x_max: Upper integration limit.
        method: "cubic" (polynomial OLS), "akima" (Akima spline), or "pchip".

    Returns:
        Definite integral float value.
    """
    if x_max <= x_min:
        return 0.0

    if method == "akima":
        if len(x) < 4:
            # Fall back to Pchip if fewer than 4 points for Akima stability
            interpolator = PchipInterpolator(x, y)
        else:
            interpolator = Akima1DInterpolator(x, y)
        val = float(interpolator.integrate(x_min, x_max))
        return val

    elif method == "pchip":
        interpolator = PchipInterpolator(x, y)
        val = float(interpolator.integrate(x_min, x_max))
        return val

    elif method == "cubic":
        # Degree 3 polynomial OLS fit (or degree min(3, len(x)-1))
        deg = min(3, len(x) - 1)
        poly_coeffs = np.polyfit(x, y, deg=deg)
        poly_integral = np.polyint(poly_coeffs)
        val = float(np.polyval(poly_integral, x_max) - np.polyval(poly_integral, x_min))
        return val

    else:
        raise ValueError(f"Unsupported method '{method}'. Supported methods are 'cubic', 'akima', 'pchip'.")


def compute_bd_rate(
    rates_ref: Union[Sequence[float], np.ndarray, torch.Tensor],
    dist_ref: Union[Sequence[float], np.ndarray, torch.Tensor],
    rates_target: Union[Sequence[float], np.ndarray, torch.Tensor],
    dist_target: Union[Sequence[float], np.ndarray, torch.Tensor],
    method: BDRateMethod = "akima",
    metric_mode: MetricMode = "higher_is_better",
    enforce_monotonic: bool = True,
    strict_overlap: bool = False,
) -> float:
    """Compute the Bjøntegaard-Delta Rate (BD-Rate) percentage difference.

    Computes the average percentage rate change of `target` relative to `ref`
    over the overlapping distortion range. Negative values indicate rate savings
    (e.g., -15.4% indicates target requires 15.4% less rate for the same quality).

    Args:
        rates_ref: Bitrates of the reference model.
        dist_ref: Distortion/quality metric values of the reference model.
        rates_target: Bitrates of the target model under evaluation.
        dist_target: Distortion/quality metric values of the target model.
        method: Interpolation method: "akima" (default), "cubic" (ITU-T VCEG-M33),
                or "pchip" (monotonic shape-preserving spline).
        metric_mode: "higher_is_better" (e.g. PSNR) or "lower_is_better" (e.g. LPIPS).
        enforce_monotonic: If True, applies Pareto monotonicity filtering.
        strict_overlap: If True, raises ValueError when curves do not overlap;
                        if False, returns float('nan').

    Returns:
        BD-Rate percentage float (e.g. -12.5). Returns NaN if distortion ranges do not overlap.
    """
    if enforce_monotonic:
        r_ref, d_ref = enforce_monotonicity(rates_ref, dist_ref, metric_mode=metric_mode)
        r_tgt, d_tgt = enforce_monotonicity(rates_target, dist_target, metric_mode=metric_mode)
    else:
        r_ref = _to_numpy_1d(rates_ref, name="rates_ref")
        d_ref = _to_numpy_1d(dist_ref, name="dist_ref")
        r_tgt = _to_numpy_1d(rates_target, name="rates_target")
        d_tgt = _to_numpy_1d(dist_target, name="dist_target")

    if len(r_ref) < 2 or len(r_tgt) < 2:
        if strict_overlap:
            raise ValueError("At least 2 points are required on each curve after monotonicity enforcement.")
        return float("nan")

    # Natural logarithm of rates
    log_r_ref = np.log(r_ref)
    log_r_tgt = np.log(r_tgt)

    # Sort coordinates by distortion ascending for integration
    idx_ref = np.argsort(d_ref)
    d_ref_sorted = d_ref[idx_ref]
    log_r_ref_sorted = log_r_ref[idx_ref]

    idx_tgt = np.argsort(d_tgt)
    d_tgt_sorted = d_tgt[idx_tgt]
    log_r_tgt_sorted = log_r_tgt[idx_tgt]

    # Overlapping distortion interval [d_min, d_max]
    d_min = max(float(d_ref_sorted.min()), float(d_tgt_sorted.min()))
    d_max = min(float(d_ref_sorted.max()), float(d_tgt_sorted.max()))

    if d_max <= d_min:
        if strict_overlap:
            raise ValueError(
                f"Distortion ranges do not overlap: ref=[{d_ref_sorted.min():.4f}, {d_ref_sorted.max():.4f}], "
                f"target=[{d_tgt_sorted.min():.4f}, {d_tgt_sorted.max():.4f}]."
            )
        return float("nan")

    # Integrate log rate over overlapping distortion range
    int_ref = _integrate_curve(d_ref_sorted, log_r_ref_sorted, d_min, d_max, method=method)
    int_tgt = _integrate_curve(d_tgt_sorted, log_r_tgt_sorted, d_min, d_max, method=method)

    avg_log_rate_diff = (int_tgt - int_ref) / (d_max - d_min)

    try:
        bd_rate_pct = math.expm1(avg_log_rate_diff) * 100.0
    except OverflowError:
        bd_rate_pct = float("inf") if avg_log_rate_diff > 0 else -100.0

    return float(bd_rate_pct)


def compute_bd_metric(
    rates_ref: Union[Sequence[float], np.ndarray, torch.Tensor],
    dist_ref: Union[Sequence[float], np.ndarray, torch.Tensor],
    rates_target: Union[Sequence[float], np.ndarray, torch.Tensor],
    dist_target: Union[Sequence[float], np.ndarray, torch.Tensor],
    method: BDRateMethod = "akima",
    metric_mode: MetricMode = "higher_is_better",
    enforce_monotonic: bool = True,
    strict_overlap: bool = False,
) -> float:
    """Compute the Bjøntegaard-Delta Metric / BD-PSNR difference.

    Computes the average distortion difference of `target` relative to `ref`
    over the overlapping log-rate range.

    For "higher_is_better" metrics (e.g. PSNR in dB):
        Positive values indicate quality gain (e.g. +0.75 dB).
    For "lower_is_better" metrics (e.g. LPIPS):
        Negative values indicate perceptual improvement (lower perceptual distance).

    Args:
        rates_ref: Bitrates of the reference model.
        dist_ref: Distortion/quality metric values of the reference model.
        rates_target: Bitrates of the target model under evaluation.
        dist_target: Distortion/quality metric values of the target model.
        method: Interpolation method: "akima", "cubic", or "pchip".
        metric_mode: "higher_is_better" or "lower_is_better".
        enforce_monotonic: If True, applies Pareto monotonicity filtering.
        strict_overlap: If True, raises ValueError when curves do not overlap;
                        if False, returns float('nan').

    Returns:
        BD-Metric float in distortion units (e.g. +0.8 dB PSNR).
        Returns NaN if rate ranges do not overlap.
    """
    if enforce_monotonic:
        r_ref, d_ref = enforce_monotonicity(rates_ref, dist_ref, metric_mode=metric_mode)
        r_tgt, d_tgt = enforce_monotonicity(rates_target, dist_target, metric_mode=metric_mode)
    else:
        r_ref = _to_numpy_1d(rates_ref, name="rates_ref")
        d_ref = _to_numpy_1d(dist_ref, name="dist_ref")
        r_tgt = _to_numpy_1d(rates_target, name="rates_target")
        d_tgt = _to_numpy_1d(dist_target, name="dist_target")

    if len(r_ref) < 2 or len(r_tgt) < 2:
        if strict_overlap:
            raise ValueError("At least 2 points are required on each curve after monotonicity enforcement.")
        return float("nan")

    log_r_ref = np.log(r_ref)
    log_r_tgt = np.log(r_tgt)

    # Sort coordinates by log rate ascending
    idx_ref = np.argsort(log_r_ref)
    log_r_ref_sorted = log_r_ref[idx_ref]
    d_ref_sorted = d_ref[idx_ref]

    idx_tgt = np.argsort(log_r_tgt)
    log_r_tgt_sorted = log_r_tgt[idx_tgt]
    d_tgt_sorted = d_tgt[idx_tgt]

    # Overlapping log rate interval [r_min, r_max]
    r_min = max(float(log_r_ref_sorted.min()), float(log_r_tgt_sorted.min()))
    r_max = min(float(log_r_ref_sorted.max()), float(log_r_tgt_sorted.max()))

    if r_max <= r_min:
        if strict_overlap:
            ref_low = float(np.exp(log_r_ref_sorted.min()))
            ref_high = float(np.exp(log_r_ref_sorted.max()))
            tgt_low = float(np.exp(log_r_tgt_sorted.min()))
            tgt_high = float(np.exp(log_r_tgt_sorted.max()))
            raise ValueError(
                f"Rate ranges do not overlap: ref=[{ref_low:.4f}, {ref_high:.4f}], "
                f"target=[{tgt_low:.4f}, {tgt_high:.4f}]."
            )
        return float("nan")

    # Integrate distortion curve over overlapping log rate range
    int_ref = _integrate_curve(log_r_ref_sorted, d_ref_sorted, r_min, r_max, method=method)
    int_tgt = _integrate_curve(log_r_tgt_sorted, d_tgt_sorted, r_min, r_max, method=method)

    avg_metric_diff = (int_tgt - int_ref) / (r_max - r_min)

    return float(avg_metric_diff)


def compute_bd_comparison(
    rates_ref: Union[Sequence[float], np.ndarray, torch.Tensor],
    dist_ref: Union[Sequence[float], np.ndarray, torch.Tensor],
    rates_target: Union[Sequence[float], np.ndarray, torch.Tensor],
    dist_target: Union[Sequence[float], np.ndarray, torch.Tensor],
    method: BDRateMethod = "akima",
    metric_mode: MetricMode = "higher_is_better",
    enforce_monotonic: bool = True,
) -> BDComparisonResult:
    """Compute complete Bjøntegaard-Delta rate and distortion comparison with interval metadata.

    Args:
        rates_ref: Bitrates of the reference model.
        dist_ref: Distortion/quality metric values of the reference model.
        rates_target: Bitrates of the target model under evaluation.
        dist_target: Distortion/quality metric values of the target model.
        method: Interpolation method: "akima", "cubic", or "pchip".
        metric_mode: "higher_is_better" or "lower_is_better".
        enforce_monotonic: If True, applies Pareto monotonicity filtering.

    Returns:
        BDComparisonResult containing bd_rate, bd_metric, and overlapping interval metadata.
    """
    if enforce_monotonic:
        r_ref, d_ref = enforce_monotonicity(rates_ref, dist_ref, metric_mode=metric_mode)
        r_tgt, d_tgt = enforce_monotonicity(rates_target, dist_target, metric_mode=metric_mode)
    else:
        r_ref = _to_numpy_1d(rates_ref, name="rates_ref")
        d_ref = _to_numpy_1d(dist_ref, name="dist_ref")
        r_tgt = _to_numpy_1d(rates_target, name="rates_target")
        d_tgt = _to_numpy_1d(dist_target, name="dist_target")

    bd_rate = compute_bd_rate(
        r_ref,
        d_ref,
        r_tgt,
        d_tgt,
        method=method,
        metric_mode=metric_mode,
        enforce_monotonic=False,
    )
    bd_metric = compute_bd_metric(
        r_ref,
        d_ref,
        r_tgt,
        d_tgt,
        method=method,
        metric_mode=metric_mode,
        enforce_monotonic=False,
    )

    overlap_r_min = max(float(r_ref.min()), float(r_tgt.min()))
    overlap_r_max = min(float(r_ref.max()), float(r_tgt.max()))
    overlap_d_min = max(float(d_ref.min()), float(d_tgt.min()))
    overlap_d_max = min(float(d_ref.max()), float(d_tgt.max()))

    r_range = (overlap_r_min, overlap_r_max) if overlap_r_max > overlap_r_min else (float("nan"), float("nan"))
    d_range = (overlap_d_min, overlap_d_max) if overlap_d_max > overlap_d_min else (float("nan"), float("nan"))

    return BDComparisonResult(
        bd_rate=bd_rate,
        bd_metric=bd_metric,
        overlap_rate_range=r_range,
        overlap_metric_range=d_range,
        method=method,
        metric_mode=metric_mode,
        ref_num_points=len(r_ref),
        target_num_points=len(r_tgt),
    )


def _compute_bootstrap_stats(
    point_est: float,
    samples: np.ndarray,
    confidence_level: float,
    beneficial_direction: Literal["negative", "positive"] = "negative",
) -> BootstrapResult:
    """Compute summary statistics, confidence interval, and p-value from bootstrap samples."""
    valid_samples = samples[np.isfinite(samples)]
    if len(valid_samples) < 10:
        return BootstrapResult(
            point_estimate=point_est,
            bootstrap_mean=float("nan"),
            std_error=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            confidence_level=confidence_level,
            p_value=1.0,
            hypothesis_confirmed=False,
            n_bootstraps=len(samples),
            bootstrap_samples=samples,
        )

    boot_mean = float(np.mean(valid_samples))
    boot_se = float(np.std(valid_samples, ddof=1))

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(valid_samples, lower_pct))
    ci_upper = float(np.percentile(valid_samples, upper_pct))

    # Davison & Hinkley (1997) Monte Carlo empirical bootstrap p-value for H0: effect == 0
    b_count = len(valid_samples)
    p_neg = (1.0 + float(np.sum(valid_samples <= 0.0))) / (b_count + 1.0)
    p_pos = (1.0 + float(np.sum(valid_samples >= 0.0))) / (b_count + 1.0)
    p_val = min(1.0, 2.0 * min(p_neg, p_pos))

    # Check hypothesis confirmation: CI excludes zero in the beneficial direction
    if beneficial_direction == "negative":
        # Rate saving or distortion reduction: CI strictly below 0
        confirmed = bool((ci_upper < 0.0) and (p_val < alpha))
    else:
        # Quality gain: CI strictly above 0
        confirmed = bool((ci_lower > 0.0) and (p_val < alpha))

    return BootstrapResult(
        point_estimate=point_est,
        bootstrap_mean=boot_mean,
        std_error=boot_se,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        p_value=p_val,
        hypothesis_confirmed=confirmed,
        n_bootstraps=len(valid_samples),
        bootstrap_samples=valid_samples,
    )


def bootstrap_bd_rate(
    rates_by_model: Mapping[str, Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor]],
    dist_by_model: Mapping[str, Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor]],
    ref_model: str = "A",
    target_model: str = "C",
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    method: BDRateMethod = "akima",
    metric_mode: MetricMode = "higher_is_better",
    seed: Optional[int] = 42,
) -> BootstrapResult:
    """Compute image-level bootstrap 95% confidence intervals for BD-Rate.

    Resamples test images with replacement across `n_bootstraps` iterations (default 1000),
    evaluating dataset-mean rate-distortion curves and empirical distributions.

    Args:
        rates_by_model: Mapping from model name to 2D array of shape (N_images, N_lambdas).
        dist_by_model: Mapping from model name to 2D array of shape (N_images, N_lambdas).
        ref_model: Name of reference model (e.g. "A" or "B").
        target_model: Name of target model (e.g. "C" or "D").
        n_bootstraps: Number of bootstrap resamples (default 1000).
        confidence_level: Confidence level (default 0.95 for 95% CI).
        method: Interpolation method ("akima", "cubic", or "pchip").
        metric_mode: "higher_is_better" or "lower_is_better".
        seed: Random seed for deterministic reproducibility.

    Returns:
        BootstrapResult with point estimate, standard error, 95% CI, and p-value.
    """
    if ref_model not in rates_by_model or target_model not in rates_by_model:
        raise KeyError(
            f"Models '{ref_model}' and '{target_model}' must exist in rates_by_model. "
            f"Available models: {list(rates_by_model.keys())}."
        )

    r_ref_all = _to_numpy_2d(rates_by_model[ref_model], name=f"rates['{ref_model}']")
    d_ref_all = _to_numpy_2d(dist_by_model[ref_model], name=f"dist['{ref_model}']")
    r_tgt_all = _to_numpy_2d(rates_by_model[target_model], name=f"rates['{target_model}']")
    d_tgt_all = _to_numpy_2d(dist_by_model[target_model], name=f"dist['{target_model}']")

    if r_ref_all.shape != d_ref_all.shape:
        raise ValueError(f"Shape mismatch for '{ref_model}': rates {r_ref_all.shape} vs dist {d_ref_all.shape}.")
    if r_tgt_all.shape != d_tgt_all.shape:
        raise ValueError(f"Shape mismatch for '{target_model}': rates {r_tgt_all.shape} vs dist {d_tgt_all.shape}.")

    n_images = r_ref_all.shape[0]
    if n_images != r_tgt_all.shape[0]:
        raise ValueError(
            f"Image count mismatch: ref '{ref_model}' has {n_images} images, "
            f"target '{target_model}' has {r_tgt_all.shape[0]} images."
        )
    if r_ref_all.shape[1] < 2:
        raise ValueError(f"At least 2 rate points required per image, got {r_ref_all.shape[1]}.")

    # 1. Point estimate on dataset-level mean RD curve
    r_ref_mean = np.mean(r_ref_all, axis=0)
    d_ref_mean = np.mean(d_ref_all, axis=0)
    r_tgt_mean = np.mean(r_tgt_all, axis=0)
    d_tgt_mean = np.mean(d_tgt_all, axis=0)

    point_est = compute_bd_rate(
        r_ref_mean,
        d_ref_mean,
        r_tgt_mean,
        d_tgt_mean,
        method=method,
        metric_mode=metric_mode,
        enforce_monotonic=True,
    )

    # 2. Bootstrap resampling across image index set
    rng = np.random.default_rng(seed)
    boot_samples: List[float] = []

    for _ in range(n_bootstraps):
        boot_idx = rng.integers(0, n_images, size=n_images)
        b_r_ref = np.mean(r_ref_all[boot_idx], axis=0)
        b_d_ref = np.mean(d_ref_all[boot_idx], axis=0)
        b_r_tgt = np.mean(r_tgt_all[boot_idx], axis=0)
        b_d_tgt = np.mean(d_tgt_all[boot_idx], axis=0)

        bdr = compute_bd_rate(
            b_r_ref,
            b_d_ref,
            b_r_tgt,
            b_d_tgt,
            method=method,
            metric_mode=metric_mode,
            enforce_monotonic=True,
        )
        if np.isfinite(bdr):
            boot_samples.append(bdr)

    if len(boot_samples) < 0.8 * n_bootstraps:
        logger.warning(
            "Low bootstrap overlap: only %d of %d resamples yielded finite BD-Rate.",
            len(boot_samples),
            n_bootstraps,
        )

    samples_arr = np.array(boot_samples, dtype=np.float64)
    return _compute_bootstrap_stats(
        point_est=point_est,
        samples=samples_arr,
        confidence_level=confidence_level,
        beneficial_direction="negative",
    )


def bootstrap_bd_metric(
    rates_by_model: Mapping[str, Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor]],
    dist_by_model: Mapping[str, Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor]],
    ref_model: str = "A",
    target_model: str = "C",
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    method: BDRateMethod = "akima",
    metric_mode: MetricMode = "higher_is_better",
    seed: Optional[int] = 42,
) -> BootstrapResult:
    """Compute image-level bootstrap 95% confidence intervals for BD-Metric / BD-PSNR.

    Args:
        rates_by_model: Mapping from model name to 2D array of shape (N_images, N_lambdas).
        dist_by_model: Mapping from model name to 2D array of shape (N_images, N_lambdas).
        ref_model: Reference model identifier.
        target_model: Target model identifier.
        n_bootstraps: Number of bootstrap iterations.
        confidence_level: Confidence level (default 0.95).
        method: Interpolation method ("akima", "cubic", or "pchip").
        metric_mode: "higher_is_better" or "lower_is_better".
        seed: Random seed.

    Returns:
        BootstrapResult with point estimate, standard error, 95% CI, and p-value.
    """
    if ref_model not in rates_by_model or target_model not in rates_by_model:
        raise KeyError(
            f"Models '{ref_model}' and '{target_model}' must exist in rates_by_model. "
            f"Available: {list(rates_by_model.keys())}."
        )

    r_ref_all = _to_numpy_2d(rates_by_model[ref_model], name=f"rates['{ref_model}']")
    d_ref_all = _to_numpy_2d(dist_by_model[ref_model], name=f"dist['{ref_model}']")
    r_tgt_all = _to_numpy_2d(rates_by_model[target_model], name=f"rates['{target_model}']")
    d_tgt_all = _to_numpy_2d(dist_by_model[target_model], name=f"dist['{target_model}']")

    if r_ref_all.shape != d_ref_all.shape:
        raise ValueError(f"Shape mismatch for '{ref_model}': rates {r_ref_all.shape} vs dist {d_ref_all.shape}.")
    if r_tgt_all.shape != d_tgt_all.shape:
        raise ValueError(f"Shape mismatch for '{target_model}': rates {r_tgt_all.shape} vs dist {d_tgt_all.shape}.")

    n_images = r_ref_all.shape[0]
    if n_images != r_tgt_all.shape[0]:
        raise ValueError(
            f"Image count mismatch: ref '{ref_model}' has {n_images} images, "
            f"target '{target_model}' has {r_tgt_all.shape[0]} images."
        )

    # Point estimate
    point_est = compute_bd_metric(
        np.mean(r_ref_all, axis=0),
        np.mean(d_ref_all, axis=0),
        np.mean(r_tgt_all, axis=0),
        np.mean(d_tgt_all, axis=0),
        method=method,
        metric_mode=metric_mode,
        enforce_monotonic=True,
    )

    rng = np.random.default_rng(seed)
    boot_samples: List[float] = []

    for _ in range(n_bootstraps):
        boot_idx = rng.integers(0, n_images, size=n_images)
        bdm = compute_bd_metric(
            np.mean(r_ref_all[boot_idx], axis=0),
            np.mean(d_ref_all[boot_idx], axis=0),
            np.mean(r_tgt_all[boot_idx], axis=0),
            np.mean(d_tgt_all[boot_idx], axis=0),
            method=method,
            metric_mode=metric_mode,
            enforce_monotonic=True,
        )
        if np.isfinite(bdm):
            boot_samples.append(bdm)

    if len(boot_samples) < 0.8 * n_bootstraps:
        logger.warning(
            "Low bootstrap overlap: only %d of %d resamples yielded finite BD-Metric.",
            len(boot_samples),
            n_bootstraps,
        )

    samples_arr = np.array(boot_samples, dtype=np.float64)
    beneficial = "positive" if metric_mode == "higher_is_better" else "negative"
    return _compute_bootstrap_stats(
        point_est=point_est,
        samples=samples_arr,
        confidence_level=confidence_level,
        beneficial_direction=beneficial,
    )


def evaluate_factorial_contrasts(
    rates_by_model: Mapping[str, Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor]],
    dist_by_model: Mapping[str, Union[Sequence[Sequence[float]], np.ndarray, torch.Tensor]],
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    method: BDRateMethod = "akima",
    metric_mode: MetricMode = "higher_is_better",
    seed: Optional[int] = 42,
) -> FactorialContrastsResult:
    """Evaluate pre-registered 4-model factorial contrasts and interaction effects.

    Pre-registered factorial design isolates causal mechanisms (Proposal Sec 8, Math Sec 5.2):
    1. Mechanism Effect under saliency loss: Contrast (C - B)
       Evaluates whether explicit allocation in Model C provides coding gain over
       saliency loss alone (Model B) at matched actual bitrate: BD-Rate(B -> C).
    2. Mechanism Effect under uniform MSE loss: Contrast (D - A)
       Evaluates whether explicit allocation in Model D provides coding gain over
       uniform baseline (Model A) without saliency-weighted distortion: BD-Rate(A -> D).
    3. Super-Additive Interaction Effect: ((C - B) - (D - A))
       Evaluates whether explicit allocation and saliency loss interact synergistically:
       BD-Rate(B -> C) - BD-Rate(A -> D).

    Args:
        rates_by_model: Mapping from model name ("A", "B", "C", "D", and optional variants)
            to 2D arrays of shape (N_images, N_lambdas).
        dist_by_model: Mapping from model name to 2D arrays of shape (N_images, N_lambdas).
        n_bootstraps: Number of bootstrap iterations (default 1000).
        confidence_level: Confidence level (default 0.95).
        method: Interpolation method ("akima", "cubic", or "pchip").
        metric_mode: "higher_is_better" or "lower_is_better".
        seed: Random seed for deterministic reproducibility.

    Returns:
        FactorialContrastsResult containing bootstrap statistics for contrasts
        (C - B), (D - A), the interaction, and all pairwise combinations.
    """
    for required in ["A", "B", "C", "D"]:
        if required not in rates_by_model or required not in dist_by_model:
            raise KeyError(
                f"Required model '{required}' missing from rates_by_model or dist_by_model. "
                f"Available: {list(rates_by_model.keys())}."
            )

    # Standardize to numpy 2D arrays with autograd detaching
    rates_np = {m: _to_numpy_2d(v, name=f"rates['{m}']") for m, v in rates_by_model.items()}
    dist_np = {m: _to_numpy_2d(v, name=f"dist['{m}']") for m, v in dist_by_model.items()}

    n_images = rates_np["A"].shape[0]

    for m in rates_np:
        if rates_np[m].shape != dist_np[m].shape:
            raise ValueError(f"Shape mismatch in model '{m}': rates {rates_np[m].shape} vs dist {dist_np[m].shape}.")
        if rates_np[m].shape[0] != n_images:
            raise ValueError(f"Image count mismatch in model '{m}': expected {n_images}, got {rates_np[m].shape[0]}.")
        if rates_np[m].shape[1] < 2:
            raise ValueError(f"At least 2 lambda points required, got {rates_np[m].shape[1]} for model '{m}'.")

    # Compute point estimates on full dataset
    def _mean_rd(m: str) -> Tuple[np.ndarray, np.ndarray]:
        return np.mean(rates_np[m], axis=0), np.mean(dist_np[m], axis=0)

    pairs = [
        ("B", "A"),
        ("C", "A"),
        ("D", "A"),
        ("C", "B"),
        ("D", "B"),
        ("C", "D"),
    ]
    if "C_transmitted" in rates_np:
        pairs.append(("C", "C_transmitted"))

    pe_pairwise: Dict[str, float] = {}
    for tgt, ref in pairs:
        r_r, d_r = _mean_rd(ref)
        r_t, d_t = _mean_rd(tgt)
        pe_pairwise[f"{tgt}_vs_{ref}"] = compute_bd_rate(
            r_r, d_r, r_t, d_t, method=method, metric_mode=metric_mode, enforce_monotonic=True
        )

    pe_c_minus_b = pe_pairwise["C_vs_B"]
    pe_d_minus_a = pe_pairwise["D_vs_A"]
    pe_interaction = pe_c_minus_b - pe_d_minus_a

    # Bootstrap resampling
    rng = np.random.default_rng(seed)
    boot_c_minus_b: List[float] = []
    boot_d_minus_a: List[float] = []
    boot_interaction: List[float] = []
    boot_pairwise: Dict[str, List[float]] = {k: [] for k in pe_pairwise}

    for _ in range(n_bootstraps):
        boot_idx = rng.integers(0, n_images, size=n_images)

        boot_rd_dict = {
            m: (np.mean(rates_np[m][boot_idx], axis=0), np.mean(dist_np[m][boot_idx], axis=0)) for m in rates_np
        }

        # Evaluate each pair once per bootstrap replicate
        cur_pair_bdr: Dict[str, float] = {}
        for tgt, ref in pairs:
            r_r, d_r = boot_rd_dict[ref]
            r_t, d_t = boot_rd_dict[tgt]
            cur_pair_bdr[f"{tgt}_vs_{ref}"] = compute_bd_rate(
                r_r, d_r, r_t, d_t, method=method, metric_mode=metric_mode, enforce_monotonic=True
            )

        b_cmb = cur_pair_bdr["C_vs_B"]
        b_dma = cur_pair_bdr["D_vs_A"]

        # If both core factorial contrasts are finite, record joint resamples
        if np.isfinite(b_cmb) and np.isfinite(b_dma):
            boot_c_minus_b.append(b_cmb)
            boot_d_minus_a.append(b_dma)
            boot_interaction.append(b_cmb - b_dma)

        for pair_name, val in cur_pair_bdr.items():
            if np.isfinite(val):
                boot_pairwise[pair_name].append(val)

    # Compute bootstrap summary statistics
    res_c_minus_b = _compute_bootstrap_stats(
        point_est=pe_c_minus_b,
        samples=np.array(boot_c_minus_b, dtype=np.float64),
        confidence_level=confidence_level,
        beneficial_direction="negative",
    )
    res_d_minus_a = _compute_bootstrap_stats(
        point_est=pe_d_minus_a,
        samples=np.array(boot_d_minus_a, dtype=np.float64),
        confidence_level=confidence_level,
        beneficial_direction="negative",
    )
    res_interaction = _compute_bootstrap_stats(
        point_est=pe_interaction,
        samples=np.array(boot_interaction, dtype=np.float64),
        confidence_level=confidence_level,
        beneficial_direction="negative",
    )

    pairwise_results: Dict[str, BootstrapResult] = {}
    for pair_name, samples_list in boot_pairwise.items():
        # Ensure C_vs_B and D_vs_A share identical statistics with factorial contrasts
        if pair_name == "C_vs_B":
            pairwise_results[pair_name] = res_c_minus_b
        elif pair_name == "D_vs_A":
            pairwise_results[pair_name] = res_d_minus_a
        else:
            pairwise_results[pair_name] = _compute_bootstrap_stats(
                point_est=pe_pairwise[pair_name],
                samples=np.array(samples_list, dtype=np.float64),
                confidence_level=confidence_level,
                beneficial_direction="negative",
            )

    return FactorialContrastsResult(
        contrast_c_minus_b=res_c_minus_b,
        contrast_d_minus_a=res_d_minus_a,
        contrast_interaction=res_interaction,
        pairwise_contrasts=pairwise_results,
        method=method,
        metric_mode=metric_mode,
        confidence_level=confidence_level,
        n_bootstraps=n_bootstraps,
    )
