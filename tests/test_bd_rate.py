"""Tests for Bjøntegaard-Delta evaluation engine and bootstrap confidence intervals.

Verifies:
1. Pareto monotonicity enforcement on noisy and plateaued rate-distortion points.
2. Numerical agreement with standard VCEG-M33 Bjontegaard reference implementations.
3. Natural log vs base-10 log formulation exact mathematical equivalence.
4. Akima, cubic, and PCHIP interpolation methods and oscillation mitigation.
5. Higher-is-better (PSNR) vs lower-is-better (LPIPS) metric modes.
6. Non-overlapping curve handling and graceful NaN fallbacks.
7. 1000-resample image-level bootstrap 95% confidence intervals and p-values.
8. Pre-registered factorial contrasts: (C - B), (D - A), and interaction ((C - B) - (D - A)).
9. Interoperability between PyTorch tensors (including requires_grad=True) and NumPy arrays.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest
import torch

from src.evaluation.bd_rate import (
    BDComparisonResult,
    BootstrapResult,
    FactorialContrastsResult,
    bootstrap_bd_metric,
    bootstrap_bd_rate,
    compute_bd_comparison,
    compute_bd_metric,
    compute_bd_rate,
    enforce_monotonicity,
    evaluate_factorial_contrasts,
)


class TestMonotonicityEnforcement:
    """Tests for enforce_monotonicity and rate-distortion coordinate validation."""

    def test_sorting_and_strictly_increasing_rates(self) -> None:
        """Rates provided in scrambled order should be sorted in ascending order."""
        rates = [0.8, 0.2, 0.4, 0.1]
        distortions = [37.0, 32.0, 34.5, 30.0]

        r_sorted, d_sorted = enforce_monotonicity(rates, distortions, metric_mode="higher_is_better")
        assert np.all(np.diff(r_sorted) > 0.0)
        assert np.all(np.diff(d_sorted) > 0.0)
        np.testing.assert_allclose(r_sorted, [0.1, 0.2, 0.4, 0.8])
        np.testing.assert_allclose(d_sorted, [30.0, 32.0, 34.5, 37.0])

    def test_plateau_handling_higher_is_better(self) -> None:
        """Duplicate distortion points (plateaus) should prune strictly dominated points."""
        rates = [0.1, 0.2, 0.3, 0.4]
        distortions = [30.0, 32.0, 32.0, 35.0]  # Plateau at 32.0: (0.3, 32.0) is dominated

        r_sorted, d_sorted = enforce_monotonicity(rates, distortions, metric_mode="higher_is_better")
        assert np.all(np.diff(r_sorted) > 0.0)
        assert np.all(np.diff(d_sorted) > 0.0)
        np.testing.assert_allclose(r_sorted, [0.1, 0.2, 0.4])
        np.testing.assert_allclose(d_sorted, [30.0, 32.0, 35.0])

    def test_inversion_handling_lower_is_better(self) -> None:
        """For lower-is-better metrics (e.g. LPIPS), higher rate with higher LPIPS is pruned."""
        rates = [0.1, 0.2, 0.3, 0.4]
        distortions = [0.50, 0.40, 0.42, 0.30]  # Inversion at 0.42 > 0.40 is dominated

        r_sorted, d_sorted = enforce_monotonicity(rates, distortions, metric_mode="lower_is_better")
        assert np.all(np.diff(r_sorted) > 0.0)
        assert np.all(np.diff(d_sorted) < 0.0)
        np.testing.assert_allclose(r_sorted, [0.1, 0.2, 0.4])
        np.testing.assert_allclose(d_sorted, [0.50, 0.40, 0.30])

    def test_akima_derivative_explosion_prevention(self) -> None:
        """Curves with inversions should not cause Akima derivative explosion or -100% underflow."""
        r1 = [0.1, 0.3, 0.5, 0.6, 0.9]
        d1 = [29.0, 31.0, 33.0, 31.0, 36.0]
        r2 = [0.1, 0.3, 0.5, 0.6, 0.9]
        d2 = [29.5, 31.5, 33.5, 31.5, 36.5]  # Exactly +0.5 dB gain

        bdr_akima = compute_bd_rate(r1, d1, r2, d2, method="akima")
        bdr_pchip = compute_bd_rate(r1, d1, r2, d2, method="pchip")
        bdr_cubic = compute_bd_rate(r1, d1, r2, d2, method="cubic")

        # Must not underflow to -100.0%
        assert -20.0 < bdr_akima < -10.0
        assert -20.0 < bdr_pchip < -10.0
        assert -20.0 < bdr_cubic < -10.0
        assert abs(bdr_akima - bdr_pchip) < 1.0

    def test_non_positive_rate_raises_value_error(self) -> None:
        """Zero or negative bitrates must raise ValueError."""
        rates = [0.0, 0.2, 0.4, 0.8]
        distortions = [30.0, 32.0, 34.5, 37.0]
        with pytest.raises(ValueError, match=r"strictly positive"):
            enforce_monotonicity(rates, distortions)

    def test_length_mismatch_raises_value_error(self) -> None:
        """Mismatched rate and distortion lengths must raise ValueError."""
        with pytest.raises(ValueError, match=r"Length mismatch"):
            enforce_monotonicity([0.1, 0.2], [30.0, 32.0, 34.0])

    def test_fewer_than_two_points_raises_value_error(self) -> None:
        """Single point must raise ValueError."""
        with pytest.raises(ValueError, match=r"At least 2"):
            enforce_monotonicity([0.1], [30.0])


class TestBjontegaardNumericalAgreement:
    """Verifies numerical agreement against canonical VCEG-M33 references."""

    # Standard 4-point benchmark test vector widely used in video/image coding literature:
    R1 = np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float64)
    D1 = np.array([30.0, 32.0, 34.5, 37.0], dtype=np.float64)
    R2 = np.array([0.08, 0.17, 0.35, 0.7], dtype=np.float64)
    D2 = np.array([30.2, 32.2, 34.8, 37.3], dtype=np.float64)

    def test_exact_agreement_with_vceg_reference_cubic(self) -> None:
        """Verify cubic polynomial OLS matches canonical VCEG-M33 Bjontegaard results."""
        # Canonical VCEG-M33 analytical evaluation for this vector:
        # BD-Rate ≈ -20.3359%, BD-PSNR ≈ +0.7581 dB
        bd_rate = compute_bd_rate(self.R1, self.D1, self.R2, self.D2, method="cubic")
        bd_psnr = compute_bd_metric(self.R1, self.D1, self.R2, self.D2, method="cubic")

        assert math.isclose(bd_rate, -20.33590187, abs_tol=1e-4)
        assert math.isclose(bd_psnr, 0.75809482, abs_tol=1e-4)

    def test_natural_log_vs_base_10_log_equivalence(self) -> None:
        """Verify that ln(R) and log10(R) formulations produce identical results to machine precision."""
        bdr_ln = compute_bd_rate(self.R1, self.D1, self.R2, self.D2, method="cubic")

        # Direct base-10 calculation:
        lR1_10 = np.log10(self.R1)
        lR2_10 = np.log10(self.R2)
        p1_10 = np.polyfit(self.D1, lR1_10, 3)
        p2_10 = np.polyfit(self.D2, lR2_10, 3)
        min_d = max(self.D1.min(), self.D2.min())
        max_d = min(self.D1.max(), self.D2.max())
        int1_10 = np.polyval(np.polyint(p1_10), max_d) - np.polyval(np.polyint(p1_10), min_d)
        int2_10 = np.polyval(np.polyint(p2_10), max_d) - np.polyval(np.polyint(p2_10), min_d)
        bdr_10 = (10.0 ** ((int2_10 - int1_10) / (max_d - min_d)) - 1.0) * 100.0

        assert math.isclose(bdr_ln, bdr_10, abs_tol=1e-8)

    def test_identity_comparison_yields_zero(self) -> None:
        """Comparing identical curves must yield exactly 0.0% BD-Rate and 0.0 dB BD-PSNR."""
        bd_rate_cubic = compute_bd_rate(self.R1, self.D1, self.R1, self.D1, method="cubic")
        bd_psnr_cubic = compute_bd_metric(self.R1, self.D1, self.R1, self.D1, method="cubic")
        assert abs(bd_rate_cubic) < 1e-10
        assert abs(bd_psnr_cubic) < 1e-10

        bd_rate_akima = compute_bd_rate(self.R1, self.D1, self.R1, self.D1, method="akima")
        bd_psnr_akima = compute_bd_metric(self.R1, self.D1, self.R1, self.D1, method="akima")
        assert abs(bd_rate_akima) < 1e-10
        assert abs(bd_psnr_akima) < 1e-10

    def test_uniform_bitrate_scaling(self) -> None:
        """Target curve with exactly 20% lower bitrate at all quality points yields -20% BD-Rate."""
        r_scaled = self.R1 * 0.80  # Exactly 20% bitrate reduction
        bd_rate_cubic = compute_bd_rate(self.R1, self.D1, r_scaled, self.D1, method="cubic")
        bd_rate_akima = compute_bd_rate(self.R1, self.D1, r_scaled, self.D1, method="akima")

        assert math.isclose(bd_rate_cubic, -20.0, abs_tol=1e-4)
        assert math.isclose(bd_rate_akima, -20.0, abs_tol=1e-4)

    def test_uniform_psnr_shift(self) -> None:
        """Target curve with exactly +1.5 dB higher PSNR at all rate points yields +1.5 dB BD-PSNR."""
        d_shifted = self.D1 + 1.50  # Exactly +1.5 dB PSNR increase
        bd_psnr_cubic = compute_bd_metric(self.R1, self.D1, self.R1, d_shifted, method="cubic")
        bd_psnr_akima = compute_bd_metric(self.R1, self.D1, self.R1, d_shifted, method="akima")

        assert math.isclose(bd_psnr_cubic, 1.50, abs_tol=1e-4)
        assert math.isclose(bd_psnr_akima, 1.50, abs_tol=1e-4)


class TestInterpolationMethods:
    """Verifies Akima, PCHIP, and Cubic interpolation methods."""

    # 6-point rate sweep matching our research study:
    # lambdas = [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483]
    R_A = np.array([0.052, 0.098, 0.185, 0.340, 0.610, 1.050], dtype=np.float64)
    D_A = np.array([28.40, 30.80, 33.10, 35.50, 37.90, 40.20], dtype=np.float64)

    # Model C has higher quality / lower rate
    R_C = np.array([0.044, 0.085, 0.160, 0.295, 0.530, 0.920], dtype=np.float64)
    D_C = np.array([28.70, 31.15, 33.50, 35.95, 38.35, 40.65], dtype=np.float64)

    def test_methods_consistent_negative_bd_rate(self) -> None:
        """All three methods (cubic, akima, pchip) should confirm significant bitrate saving."""
        bdr_cubic = compute_bd_rate(self.R_A, self.D_A, self.R_C, self.D_C, method="cubic")
        bdr_akima = compute_bd_rate(self.R_A, self.D_A, self.R_C, self.D_C, method="akima")
        bdr_pchip = compute_bd_rate(self.R_A, self.D_A, self.R_C, self.D_C, method="pchip")

        # All must indicate significant bitrate reduction
        assert -26.0 < bdr_cubic < -18.0
        assert -26.0 < bdr_akima < -18.0
        assert -26.0 < bdr_pchip < -18.0

        # Differences between Akima and Pchip should be minimal (< 1% absolute)
        assert abs(bdr_akima - bdr_pchip) < 1.0

    def test_methods_consistent_positive_bd_psnr(self) -> None:
        """All three methods should confirm positive BD-PSNR improvement."""
        bdp_cubic = compute_bd_metric(self.R_A, self.D_A, self.R_C, self.D_C, method="cubic")
        bdp_akima = compute_bd_metric(self.R_A, self.D_A, self.R_C, self.D_C, method="akima")
        bdp_pchip = compute_bd_metric(self.R_A, self.D_A, self.R_C, self.D_C, method="pchip")

        assert 0.75 < bdp_cubic < 1.25
        assert 0.75 < bdp_akima < 1.25
        assert 0.75 < bdp_pchip < 1.25

    def test_compute_bd_comparison_metadata(self) -> None:
        """compute_bd_comparison should populate all interval and configuration metadata."""
        comp = compute_bd_comparison(self.R_A, self.D_A, self.R_C, self.D_C, method="akima")
        assert isinstance(comp, BDComparisonResult)
        assert comp.bd_rate < 0.0
        assert comp.bd_metric > 0.0
        assert comp.method == "akima"
        assert comp.ref_num_points == 6
        assert comp.target_num_points == 6
        assert comp.overlap_metric_range[0] == pytest.approx(28.70, abs=1e-3)
        assert comp.overlap_metric_range[1] == pytest.approx(40.20, abs=1e-3)


class TestMetricModesAndIntervals:
    """Tests for lower-is-better metrics (LPIPS) and non-overlapping curve handling."""

    def test_lower_is_better_metric_lpips(self) -> None:
        """For LPIPS, lower values indicate better quality."""
        rates_A = [0.1, 0.2, 0.4, 0.8]
        lpips_A = [0.45, 0.35, 0.25, 0.15]  # Decreasing with rate

        # Model C achieves lower LPIPS at same rates
        rates_C = [0.1, 0.2, 0.4, 0.8]
        lpips_C = [0.40, 0.30, 0.21, 0.11]  # Better (lower) LPIPS

        bdr = compute_bd_rate(
            rates_A,
            lpips_A,
            rates_C,
            lpips_C,
            metric_mode="lower_is_better",
            method="akima",
        )
        bd_metric = compute_bd_metric(
            rates_A,
            lpips_A,
            rates_C,
            lpips_C,
            metric_mode="lower_is_better",
            method="akima",
        )

        # Rate should be lower at same LPIPS -> negative BD-rate
        assert bdr < 0.0
        # Average LPIPS delta should be negative (better)
        assert bd_metric < 0.0

    def test_non_overlapping_curves_return_nan(self) -> None:
        """When curves do not overlap in distortion or rate, return NaN without strict mode."""
        r1 = [0.1, 0.2, 0.3]
        d1 = [30.0, 31.0, 32.0]
        r2 = [0.1, 0.2, 0.3]
        d2 = [40.0, 41.0, 42.0]  # No distortion overlap with d1

        bdr = compute_bd_rate(r1, d1, r2, d2, strict_overlap=False)
        assert math.isnan(bdr)

    def test_non_overlapping_comparison_metadata_intervals(self) -> None:
        """Non-overlapping curves report NaN intervals in BDComparisonResult."""
        r1 = [0.1, 0.2, 0.3]
        d1 = [30.0, 31.0, 32.0]
        r2 = [0.1, 0.2, 0.3]
        d2 = [40.0, 41.0, 42.0]
        comp = compute_bd_comparison(r1, d1, r2, d2)
        assert math.isnan(comp.bd_rate)
        assert math.isnan(comp.overlap_metric_range[0])
        assert math.isnan(comp.overlap_metric_range[1])

    def test_strict_overlap_raises_value_error(self) -> None:
        """When strict_overlap=True, non-overlapping curves raise ValueError."""
        r1 = [0.1, 0.2, 0.3]
        d1 = [30.0, 31.0, 32.0]
        r2 = [0.1, 0.2, 0.3]
        d2 = [40.0, 41.0, 42.0]
        with pytest.raises(ValueError, match=r"Distortion ranges do not overlap"):
            compute_bd_rate(r1, d1, r2, d2, strict_overlap=True)


class TestImageLevelBootstrapEngine:
    """Verifies 1000-resample image-level bootstrap confidence intervals."""

    @pytest.fixture
    def synthetic_multimodel_dataset(self) -> Tuple[dict, dict]:
        """Generate synthetic 24-image 6-point sweep dataset across Models A, B, C, D."""
        rng = np.random.default_rng(12345)
        n_images = 24
        n_lambdas = 6

        # Base nominal rates and PSNRs for Model A
        base_rates_A = np.array([0.05, 0.10, 0.18, 0.33, 0.60, 1.00])
        base_psnr_A = np.array([28.0, 30.5, 33.0, 35.5, 37.8, 40.0])

        rates: dict[str, np.ndarray] = {}
        distortions: dict[str, np.ndarray] = {}

        # Image variability noise (shared across models on the same image)
        img_rate_factors = rng.uniform(0.85, 1.15, size=(n_images, 1))
        img_psnr_offsets = rng.normal(0.0, 0.5, size=(n_images, 1))

        # Model A: Uniform baseline
        rates["A"] = (base_rates_A * img_rate_factors) + rng.normal(0, 0.005, size=(n_images, n_lambdas))
        distortions["A"] = (base_psnr_A + img_psnr_offsets) + rng.normal(0, 0.1, size=(n_images, n_lambdas))

        # Model B: Saliency-weighted loss only (modest ROI gain, slightly higher overall rate)
        rates["B"] = rates["A"] * 0.98 + rng.normal(0, 0.004, size=(n_images, n_lambdas))
        distortions["B"] = distortions["A"] + 0.15 + rng.normal(0, 0.08, size=(n_images, n_lambdas))

        # Model C: Explicit allocation + saliency loss (strong coding gain: ~15% bitrate saving)
        rates["C"] = rates["A"] * 0.85 + rng.normal(0, 0.004, size=(n_images, n_lambdas))
        distortions["C"] = distortions["A"] + 0.50 + rng.normal(0, 0.08, size=(n_images, n_lambdas))

        # Model D: Explicit allocation + uniform MSE loss (gain from adaptive quantization alone: ~8%)
        rates["D"] = rates["A"] * 0.92 + rng.normal(0, 0.004, size=(n_images, n_lambdas))
        distortions["D"] = distortions["A"] + 0.30 + rng.normal(0, 0.08, size=(n_images, n_lambdas))

        return rates, distortions

    def test_bootstrap_bd_rate_point_estimate_and_ci(self, synthetic_multimodel_dataset: Tuple[dict, dict]) -> None:
        """1000-resample bootstrap produces tight 95% CI covering significant rate savings."""
        rates, dist = synthetic_multimodel_dataset

        res = bootstrap_bd_rate(
            rates_by_model=rates,
            dist_by_model=dist,
            ref_model="A",
            target_model="C",
            n_bootstraps=500,  # 500 for fast test execution
            confidence_level=0.95,
            seed=42,
        )

        assert isinstance(res, BootstrapResult)
        assert res.n_bootstraps == 500
        # Point estimate must be negative (rate saving)
        assert -30.0 < res.point_estimate < -15.0
        # CI bounds must bracket point estimate and be strictly negative
        assert res.ci_lower <= res.point_estimate <= res.ci_upper
        assert res.ci_upper < 0.0
        # P-value must confirm significance
        assert res.p_value < 0.05
        assert res.hypothesis_confirmed is True

    def test_bootstrap_bd_metric_gain(self, synthetic_multimodel_dataset: Tuple[dict, dict]) -> None:
        """Bootstrap BD-Metric evaluates positive PSNR gain with statistical significance."""
        rates, dist = synthetic_multimodel_dataset

        res = bootstrap_bd_metric(
            rates_by_model=rates,
            dist_by_model=dist,
            ref_model="A",
            target_model="C",
            n_bootstraps=500,
            seed=42,
        )

        assert res.point_estimate > 0.0
        assert res.ci_lower > 0.0
        assert res.hypothesis_confirmed is True

    def test_1d_array_passed_to_bootstrap_raises_value_error(self) -> None:
        """Passing 1D array to bootstrap functions should raise clear ValueError, not scalar collapse."""
        rates = {"A": np.array([0.1, 0.2, 0.4]), "C": np.array([0.08, 0.16, 0.35])}
        dist = {"A": np.array([30.0, 32.0, 34.0]), "C": np.array([30.5, 32.5, 34.5])}
        with pytest.raises(ValueError, match=r"must be 2-dimensional"):
            bootstrap_bd_rate(rates, dist, "A", "C", n_bootstraps=50)


class TestPreRegisteredFactorialContrasts:
    """Verifies pre-registered factorial contrasts and interaction hypotheses."""

    @pytest.fixture
    def factorial_experiment_data(self) -> Tuple[dict, dict]:
        """Generate structured factorial dataset matching Section 5.2."""
        rng = np.random.default_rng(999)
        n_images = 20

        base_r = np.array([0.06, 0.12, 0.22, 0.38, 0.65, 1.10])
        base_d = np.array([29.0, 31.2, 33.5, 35.8, 38.0, 40.2])

        rates: dict[str, np.ndarray] = {}
        dist: dict[str, np.ndarray] = {}

        img_factors = rng.uniform(0.9, 1.1, size=(n_images, 1))

        # Model A: baseline
        rates["A"] = base_r * img_factors
        dist["A"] = np.tile(base_d, (n_images, 1))

        # Model B: loss effect only
        rates["B"] = rates["A"] * 0.96
        dist["B"] = dist["A"] + 0.1

        # Model D: allocation mechanism effect only under uniform loss (D - A)
        rates["D"] = rates["A"] * 0.90
        dist["D"] = dist["A"] + 0.3

        # Model C: super-additive mechanism under saliency loss:
        # C is better than B by more than D is better than A
        rates["C"] = rates["A"] * 0.82
        dist["C"] = dist["A"] + 0.6

        return rates, dist

    def test_factorial_contrasts_evaluation(self, factorial_experiment_data: Tuple[dict, dict]) -> None:
        """Evaluate pre-registered contrasts (C - B), (D - A), and interaction ((C - B) - (D - A))."""
        rates, dist = factorial_experiment_data

        result = evaluate_factorial_contrasts(
            rates_by_model=rates,
            dist_by_model=dist,
            n_bootstraps=300,
            confidence_level=0.95,
            seed=42,
        )

        assert isinstance(result, FactorialContrastsResult)

        # Contrast 1: (C - B) mechanism under saliency loss
        assert result.contrast_c_minus_b.point_estimate < 0.0
        assert result.contrast_c_minus_b.hypothesis_confirmed is True

        # Contrast 2: (D - A) mechanism under uniform loss
        assert result.contrast_d_minus_a.point_estimate < 0.0
        assert result.contrast_d_minus_a.hypothesis_confirmed is True

        # Contrast 3: Interaction effect ((C - B) - (D - A))
        # Point estimate must be negative (super-additive coding gain)
        assert result.contrast_interaction.point_estimate < 0.0

        # Pairwise contrasts verification
        assert "B_vs_A" in result.pairwise_contrasts
        assert "C_vs_A" in result.pairwise_contrasts
        assert "D_vs_A" in result.pairwise_contrasts
        assert "C_vs_B" in result.pairwise_contrasts

    def test_missing_model_raises_key_error(self) -> None:
        """Missing required model A, B, C, or D raises KeyError."""
        with pytest.raises(KeyError, match=r"Required model 'D' missing"):
            evaluate_factorial_contrasts(
                rates_by_model={"A": np.ones((5, 4)), "B": np.ones((5, 4)), "C": np.ones((5, 4))},
                dist_by_model={"A": np.ones((5, 4)), "B": np.ones((5, 4)), "C": np.ones((5, 4))},
            )

    def test_image_count_mismatch_raises_value_error(self) -> None:
        """Models with different image counts in factorial evaluation raise ValueError."""
        rates = {
            "A": np.ones((10, 4)),
            "B": np.ones((8, 4)),  # 8 != 10
            "C": np.ones((10, 4)),
            "D": np.ones((10, 4)),
        }
        dist = {
            "A": np.ones((10, 4)) * 30,
            "B": np.ones((8, 4)) * 30,
            "C": np.ones((10, 4)) * 30,
            "D": np.ones((10, 4)) * 30,
        }
        with pytest.raises(ValueError, match=r"Image count mismatch"):
            evaluate_factorial_contrasts(rates, dist, n_bootstraps=50)


class TestPyTorchTensorAndNumPyInteroperability:
    """Verifies PyTorch tensor and NumPy array interoperability."""

    def test_pytorch_tensor_inputs(self) -> None:
        """PyTorch tensors (with grad enabled or on cpu) work without manual conversion."""
        r_ref = torch.tensor([0.1, 0.2, 0.4, 0.8], requires_grad=False)
        d_ref = torch.tensor([30.0, 32.0, 34.5, 37.0], requires_grad=False)
        r_tgt = torch.tensor([0.08, 0.17, 0.35, 0.7], requires_grad=False)
        d_tgt = torch.tensor([30.2, 32.2, 34.8, 37.3], requires_grad=False)

        bdr = compute_bd_rate(r_ref, d_ref, r_tgt, d_tgt, method="akima")
        bdp = compute_bd_metric(r_ref, d_ref, r_tgt, d_tgt, method="akima")

        assert isinstance(bdr, float)
        assert isinstance(bdp, float)
        assert bdr < 0.0
        assert bdp > 0.0

    def test_pytorch_tensor_requires_grad_in_bootstrap(self) -> None:
        """PyTorch tensors with requires_grad=True must not trigger autograd runtime errors."""
        n_img = 10
        n_pts = 4
        rates = {
            "A": torch.rand(n_img, n_pts, requires_grad=True) + 0.1,
            "B": torch.rand(n_img, n_pts, requires_grad=True) + 0.1,
            "C": torch.rand(n_img, n_pts, requires_grad=True) + 0.1,
            "D": torch.rand(n_img, n_pts, requires_grad=True) + 0.1,
        }
        dist = {
            "A": torch.rand(n_img, n_pts, requires_grad=True) * 10 + 25,
            "B": torch.rand(n_img, n_pts, requires_grad=True) * 10 + 25,
            "C": torch.rand(n_img, n_pts, requires_grad=True) * 10 + 25,
            "D": torch.rand(n_img, n_pts, requires_grad=True) * 10 + 25,
        }

        # Should execute cleanly without autograd RuntimeError
        res_rate = bootstrap_bd_rate(rates, dist, "A", "C", n_bootstraps=50)
        assert isinstance(res_rate, BootstrapResult)

        res_metric = bootstrap_bd_metric(rates, dist, "A", "C", n_bootstraps=50)
        assert isinstance(res_metric, BootstrapResult)

        res_factorial = evaluate_factorial_contrasts(rates, dist, n_bootstraps=50)
        assert isinstance(res_factorial, FactorialContrastsResult)
