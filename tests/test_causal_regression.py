"""Tests for causal regression with image fixed effects and partial correlation analysis.

Verifies:
1. Exact mathematical equivalence between Frisch-Waugh-Lovell within-estimator and explicit dummy variable OLS.
2. Accurate recovery of synthetic ground truth parameters (beta_S, beta_base, alpha_k).
3. Clustered covariance estimation across image clusters, degrees of freedom, and single-cluster fallback.
4. Elimination of texture complexity bias via partial correlation PartialCorr(S_pred, S | bits_A).
5. High-level run_causal_regression and run_causal_partial_correlation interface with batched tensors,
   SpatialBitMap objects, unpadded valid masks, and multi-channel inputs.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.evaluation.causal import (
    compute_partial_correlation,
    fit_image_fixed_effects_regression,
    run_causal_partial_correlation,
    run_causal_regression,
)
from src.evaluation.spatial_allocation import SpatialBitMap


class TestFixedEffectsRegressionEquivalence:
    """Tests mathematical agreement between within-transformation and explicit dummy OLS."""

    def test_within_estimator_matches_explicit_dummy_ols(self) -> None:
        """Verifies within-estimator beta and alpha identically match full dummy OLS [D, X]."""
        torch.manual_seed(42)

        num_images = 4
        cells_per_image = [30, 45, 50, 40]
        total_cells = sum(cells_per_image)

        # True parameters
        true_alpha = torch.tensor([2.5, -1.8, 4.0, 0.2], dtype=torch.float64)
        true_beta = torch.tensor([3.5, -0.4], dtype=torch.float64)  # beta_S, beta_base

        y_list: list[torch.Tensor] = []
        x_mat_list: list[torch.Tensor] = []

        d_rows: list[torch.Tensor] = []
        x_rows: list[torch.Tensor] = []
        y_rows: list[torch.Tensor] = []

        for k, n_k in enumerate(cells_per_image):
            # Regressors: S in [0, 1], bits_base around 50
            s_k = torch.rand(n_k, dtype=torch.float64)
            b_k = 40.0 + 20.0 * torch.rand(n_k, dtype=torch.float64)
            x_k = torch.stack([s_k, b_k], dim=1)

            noise = 0.05 * torch.randn(n_k, dtype=torch.float64)
            y_k = true_alpha[k] + x_k @ true_beta + noise

            y_list.append(y_k)
            x_mat_list.append(x_k)

            # Build explicit dummy matrix row by row
            dummy_block = torch.zeros(n_k, num_images, dtype=torch.float64)
            dummy_block[:, k] = 1.0

            d_rows.append(dummy_block)
            x_rows.append(x_k)
            y_rows.append(y_k)

        full_d = torch.cat(d_rows, dim=0)
        full_x = torch.cat(x_rows, dim=0)
        full_y = torch.cat(y_rows, dim=0)

        # Explicit dummy regression: Z = [D, X]
        z_mat = torch.cat([full_d, full_x], dim=1)
        # OLS: theta = (Z^T Z)^(-1) Z^T y
        theta_explicit = torch.linalg.pinv(z_mat) @ full_y

        explicit_alpha = theta_explicit[:num_images]
        explicit_beta = theta_explicit[num_images:]

        # Run within-estimator
        res = fit_image_fixed_effects_regression(y_list, x_mat_list)

        # 1. Compare beta coefficients (should match to machine precision < 1e-7)
        assert res.beta_s == pytest.approx(float(explicit_beta[0].item()), abs=1e-7)
        assert res.beta_base == pytest.approx(float(explicit_beta[1].item()), abs=1e-7)

        # 2. Compare recovered image fixed effects alpha_k
        for k in range(num_images):
            assert res.alpha_images[k] == pytest.approx(float(explicit_alpha[k].item()), abs=1e-7)

        # 3. Parameter accuracy against known truth
        assert res.beta_s == pytest.approx(3.5, abs=0.1)
        assert res.beta_base == pytest.approx(-0.4, abs=0.05)
        assert res.hypothesis_confirmed is True
        assert res.p_s < 0.001
        assert res.num_images == num_images
        assert res.num_cells_total == total_cells


class TestClusteredCovarianceAndInference:
    """Tests cluster-robust standard errors, degrees of freedom, and hypothesis confirmation."""

    def test_clustered_standard_errors_positive_and_finite(self) -> None:
        """Verifies standard errors are positive, finite, and scale with residual variance."""
        torch.manual_seed(99)
        num_images = 6
        n_per_img = 64

        y_list = []
        x_list = []
        for _ in range(num_images):
            s = torch.rand(n_per_img, dtype=torch.float64)
            b = torch.randn(n_per_img, dtype=torch.float64)
            x = torch.stack([s, b], dim=1)
            y = 2.0 * s - 0.1 * b + 0.5 * torch.randn(n_per_img, dtype=torch.float64)
            y_list.append(y)
            x_list.append(x)

        res = fit_image_fixed_effects_regression(y_list, x_list)

        assert res.se_s > 0.0
        assert res.se_base > 0.0
        assert not math.isnan(res.se_s)
        assert not math.isnan(res.se_base)
        assert res.degrees_of_freedom == (num_images * n_per_img) - num_images - 2
        assert 0.0 <= res.p_s <= 1.0
        assert 0.0 <= res.p_base <= 1.0
        assert 0.0 <= res.r2_within <= 1.0
        assert 0.0 <= res.r2_overall <= 1.0

    def test_single_image_cluster_fallback_hc1_covariance(self) -> None:
        """Verifies single image cluster (K=1) uses HC1 cell-level covariance and does not falsely confirm noise."""
        torch.manual_seed(42)
        n = 200

        # Pure Gaussian noise: y has zero true correlation with s
        s = torch.rand(n, dtype=torch.float64)
        b = torch.randn(n, dtype=torch.float64)
        x = torch.stack([s, b], dim=1)
        y = torch.randn(n, dtype=torch.float64)

        res = fit_image_fixed_effects_regression([y], [x])
        assert res.num_images == 1
        # Crucial check: Standard error must NOT be the 1e-6 clamp artifact!
        assert res.se_s > 0.05, f"Expected realistic HC1 standard error, got {res.se_s}"
        # On pure noise, hypothesis must NOT be confirmed
        assert res.hypothesis_confirmed is False
        assert res.p_s > 0.05

    def test_unidentifiable_regressor_yields_nan_se(self) -> None:
        """Verifies regressor with no within-image variation produces NaN SE and hypothesis_confirmed=False."""
        n = 100
        # s is constant (zero within-image variance)
        s = torch.full((n,), 0.5, dtype=torch.float64)
        b = torch.randn(n, dtype=torch.float64)
        x = torch.stack([s, b], dim=1)
        y = 3.0 * b + torch.randn(n, dtype=torch.float64)

        res = fit_image_fixed_effects_regression([y], [x])
        assert math.isnan(res.se_s)
        assert res.p_s == 1.0
        assert res.hypothesis_confirmed is False

    def test_negative_beta_hypothesis_unconfirmed(self) -> None:
        """Verifies hypothesis_confirmed is False when beta_s <= 0."""
        n = 80
        s = torch.rand(n, dtype=torch.float64)
        b = torch.randn(n, dtype=torch.float64)
        x = torch.stack([s, b], dim=1)
        # Deliberately negative relationship
        y = -2.0 * s + 0.1 * torch.randn(n, dtype=torch.float64)

        res = fit_image_fixed_effects_regression([y, y], [x, x])
        assert res.beta_s < 0.0
        assert res.hypothesis_confirmed is False


class TestPartialCorrelationAnalysis:
    """Tests first-order partial correlation PartialCorr(X, Y | Z)."""

    def test_spurious_correlation_elimination(self) -> None:
        """Proves partial correlation eliminates spurious correlation induced by texture complexity confounder Z."""
        torch.manual_seed(1234)
        n = 5000

        # Z is texture complexity (e.g. baseline bits)
        z = torch.randn(n)

        # X (predicted saliency) and Y (ground truth saliency) are both conditionally independent given Z:
        # X = 0.8 * Z + eps_X
        # Y = 0.8 * Z + eps_Y
        x = 0.8 * z + 0.3 * torch.randn(n)
        y = 0.8 * z + 0.3 * torch.randn(n)

        # Raw marginal correlation between X and Y is strong (spurious!)
        res = compute_partial_correlation(x, y, z)

        assert res.raw_corr_xy > 0.75, f"Expected high raw correlation, got {res.raw_corr_xy}"
        # Partial correlation controlling for Z should drop to near zero
        assert abs(res.partial_corr) < 0.05, f"Expected partial correlation near 0, got {res.partial_corr}"
        assert res.num_cells == n

    def test_genuine_saliency_signal_preservation(self) -> None:
        """Verifies genuine direct relationship survives conditioning on texture complexity."""
        torch.manual_seed(5678)
        n = 5000

        z = torch.randn(n)  # Texture complexity
        x = torch.randn(n)  # Predicted saliency independent of texture
        # Y (ground truth saliency) depends directly on X plus some texture noise
        y = 0.7 * x + 0.3 * z + 0.2 * torch.randn(n)

        res = compute_partial_correlation(x, y, z)

        # Partial correlation should remain strong
        assert res.partial_corr > 0.85
        assert res.raw_corr_xz == pytest.approx(0.0, abs=0.08)

    def test_1d_vector_inputs_to_partial_correlation(self) -> None:
        """Verifies 1D vector inputs are accepted without spatial interpolation errors."""
        x = torch.randn(200)
        y = x + 0.1 * torch.randn(200)
        z = torch.randn(200)

        res = compute_partial_correlation(x, y, z)
        assert res.num_cells == 200
        assert res.partial_corr > 0.9

    def test_multi_channel_4d_tensors(self) -> None:
        """Verifies multi-channel inputs (C=3) are channel-averaged properly without shape crashes."""
        x = torch.randn(2, 3, 16, 16)
        y = torch.randn(2, 1, 16, 16)
        z = torch.randn(2, 3, 16, 16)

        res = compute_partial_correlation(x, y, z)
        assert res.num_cells == 2 * 16 * 16

    def test_collinear_and_flat_inputs_avoid_nan(self) -> None:
        """Verifies edge cases (collinear Z, zero-variance maps) handle degenerate states safely."""
        x = torch.randn(50)
        y = torch.randn(50)
        # Z perfectly collinear with X
        res_collinear = compute_partial_correlation(x, y, x)
        assert res_collinear.partial_corr == 0.0

        # Flat map
        flat = torch.zeros(50)
        res_flat = compute_partial_correlation(flat, y, x)
        assert res_flat.partial_corr == 0.0

    def test_valid_mask_filtering_partial_correlation(self) -> None:
        """Verifies valid mask isolates valid cells in partial correlation."""
        x = torch.randn(10, 10)
        y = x.clone()
        z = torch.randn(10, 10)

        # Corrupt border cells
        x[0, :] = 1000.0
        y[0, :] = -1000.0

        mask = torch.ones(10, 10, dtype=torch.bool)
        mask[0, :] = False

        res = compute_partial_correlation(x, y, z, valid_mask=mask)
        # When border is masked out, x == y, so partial correlation is 1.0
        assert res.partial_corr == pytest.approx(1.0, abs=1e-5)
        assert res.num_cells == 90


class TestHighLevelCausalPipeline:
    """Tests run_causal_regression and run_causal_partial_correlation end-to-end."""

    def test_run_causal_regression_with_spatial_bit_maps(self) -> None:
        """Verifies run_causal_regression accepts SpatialBitMap containers."""
        torch.manual_seed(42)
        b, c, h, w = 3, 1, 16, 16

        # Construct synthetic baseline bits (Model A) and target bits (Model C)
        saliency = torch.rand(b, c, h, w)
        bits_a = 50.0 + 10.0 * torch.randn(b, c, h, w)

        # Model C allocates more bits where saliency is high
        bits_c = bits_a + 8.0 * saliency + torch.randn(b, c, h, w)

        mask = torch.ones(b, c, h, w, dtype=torch.bool)

        sbm_c = SpatialBitMap(
            y_bits=bits_c,
            z_bits=torch.zeros_like(bits_c),
            total_bits=bits_c,
            valid_mask=mask,
        )
        sbm_a = SpatialBitMap(
            y_bits=bits_a,
            z_bits=torch.zeros_like(bits_a),
            total_bits=bits_a,
            valid_mask=mask,
        )

        res = run_causal_regression(
            bits_target=sbm_c,
            bits_baseline=sbm_a,
            saliency=saliency,
        )

        assert res.beta_s > 6.0
        assert res.hypothesis_confirmed is True
        assert res.p_s < 0.001
        assert res.num_images == b
        assert res.num_cells_total == b * h * w

    def test_mask_fallback_from_bits_baseline(self) -> None:
        """Verifies valid mask is extracted from bits_baseline SpatialBitMap when bits_target is a raw Tensor."""
        b, c, h, w = 2, 1, 16, 16
        bits_t = torch.randn(b, c, h, w)
        bits_b = torch.randn(b, c, h, w)
        sal = torch.rand(b, c, h, w)
        mask = torch.zeros(b, c, h, w, dtype=torch.bool)
        mask[:, :, 2:14, 2:14] = True

        sbm_baseline = SpatialBitMap(
            y_bits=bits_b,
            z_bits=torch.zeros_like(bits_b),
            total_bits=bits_b,
            valid_mask=mask,
        )

        res = run_causal_regression(
            bits_target=bits_t,
            bits_baseline=sbm_baseline,
            saliency=sal,
        )

        assert res.num_cells_total == 2 * 12 * 12

    def test_run_causal_partial_correlation_multi_image(self) -> None:
        """Verifies run_causal_partial_correlation with per-image demeaning."""
        b, c, h, w = 2, 1, 16, 16
        s_pred = torch.rand(b, c, h, w)
        s_ref = s_pred + 0.1 * torch.randn(b, c, h, w)
        bits_a = torch.randn(b, c, h, w)

        res = run_causal_partial_correlation(
            s_pred=s_pred,
            s_ref=s_ref,
            bits_baseline=bits_a,
            demean_per_image=True,
        )

        assert res.partial_corr > 0.8
        assert res.num_cells == b * h * w
