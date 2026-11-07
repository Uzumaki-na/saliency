"""Comprehensive unit test suite for auxiliary saliency loss and representation quality gate.

Verifies Story 4.2 requirements:
- auxiliary_saliency_loss parity with F.binary_cross_entropy on fully valid masks.
- Reflection padding isolation: zero loss and zero gradient outside M_valid.
- Gradient flow exclusively into S_pred; target S_target is detached.
- Spatial downsampling alignment between image-res target and latent-res prediction.
- RateDistortionLoss integration with constant lambda_aux=0.1 and backward compatibility when s_pred is None.
- Continuous linear correlation coefficient (compute_correlation_coefficient): 1.0, -1.0, 0.0.
- Kullback-Leibler divergence (compute_kl_divergence): identical 0.0, positive divergence, zero-stability.
- AUC-Judd metric (compute_auc_judd): perfect 1.0, random ~0.50, continuous and fixation inputs.
- Representation quality gate (evaluate_representation_quality): passes when CC >= 0.60 and KL <= 0.50.
- Error handling: non-tensors, dimension mismatches, empty masks, NaN/Inf detection.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.training import (
    RateDistortionLoss,
    auxiliary_saliency_loss,
    compute_auc_judd,
    compute_correlation_coefficient,
    compute_kl_divergence,
    evaluate_representation_quality,
)


class TestAuxiliarySaliencyLoss:
    """Tests verifying auxiliary_saliency_loss mathematical properties, alignment, and gradient flow."""

    def test_parity_with_pytorch_bce_on_full_valid_mask(self) -> None:
        """When all cells are valid, auxiliary_saliency_loss matches F.binary_cross_entropy within 1e-4."""
        torch.manual_seed(42)
        s_pred = torch.rand(2, 1, 16, 16, dtype=torch.float32)
        s_target = torch.rand(2, 1, 16, 16, dtype=torch.float32)

        loss_custom = auxiliary_saliency_loss(s_pred, s_target)
        loss_torch = F.binary_cross_entropy(s_pred, s_target)

        assert torch.isclose(loss_custom, loss_torch, atol=1e-4)

    def test_padded_cells_contribute_zero_loss_and_zero_gradient(self) -> None:
        """Outside M_valid, cells contribute zero loss and zero gradient."""
        torch.manual_seed(101)
        s_pred = torch.rand(1, 1, 16, 16, dtype=torch.float32, requires_grad=True)
        s_target = torch.rand(1, 1, 16, 16, dtype=torch.float32)

        # Valid mask: central 8x8 region
        mask = torch.zeros(1, 1, 16, 16, dtype=torch.bool)
        mask[0, 0, 4:12, 4:12] = True

        loss = auxiliary_saliency_loss(s_pred, s_target, valid_mask=mask)
        loss.backward()

        assert s_pred.grad is not None
        # Gradients inside valid region must be non-zero
        assert torch.any(s_pred.grad[0, 0, 4:12, 4:12] != 0.0)
        # Gradients outside valid region must be strictly zero
        outside = ~mask
        assert torch.all(s_pred.grad[outside] == 0.0)

    def test_target_is_detached_and_receives_no_gradient(self) -> None:
        """Target saliency map s_target is detached; gradients flow exclusively into s_pred."""
        torch.manual_seed(202)
        s_pred = torch.rand(2, 1, 8, 8, dtype=torch.float32, requires_grad=True)
        s_target = torch.rand(2, 1, 8, 8, dtype=torch.float32, requires_grad=True)

        loss = auxiliary_saliency_loss(s_pred, s_target)
        loss.backward()

        assert s_pred.grad is not None
        assert torch.isfinite(s_pred.grad).all()
        assert s_target.grad is None

    def test_spatial_downsampling_alignment_image_to_latent(self) -> None:
        """Automatically downsamples image-resolution target (e.g. 64x64) to latent-resolution pred (16x16)."""
        torch.manual_seed(303)
        s_pred = torch.rand(2, 1, 16, 16, dtype=torch.float32, requires_grad=True)
        s_target_highres = torch.rand(2, 1, 64, 64, dtype=torch.float32)

        loss = auxiliary_saliency_loss(s_pred, s_target_highres)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

        loss.backward()
        assert s_pred.grad is not None
        assert s_pred.grad.shape == (2, 1, 16, 16)

    def test_numerical_stability_at_extreme_values(self) -> None:
        """Evaluates stably at exact 0.0 and 1.0 extremes without NaN or Inf."""
        s_pred = torch.tensor([[[[0.0, 1.0], [0.5, 0.0]]]], dtype=torch.float32, requires_grad=True)
        s_target = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]], dtype=torch.float32)

        loss = auxiliary_saliency_loss(s_pred, s_target)
        assert torch.isfinite(loss)

        loss.backward()
        assert s_pred.grad is not None
        assert torch.isfinite(s_pred.grad).all()

    def test_non_negative_auxiliary_loss(self) -> None:
        """Auxiliary loss is strictly non-negative, including at p=1.0 and t=1.0."""
        s = torch.ones(4, 4, dtype=torch.float32)
        loss = auxiliary_saliency_loss(s, s)
        assert float(loss.item()) >= 0.0


class TestRateDistortionLossAuxiliaryIntegration:
    """Tests verifying RateDistortionLoss auxiliary loss integration and backward compatibility."""

    def test_rate_distortion_loss_with_s_pred_and_lambda_aux(self) -> None:
        """RateDistortionLoss adds lambda_aux * aux_loss to total loss when s_pred is present."""
        torch.manual_seed(404)
        b, h, w = 2, 32, 32
        target = torch.rand(b, 3, h, w)
        x_hat = torch.rand(b, 3, h, w)
        s = torch.rand(b, 1, h, w)
        s_pred = torch.rand(b, 1, h, w)

        y_lik = torch.full((b, 192, 4, 4), 0.05)
        z_lik = torch.full((b, 128, 1, 1), 0.02)

        output = {
            "x_hat": x_hat,
            "y": torch.zeros(b, 192, 4, 4),
            "z": torch.zeros(b, 128, 1, 1),
            "log_sigma": torch.zeros(b, 192, 4, 4),
            "y_likelihoods": y_lik,
            "z_likelihoods": z_lik,
            "s_pred": s_pred,
        }

        criterion = RateDistortionLoss(lmbda=0.01, lambda_aux=0.1)
        metrics = criterion(output, target, s=s)

        expected_aux = auxiliary_saliency_loss(s_pred, s)
        assert torch.isclose(metrics["aux_loss"], expected_aux, atol=1e-6)
        assert torch.isclose(metrics["lambda_aux"], torch.tensor(0.1), atol=1e-6)

        # Total loss formula: bpp_loss + lmbda * 255^2 * distortion + lambda_aux * aux_loss
        expected_total = metrics["bpp_loss"] + 0.01 * (255.0**2) * metrics["distortion"] + 0.1 * expected_aux
        assert torch.isclose(metrics["loss"], expected_total, atol=1e-5)

        # Also verify with s_pred passed as explicit keyword argument
        output_without_pred = {k: v for k, v in output.items() if k != "s_pred"}
        metrics_kw = criterion(output_without_pred, target, s=s, s_pred=s_pred)
        assert torch.isclose(metrics_kw["aux_loss"], expected_aux, atol=1e-6)
        assert torch.isclose(metrics_kw["loss"], expected_total, atol=1e-5)

    def test_backward_compatibility_without_s_pred(self) -> None:
        """When s_pred is omitted (Models A & B), aux_loss is 0.0 and loss formula is unchanged."""
        torch.manual_seed(505)
        b, h, w = 1, 16, 16
        target = torch.rand(b, 3, h, w)
        x_hat = torch.rand(b, 3, h, w)
        s = torch.rand(b, 1, h, w)

        output = {
            "x_hat": x_hat,
            "y_likelihoods": torch.full((b, 16, 2, 2), 0.1),
            "z_likelihoods": torch.full((b, 16, 1, 1), 0.1),
        }

        criterion = RateDistortionLoss(lmbda=0.02, lambda_aux=0.1)
        metrics = criterion(output, target, s=s)

        assert metrics["aux_loss"].item() == 0.0
        expected_total = metrics["bpp_loss"] + 0.02 * (255.0**2) * metrics["distortion"]
        assert torch.isclose(metrics["loss"], expected_total, atol=1e-5)

    def test_lambda_aux_constant_across_multi_lambda(self) -> None:
        """Verifies lambda_aux remains constant and is not scaled by per-call lmbda."""
        criterion = RateDistortionLoss(lmbda=0.01, lambda_aux=0.1)
        assert criterion.lambda_aux == 0.1

        target = torch.rand(1, 3, 16, 16)
        output = {
            "x_hat": torch.rand(1, 3, 16, 16),
            "y_likelihoods": torch.full((1, 8, 2, 2), 0.1),
            "z_likelihoods": torch.full((1, 8, 1, 1), 0.1),
            "s_pred": torch.rand(1, 1, 16, 16),
        }
        s = torch.rand(1, 1, 16, 16)

        # Call with different lmbda overrides
        for test_lmbda in [0.0018, 0.0130, 0.0483]:
            res = criterion(output, target, s=s, lmbda=test_lmbda)
            expected_loss = res["bpp_loss"] + test_lmbda * (255.0**2) * res["distortion"] + 0.1 * res["aux_loss"]
            assert torch.isclose(res["loss"], expected_loss, atol=1e-5)


class TestCorrelationCoefficientMetric:
    """Tests verifying compute_correlation_coefficient."""

    def test_perfect_positive_correlation(self) -> None:
        """Identical saliency maps have CC == 1.0 within 1e-6."""
        s = torch.tensor([[0.1, 0.4], [0.7, 0.9]], dtype=torch.float32)
        cc = compute_correlation_coefficient(s, s)
        assert math.isclose(cc, 1.0, abs_tol=1e-6)

    def test_inverted_negative_correlation(self) -> None:
        """Inverted saliency maps (s_pred = 1 - s) have CC == -1.0 within 1e-6."""
        s = torch.tensor([[0.1, 0.3], [0.6, 0.8]], dtype=torch.float32)
        s_inv = 1.0 - s
        cc = compute_correlation_coefficient(s, s_inv)
        assert math.isclose(cc, -1.0, abs_tol=1e-5)

    def test_uncorrelated_orthogonal_maps(self) -> None:
        """Uncorrelated saliency maps have CC near 0.0."""
        s1 = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
        s2 = torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float32)
        cc = compute_correlation_coefficient(s1, s2)
        assert math.isclose(cc, 0.0, abs_tol=1e-6)

    def test_constant_identical_maps(self) -> None:
        """Constant identical maps return 1.0."""
        s = torch.full((16, 16), 0.5, dtype=torch.float32)
        cc = compute_correlation_coefficient(s, s)
        assert math.isclose(cc, 1.0, abs_tol=1e-6)

    def test_masked_region_evaluation(self) -> None:
        """Computes correlation exclusively over valid unpadded cells."""
        s1 = torch.tensor([[0.1, 0.9], [0.0, 0.0]], dtype=torch.float32)
        s2 = torch.tensor([[0.1, 0.9], [1.0, 1.0]], dtype=torch.float32)
        mask = torch.tensor([[True, True], [False, False]], dtype=torch.bool)

        cc = compute_correlation_coefficient(s1, s2, valid_mask=mask)
        assert math.isclose(cc, 1.0, abs_tol=1e-5)


class TestKLDivergenceMetric:
    """Tests verifying compute_kl_divergence."""

    def test_identical_distributions_have_zero_kl(self) -> None:
        """Identical saliency distributions have KL == 0.0 exactly."""
        s = torch.tensor([[0.2, 0.5], [0.1, 0.8]], dtype=torch.float32)
        kl = compute_kl_divergence(s, s)
        assert math.isclose(kl, 0.0, abs_tol=1e-6)

    def test_positive_divergence_for_differing_maps(self) -> None:
        """Differing distributions have strictly positive KL divergence."""
        s1 = torch.tensor([[0.8, 0.1], [0.1, 0.0]], dtype=torch.float32)
        s2 = torch.tensor([[0.1, 0.1], [0.1, 0.7]], dtype=torch.float32)
        kl = compute_kl_divergence(s1, s2)
        assert kl > 0.0

    def test_sparse_zero_stability(self) -> None:
        """Sparse maps with multiple zero entries evaluate stably without NaN or Inf."""
        s1 = torch.zeros(16, 16, dtype=torch.float32)
        s1[8, 8] = 1.0
        s2 = torch.full((16, 16), 0.01, dtype=torch.float32)
        kl = compute_kl_divergence(s2, s1)
        assert math.isfinite(kl)
        assert kl >= 0.0


class TestAUCJuddMetric:
    """Tests verifying compute_auc_judd."""

    def test_perfect_discriminator_has_auc_one(self) -> None:
        """Perfect predictor that ranks all fixations above non-fixations has AUC == 1.0."""
        target = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
        pred = torch.tensor([[0.9, 0.8], [0.2, 0.1]], dtype=torch.float32)

        auc = compute_auc_judd(pred, target, threshold=0.5)
        assert math.isclose(auc, 1.0, abs_tol=1e-6)

    def test_inverted_discriminator_has_auc_zero(self) -> None:
        """Inverted predictor that ranks non-fixations above fixations has AUC == 0.0."""
        target = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
        pred = torch.tensor([[0.1, 0.2], [0.8, 0.9]], dtype=torch.float32)

        auc = compute_auc_judd(pred, target, threshold=0.5)
        assert math.isclose(auc, 0.0, abs_tol=1e-6)

    def test_random_uniform_predictions_have_auc_around_half(self) -> None:
        """Random predictions evaluate to ~0.50 AUC."""
        torch.manual_seed(777)
        target = (torch.rand(32, 32) > 0.7).float()
        pred = torch.rand(32, 32)

        auc = compute_auc_judd(pred, target, threshold=0.5)
        assert 0.40 <= auc <= 0.60

    def test_constant_prediction_has_auc_half(self) -> None:
        """Constant/uniform predictions must evaluate to AUC == 0.50 (acceptance criterion 5)."""
        target = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)
        pred_constant = torch.full((2, 2), 0.5, dtype=torch.float32)

        auc = compute_auc_judd(pred_constant, target, threshold=0.5)
        assert math.isclose(auc, 0.50, abs_tol=1e-5)

    def test_batched_evaluation_per_sample_average(self) -> None:
        """Batched evaluations (B > 1) average per-sample metrics independently without cross-image distortion."""
        p1 = torch.tensor([[0.9, 0.9], [0.2, 0.2]])
        t1 = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        p2 = torch.tensor([[0.15, 0.15], [0.01, 0.01]])
        t2 = torch.tensor([[1.0, 1.0], [0.0, 0.0]])

        b_p = torch.stack([p1, p2], dim=0)  # (2, 2, 2)
        b_t = torch.stack([t1, t2], dim=0)

        # Batch AUC must remain 1.0 (mean of two 1.0 AUCs)
        auc = compute_auc_judd(b_p, b_t, threshold=0.5)
        assert math.isclose(auc, 1.0, abs_tol=1e-5)

        # Batch CC must remain 1.0 (mean of two 1.0 CCs)
        cc = compute_correlation_coefficient(b_p, b_t)
        assert math.isclose(cc, 1.0, abs_tol=1e-5)

        # Batch KL divergence
        kl = compute_kl_divergence(b_p, b_p)
        assert math.isclose(kl, 0.0, abs_tol=1e-6)


class TestRepresentationQualityGate:
    """Tests verifying evaluate_representation_quality gate decision boundaries."""

    def test_gate_passes_when_both_criteria_met(self) -> None:
        """Gate passes when CC >= 0.60 and KL <= 0.50."""
        s = torch.tensor([[0.2, 0.4], [0.6, 0.8]], dtype=torch.float32)
        # Identical map has CC=1.0 and KL=0.0
        res = evaluate_representation_quality(s, s, cc_threshold=0.60, kl_threshold=0.50)

        assert res["passed"] is True
        assert res["cc"] >= 0.60
        assert res["kl"] <= 0.50
        assert "auc_judd" in res

    def test_gate_fails_when_cc_insufficient(self) -> None:
        """Gate fails when CC < 0.60 even if KL is small."""
        s1 = torch.tensor([[1.0, 1.0], [-1.0, -1.0]], dtype=torch.float32)
        s2 = torch.tensor([[1.0, -1.0], [1.0, -1.0]], dtype=torch.float32)

        res = evaluate_representation_quality(s1, s2, cc_threshold=0.60, kl_threshold=2.0)
        assert res["passed"] is False

    def test_gate_fails_when_kl_excessive(self) -> None:
        """Gate fails when KL > 0.50 even if CC is high."""
        s = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
        # Stricter KL threshold of 0.0
        res = evaluate_representation_quality(s, s * 0.5 + 0.1, cc_threshold=0.0, kl_threshold=0.0001)
        assert res["passed"] is False


class TestErrorHandlingAndSanitization:
    """Tests verifying robust error handling across auxiliary loss and quality metrics."""

    def test_non_tensor_inputs_raise_type_error(self) -> None:
        """Non-tensor inputs raise TypeError."""
        with pytest.raises(TypeError):
            auxiliary_saliency_loss([[0.5]], torch.rand(2, 2))  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            compute_correlation_coefficient([1.0], [1.0])  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            compute_kl_divergence(torch.rand(4, 4), "not a tensor")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            compute_auc_judd(torch.rand(4, 4), None)  # type: ignore[arg-type]

    def test_nan_or_inf_raises_value_error(self) -> None:
        """NaN or Inf inputs raise ValueError."""
        s_nan = torch.tensor([[float("nan"), 0.5], [0.2, 0.3]])
        s_valid = torch.tensor([[0.5, 0.5], [0.2, 0.3]])

        with pytest.raises(ValueError, match="NaN or Inf"):
            auxiliary_saliency_loss(s_nan, s_valid)
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_correlation_coefficient(s_valid, s_nan)
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_kl_divergence(s_nan, s_valid)

    def test_empty_valid_mask_raises_value_error(self) -> None:
        """Valid mask with no valid cells raises ValueError."""
        s = torch.rand(4, 4)
        mask = torch.zeros(4, 4, dtype=torch.bool)

        with pytest.raises(ValueError, match="no valid cells"):
            auxiliary_saliency_loss(s, s, valid_mask=mask)
        with pytest.raises(ValueError, match="no valid cells"):
            compute_correlation_coefficient(s, s, valid_mask=mask)

    def test_negative_lambda_aux_raises_value_error(self) -> None:
        """Negative lambda_aux raises ValueError."""
        with pytest.raises(ValueError, match="lambda_aux must be non-negative"):
            RateDistortionLoss(lambda_aux=-0.5)

    def test_missing_s_when_s_pred_present_raises_value_error(self) -> None:
        """When s_pred is provided but target s is None, raises ValueError."""
        criterion = RateDistortionLoss()
        output = {
            "x_hat": torch.rand(1, 3, 16, 16),
            "y_likelihoods": torch.full((1, 8, 2, 2), 0.1),
            "s_pred": torch.rand(1, 1, 16, 16),
        }
        with pytest.raises(ValueError, match="saliency map 's' must be provided"):
            criterion(output, target=torch.rand(1, 3, 16, 16), s=None)
