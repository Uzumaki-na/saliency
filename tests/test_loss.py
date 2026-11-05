"""Comprehensive unit test suite for saliency-weighted distortion and RateDistortionLoss.

Verifies:
- Uniform saliency equivalence (S=1.0, S=0.0, S=None) to standard MSE within 1e-6.
- Explicit 0.2 background floor bounding (S=0 -> 0.2, S=1 -> 1.0).
- Saliency gradient isolation (requires_grad on S produces no gradients into S).
- Valid mask isolation for reflection-padded inputs (zero loss impact and zero grads outside M_valid).
- RateDistortionLoss formulation (loss == bpp_loss + lmbda * 255^2 * distortion) across multi-lambda.
- Bpp conversion correctness (-sum log2(p) / (B * H * W)).
- Strict FP32 precision execution under autocast.
- Robust error handling for dimension mismatches and invalid inputs.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.codec.model_a import ModelA, ModelOutput
from src.data.padding import (
    pad_image_and_saliency_centered,
    pad_symmetric_centered,
    unpad_centered,
)
from src.training.loss import RateDistortionLoss, saliency_weighted_distortion

LAMBDA_RATE_POINTS = [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483]


class TestSaliencyWeightedDistortionUniformEquivalence:
    """Tests verifying that uniform or None saliency maps are mathematically equivalent to unweighted MSE."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_saliency_none_equals_mse(self, batch_size: int) -> None:
        """When S is None, D_saliency must match standard unweighted MSE within 1e-6."""
        torch.manual_seed(42)
        x = torch.rand(batch_size, 3, 64, 64)
        x_hat = torch.rand(batch_size, 3, 64, 64)

        expected_mse = torch.mean((x - x_hat) ** 2)
        d_val = saliency_weighted_distortion(x, x_hat, s=None)

        assert torch.isclose(d_val, expected_mse, atol=1e-6)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_saliency_ones_equals_mse(self, batch_size: int) -> None:
        """When S is all 1.0, D_saliency must match standard unweighted MSE within 1e-6."""
        torch.manual_seed(42)
        x = torch.rand(batch_size, 3, 64, 64)
        x_hat = torch.rand(batch_size, 3, 64, 64)
        s = torch.ones(batch_size, 1, 64, 64)

        expected_mse = torch.mean((x - x_hat) ** 2)
        d_val = saliency_weighted_distortion(x, x_hat, s=s)

        assert torch.isclose(d_val, expected_mse, atol=1e-6)

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_saliency_zeros_equals_mse_via_cancellation(self, batch_size: int) -> None:
        """When S is all 0.0 (S'=0.2), spatial normalization cancels 0.2, matching MSE within 1e-6."""
        torch.manual_seed(42)
        x = torch.rand(batch_size, 3, 64, 64)
        x_hat = torch.rand(batch_size, 3, 64, 64)
        s = torch.zeros(batch_size, 1, 64, 64)

        expected_mse = torch.mean((x - x_hat) ** 2)
        d_val = saliency_weighted_distortion(x, x_hat, s=s)

        assert torch.isclose(d_val, expected_mse, atol=1e-6)

    def test_saliency_constant_arbitrary_value_equals_mse(self) -> None:
        """Any spatially uniform saliency map (e.g. S=0.5) must cancel and equal unweighted MSE."""
        torch.manual_seed(123)
        x = torch.rand(2, 3, 64, 64)
        x_hat = torch.rand(2, 3, 64, 64)
        s = torch.full((2, 1, 64, 64), 0.5)

        expected_mse = torch.mean((x - x_hat) ** 2)
        d_val = saliency_weighted_distortion(x, x_hat, s=s)

        assert torch.isclose(d_val, expected_mse, atol=1e-6)

    def test_3d_input_without_batch_dim_equals_mse(self) -> None:
        """Supports 3D tensor inputs (C, H, W) matching unweighted MSE within 1e-6."""
        torch.manual_seed(99)
        x = torch.rand(3, 32, 32)
        x_hat = torch.rand(3, 32, 32)
        s = torch.ones(1, 32, 32)

        expected_mse = torch.mean((x - x_hat) ** 2)
        d_val = saliency_weighted_distortion(x, x_hat, s=s)

        assert torch.isclose(d_val, expected_mse, atol=1e-6)


class TestSaliencyFloorAndWeighting:
    """Tests verifying the 0.2 background floor and relative weighting behavior."""

    def test_relative_weight_ratio_between_zero_and_one_saliency(self) -> None:
        r"""Verify that a pixel with S=0 has exactly 0.2x the weight of a pixel with S=1.

        Construct an image where pixel A has S=0 and pixel B has S=1.
        Loss contribution from pixel A alone vs pixel B alone must have ratio 0.2 / 1.0 = 0.2.
        """
        h, w = 32, 32
        x = torch.zeros(1, 1, h, w)
        s = torch.full((1, 1, h, w), 0.5)
        s[0, 0, 5, 5] = 0.0  # Pixel A: S=0 -> S'=0.2
        s[0, 0, 10, 10] = 1.0  # Pixel B: S=1 -> S'=1.0

        # Scenario 1: only pixel A has error 1.0
        x_hat_a = torch.zeros(1, 1, h, w)
        x_hat_a[0, 0, 5, 5] = 1.0
        d_a = saliency_weighted_distortion(x, x_hat_a, s=s)

        # Scenario 2: only pixel B has error 1.0
        x_hat_b = torch.zeros(1, 1, h, w)
        x_hat_b[0, 0, 10, 10] = 1.0
        d_b = saliency_weighted_distortion(x, x_hat_b, s=s)

        ratio = (d_a / d_b).item()
        assert math.isclose(ratio, 0.2, rel_tol=1e-5)

    def test_saliency_out_of_bounds_clamping(self) -> None:
        """Values of S < 0 are clamped to 0 (S'=0.2), and S > 1 are clamped to 1 (S'=1.0)."""
        h, w = 16, 16
        x = torch.zeros(1, 1, h, w)
        x_hat = torch.ones(1, 1, h, w)

        s_negative = torch.full((1, 1, h, w), -2.0)
        s_positive_excess = torch.full((1, 1, h, w), 5.0)

        d_neg = saliency_weighted_distortion(x, x_hat, s=s_negative)
        d_pos = saliency_weighted_distortion(x, x_hat, s=s_positive_excess)

        # Since each is spatially uniform after clamping, each must equal 1.0
        assert torch.isclose(d_neg, torch.tensor(1.0), atol=1e-6)
        assert torch.isclose(d_pos, torch.tensor(1.0), atol=1e-6)

    def test_arbitrary_saliency_manual_calculation(self) -> None:
        """Verify weighted MSE against manual explicit reference calculation on a 2x2 grid."""
        x = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]])  # (1, 1, 2, 2)
        x_hat = torch.tensor([[[[0.3, 0.1], [0.9, 0.5]]]])
        s = torch.tensor([[[[0.0, 0.25], [0.5, 1.0]]]])

        # S' = 0.2 + 0.8 * S
        s_prime = 0.2 + 0.8 * s
        sq_err = (x - x_hat) ** 2
        expected = (torch.sum(s_prime * sq_err) / torch.sum(s_prime)).item()

        actual = saliency_weighted_distortion(x, x_hat, s=s).item()
        assert math.isclose(actual, expected, rel_tol=1e-6)


class TestGradientIsolation:
    """Tests verifying that gradients flow through x_hat and never into saliency map S."""

    def test_saliency_receives_no_gradients_in_functional_distortion(self) -> None:
        """When s has requires_grad=True, x_hat receives grads while s receives none."""
        x = torch.rand(2, 3, 32, 32)
        x_hat = torch.rand(2, 3, 32, 32, requires_grad=True)
        s = torch.rand(2, 1, 32, 32, requires_grad=True)

        d_val = saliency_weighted_distortion(x, x_hat, s=s)
        d_val.backward()

        assert x_hat.grad is not None
        assert torch.any(x_hat.grad != 0.0)
        assert s.grad is None

    def test_saliency_receives_no_gradients_in_rate_distortion_loss(self) -> None:
        """RateDistortionLoss preserves gradient isolation on saliency map."""
        x = torch.rand(2, 3, 64, 64)
        x_hat = torch.rand(2, 3, 64, 64, requires_grad=True)
        y_lik = torch.full((2, 192, 4, 4), 0.05, requires_grad=True)
        z_lik = torch.full((2, 128, 1, 1), 0.05, requires_grad=True)
        s = torch.rand(2, 1, 64, 64, requires_grad=True)

        output = {
            "x_hat": x_hat,
            "y_likelihoods": y_lik,
            "z_likelihoods": z_lik,
        }

        criterion = RateDistortionLoss(lmbda=0.0130)
        res = criterion(output, x, s=s)
        res["loss"].backward()

        assert x_hat.grad is not None
        assert y_lik.grad is not None
        assert z_lik.grad is not None
        assert s.grad is None


class TestValidMaskIsolation:
    """Tests verifying that pixel errors outside M_valid do not affect loss or gradients."""

    def test_errors_outside_valid_mask_do_not_affect_distortion_value(self) -> None:
        """Perturbing pixels outside M_valid produces zero change in calculated distortion."""
        h, w = 64, 64
        x = torch.rand(1, 3, h, w)
        x_hat_base = torch.rand(1, 3, h, w)
        s = torch.rand(1, 1, h, w)

        # Valid mask: central region [16:48, 16:48]
        mask = torch.zeros(1, 1, h, w, dtype=torch.bool)
        mask[0, 0, 16:48, 16:48] = True

        d_base = saliency_weighted_distortion(x, x_hat_base, s=s, valid_mask=mask)

        # Corrupt pixels strictly outside mask
        x_hat_corrupted = x_hat_base.clone()
        x_hat_corrupted[0, :, :16, :] += 5.0
        x_hat_corrupted[0, :, 48:, :] -= 10.0
        x_hat_corrupted[0, :, :, :16] = 99.0
        x_hat_corrupted[0, :, :, 48:] = -99.0

        d_corrupted = saliency_weighted_distortion(x, x_hat_corrupted, s=s, valid_mask=mask)

        assert torch.isclose(d_base, d_corrupted, atol=1e-7)

    def test_gradients_outside_valid_mask_are_strictly_zero(self) -> None:
        """Gradients outside M_valid must be identically 0.0."""
        h, w = 64, 64
        x = torch.rand(1, 3, h, w)
        x_hat = torch.rand(1, 3, h, w, requires_grad=True)
        s = torch.rand(1, 1, h, w)

        mask = torch.zeros(1, 1, h, w, dtype=torch.bool)
        mask[0, 0, 16:48, 16:48] = True

        d_val = saliency_weighted_distortion(x, x_hat, s=s, valid_mask=mask)
        d_val.backward()

        assert x_hat.grad is not None
        # Gradients outside valid mask must be all zero
        outside_mask = ~mask.expand_as(x_hat)
        inside_mask = mask.expand_as(x_hat)

        assert torch.all(x_hat.grad[outside_mask] == 0.0)
        assert torch.any(x_hat.grad[inside_mask] != 0.0)

    def test_masked_distortion_matches_unpadded_crop(self) -> None:
        """Compare masked distortion on padded tensor with unmasked distortion on unpadded crop."""
        orig_h, orig_w = 48, 50
        x_orig = torch.rand(1, 3, orig_h, orig_w)
        s_orig = torch.rand(1, 1, orig_h, orig_w)
        x_hat_orig = torch.rand(1, 3, orig_h, orig_w)

        # Pad using centered symmetric padding to multiple of 64
        x_pad, s_pad, bounds = pad_image_and_saliency_centered(x_orig, s_orig, factor=64)
        x_hat_pad, _ = pad_symmetric_centered(x_hat_orig, factor=64, pad_bounds=bounds)

        # Build valid image mask matching unpadded bounds
        h_pad, w_pad = x_pad.shape[-2], x_pad.shape[-1]
        mask = torch.zeros(1, 1, h_pad, w_pad, dtype=torch.bool)
        mask[
            0,
            0,
            bounds.pad_top : h_pad - bounds.pad_bottom,
            bounds.pad_left : w_pad - bounds.pad_right,
        ] = True

        d_padded = saliency_weighted_distortion(x_pad, x_hat_pad, s=s_pad, valid_mask=mask)

        # Crop back using unpad_centered
        x_crop = unpad_centered(x_pad, bounds)
        x_hat_crop = unpad_centered(x_hat_pad, bounds)
        s_crop = unpad_centered(s_pad, bounds)
        d_crop = saliency_weighted_distortion(x_crop, x_hat_crop, s=s_crop)

        assert torch.isclose(d_padded, d_crop, atol=1e-6)


class TestRateDistortionLossModule:
    """Tests verifying RateDistortionLoss module behavior, metrics, and multi-lambda evaluations."""

    @pytest.mark.parametrize("lmbda", LAMBDA_RATE_POINTS)
    def test_loss_equals_bpp_plus_scaled_distortion(self, lmbda: float) -> None:
        """Acceptance: loss == bpp_loss + lmbda * 255^2 * distortion across all 6 lambda points."""
        torch.manual_seed(42)
        b, h, w = 2, 64, 64
        target = torch.rand(b, 3, h, w)
        x_hat = torch.rand(b, 3, h, w)
        y_lik = torch.full((b, 192, 4, 4), 0.05)
        z_lik = torch.full((b, 128, 1, 1), 0.02)
        s = torch.rand(b, 1, h, w)

        output = ModelOutput(
            x_hat=x_hat,
            y=torch.zeros(b, 192, 4, 4),
            z=torch.zeros(b, 128, 1, 1),
            log_sigma=torch.zeros(b, 192, 4, 4),
            y_likelihoods=y_lik,
            z_likelihoods=z_lik,
        )

        criterion = RateDistortionLoss(lmbda=lmbda)
        metrics = criterion(output, target, s=s)

        expected_distortion = saliency_weighted_distortion(target, x_hat, s=s)
        expected_mse = torch.mean((target - x_hat) ** 2)

        num_pixels = b * h * w
        expected_y_bpp = -torch.sum(torch.log2(y_lik)) / num_pixels
        expected_z_bpp = -torch.sum(torch.log2(z_lik)) / num_pixels
        expected_bpp = expected_y_bpp + expected_z_bpp
        expected_loss = expected_bpp + lmbda * (255.0**2) * expected_distortion

        assert torch.isclose(metrics["y_bpp"], expected_y_bpp, atol=1e-5)
        assert torch.isclose(metrics["z_bpp"], expected_z_bpp, atol=1e-5)
        assert torch.isclose(metrics["bpp_loss"], expected_bpp, atol=1e-5)
        assert torch.isclose(metrics["distortion"], expected_distortion, atol=1e-6)
        assert torch.isclose(metrics["mse_loss"], expected_mse, atol=1e-6)
        assert torch.isclose(metrics["loss"], expected_loss, atol=1e-4)

    def test_per_call_lambda_override(self) -> None:
        """RateDistortionLoss allows overriding lmbda per call."""
        b, h, w = 1, 64, 64
        target = torch.rand(b, 3, h, w)
        output = {
            "x_hat": torch.rand(b, 3, h, w),
            "y_likelihoods": torch.full((b, 192, 4, 4), 0.1),
            "z_likelihoods": torch.full((b, 128, 1, 1), 0.1),
        }

        criterion = RateDistortionLoss(lmbda=0.0018)
        out_default = criterion(output, target)
        out_override = criterion(output, target, lmbda=0.0483)

        assert not torch.isclose(out_default["loss"], out_override["loss"])
        # Check math of override
        diff = out_override["loss"] - out_override["bpp_loss"]
        expected_scaled_d = 0.0483 * (255.0**2) * out_override["distortion"]
        assert torch.isclose(diff, expected_scaled_d, atol=1e-4)

    def test_nested_likelihood_dictionary_support(self) -> None:
        """Supports CompressAI standard nested likelihoods dict {'likelihoods': {'y': ..., 'z': ...}}."""
        b, h, w = 1, 64, 64
        target = torch.rand(b, 3, h, w)
        output = {
            "x_hat": torch.rand(b, 3, h, w),
            "likelihoods": {
                "y": torch.full((b, 192, 4, 4), 0.08),
                "z": torch.full((b, 128, 1, 1), 0.04),
            },
        }

        criterion = RateDistortionLoss(lmbda=0.01)
        res = criterion(output, target)

        assert "loss" in res
        assert "y_bpp" in res
        assert "z_bpp" in res
        assert res["y_bpp"] > 0
        assert res["z_bpp"] > 0

    def test_model_a_integration(self) -> None:
        """Full end-to-end forward integration with ModelA instance."""
        model = ModelA()
        x = torch.rand(1, 3, 64, 64)
        s = torch.rand(1, 1, 64, 64)

        output = model(x)
        criterion = RateDistortionLoss(lmbda=0.0067)
        res = criterion(output, x, s=s)

        for key in ["loss", "bpp_loss", "y_bpp", "z_bpp", "distortion", "mse_loss"]:
            assert key in res
            assert isinstance(res[key], torch.Tensor)
            assert res[key].ndim == 0
            assert not torch.isnan(res[key])
            assert not torch.isinf(res[key])


class TestNumericalPrecisionAndAutocast:
    """Tests verifying strict FP32 precision under autocast contexts."""

    def test_strict_fp32_under_cpu_autocast(self) -> None:
        """Distortion and loss outputs must be FP32 even when executed under bfloat16 autocast."""
        x = torch.rand(2, 3, 32, 32)
        x_hat = torch.rand(2, 3, 32, 32)
        s = torch.rand(2, 1, 32, 32)

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            d = saliency_weighted_distortion(x, x_hat, s=s)

        assert d.dtype == torch.float32

    def test_bf16_inputs_promoted_to_fp32(self) -> None:
        """Input tensors provided in bfloat16 are safely promoted to float32 without underflow."""
        x = torch.rand(1, 3, 32, 32, dtype=torch.bfloat16)
        x_hat = torch.rand(1, 3, 32, 32, dtype=torch.bfloat16)
        s = torch.rand(1, 1, 32, 32, dtype=torch.bfloat16)

        d = saliency_weighted_distortion(x, x_hat, s=s)
        assert d.dtype == torch.float32


class TestErrorHandlingAndValidation:
    """Tests verifying proper exceptions for mismatched shapes, invalid types, and arguments."""

    def test_shape_mismatch_raises_value_error(self) -> None:
        """Target x and reconstruction x_hat with differing spatial shapes raise ValueError."""
        x = torch.rand(2, 3, 64, 64)
        x_hat = torch.rand(2, 3, 32, 32)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            saliency_weighted_distortion(x, x_hat)

    def test_saliency_spatial_mismatch_raises_value_error(self) -> None:
        """Saliency map with mismatched spatial dimensions raises ValueError."""
        x = torch.rand(1, 3, 64, 64)
        x_hat = torch.rand(1, 3, 64, 64)
        s = torch.rand(1, 1, 32, 32)

        with pytest.raises(ValueError, match="Spatial dimension mismatch"):
            saliency_weighted_distortion(x, x_hat, s=s)

    def test_valid_mask_spatial_mismatch_raises_value_error(self) -> None:
        """Valid mask with mismatched spatial dimensions raises ValueError."""
        x = torch.rand(1, 3, 64, 64)
        x_hat = torch.rand(1, 3, 64, 64)
        mask = torch.ones(32, 32, dtype=torch.bool)

        with pytest.raises(ValueError, match="Spatial dimension mismatch"):
            saliency_weighted_distortion(x, x_hat, valid_mask=mask)

    def test_non_tensor_inputs_raise_type_error(self) -> None:
        """Passing non-torch.Tensor instances raises TypeError."""
        with pytest.raises(TypeError, match=r"must be a torch\.Tensor"):
            saliency_weighted_distortion([1, 2, 3], torch.rand(3, 10, 10))  # type: ignore[arg-type]

    def test_negative_lambda_raises_value_error(self) -> None:
        """Initializing RateDistortionLoss with negative lambda raises ValueError."""
        with pytest.raises(ValueError, match="lmbda must be non-negative"):
            RateDistortionLoss(lmbda=-0.01)

    def test_missing_x_hat_raises_value_error(self) -> None:
        """RateDistortionLoss raises ValueError if output lacks x_hat."""
        criterion = RateDistortionLoss(lmbda=0.01)
        with pytest.raises(ValueError, match="output must contain 'x_hat'"):
            criterion({}, torch.rand(1, 3, 16, 16))
