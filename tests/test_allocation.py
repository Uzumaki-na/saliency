"""Comprehensive unit test suite for log-domain adaptive quantization step module.

Validates Story 4.1 requirements:
- Uniform saliency equivalence: constant S_pred produces log_delta == 0.0 and delta == 1.0.
- Exact zero log-mean constraint: continuous log-mean == 0.0 within 1e-6; discrete log-mean <= 1/32.
- Step bounds: delta in [0.25, 4.0] and log_delta in [-log 4.0, +log 4.0] across extreme contrast.
- Reflection padding / M_valid isolation: padded cells have log_delta == 0.0 and delta == 1.0.
- Straight-Through Estimator (STE) gradient flow: unity backward gradient through quantizer and finite grads to S_pred.
- Static 129-entry offline table parity: table lookup bit-exact or matches torch.exp(log_delta) within 1e-6.
- Effective log-scale computation: log_sigma_eff = log_sigma - log_delta.
- Adaptive latent quantization: quantize_latents_adaptive in training (noise) and eval (round) modes.
- Batch and dimension invariance: (H, W), (1, H, W), (B, 1, H, W), (B, C, H, W).
- Error handling: dimension mismatches, empty masks, NaN/inf detection, and range validation.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.codec.allocation import (
    DELTA_TABLE_129,
    STE1Over32Quantizer,
    build_delta_table,
    compute_effective_log_scale,
    compute_log_delta,
    lookup_delta_from_table,
    quantize_latents_adaptive,
    quantize_ste_1_32,
)
from src.data.padding import compute_valid_latent_mask


class TestDeltaTableAndQuantizerSTE:
    """Tests for build_delta_table, DELTA_TABLE_129, and quantize_ste_1_32."""

    def test_delta_table_structure_and_constants(self) -> None:
        """Validates structure, bounds, symmetry, and constants of precomputed 129-entry table."""
        table = DELTA_TABLE_129
        assert isinstance(table, torch.Tensor)
        assert table.shape == (129,)
        assert table.dtype == torch.float32
        assert table.device.type == "cpu"

        # Bounds: table must strictly span [0.25, 4.0]
        assert (table >= 0.25).all()
        assert (table <= 4.0).all()
        assert math.isclose(table.min().item(), 0.25, abs_tol=1e-6)
        assert math.isclose(table.max().item(), 4.0, abs_tol=1e-6)
        assert math.isclose(table[0].item(), 0.25, abs_tol=1e-6)
        assert math.isclose(table[128].item(), 4.0, abs_tol=1e-6)

        # Center element (k=0, idx=64) must be exactly 1.0
        assert math.isclose(table[64].item(), 1.0, abs_tol=1e-6)

        # Symmetry: T[64 - k] * T[64 + k] == 1.0 across all k
        for k in range(1, 65):
            val_neg = table[64 - k].item()
            val_pos = table[64 + k].item()
            assert math.isclose(val_neg * val_pos, 1.0, abs_tol=1e-5), f"Asymmetry at k={k}: {val_neg} vs {val_pos}"

    def test_build_delta_table_custom_parameters(self) -> None:
        """Validates build_delta_table with custom odd entries and max factors."""
        table_65 = build_delta_table(num_entries=65, max_factor=2.0)
        assert table_65.shape == (65,)
        assert math.isclose(table_65[32].item(), 1.0, abs_tol=1e-6)
        assert math.isclose(table_65[0].item(), 0.5, abs_tol=1e-6)
        assert math.isclose(table_65[64].item(), 2.0, abs_tol=1e-6)

        with pytest.raises(ValueError, match="odd positive integer"):
            build_delta_table(num_entries=128)
        with pytest.raises(ValueError, match="odd positive integer"):
            build_delta_table(num_entries=1)
        with pytest.raises(ValueError, match="odd positive integer"):
            build_delta_table(num_entries=-5)
        with pytest.raises(ValueError, match=r"greater than 1\.0"):
            build_delta_table(max_factor=0.5)

    def test_quantize_ste_1_32_forward_rounding(self) -> None:
        """Verifies 1/32 grid rounding in the forward pass."""
        inputs = torch.tensor([-0.035, -0.015, 0.0, 0.015, 0.035, 0.0625], dtype=torch.float32)
        expected = torch.round(inputs * 32.0) / 32.0
        out = quantize_ste_1_32(inputs)
        assert torch.allclose(out, expected, atol=1e-7)

    def test_quantize_ste_1_32_backward_unity_jacobian(self) -> None:
        """Verifies Straight-Through Estimator backward pass has unity Jacobian dy/dx = 1.0."""
        x = torch.randn(20, dtype=torch.float32, requires_grad=True)
        y = quantize_ste_1_32(x)
        # Random arbitrary downstream gradient
        grad_out = torch.randn_like(x)
        (grad_in,) = torch.autograd.grad(y, x, grad_outputs=grad_out)
        assert torch.allclose(grad_in, grad_out, atol=1e-7)

        # Test Module wrapper
        module = STE1Over32Quantizer()
        x2 = torch.randn(10, dtype=torch.float32, requires_grad=True)
        y2 = module(x2)
        assert torch.allclose(y2, quantize_ste_1_32(x2), atol=1e-7)

    def test_lookup_delta_from_table_parity(self) -> None:
        """Verifies lookup_delta_from_table across full range k in [-64, 64]."""
        # Test across the entire integer grid k in [-64, 64]
        k = torch.arange(-64, 65, dtype=torch.float32)
        log_delta = k / 32.0
        delta_table = lookup_delta_from_table(log_delta)

        # Bounds: all entries must be in [0.25, 4.0]
        assert (delta_table >= 0.25).all()
        assert (delta_table <= 4.0).all()
        assert math.isclose(delta_table[0].item(), 0.25, abs_tol=1e-6)
        assert math.isclose(delta_table[128].item(), 4.0, abs_tol=1e-6)

        # For all |k| <= 44, must match torch.exp(log_delta) within 1e-6
        mask_44 = torch.abs(k) <= 44
        assert torch.allclose(delta_table[mask_44], torch.exp(log_delta[mask_44]), atol=1e-6)

        # For |k| >= 45, clamped to 0.25 and 4.0
        assert (delta_table[k <= -45] == 0.25).all()
        assert (delta_table[k >= 45] == 4.0).all()

    def test_lookup_delta_table_validation_and_inv_grid(self) -> None:
        """Verifies table validation (odd, 1D, >=3) and custom inv_grid in lookup_delta_from_table."""
        log_delta = torch.zeros(4, dtype=torch.float32)

        # Even number of elements
        with pytest.raises(ValueError, match="odd number of elements"):
            lookup_delta_from_table(log_delta, table=torch.ones(10))

        # Fewer than 3 elements
        with pytest.raises(ValueError, match="odd number of elements"):
            lookup_delta_from_table(log_delta, table=torch.ones(1))

        # Not 1D
        with pytest.raises(ValueError, match="odd number of elements"):
            lookup_delta_from_table(log_delta, table=torch.ones(3, 3))

        # Invalid inv_grid
        with pytest.raises(ValueError, match="greater than 0"):
            lookup_delta_from_table(log_delta, inv_grid=0.0)

        # Valid custom table and inv_grid
        custom_table = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        out = lookup_delta_from_table(torch.tensor([-0.1, 0.0, 0.1]), table=custom_table, inv_grid=10.0)
        assert torch.allclose(out, torch.tensor([0.5, 1.0, 2.0]))

    def test_lookup_delta_from_table_device_and_shape_invariance(self) -> None:
        """Verifies lookup_delta_from_table preserves device and multi-dimensional tensor shape."""
        shapes = [(16, 16), (1, 16, 16), (2, 1, 16, 16), (2, 4, 16, 16)]
        for shp in shapes:
            log_delta = torch.zeros(shp, dtype=torch.float32)
            out = lookup_delta_from_table(log_delta)
            assert out.shape == shp
            assert torch.allclose(out, torch.ones(shp), atol=1e-7)


class TestUniformSaliencyEquivalence:
    """Verifies that uniform saliency distributions produce log_delta == 0.0 and delta == 1.0."""

    @pytest.mark.parametrize("constant_val", [0.0, 0.2, 0.5, 0.8, 1.0])
    @pytest.mark.parametrize("shape", [(16, 16), (1, 16, 16), (2, 1, 16, 16)])
    def test_uniform_saliency_produces_identity_step(
        self,
        constant_val: float,
        shape: tuple[int, ...],
    ) -> None:
        """Uniform saliency across any constant value must produce log_delta=0 and delta=1."""
        s_pred = torch.full(shape, constant_val, dtype=torch.float32)
        log_delta, delta = compute_log_delta(s_pred)

        assert torch.allclose(log_delta, torch.zeros_like(log_delta), atol=1e-6)
        assert torch.allclose(delta, torch.ones_like(delta), atol=1e-6)

    def test_uniform_saliency_with_valid_mask(self) -> None:
        """Uniform saliency inside a valid mask must yield log_delta=0, delta=1 inside and outside."""
        s_pred = torch.full((2, 1, 16, 16), 0.7, dtype=torch.float32)
        mask = torch.zeros((16, 16), dtype=torch.bool)
        mask[2:14, 2:14] = True  # Central valid region

        log_delta, delta = compute_log_delta(s_pred, valid_mask=mask)

        assert torch.allclose(log_delta, torch.zeros_like(log_delta), atol=1e-6)
        assert torch.allclose(delta, torch.ones_like(delta), atol=1e-6)


class TestZeroLogMeanConstraint:
    """Verifies the zero log-mean constraint over valid cells."""

    @pytest.mark.parametrize("seed", [101, 202, 303])
    def test_continuous_log_mean_is_zero_without_padding(self, seed: int) -> None:
        """Over fully valid cells, the continuous log-mean sum(log_delta_cont)/N == 0 within 1e-6."""
        torch.manual_seed(seed)
        s_pred = torch.rand(4, 1, 32, 32, dtype=torch.float32)

        log_delta_cont, _ = compute_log_delta(s_pred, apply_ste=False)

        # Per-batch spatial mean must be 0 within 1e-6
        mean_per_batch = log_delta_cont.mean(dim=(-2, -1))
        assert torch.allclose(mean_per_batch, torch.zeros_like(mean_per_batch), atol=1e-6)

    @pytest.mark.parametrize("seed", [404, 505])
    def test_continuous_log_mean_is_zero_with_valid_mask(self, seed: int) -> None:
        """Over valid cells in M_valid, continuous log-mean must equal 0 within 1e-6."""
        torch.manual_seed(seed)
        s_pred = torch.rand(3, 1, 24, 24, dtype=torch.float32)

        # Create valid mask with borders excluded
        mask = torch.zeros((24, 24), dtype=torch.bool)
        mask[3:21, 4:20] = True
        n_valid = mask.sum().item()

        log_delta_cont, _ = compute_log_delta(s_pred, valid_mask=mask, apply_ste=False)

        # Mean strictly over valid cells
        for b in range(3):
            valid_slice = log_delta_cont[b, 0][mask]
            mean_valid = valid_slice.sum() / float(n_valid)
            assert math.isclose(mean_valid.item(), 0.0, abs_tol=1e-6)

    def test_quantized_discrete_log_mean_within_grid_resolution(self) -> None:
        """Discrete log_delta mean over valid cells must be <= 1/32 grid resolution."""
        torch.manual_seed(999)
        s_pred = torch.rand(4, 1, 32, 32, dtype=torch.float32)
        log_delta_disc, _ = compute_log_delta(s_pred, apply_ste=True)

        for b in range(4):
            batch_mean = log_delta_disc[b, 0].mean().item()
            assert abs(batch_mean) <= (1.0 / 32.0) + 1e-7


class TestStepBoundsAndExtremeContrast:
    """Verifies that delta in [0.25, 4.0] and log_delta in [-log 4.0, log 4.0] across extreme contrast."""

    def test_extreme_bimodal_contrast(self) -> None:
        """Bimodal saliency map with extreme 0.0 and 1.0 regions contracts gracefully within [0.25, 4.0]."""
        s_pred = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
        s_pred[:, :, :16, :] = 1.0  # Half 1.0, half 0.0

        log_delta, delta = compute_log_delta(s_pred)
        max_log = math.log(4.0)

        assert (log_delta >= -max_log - 1e-6).all()
        assert (log_delta <= max_log + 1e-6).all()
        assert (delta >= 0.25 - 1e-6).all()
        assert (delta <= 4.0 + 1e-6).all()

    def test_checkerboard_contrast(self) -> None:
        """High-frequency alternating checkerboard pattern."""
        s_pred = torch.zeros(1, 1, 16, 16, dtype=torch.float32)
        s_pred[:, :, ::2, ::2] = 1.0
        s_pred[:, :, 1::2, 1::2] = 1.0

        log_delta, delta = compute_log_delta(s_pred)
        max_log = math.log(4.0)

        assert (log_delta >= -max_log - 1e-6).all()
        assert (log_delta <= max_log + 1e-6).all()
        assert (delta >= 0.25 - 1e-6).all()
        assert (delta <= 4.0 + 1e-6).all()

    def test_single_spike_dirac_saliency(self) -> None:
        """Sparse single-spike saliency map (Dirac impulse on zero background)."""
        s_pred = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
        s_pred[0, 0, 16, 16] = 1.0

        log_delta, delta = compute_log_delta(s_pred)
        max_log = math.log(4.0)

        assert (log_delta >= -max_log - 1e-6).all()
        assert (log_delta <= max_log + 1e-6).all()
        assert (delta >= 0.25 - 1e-6).all()
        assert (delta <= 4.0 + 1e-6).all()

    @pytest.mark.parametrize("max_factor", [2.0, 3.0, 4.0, 8.0])
    def test_custom_max_factor_bounds(self, max_factor: float) -> None:
        """Verifies step bounding for user-specified max_factor values."""
        s_pred = torch.zeros(1, 1, 16, 16, dtype=torch.float32)
        s_pred[:, :, :8, :] = 1.0

        log_delta, delta = compute_log_delta(s_pred, max_factor=max_factor)
        max_log = math.log(max_factor)

        assert (log_delta >= -max_log - 1e-6).all()
        assert (log_delta <= max_log + 1e-6).all()
        assert (delta >= (1.0 / max_factor) - 1e-6).all()
        assert (delta <= max_factor + 1e-6).all()


class TestValidMaskAndReflectionPaddingIsolation:
    """Verifies isolation of valid cells M_valid from reflection-padded boundary regions."""

    def test_padded_cells_strictly_identity(self) -> None:
        """Outside M_valid, log_delta == 0.0 and delta == 1.0 strictly."""
        torch.manual_seed(1234)
        s_pred = torch.rand(2, 1, 24, 24, dtype=torch.float32)

        mask = torch.zeros((24, 24), dtype=torch.bool)
        mask[4:20, 4:20] = True
        padded_region = ~mask

        log_delta, delta = compute_log_delta(s_pred, valid_mask=mask)

        for b in range(2):
            padded_log_delta = log_delta[b, 0][padded_region]
            padded_delta = delta[b, 0][padded_region]

            assert torch.all(padded_log_delta == 0.0)
            assert torch.all(padded_delta == 1.0)

    def test_padded_region_values_do_not_contaminate_valid_cells(self) -> None:
        """Varying values in padded cells must have zero impact on valid cell log_delta."""
        torch.manual_seed(5678)
        s_valid = torch.rand(1, 1, 16, 16, dtype=torch.float32)

        # Tensor with zeros in padding
        s_padded_zero = torch.zeros(1, 1, 24, 24, dtype=torch.float32)
        s_padded_zero[:, :, 4:20, 4:20] = s_valid

        # Tensor with ones in padding
        s_padded_one = torch.ones(1, 1, 24, 24, dtype=torch.float32)
        s_padded_one[:, :, 4:20, 4:20] = s_valid

        mask = torch.zeros((24, 24), dtype=torch.bool)
        mask[4:20, 4:20] = True

        log_delta_zero, _ = compute_log_delta(s_padded_zero, valid_mask=mask)
        log_delta_one, _ = compute_log_delta(s_padded_one, valid_mask=mask)

        # Valid cells must have identical allocation despite different padding contents
        assert torch.allclose(
            log_delta_zero[:, :, 4:20, 4:20],
            log_delta_one[:, :, 4:20, 4:20],
            atol=1e-7,
        )

    def test_integration_with_compute_valid_latent_mask(self) -> None:
        """Integration with padding.py compute_valid_latent_mask."""
        # Unpadded image 200x300 padded to multiples of 64 -> 256x320
        # Latents downsampled by 16 -> 16x20
        mask = compute_valid_latent_mask(
            H_orig=200,
            W_orig=300,
            H_pad=256,
            W_pad=320,
            stride=16,
            pad_bounds=(28, 28, 10, 10),
        )
        assert mask.shape == (16, 20)
        assert mask.any()
        assert not mask.all()  # Has padded borders

        s_pred = torch.rand(2, 1, 16, 20, dtype=torch.float32)
        log_delta, delta = compute_log_delta(s_pred, valid_mask=mask)

        # Check padded borders are identity
        assert torch.all(log_delta[:, :, ~mask] == 0.0)
        assert torch.all(delta[:, :, ~mask] == 1.0)


class TestGradientFlowAndSTE:
    """Verifies gradient flow through compute_log_delta and STE."""

    def test_gradient_flow_to_s_pred_with_ste(self) -> None:
        """Backward pass from log_delta produces finite, non-zero gradients into s_pred."""
        torch.manual_seed(42)
        s_pred = torch.rand(2, 1, 16, 16, dtype=torch.float32, requires_grad=True)

        log_delta, _ = compute_log_delta(s_pred, apply_ste=True)
        loss = log_delta.sum()
        loss.backward()

        assert s_pred.grad is not None
        assert torch.isfinite(s_pred.grad).all()
        assert not torch.allclose(s_pred.grad, torch.zeros_like(s_pred.grad))

    def test_gradient_flow_with_valid_mask(self) -> None:
        """Gradients to s_pred must be strictly zero in masked (padded) cells."""
        torch.manual_seed(42)
        s_pred = torch.rand(1, 1, 16, 16, dtype=torch.float32, requires_grad=True)

        mask = torch.zeros((16, 16), dtype=torch.bool)
        mask[4:12, 4:12] = True

        log_delta, _ = compute_log_delta(s_pred, valid_mask=mask, apply_ste=True)
        loss = log_delta.sum()
        loss.backward()

        assert s_pred.grad is not None
        # Valid cells have non-zero gradients
        assert not torch.allclose(s_pred.grad[0, 0, 4:12, 4:12], torch.zeros(8, 8))
        # Padded cells have zero gradients
        assert torch.all(s_pred.grad[0, 0, ~mask] == 0.0)

    def test_gradient_flow_through_delta_exponential(self) -> None:
        """Backward pass through delta = exp(log_delta) produces valid gradients."""
        torch.manual_seed(42)
        s_pred = torch.rand(1, 1, 12, 12, dtype=torch.float32, requires_grad=True)

        _, delta = compute_log_delta(s_pred, apply_ste=True)
        loss = delta.sum()
        loss.backward()

        assert s_pred.grad is not None
        assert torch.isfinite(s_pred.grad).all()
        assert not torch.allclose(s_pred.grad, torch.zeros_like(s_pred.grad))


class TestEffectiveLogScaleComputation:
    """Verifies compute_effective_log_scale."""

    def test_effective_log_scale_formulation(self) -> None:
        """Verifies log_sigma_eff = log_sigma - log_delta."""
        log_sigma = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
        log_delta = torch.tensor([[[[0.5, -0.5], [0.0, 0.25]]]], dtype=torch.float32)

        expected = torch.tensor([[[[0.5, 2.5], [3.0, 3.75]]]], dtype=torch.float32)
        out = compute_effective_log_scale(log_sigma, log_delta)

        assert torch.allclose(out, expected, atol=1e-7)

    def test_effective_scale_broadcasting(self) -> None:
        """Verifies channel broadcasting: (B, C, H, W) and (B, 1, H, W)."""
        log_sigma = torch.randn(2, 64, 16, 16, dtype=torch.float32)
        log_delta = torch.randn(2, 1, 16, 16, dtype=torch.float32)

        out = compute_effective_log_scale(log_sigma, log_delta)
        assert out.shape == (2, 64, 16, 16)
        assert torch.allclose(out, log_sigma - log_delta, atol=1e-7)

    def test_effective_scale_2d_and_3d_delta(self) -> None:
        """Verifies 2D (H, W) or 3D (1, H, W) log_delta aligns with 4D log_sigma."""
        log_sigma = torch.randn(2, 16, 8, 8, dtype=torch.float32)
        log_delta_2d = torch.randn(8, 8, dtype=torch.float32)

        out_2d = compute_effective_log_scale(log_sigma, log_delta_2d)
        assert out_2d.shape == (2, 16, 8, 8)
        assert torch.allclose(out_2d, log_sigma - log_delta_2d.unsqueeze(0).unsqueeze(0), atol=1e-7)

    def test_effective_scale_channel_3d_broadcasting(self) -> None:
        """Verifies 3D channel tensor (C, H, W) broadcasts with 4D (B, C, H, W)."""
        log_sigma = torch.randn(3, 16, 8, 8, dtype=torch.float32)
        log_delta_c = torch.randn(16, 8, 8, dtype=torch.float32)

        out = compute_effective_log_scale(log_sigma, log_delta_c)
        assert out.shape == (3, 16, 8, 8)
        assert torch.allclose(out, log_sigma - log_delta_c.unsqueeze(0), atol=1e-7)


class TestAdaptiveLatentQuantization:
    """Verifies quantize_latents_adaptive."""

    def test_eval_mode_deterministic_rounding(self) -> None:
        """In eval mode (training=False), quantize_latents_adaptive applies round(y / delta) * delta."""
        y = torch.tensor([[[[1.3, -2.4], [0.7, -0.2]]]], dtype=torch.float32)
        delta = torch.tensor([[[[0.5, 2.0], [1.0, 0.25]]]], dtype=torch.float32)

        # Expected: round(1.3/0.5)*0.5 = 3*0.5 = 1.5
        # round(-2.4/2.0)*2.0 = -1*2.0 = -2.0
        # round(0.7/1.0)*1.0 = 1*1.0 = 1.0
        # round(-0.2/0.25)*0.25 = -1*0.25 = -0.25
        expected = torch.tensor([[[[1.5, -2.0], [1.0, -0.25]]]], dtype=torch.float32)

        out = quantize_latents_adaptive(y, delta, training=False)
        assert torch.allclose(out, expected, atol=1e-7)

    def test_training_mode_uniform_noise(self) -> None:
        """In training mode (training=True), relaxation noise U(-0.5, 0.5) is added to y / delta."""
        torch.manual_seed(42)
        y = torch.full((10, 10), 5.0, dtype=torch.float32)
        delta = torch.full((10, 10), 2.0, dtype=torch.float32)

        out = quantize_latents_adaptive(y, delta, training=True)
        # Scaled should be 2.5 + U(-0.5, 0.5) in [2.0, 3.0], then * 2.0 -> in [4.0, 6.0]
        assert (out >= 4.0 - 1e-6).all()
        assert (out <= 6.0 + 1e-6).all()
        # Verify noise is actually non-deterministic
        out2 = quantize_latents_adaptive(y, delta, training=True)
        assert not torch.allclose(out, out2)

    def test_delta_one_equivalence_to_standard_quantization(self) -> None:
        """When delta == 1.0, matches standard round(y)."""
        y = torch.randn(4, 16, 8, 8, dtype=torch.float32)
        delta = torch.ones(4, 1, 8, 8, dtype=torch.float32)

        out = quantize_latents_adaptive(y, delta, training=False)
        expected = torch.round(y)
        assert torch.allclose(out, expected, atol=1e-7)

    def test_quantize_latents_channel_3d_broadcasting(self) -> None:
        """Verifies 3D channel tensor (C, H, W) broadcasts with 4D (B, C, H, W)."""
        y = torch.randn(3, 16, 8, 8, dtype=torch.float32)
        delta_c = torch.full((16, 8, 8), 0.5, dtype=torch.float32)

        out = quantize_latents_adaptive(y, delta_c, training=False)
        assert out.shape == (3, 16, 8, 8)
        expected = torch.round(y / delta_c.unsqueeze(0)) * delta_c.unsqueeze(0)
        assert torch.allclose(out, expected, atol=1e-7)


class TestBatchAndDimensionInvariance:
    """Verifies that compute_log_delta produces identical results regardless of batch or input dimension format."""

    def test_per_sample_batch_invariance(self) -> None:
        """Processing two samples in a batch must yield identical log_delta as processing each independently."""
        torch.manual_seed(777)
        s1 = torch.rand(1, 1, 16, 16, dtype=torch.float32)
        s2 = torch.rand(1, 1, 16, 16, dtype=torch.float32)
        batch = torch.cat([s1, s2], dim=0)

        log_delta_s1, delta_s1 = compute_log_delta(s1)
        log_delta_s2, delta_s2 = compute_log_delta(s2)
        log_delta_b, delta_b = compute_log_delta(batch)

        assert torch.allclose(log_delta_b[0:1], log_delta_s1, atol=1e-7)
        assert torch.allclose(log_delta_b[1:2], log_delta_s2, atol=1e-7)
        assert torch.allclose(delta_b[0:1], delta_s1, atol=1e-7)
        assert torch.allclose(delta_b[1:2], delta_s2, atol=1e-7)

    @pytest.mark.parametrize("shape", [(16, 16), (1, 16, 16), (1, 1, 16, 16), (1, 3, 16, 16)])
    def test_dimension_invariance(self, shape: tuple[int, ...]) -> None:
        """2D, 3D, and 4D tensors with same spatial values yield matching allocation."""
        torch.manual_seed(888)
        base = torch.rand(16, 16, dtype=torch.float32)
        if len(shape) == 2:
            s_pred = base
        elif len(shape) == 3:
            s_pred = base.unsqueeze(0).expand(shape)
        else:
            s_pred = base.unsqueeze(0).unsqueeze(0).expand(shape)

        log_delta, delta = compute_log_delta(s_pred)
        assert log_delta.shape == shape
        assert delta.shape == shape

        log_delta_2d, _ = compute_log_delta(base)
        for i in range(shape[1] if len(shape) == 4 else (shape[0] if len(shape) == 3 else 1)):
            if len(shape) == 4:
                slice_log = log_delta[0, i]
            elif len(shape) == 3:
                slice_log = log_delta[i]
            else:
                slice_log = log_delta
            assert torch.allclose(slice_log, log_delta_2d, atol=1e-7)


class TestErrorHandlingAndSanitization:
    """Verifies robust error handling on invalid inputs."""

    def test_invalid_type_raises_type_error(self) -> None:
        """Non-tensor inputs raise TypeError."""
        with pytest.raises(TypeError):
            compute_log_delta([[0.5, 0.5], [0.5, 0.5]])  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            quantize_ste_1_32([1.0, 2.0])  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            lookup_delta_from_table([1.0])  # type: ignore[arg-type]

    def test_invalid_ndim_raises_value_error(self) -> None:
        """1D or 5D tensors raise ValueError."""
        with pytest.raises(ValueError, match="2D, 3D, or 4D tensor"):
            compute_log_delta(torch.rand(16))
        with pytest.raises(ValueError, match="2D, 3D, or 4D tensor"):
            compute_log_delta(torch.rand(1, 1, 1, 16, 16))

    def test_nan_and_inf_detection(self) -> None:
        """NaN and Inf inputs raise ValueError."""
        s_nan = torch.rand(8, 8)
        s_nan[2, 2] = float("nan")
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_log_delta(s_nan)

        s_inf = torch.rand(8, 8)
        s_inf[2, 2] = float("inf")
        with pytest.raises(ValueError, match="NaN or Inf"):
            compute_log_delta(s_inf)

    def test_out_of_range_saliency_raises_value_error(self) -> None:
        """Saliency values significantly outside [0, 1] raise ValueError."""
        with pytest.raises(ValueError, match="s_pred values must be in \\[0, 1\\]"):
            compute_log_delta(torch.tensor([[0.5, 1.5], [0.2, 0.3]]))
        with pytest.raises(ValueError, match="s_pred values must be in \\[0, 1\\]"):
            compute_log_delta(torch.tensor([[-0.5, 0.5], [0.2, 0.3]]))

    def test_valid_mask_mismatch_raises_value_error(self) -> None:
        """Mismatch between spatial dimensions of valid_mask and s_pred raises ValueError."""
        s_pred = torch.rand(16, 16)
        mask = torch.ones(12, 12, dtype=torch.bool)
        with pytest.raises(ValueError, match="Spatial dimensions"):
            compute_log_delta(s_pred, valid_mask=mask)

    def test_all_zero_valid_mask_raises_value_error(self) -> None:
        """A valid_mask with no valid cells raises ValueError."""
        s_pred = torch.rand(16, 16)
        mask = torch.zeros(16, 16, dtype=torch.bool)
        with pytest.raises(ValueError, match="no valid cells"):
            compute_log_delta(s_pred, valid_mask=mask)

    def test_effective_log_scale_mismatch_raises_value_error(self) -> None:
        """Spatial dimension mismatch in compute_effective_log_scale raises ValueError."""
        log_sigma = torch.randn(2, 16, 16, 16)
        log_delta = torch.randn(2, 1, 12, 12)
        with pytest.raises(ValueError, match="Spatial dimensions must match"):
            compute_effective_log_scale(log_sigma, log_delta)

    def test_device_mismatch_raises_value_error(self) -> None:
        """Tensors on different devices raise ValueError."""
        log_sigma_meta = torch.randn(2, 4, 8, 8, device="meta")
        log_delta_cpu = torch.randn(2, 4, 8, 8, device="cpu")
        with pytest.raises(ValueError, match="Device mismatch:"):
            compute_effective_log_scale(log_sigma_meta, log_delta_cpu)

        y_meta = torch.randn(2, 4, 8, 8, device="meta")
        delta_cpu = torch.randn(2, 4, 8, 8, device="cpu")
        with pytest.raises(ValueError, match="Device mismatch:"):
            quantize_latents_adaptive(y_meta, delta_cpu)
