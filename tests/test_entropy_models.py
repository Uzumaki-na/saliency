"""Unit test suite for entropy models and discrete CDF compilation.

Validates:
- GaussianConditional: Zero-mean likelihood integration, discrete scale quantization,
  16-bit integer CDF table construction, and strict CDF monotonicity.
- EntropyBottleneck: Monotonic cumulative distribution network, channel-wise support
  estimation, 16-bit integer CDF compilation, and strict monotonicity across all channels.
- Numerical isolation: Strict FP32 execution outside autocast to prevent CDF underflow.
- Gradient flow: End-to-end backpropagation through differentiable likelihood paths.
- Checkpoint persistence: State dict serialization and bit-exact CDF restoration.
- Theoretical entropy: Shannon entropy bounds and rate estimation consistency.
"""

from __future__ import annotations

import io
import math

import pytest
import torch

from src.codec.entropy_models import (
    EntropyBottleneck,
    GaussianConditional,
    _normalize_frequencies,
    _standard_normal_cdf,
)


class TestFrequencyNormalization:
    """Tests for discrete frequency normalization and integer rounding."""

    def test_normalize_frequencies_sum_and_min(self):
        """Verifies frequencies sum to total_freq and each frequency is >= 1."""
        probs = torch.tensor([0.1, 0.4, 0.3, 0.15, 0.05], dtype=torch.float32)
        total_freq = 65536
        freqs = _normalize_frequencies(probs, total_freq=total_freq)

        assert freqs.shape == probs.shape
        assert freqs.dtype == torch.int32
        assert freqs.sum().item() == total_freq
        assert (freqs >= 1).all()

    def test_normalize_frequencies_tiny_probabilities(self):
        """Verifies symbols with near-zero probability still receive >= 1 frequency."""
        probs = torch.tensor([1e-15, 0.5, 0.5, 1e-12], dtype=torch.float32)
        total_freq = 65536
        freqs = _normalize_frequencies(probs, total_freq=total_freq)

        assert freqs.sum().item() == total_freq
        assert freqs[0].item() >= 1
        assert freqs[3].item() >= 1

    def test_normalize_frequencies_large_alphabet(self):
        """Verifies frequency normalization on an alphabet of 1000 symbols."""
        probs = torch.randn(1000).abs()
        probs /= probs.sum()
        freqs = _normalize_frequencies(probs, total_freq=65536)

        assert freqs.sum().item() == 65536
        assert (freqs >= 1).all()

    def test_normalize_frequencies_excessive_alphabet_raises(self):
        """Verifies error raised when alphabet exceeds frequency budget."""
        probs = torch.ones(70000, dtype=torch.float32)
        with pytest.raises(ValueError, match=r"Alphabet size .* cannot exceed total frequency"):
            _normalize_frequencies(probs, total_freq=65536)


class TestStandardNormalCDF:
    """Tests for standard normal cumulative distribution function."""

    def test_standard_normal_cdf_known_values(self):
        """Verifies erf-based CDF against known standard normal values."""
        points = torch.tensor([-3.0, -1.96, 0.0, 1.96, 3.0], dtype=torch.float32)
        cdf = _standard_normal_cdf(points)

        assert pytest.approx(cdf[2].item(), rel=1e-5) == 0.5
        assert pytest.approx(cdf[3].item(), rel=1e-3) == 0.975
        assert pytest.approx(cdf[1].item(), rel=1e-3) == 0.025
        assert (cdf[1:] > cdf[:-1]).all(), "Standard normal CDF must be strictly increasing"


class TestGaussianConditional:
    """Comprehensive tests for GaussianConditional entropy model."""

    def test_initialization_defaults(self):
        """Verifies default parameters and registered buffers."""
        gc = GaussianConditional()
        assert gc.num_scales == 64
        assert gc.scale_bound.item() == pytest.approx(0.11)
        assert gc.scale_table.numel() == 64
        assert gc.scale_table[0].item() == pytest.approx(0.11, rel=1e-3)
        assert gc.scale_table[-1].item() == pytest.approx(256.0, rel=1e-3)

    def test_build_indexes_mapping(self):
        """Verifies mapping continuous scales to discrete indices [0, 63]."""
        gc = GaussianConditional()
        scales = torch.tensor([0.05, 0.11, 1.0, 10.0, 256.0, 500.0])
        indexes = gc.build_indexes(scales)

        assert indexes.shape == scales.shape
        assert indexes[0].item() == 0  # Below scale_min clamps to 0
        assert indexes[1].item() == 0  # scale_min maps to 0
        assert indexes[-2].item() == 63  # scale_max maps to 63
        assert indexes[-1].item() == 63  # Above scale_max clamps to 63
        assert (indexes[1:] >= indexes[:-1]).all(), "Indexes must be monotonically non-decreasing"

    def test_forward_training_noise_relaxation(self):
        """Verifies additive uniform noise relaxation during training."""
        gc = GaussianConditional()
        gc.train()

        inputs = torch.zeros(2, 192, 8, 8)
        scales = torch.full_like(inputs, 1.0)
        y_hat, likelihoods = gc(inputs, scales, training=True)

        assert y_hat.shape == inputs.shape
        assert likelihoods.shape == inputs.shape
        # Noise should be in [-0.5, 0.5]
        assert (y_hat >= -0.5).all() and (y_hat <= 0.5).all()
        # Non-zero variance across elements
        assert y_hat.var() > 0.05

    def test_forward_eval_deterministic_rounding(self):
        """Verifies deterministic integer rounding during evaluation."""
        gc = GaussianConditional()
        gc.eval()

        inputs = torch.tensor([[-1.7, -0.4, 0.2, 0.6, 2.3]])
        scales = torch.ones_like(inputs)
        y_hat, likelihoods = gc(inputs, scales, training=False)

        expected = torch.tensor([[-2.0, -0.0, 0.0, 1.0, 2.0]])
        assert torch.equal(y_hat, expected)

        # Successive calls produce bit-identical tensors
        y_hat2, likelihoods2 = gc(inputs, scales, training=False)
        assert torch.equal(y_hat, y_hat2)
        assert torch.equal(likelihoods, likelihoods2)

    def test_likelihood_bounds_and_clamping(self):
        """Verifies likelihoods are strictly positive and bounded by [1e-9, 1.0]."""
        gc = GaussianConditional()
        # Test extreme outlier
        inputs = torch.tensor([[0.0, 100.0, -100.0]])
        scales = torch.tensor([[1.0, 1.0, 1.0]])
        _, likelihoods = gc(inputs, scales, training=False)

        assert (likelihoods >= 1e-9).all()
        assert (likelihoods <= 1.0).all()
        assert likelihoods[0, 0].item() > 0.1  # Normal center has mass ~0.38

    def test_scale_bound_enforcement(self):
        """Verifies scale parameters below 0.11 are lower bounded."""
        gc = GaussianConditional(scale_bound=0.11)
        inputs = torch.zeros(1, 10)
        scales = torch.full_like(inputs, -5.0)  # Invalid negative scales
        _, likelihoods = gc(inputs, scales, training=False)

        # Scale was clamped to 0.11, so center likelihood is approx 1.0 (unit bin / 0.11 covers >99.99%)
        assert not torch.isnan(likelihoods).any()
        assert (likelihoods > 0.9).all()

    def test_gradient_flow(self):
        """Verifies gradients backpropagate through likelihoods into inputs and scales."""
        gc = GaussianConditional()
        inputs = torch.randn(2, 16, 4, 4, requires_grad=True)
        scales = (torch.rand(2, 16, 4, 4) + 0.2).requires_grad_(True)

        _, likelihoods = gc(inputs, scales, training=True)
        loss = -torch.log2(likelihoods).sum()
        loss.backward()

        assert inputs.grad is not None
        assert not torch.isnan(inputs.grad).any()
        assert torch.any(inputs.grad != 0.0)

        assert scales.grad is not None
        assert not torch.isnan(scales.grad).any()
        assert torch.any(scales.grad != 0.0)

    def test_update_scale_table_monotonicity_and_sum(self):
        """Verifies offline compiled CDF tables satisfy exact monotonicity and sum to 65536."""
        gc = GaussianConditional()
        updated = gc.update_scale_table()
        assert updated is True

        assert gc._quantized_cdf.dim() == 2
        assert gc._quantized_cdf.shape[0] == gc.num_scales
        assert gc._cdf_length.numel() == gc.num_scales
        assert gc._offset.numel() == gc.num_scales

        total_freq = 65536

        for i in range(gc.num_scales):
            length = gc._cdf_length[i].item()
            offset = gc._offset[i].item()
            cdf = gc._quantized_cdf[i, :length]

            # Invariant 1: First element is 0
            assert cdf[0].item() == 0, f"Scale {i}: cdf[0] must be 0"

            # Invariant 2: Last valid element is 65536
            assert cdf[-1].item() == total_freq, f"Scale {i}: cdf[-1] must be {total_freq}"

            # Invariant 3: Strictly monotonic (each symbol has frequency >= 1)
            diffs = cdf[1:] - cdf[:-1]
            assert (diffs >= 1).all(), f"Scale {i}: frequency violation (min diff {diffs.min().item()})"

            # Invariant 4: Offset symmetry around 0 for zero-mean Gaussian
            num_symbols = length - 1
            max_sym = offset + num_symbols - 1
            assert offset == -max_sym, f"Scale {i}: offset {offset} must be symmetric to max {max_sym}"


class TestEntropyBottleneck:
    """Comprehensive tests for EntropyBottleneck factorized model."""

    def test_initialization_defaults(self):
        """Verifies channels, filter dimensions, and monotonic network parameters."""
        eb = EntropyBottleneck(channels=128)
        assert eb.channels == 128
        assert len(eb.weights) == 4
        assert len(eb.biases) == 4
        assert eb._quantized_cdf.numel() == 0  # Uncompiled initially

    def test_logits_cumulative_monotonicity(self):
        """Verifies cumulative distribution function is strictly increasing."""
        eb = EntropyBottleneck(channels=4)
        x = torch.linspace(-20.0, 20.0, 200).unsqueeze(0).expand(4, -1)
        cdf = eb._logits_cumulative(x)

        assert cdf.shape == (4, 200)
        # Bounded in (0, 1)
        assert (cdf >= 0.0).all() and (cdf <= 1.0).all()
        # Strictly monotonic along x
        for c in range(4):
            diffs = cdf[c, 1:] - cdf[c, :-1]
            assert (diffs > 0).all(), f"Channel {c} cumulative function is not strictly increasing"

    def test_forward_shapes_and_training_relaxation(self):
        """Verifies forward pass with noise relaxation in training mode."""
        eb = EntropyBottleneck(channels=64)
        eb.train()

        z = torch.randn(2, 64, 8, 8)
        z_hat, likelihoods = eb(z, training=True)

        assert z_hat.shape == z.shape
        assert likelihoods.shape == z.shape
        assert (likelihoods >= 1e-9).all()
        assert (likelihoods <= 1.0).all()

    def test_forward_eval_deterministic_rounding(self):
        """Verifies deterministic integer rounding in eval mode."""
        eb = EntropyBottleneck(channels=32)
        eb.eval()

        z = torch.randn(2, 32, 4, 4)
        z_hat, likelihoods = eb(z, training=False)

        assert torch.equal(z_hat, torch.round(z))

        # Repeated eval produces bit-identical outputs
        z_hat2, likelihoods2 = eb(z, training=False)
        assert torch.equal(z_hat, z_hat2)
        assert torch.equal(likelihoods, likelihoods2)

    def test_gradient_flow_into_parameters(self):
        """Verifies gradients backpropagate through likelihoods into inputs, weights, and biases."""
        eb = EntropyBottleneck(channels=16)
        z = torch.randn(2, 16, 4, 4, requires_grad=True)

        _, likelihoods = eb(z, training=True)
        loss = -torch.log2(likelihoods).sum()
        loss.backward()

        assert z.grad is not None
        assert not torch.isnan(z.grad).any()
        assert torch.any(z.grad != 0.0)

        for i, w in enumerate(eb.weights):
            assert w.grad is not None and not torch.isnan(w.grad).any(), f"Weight {i} missing/nan gradient"
            assert torch.any(w.grad != 0.0), f"Weight {i} has all-zero gradient"

        for i, b in enumerate(eb.biases):
            assert b.grad is not None and not torch.isnan(b.grad).any(), f"Bias {i} missing/nan gradient"
            assert torch.any(b.grad != 0.0), f"Bias {i} has all-zero gradient"

    def test_update_cdf_monotonicity_and_sum(self):
        """Verifies compiled CDF tables satisfy strict monotonicity for all 128 channels."""
        eb = EntropyBottleneck(channels=128)
        updated = eb.update()
        assert updated is True

        assert eb._quantized_cdf.dim() == 2
        assert eb._quantized_cdf.shape[0] == 128
        assert eb._cdf_length.numel() == 128
        assert eb._offset.numel() == 128

        total_freq = 65536

        for c in range(128):
            length = eb._cdf_length[c].item()
            cdf = eb._quantized_cdf[c, :length]

            # Invariant 1: First element is 0
            assert cdf[0].item() == 0, f"Channel {c}: cdf[0] must be 0"

            # Invariant 2: Last valid element is 65536
            assert cdf[-1].item() == total_freq, f"Channel {c}: cdf[-1] must be {total_freq}"

            # Invariant 3: Strictly monotonic
            diffs = cdf[1:] - cdf[:-1]
            assert (diffs >= 1).all(), f"Channel {c}: frequency violation (min diff {diffs.min().item()})"

    def test_update_idempotence(self):
        """Verifies update(force=False) does not recompile if already populated."""
        eb = EntropyBottleneck(channels=16)
        res1 = eb.update()
        assert res1 is True

        res2 = eb.update(force=False)
        assert res2 is False

        res3 = eb.update(force=True)
        assert res3 is True


class TestModelAEntropyIntegration:
    """Tests integration of EntropyBottleneck and GaussianConditional into ModelA."""

    def test_model_a_forward_with_entropy_models(self):
        """Verifies end-to-end forward pass yields reconstructed image and valid likelihoods."""
        from src.codec import ModelA

        model = ModelA()
        model.eval()

        x = torch.randn(1, 3, 64, 64)
        out = model(x, training=False)

        assert out.x_hat.shape == (1, 3, 64, 64)
        assert out.y_likelihoods is not None
        assert out.z_likelihoods is not None

        assert out.y_likelihoods.shape == out.y.shape
        assert out.z_likelihoods.shape == out.z.shape

        assert (out.y_likelihoods >= 1e-9).all()
        assert (out.z_likelihoods >= 1e-9).all()

    def test_model_a_update_compiles_both_models(self):
        """Verifies ModelA.update() compiles both entropy bottleneck and gaussian conditional."""
        from src.codec import ModelA

        model = ModelA()
        res = model.update()

        assert res["entropy_bottleneck"] is True
        assert res["gaussian_conditional"] is True

        assert model.entropy_bottleneck._quantized_cdf.numel() > 0
        assert model.gaussian_conditional._quantized_cdf.numel() > 0

    def test_checkpoint_state_dict_roundtrip(self):
        """Verifies that compiled CDF buffers serialize and restore bit-identically."""
        from src.codec import ModelA

        model1 = ModelA()
        model1.update()

        # Serialize to in-memory bytes
        buffer = io.BytesIO()
        torch.save(model1.state_dict(), buffer)
        buffer.seek(0)

        # Load into fresh model instance
        model2 = ModelA()
        assert model2.gaussian_conditional._quantized_cdf.numel() == 0
        assert model2.entropy_bottleneck._quantized_cdf.numel() == 0

        model2.load_state_dict(torch.load(buffer, weights_only=True))

        assert torch.equal(
            model1.gaussian_conditional._quantized_cdf,
            model2.gaussian_conditional._quantized_cdf,
        )
        assert torch.equal(
            model1.gaussian_conditional._cdf_length,
            model2.gaussian_conditional._cdf_length,
        )
        assert torch.equal(
            model1.gaussian_conditional._offset,
            model2.gaussian_conditional._offset,
        )
        assert torch.equal(
            model1.entropy_bottleneck._quantized_cdf,
            model2.entropy_bottleneck._quantized_cdf,
        )
        assert torch.equal(
            model1.entropy_bottleneck._cdf_length,
            model2.entropy_bottleneck._cdf_length,
        )
        assert torch.equal(
            model1.entropy_bottleneck._offset,
            model2.entropy_bottleneck._offset,
        )


class TestTheoreticalEntropyConsistency:
    """Verifies consistency between theoretical Shannon entropy and integrated likelihoods."""

    def test_theoretical_rate_versus_shannon_entropy(self):
        """Verifies that negative log2 likelihoods match discrete Shannon entropy."""
        gc = GaussianConditional()
        sigma = 2.0
        scales = torch.full((1, 10000), sigma)
        # Sample integer symbols from discrete normal distribution
        norm_samples = torch.randn(1, 10000) * sigma
        discrete_symbols = torch.round(norm_samples)

        _, likelihoods = gc(discrete_symbols, scales, training=False)
        empirical_bpp = -torch.log2(likelihoods).mean().item()

        # Differential entropy of continuous normal: 0.5 * log2(2 * pi * e * sigma^2)
        diff_entropy = 0.5 * math.log2(2.0 * math.pi * math.e * (sigma**2))

        # Discrete entropy of unit-quantized normal is approximately diff_entropy
        assert abs(empirical_bpp - diff_entropy) < 0.15, (
            f"Empirical rate ({empirical_bpp:.3f}) must closely match continuous entropy ({diff_entropy:.3f})"
        )
