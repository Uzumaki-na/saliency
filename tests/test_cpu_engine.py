"""Unit tests for deterministic single-threaded CPU entropy parameter execution engine.

Verifies:
- Bit-identical discrete scale index and parameter derivation across repeated CPU runs.
- Cross-device input invariance (CUDA input vs CPU input producing exact parity).
- Environment determinism configuration (single-threading, AVX2, MKL-DNN disabled).
- Spatial dimension handling (4D batches and unbatched 3D tensors).
- Adaptive quantization step delta shifting.
- Bounded scale index guarantee under extreme latent inputs.
"""

from __future__ import annotations

import os

import pytest
import torch

from src.codec import ModelA
from src.codec.model_a import HyperSynthesisTransform
from src.entropy import (
    DeterministicCPUEngine,
    configure_cpu_determinism,
)


class TestCPUDeterminismEnvironment:
    """Tests for CPU determinism configuration helper."""

    def test_configure_cpu_determinism_settings(self):
        """Verifies environment variables, thread counts, and backend flags."""
        configure_cpu_determinism()

        assert os.environ.get("MKL_ENABLE_INSTRUCTIONS") == "AVX2"
        assert torch.get_num_threads() == 1
        if hasattr(torch.backends, "mkldnn"):
            assert torch.backends.mkldnn.enabled is False
        if hasattr(torch.backends.cuda, "matmul"):
            assert torch.backends.cuda.matmul.allow_tf32 is False


class TestDeterministicCPUEngine:
    """Comprehensive tests for DeterministicCPUEngine."""

    def test_initialization_defaults(self):
        """Verifies default engine configuration and internal CPU module state."""
        engine = DeterministicCPUEngine()
        assert engine.num_scales == 64
        assert engine.scale_min == pytest.approx(0.11)
        assert engine.scale_max == pytest.approx(256.0)
        assert engine.scale_table.numel() == 64
        assert engine.scale_table.device.type == "cpu"

        # Module on CPU in eval mode without grad
        assert next(engine.h_s.parameters()).device.type == "cpu"
        assert next(engine.h_s.parameters()).dtype == torch.float32
        assert not next(engine.h_s.parameters()).requires_grad
        assert not engine.h_s.training

    def test_from_model_construction(self):
        """Verifies constructing engine directly from a ModelA instance."""
        model = ModelA()
        engine = DeterministicCPUEngine.from_model(model)

        assert engine.num_scales == 64
        assert isinstance(engine.h_s, HyperSynthesisTransform)

    def test_forward_shapes_4d_batch(self):
        """Verifies 4D batch input yields 4x spatially upsampled log_sigma and scale_indexes."""
        engine = DeterministicCPUEngine()
        z_hat = torch.randint(-5, 6, (2, 128, 4, 4), dtype=torch.float32)
        log_sigma, scale_idx = engine(z_hat)

        assert log_sigma.shape == (2, 192, 16, 16)
        assert scale_idx.shape == (2, 192, 16, 16)
        assert log_sigma.device.type == "cpu"
        assert scale_idx.device.type == "cpu"
        assert scale_idx.dtype == torch.int32
        assert (scale_idx >= 0).all() and (scale_idx <= 63).all()

    def test_forward_shapes_3d_unbatched(self):
        """Verifies unbatched 3D tensor input yields unbatched 3D outputs."""
        engine = DeterministicCPUEngine()
        z_hat = torch.randint(-5, 6, (128, 6, 8), dtype=torch.float32)
        log_sigma, scale_idx = engine(z_hat)

        assert log_sigma.shape == (192, 24, 32)
        assert scale_idx.shape == (192, 24, 32)
        assert (scale_idx >= 0).all() and (scale_idx <= 63).all()

    def test_invalid_dimensions_raise_value_error(self):
        """Verifies error raised when input has invalid dimensions."""
        engine = DeterministicCPUEngine()
        invalid_2d = torch.zeros(128, 16)
        with pytest.raises(ValueError, match="Expected 3D or 4D tensor"):
            engine(invalid_2d)

    def test_bit_exact_determinism_across_100_runs(self):
        """Verifies 100 consecutive executions yield bit-identical scale indices and log-scales."""
        engine = DeterministicCPUEngine()
        z_hat = torch.randint(-10, 11, (1, 128, 8, 8), dtype=torch.float32)

        base_log_sigma, base_scale_idx = engine(z_hat)

        for _ in range(100):
            log_sigma, scale_idx = engine(z_hat)
            # Exact bit equality
            assert torch.equal(scale_idx, base_scale_idx), "Scale indices diverged across CPU runs"
            assert torch.equal(log_sigma, base_log_sigma), "Log-scales diverged across CPU runs"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for cross-device parity test")
    def test_cross_device_input_parity(self):
        """Verifies identical scale indices whether z_hat originates on CPU or CUDA."""
        engine = DeterministicCPUEngine()
        z_cpu = torch.randint(-15, 16, (2, 128, 6, 6), dtype=torch.float32)
        z_cuda = z_cpu.cuda()

        log_sigma_cpu, idx_cpu = engine(z_cpu)
        log_sigma_cuda, idx_cuda = engine(z_cuda)

        assert torch.equal(idx_cpu, idx_cuda), "Scale indices differed between CPU and CUDA input sources"
        assert torch.equal(log_sigma_cpu, log_sigma_cuda), "Log-scales differed between CPU and CUDA input sources"

    def test_quantization_step_delta_shift(self):
        """Verifies log_delta modifies effective scale and shifts discrete scale indices."""
        engine = DeterministicCPUEngine()
        z_hat = torch.zeros(1, 128, 4, 4)

        _, idx_baseline = engine(z_hat)

        # Positive log_delta (finer step) increases effective rate / increases precision
        log_delta = torch.full((1, 192, 16, 16), 0.5)
        _, idx_shifted = engine(z_hat, log_delta=log_delta)

        # Baseline log_sigma - 0.5 results in smaller effective sigma, shifting indices down
        assert (idx_shifted <= idx_baseline).all()
        assert (idx_shifted < idx_baseline).any(), "Index must shift to a smaller scale table index"

    def test_extreme_latent_values_remain_bounded(self):
        """Verifies extreme positive and negative hyper-latents do not overflow index bounds."""
        engine = DeterministicCPUEngine()
        z_extreme = torch.tensor([-500.0, 500.0]).repeat_interleave(64).reshape(1, 128, 1, 1)

        _, scale_idx = engine(z_extreme)

        assert (scale_idx >= 0).all()
        assert (scale_idx <= 63).all()

    def test_sync_parameters_updates_weights(self):
        """Verifies engine.sync_parameters copies updated weights correctly."""
        model = ModelA()
        engine = DeterministicCPUEngine.from_model(model)

        # Mutate model h_s conv weights
        with torch.no_grad():
            model.h_s.conv3.weight.add_(1.0)

        # Sync into engine
        engine.sync_parameters(model.h_s)

        # Assert internal weights match mutated model weights
        assert torch.equal(engine.h_s.conv3.weight, model.h_s.conv3.weight.cpu())

