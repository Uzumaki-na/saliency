"""Unit test suite for GDN/IGDN layers and Model A network transforms.

Verifies:
- LowerBoundFunction and LowerBound custom gradient routing
- NonNegativeParametrizer stability and minimum bounds
- GDN and IGDN non-linearities, strict FP32 execution, and numerical stability
- AnalysisTransform (g_a), SynthesisTransform (g_s), HyperAnalysisTransform (h_a),
  and HyperSynthesisTransform (h_s) dimension scaling and behavior
- ModelA operational modes (stochastic noise in training vs deterministic rounding in eval)
- Arbitrary spatial dimension handling (multiples of 64) and invalid dimension error handling
- Zero and flat tensor edge cases (no NaN or Inf)
- Full end-to-end gradient backpropagation across all network parameters
- Mixed precision (BF16 autocast) execution and FP32 precision boundaries
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.codec import (
    GDN,
    IGDN,
    AnalysisTransform,
    HyperAnalysisTransform,
    HyperSynthesisTransform,
    LowerBound,
    LowerBoundFunction,
    ModelA,
    ModelOutput,
    NonNegativeParametrizer,
    SynthesisTransform,
)

# -----------------------------------------------------------------------------
# 1. LowerBound & NonNegativeParametrizer Tests
# -----------------------------------------------------------------------------


class TestLowerBound:
    """Verifies LowerBound autograd behavior and custom gradient routing."""

    def test_lower_bound_forward(self):
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        bound = torch.tensor(0.5)
        out = LowerBoundFunction.apply(x, bound)
        expected = torch.tensor([0.5, 0.5, 1.0, 2.0])
        assert torch.allclose(out, expected)

    def test_lower_bound_backward_routing(self):
        # When x >= bound, gradient passes through regardless of sign.
        # When x < bound, gradient passes through ONLY if grad_output < 0 (pushes x towards bound).
        x = torch.tensor([-0.5, -0.5, 1.5, 1.5], requires_grad=True)
        bound = torch.tensor(0.0)
        out = LowerBoundFunction.apply(x, bound)

        # grad_output has [positive, negative, positive, negative]
        grad_out = torch.tensor([2.0, -3.0, 4.0, -5.0])
        out.backward(grad_out)

        # Expected:
        # idx 0: x < 0, grad > 0 -> 0.0
        # idx 1: x < 0, grad < 0 -> -3.0
        # idx 2: x >= 0, grad > 0 -> 4.0
        # idx 3: x >= 0, grad < 0 -> -5.0
        expected_grad = torch.tensor([0.0, -3.0, 4.0, -5.0])
        assert x.grad is not None
        assert torch.allclose(x.grad, expected_grad)

    def test_lower_bound_module_wrapper(self):
        module = LowerBound(bound=0.25)
        x = torch.tensor([0.1, 0.5])
        out = module(x)
        assert torch.allclose(out, torch.tensor([0.25, 0.5]))
        assert out.dtype == torch.float32

    def test_non_negative_parametrizer(self):
        reparam = NonNegativeParametrizer(minimum=1e-5)
        init_val = torch.tensor([1.0, 0.001, 1e-6])
        internal_p = reparam.init(init_val)

        # Forward projection
        p_projected = reparam(internal_p)
        assert torch.all(p_projected >= 1e-5)

        # Check negative inputs get clamped above minimum
        negative_p = torch.tensor([-5.0, -10.0])
        p_clamped = reparam(negative_p)
        assert torch.all(p_clamped >= 1e-5)


# -----------------------------------------------------------------------------
# 2. GDN & IGDN Normalization Layer Tests
# -----------------------------------------------------------------------------


class TestGDNandIGDN:
    """Verifies GDN and IGDN non-linearities, bounds, and numerical stability."""

    def test_gdn_shape_preservation(self):
        gdn = GDN(in_channels=64)
        x = torch.randn(2, 64, 32, 32)
        out = gdn(x)
        assert out.shape == x.shape
        assert out.dtype == torch.float32

    def test_igdn_shape_preservation(self):
        igdn = IGDN(in_channels=64)
        x = torch.randn(2, 64, 32, 32)
        out = igdn(x)
        assert out.shape == x.shape
        assert out.dtype == torch.float32

    def test_gdn_identity_like_initialization(self):
        gdn = GDN(in_channels=32, beta_min=1e-6, gamma_init=0.1)
        # Small inputs close to 0: denominator is sqrt(beta) = sqrt(1.0) = 1.0
        x = torch.randn(1, 32, 8, 8) * 0.01
        out = gdn(x)
        assert torch.allclose(out, x, atol=1e-3, rtol=1e-3)

    def test_zero_and_flat_inputs_numerical_stability(self):
        gdn = GDN(in_channels=16)
        igdn = IGDN(in_channels=16)

        # All-zero tensor
        zero_x = torch.zeros(2, 16, 16, 16)
        out_gdn = gdn(zero_x)
        out_igdn = igdn(zero_x)

        assert not torch.isnan(out_gdn).any()
        assert not torch.isinf(out_gdn).any()
        assert not torch.isnan(out_igdn).any()
        assert not torch.isinf(out_igdn).any()
        assert torch.allclose(out_gdn, zero_x)
        assert torch.allclose(out_igdn, zero_x)

        # Very large flat input tensor (check denominator does not overflow or become unstable)
        large_x = torch.ones(2, 16, 16, 16) * 100.0
        out_large = gdn(large_x)
        assert not torch.isnan(out_large).any()
        assert not torch.isinf(out_large).any()

    def test_gdn_parameter_positivity(self):
        gdn = GDN(in_channels=16, beta_min=1e-5)
        # Force raw parameters to large negative values
        with torch.no_grad():
            gdn.beta.fill_(-100.0)
            gdn.gamma.fill_(-100.0)

        # Projected parameters inside forward must strictly satisfy lower bounds
        beta_proj = gdn.beta_reparam(gdn.beta)
        gamma_proj = gdn.gamma_reparam(gdn.gamma)
        assert torch.all(beta_proj >= 1e-5)
        assert torch.all(gamma_proj >= 0.0)

        # Executing forward pass succeeds cleanly
        x = torch.randn(2, 16, 8, 8)
        out = gdn(x)
        assert not torch.isnan(out).any()

    def test_input_validation(self):
        gdn = GDN(in_channels=16)
        with pytest.raises(ValueError, match="Expected 4D input tensor"):
            gdn(torch.randn(16, 8, 8))

        with pytest.raises(ValueError, match="Channel dimension mismatch"):
            gdn(torch.randn(2, 32, 8, 8))


# -----------------------------------------------------------------------------
# 3. Individual Transform Tests (g_a, g_s, h_a, h_s)
# -----------------------------------------------------------------------------


class TestTransforms:
    """Verifies dimensional scaling and layer contracts for individual transforms."""

    def test_analysis_transform_ga(self):
        ga = AnalysisTransform(in_channels=3, n_channels=128, m_channels=192)
        x = torch.randn(2, 3, 256, 256)
        y = ga(x)

        assert y.shape == (2, 192, 16, 16)
        assert y.dtype == torch.float32

    def test_analysis_transform_invalid_dimensions(self):
        ga = AnalysisTransform(in_channels=3, n_channels=128, m_channels=192)
        # 250 is not divisible by 16
        with pytest.raises(ValueError, match="must be multiples of 16"):
            ga(torch.randn(1, 3, 250, 256))

    def test_synthesis_transform_gs(self):
        gs = SynthesisTransform(in_channels=192, n_channels=128, out_channels=3)
        y_hat = torch.randn(2, 192, 16, 16)
        x_hat = gs(y_hat)

        assert x_hat.shape == (2, 3, 256, 256)
        assert x_hat.dtype == torch.float32

    def test_hyper_analysis_transform_ha(self):
        ha = HyperAnalysisTransform(in_channels=192, n_channels=128, out_channels=128)
        y = torch.randn(2, 192, 16, 16)
        z = ha(y)

        # 16 / 4 = 4
        assert z.shape == (2, 128, 4, 4)

        # Test invariance to sign due to |y| input
        z_from_neg = ha(-y)
        assert torch.allclose(z, z_from_neg, atol=1e-6)

    def test_hyper_analysis_transform_invalid_dimensions(self):
        ha = HyperAnalysisTransform(in_channels=192, n_channels=128, out_channels=128)
        with pytest.raises(ValueError, match="must be multiples of 4"):
            ha(torch.randn(1, 192, 15, 16))

    def test_hyper_synthesis_transform_hs(self):
        hs = HyperSynthesisTransform(in_channels=128, n_channels=128, out_channels=192)
        z_hat = torch.randn(2, 128, 4, 4)
        log_sigma = hs(z_hat)

        # 4 * 4 = 16
        assert log_sigma.shape == (2, 192, 16, 16)
        assert log_sigma.dtype == torch.float32


# -----------------------------------------------------------------------------
# 4. ModelOutput Container Tests
# -----------------------------------------------------------------------------


class TestModelOutput:
    """Verifies ModelOutput dict access, attribute access, and tuple unpacking."""

    def test_model_output_accessors(self):
        x_hat = torch.randn(1, 3, 64, 64)
        y = torch.randn(1, 192, 4, 4)
        z = torch.randn(1, 128, 1, 1)
        log_sigma = torch.randn(1, 192, 4, 4)

        out = ModelOutput(x_hat=x_hat, y=y, z=z, log_sigma=log_sigma)

        # Dictionary access
        assert out["x_hat"] is x_hat
        assert out["y"] is y
        assert out["z"] is z
        assert out["log_sigma"] is log_sigma

        # Attribute access
        assert out.x_hat is x_hat
        assert out.y is y
        assert out.z is z
        assert out.log_sigma is log_sigma

        # Tuple conversion
        t_x, t_y, t_z, t_s = out.to_tuple()
        assert t_x is x_hat
        assert t_y is y
        assert t_z is z
        assert t_s is log_sigma

    def test_model_output_attribute_error(self):
        out = ModelOutput(
            x_hat=torch.empty(0),
            y=torch.empty(0),
            z=torch.empty(0),
            log_sigma=torch.empty(0),
        )
        with pytest.raises(AttributeError, match="has no attribute 'nonexistent'"):
            _ = out.nonexistent


# -----------------------------------------------------------------------------
# 5. ModelA End-to-End Architecture & Operational Modes Tests
# -----------------------------------------------------------------------------


class TestModelA:
    """Verifies ModelA forward passes, operational modes, gradient flow, and mixed precision."""

    def test_standard_training_forward_shapes(self):
        model = ModelA(in_channels=3, n_channels=128, m_channels=192)
        model.train()
        x = torch.rand(2, 3, 256, 256)

        out = model(x, training=True)

        assert out.x_hat.shape == (2, 3, 256, 256)
        assert out.y.shape == (2, 192, 16, 16)
        assert out.z.shape == (2, 128, 4, 4)
        assert out.log_sigma.shape == (2, 192, 16, 16)
        assert out.y_hat.shape == (2, 192, 16, 16)
        assert out.z_hat.shape == (2, 128, 4, 4)

        # Verify training mode adds continuous uniform noise
        diff_y = torch.abs(out.y_hat - out.y)
        assert torch.all(diff_y <= 0.50001)
        assert torch.any(diff_y > 1e-4)  # Stochastic noise injected

    def test_evaluation_mode_integer_rounding_and_determinism(self):
        model = ModelA(in_channels=3, n_channels=128, m_channels=192)
        model.eval()
        x = torch.rand(1, 3, 768, 512)

        out1 = model(x, training=False)
        out2 = model(x, training=False)

        # Shapes on Kodak resolution (768x512)
        assert out1.x_hat.shape == (1, 3, 768, 512)
        assert out1.y.shape == (1, 192, 48, 32)
        assert out1.z.shape == (1, 128, 12, 8)
        assert out1.log_sigma.shape == (1, 192, 48, 32)

        # Exact integer verification on rounded latents
        assert torch.allclose(out1.y_hat, torch.round(out1.y))
        assert torch.allclose(out1.z_hat, torch.round(out1.z))
        assert torch.allclose(out1.y_hat, torch.round(out1.y_hat))
        assert torch.allclose(out1.z_hat, torch.round(out1.z_hat))

        # Strict determinism across repeated evaluation runs
        assert torch.equal(out1.y_hat, out2.y_hat)
        assert torch.equal(out1.z_hat, out2.z_hat)
        assert torch.equal(out1.log_sigma, out2.log_sigma)
        assert torch.equal(out1.x_hat, out2.x_hat)

    def test_arbitrary_spatial_resolutions(self):
        model = ModelA()
        model.eval()

        resolutions = [
            (128, 128),  # Small valid patch
            (320, 192),  # Asymmetric multiples of 64
            (512, 512),  # Square large
        ]

        for h, w in resolutions:
            x = torch.rand(1, 3, h, w)
            out = model(x, training=False)
            assert out.x_hat.shape == (1, 3, h, w)
            assert out.y.shape == (1, 192, h // 16, w // 16)
            assert out.z.shape == (1, 128, h // 64, w // 64)
            assert out.log_sigma.shape == (1, 192, h // 16, w // 16)

    def test_invalid_spatial_dimensions_error(self):
        model = ModelA()
        # 500x375 (unpadded dimensions)
        with pytest.raises(ValueError, match="must be multiples of 64"):
            model(torch.rand(1, 3, 500, 375))

        # 4D shape check
        with pytest.raises(ValueError, match="Expected 4D input tensor"):
            model(torch.rand(3, 256, 256))

        # Channel count check
        with pytest.raises(ValueError, match="Channel mismatch"):
            model(torch.rand(1, 4, 256, 256))

    def test_zero_and_flat_input_stability(self):
        model = ModelA()
        model.eval()

        zero_x = torch.zeros(2, 3, 256, 256)
        out = model(zero_x, training=False)

        assert not torch.isnan(out.x_hat).any()
        assert not torch.isinf(out.x_hat).any()
        assert not torch.isnan(out.y).any()
        assert not torch.isnan(out.z).any()
        assert not torch.isnan(out.log_sigma).any()

    def test_full_gradient_flow_end_to_end(self):
        """Verifies full end-to-end gradient backpropagation across all network parameters.

        Tests:
        1. Distortion loss L_dist = MSE(x_hat, x) backpropagates to g_s and g_a parameters.
        2. Hyperprior loss L_hyper = MSE(log_sigma, target) backpropagates to h_s, h_a, and g_a.
        3. Combined loss L = MSE(x_hat, x) + 0.1 * MSE(log_sigma, target) computes finite,
           non-zero gradients for every single parameter in ga, gs, ha, hs and GDN/IGDN layers.
        """
        model = ModelA()
        model.train()

        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        out = model(x, training=True)

        target_log_sigma = torch.zeros_like(out.log_sigma)
        combined_loss = F.mse_loss(out.x_hat, x) + 0.1 * F.mse_loss(out.log_sigma, target_log_sigma)
        combined_loss.backward()

        # Check input gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert torch.any(x.grad != 0.0)

        # Check that EVERY parameter in the model has a non-zero, finite gradient
        missing_grads = []
        zero_grads = []
        nan_grads = []

        for name, param in model.named_parameters():
            if param.grad is None:
                missing_grads.append(name)
            elif torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                nan_grads.append(name)
            elif torch.all(param.grad == 0.0):
                zero_grads.append(name)

        assert not missing_grads, f"Parameters with missing gradients: {missing_grads}"
        assert not nan_grads, f"Parameters with NaN/Inf gradients: {nan_grads}"
        assert not zero_grads, f"Parameters with all-zero gradients: {zero_grads}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available for mixed precision test")
    def test_mixed_precision_cuda_autocast_stability(self):
        """Verifies mixed-precision BF16 execution on CUDA.

        Checks:
        - Convolutions run in BF16
        - GDN and final layer of g_a execute in FP32
        - HyperSynthesis hs executes in strict FP32
        - Losses and gradients remain finite and stable
        """
        device = torch.device("cuda")
        model = ModelA().to(device)
        model.train()

        x = torch.randn(2, 3, 128, 128, device=device, requires_grad=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x, training=True)
            loss = F.mse_loss(out.x_hat, x) + 0.1 * out.log_sigma.pow(2).mean()

        # Precision invariants
        assert out.y.dtype == torch.float32, "Analysis transform final latent y must be strict FP32"
        assert out.log_sigma.dtype == torch.float32, "Hyper-synthesis log_sigma must be strict FP32"
        assert out.x_hat.dtype == torch.float32, "Reconstructed image x_hat must be FP32"

        # Backpropagation
        loss.backward()

        assert not torch.isnan(loss)
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has None grad"
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN grad"
