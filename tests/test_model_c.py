"""Comprehensive unit test suite for Model C (explicit adaptive allocation architecture).

Verifies Story 4.3 requirements:
- DualHeadHyperSynthesisTransform dimensions: z_hat (B, 128, H/64, W/64) ->
  log_sigma (B, 192, H/16, W/16) and s_pred (B, 1, H/16, W/16) in [0, 1].
- ModelCOutput container: attribute and dictionary access, to_tuple() 5-element tuple.
- ModelC forward execution in dual operational modes (training noise vs evaluation rounding).
- Spatial dimension validation (must be divisible by 64).
- Detached Rate-Distortion gradient flow:
  * Reconstruction and rate losses produce zero gradients to s_pred and conv_saliency.
  * Auxiliary saliency loss produces finite gradients in conv_saliency, h_s trunk, and h_a,
    while g_a receives strictly zero gradients when detach_y_for_ha=True.
- Reflection padding / valid mask integration: outside M_valid, log_delta == 0 and delta == 1.
- Pretrained bmshj2018 weight loading into ModelC with tolerance for freshly initialized conv_saliency.
- CDF compilation gate (model.update()) and checkpoint persistence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from src.codec import (
    DualHeadHyperSynthesisTransform,
    ModelA,
    ModelC,
    ModelCOutput,
    load_pretrained_bmshj2018,
)
from src.data.padding import compute_valid_latent_mask
from src.training.loss import auxiliary_saliency_loss
from src.training.tracker import save_checkpoint


class TestDualHeadHyperSynthesisTransform:
    """Tests for DualHeadHyperSynthesisTransform structure, outputs, and validation."""

    def test_output_shapes_and_saliency_bounds(self) -> None:
        """Emits log_sigma (B, 192, H/16, W/16) and s_pred (B, 1, H/16, W/16) bounded in [0, 1]."""
        h_s = DualHeadHyperSynthesisTransform(in_channels=128, n_channels=128, out_channels=192)
        z_hat = torch.randn(2, 128, 2, 2, dtype=torch.float32)

        log_sigma, s_pred = h_s(z_hat)

        assert log_sigma.shape == (2, 192, 8, 8)
        assert s_pred.shape == (2, 1, 8, 8)
        assert (s_pred >= 0.0).all()
        assert (s_pred <= 1.0).all()

    def test_layer_names_compatible_with_model_a(self) -> None:
        """Shared layers must match ModelA h_s names: deconv1, deconv2, conv3."""
        h_s = DualHeadHyperSynthesisTransform()
        assert hasattr(h_s, "deconv1")
        assert hasattr(h_s, "deconv2")
        assert hasattr(h_s, "conv3")
        assert hasattr(h_s, "conv_saliency")
        assert h_s.conv_scale is h_s.conv3

    def test_input_validation(self) -> None:
        """Validates 4D tensor and channel count."""
        h_s = DualHeadHyperSynthesisTransform(in_channels=128)
        with pytest.raises(ValueError, match="4D input tensor"):
            h_s(torch.randn(128, 2, 2))
        with pytest.raises(ValueError, match="Channel mismatch"):
            h_s(torch.randn(1, 64, 2, 2))


class TestModelCOutputContainer:
    """Tests for ModelCOutput container."""

    def test_container_access_and_tuple_unpacking(self) -> None:
        """Supports attribute access, dict access, and 5-tuple conversion."""
        tensors = {
            "x_hat": torch.zeros(1, 3, 64, 64),
            "y": torch.zeros(1, 192, 4, 4),
            "z": torch.zeros(1, 128, 1, 1),
            "log_sigma": torch.zeros(1, 192, 4, 4),
            "log_sigma_eff": torch.zeros(1, 192, 4, 4),
            "s_pred": torch.full((1, 1, 4, 4), 0.5),
            "log_delta": torch.zeros(1, 1, 4, 4),
            "delta": torch.ones(1, 1, 4, 4),
            "y_hat": torch.zeros(1, 192, 4, 4),
            "z_hat": torch.zeros(1, 128, 1, 1),
            "y_likelihoods": torch.full((1, 192, 4, 4), 0.05),
            "z_likelihoods": torch.full((1, 128, 1, 1), 0.02),
        }
        out = ModelCOutput(**tensors)

        # Dict access
        assert out["s_pred"].shape == (1, 1, 4, 4)
        assert out["delta"].shape == (1, 1, 4, 4)
        assert out["log_sigma_eff"].shape == (1, 192, 4, 4)

        # Attribute access
        assert out.s_pred.shape == (1, 1, 4, 4)
        assert out.delta.shape == (1, 1, 4, 4)
        assert out.log_sigma_eff.shape == (1, 192, 4, 4)

        # Tuple conversion: 5-tuple (x_hat, y, z, log_sigma, s_pred)
        x_hat, y, z, log_sigma, s_pred = out.to_tuple()
        assert x_hat.shape == (1, 3, 64, 64)
        assert y.shape == (1, 192, 4, 4)
        assert z.shape == (1, 128, 1, 1)
        assert log_sigma.shape == (1, 192, 4, 4)
        assert s_pred.shape == (1, 1, 4, 4)


class TestModelCArchitectureAndForward:
    """Tests for ModelC forward pass, operational modes, and dimension validation."""

    @pytest.mark.parametrize("shape", [(1, 3, 64, 64), (2, 3, 128, 128)])
    @pytest.mark.parametrize("training", [True, False])
    def test_forward_pass_modes_and_shapes(self, shape: tuple[int, ...], training: bool) -> None:
        """Executes forward pass in training and evaluation modes across spatial resolutions."""
        torch.manual_seed(42)
        model = ModelC()
        x = torch.rand(shape, dtype=torch.float32)

        out = model(x, training=training)
        assert isinstance(out, ModelCOutput)

        b, _, h, w = shape
        assert out.x_hat.shape == (b, 3, h, w)
        assert out.y.shape == (b, 192, h // 16, w // 16)
        assert out.z.shape == (b, 128, h // 64, w // 64)
        assert out.log_sigma.shape == (b, 192, h // 16, w // 16)
        assert out.log_sigma_eff.shape == (b, 192, h // 16, w // 16)
        assert out.s_pred.shape == (b, 1, h // 16, w // 16)
        assert (out.s_pred >= 0.0).all()
        assert (out.s_pred <= 1.0).all()

        assert out.delta.shape == (b, 1, h // 16, w // 16)
        assert (out.delta >= 0.25 - 1e-6).all()
        assert (out.delta <= 4.0 + 1e-6).all()

        assert out.y_hat.shape == (b, 192, h // 16, w // 16)
        assert out.z_hat.shape == (b, 128, h // 64, w // 64)
        assert out.y_likelihoods.shape == (b, 192, h // 16, w // 16)
        assert out.z_likelihoods.shape == (b, 128, h // 64, w // 64)

    def test_unpadded_input_raises_value_error(self) -> None:
        """Inputs not divisible by 64 raise ValueError."""
        model = ModelC()
        with pytest.raises(ValueError, match="multiples of 64"):
            model(torch.rand(1, 3, 50, 50))

    def test_synthesize_scales_methods(self) -> None:
        """Verifies synthesize_scales and synthesize_scales_and_saliency helper methods."""
        model = ModelC()
        z_hat = torch.randn(1, 128, 2, 2)

        log_sigma, s_pred = model.synthesize_scales_and_saliency(z_hat)
        assert log_sigma.shape == (1, 192, 8, 8)
        assert s_pred.shape == (1, 1, 8, 8)

        log_sigma_only = model.synthesize_scales(z_hat)
        assert log_sigma_only.shape == (1, 192, 8, 8)
        assert torch.allclose(log_sigma_only, log_sigma)

    def test_forward_with_batched_3d_valid_mask(self) -> None:
        """Batched 3D valid_mask (B, H, W) executes cleanly without dimension expansion errors."""
        model = ModelC()
        x = torch.rand(2, 3, 64, 64)
        mask_3d = torch.ones(2, 4, 4, dtype=torch.bool)
        mask_3d[0, 3, :] = False

        out = model(x, valid_mask=mask_3d)
        assert isinstance(out, ModelCOutput)
        assert out.delta.shape == (2, 1, 4, 4)
        # Verify padded cells have delta == 1.0
        assert torch.allclose(out.delta[0, 0, 3, :], torch.tensor(1.0))

    def test_forward_with_saliency_map_override_oracle(self) -> None:
        """Explicit ground-truth saliency_map overrides predicted saliency for Oracle O1."""
        model = ModelC()
        x = torch.rand(1, 3, 64, 64)
        # Construct contrasting saliency map
        gt_sal = torch.zeros(1, 1, 64, 64)
        gt_sal[:, :, :32, :] = 1.0

        out_normal = model(x, training=False)
        out_oracle = model(x, saliency_map=gt_sal, training=False)

        # Delta maps must differ because Oracle allocation used gt_sal
        assert not torch.allclose(out_normal.delta, out_oracle.delta)

    def test_autocast_numerical_fidelity(self) -> None:
        """Under mixed precision autocast, critical outputs and scales remain strict float32."""
        model = ModelC()
        x = torch.rand(1, 3, 64, 64)

        device_type = "cpu"
        with torch.autocast(device_type=device_type, enabled=True):
            out = model(x)

        assert out.x_hat.dtype == torch.float32
        assert out.log_sigma.dtype == torch.float32
        assert out.log_sigma_eff.dtype == torch.float32
        assert out.delta.dtype == torch.float32
        assert out.s_pred.dtype == torch.float32

    def test_forward_input_validation(self) -> None:
        """Non-4D inputs or channel mismatches raise ValueError."""
        model = ModelC()
        with pytest.raises(ValueError, match="Expected 4D input tensor"):
            model(torch.rand(3, 64, 64))
        with pytest.raises(ValueError, match="Channel mismatch"):
            model(torch.rand(1, 1, 64, 64))


class TestGradientFlowAndIsolation:
    """Verifies detached rate-distortion gradients and auxiliary gradient cutting."""

    def test_rd_loss_produces_zero_gradient_to_saliency_head(self) -> None:
        """Reconstruction and rate losses produce zero gradient to Delta, s_pred, and conv_saliency."""
        torch.manual_seed(999)
        model = ModelC()
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)

        out = model(x, training=True)
        # Compute loss purely from rate and distortion
        distortion = torch.mean((out.x_hat - x) ** 2)
        rate = -torch.sum(torch.log2(out.y_likelihoods)) / (64 * 64)
        loss = distortion + 0.01 * rate
        loss.backward()

        # The RD loss must NOT propagate back through Delta into s_pred or conv_saliency
        assert model.h_s.conv_saliency.weight.grad is None
        assert model.h_s.conv_saliency.bias.grad is None

        # But RD loss must propagate into main transforms and scale head
        assert model.g_a.conv1.weight.grad is not None
        assert model.g_s.deconv1.weight.grad is not None
        assert model.h_s.conv3.weight.grad is not None

    def test_auxiliary_loss_gradient_flow_and_ga_cutting(self) -> None:
        """Auxiliary loss trains conv_saliency, h_s trunk, and h_a, but cuts gradients before g_a."""
        torch.manual_seed(888)
        model = ModelC(detach_y_for_ha=True)
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        target_s = torch.rand(1, 1, 4, 4, dtype=torch.float32)

        out = model(x, training=True)
        aux_loss = auxiliary_saliency_loss(out.s_pred, target_s)
        aux_loss.backward()

        saliency_grad = model.h_s.conv_saliency.weight.grad
        assert saliency_grad is not None
        assert torch.isfinite(saliency_grad).all()
        assert not torch.allclose(saliency_grad, torch.zeros_like(saliency_grad))

        assert model.h_s.deconv2.weight.grad is not None
        assert model.h_s.deconv1.weight.grad is not None
        assert model.h_a.conv3.weight.grad is not None

        # Main analysis transform g_a must receive ZERO gradient due to y.detach()
        assert model.g_a.conv1.weight.grad is None
        assert model.g_a.conv4.weight.grad is None

    def test_detach_y_for_ha_false_allows_ga_gradients(self) -> None:
        """When detach_y_for_ha=False, auxiliary loss propagates back into g_a."""
        torch.manual_seed(777)
        model = ModelC(detach_y_for_ha=False)
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        target_s = torch.rand(1, 1, 4, 4, dtype=torch.float32)

        out = model(x, training=True)
        aux_loss = auxiliary_saliency_loss(out.s_pred, target_s)
        aux_loss.backward()

        # When detach_y_for_ha=False, g_a receives non-zero gradient from auxiliary loss
        assert model.g_a.conv4.weight.grad is not None


class TestValidMaskIntegration:
    """Verifies reflection padding and valid mask isolation."""

    def test_valid_mask_isolates_padded_cells(self) -> None:
        """Valid mask isolates padded cells, forcing log_delta == 0.0 and delta == 1.0 outside."""
        model = ModelC()
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)

        # 4x4 latent grid: top-left 2x2 is valid
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[:2, :2] = True

        out = model(x, valid_mask=mask)
        # Outside mask, delta must be strictly 1.0 and log_delta strictly 0.0
        padded_region = ~mask
        assert torch.all(out.log_delta[0, 0][padded_region] == 0.0)
        assert torch.all(out.delta[0, 0][padded_region] == 1.0)

    def test_compute_valid_latent_mask_integration(self) -> None:
        """Integrates seamlessly with padding.compute_valid_latent_mask."""
        mask = compute_valid_latent_mask(
            H_orig=50,
            W_orig=50,
            H_pad=64,
            W_pad=64,
            stride=16,
            pad_bounds=(7, 7, 7, 7),
        )
        assert mask.shape == (4, 4)

        model = ModelC()
        x = torch.rand(1, 3, 64, 64)
        out = model(x, valid_mask=mask)

        assert torch.all(out.log_delta[0, 0][~mask] == 0.0)
        assert torch.all(out.delta[0, 0][~mask] == 1.0)


class TestPretrainedWeightLoadingAndCDFs:
    """Verifies weight restoration from ModelA / bmshj2018 and CDF compilation."""

    def test_load_state_dict_from_model_a(self) -> None:
        """ModelC loads ModelA state dict with strict=True (gracefully tolerating conv_saliency)."""
        model_a = ModelA()
        state_dict_a = model_a.state_dict()

        model_c = ModelC()
        # Loading ModelA state dict into ModelC
        model_c.load_state_dict(state_dict_a, strict=True)

        # Shared weights must match bit-exactly
        assert torch.allclose(model_c.g_a.conv1.weight, model_a.g_a.conv1.weight)
        assert torch.allclose(model_c.g_s.deconv4.weight, model_a.g_s.deconv4.weight)
        assert torch.allclose(model_c.h_s.deconv1.weight, model_a.h_s.deconv1.weight)
        assert torch.allclose(model_c.h_s.conv3.weight, model_a.h_s.conv3.weight)
        # Saliency head remains intact and initialized
        assert model_c.h_s.conv_saliency.weight.norm() > 0.0

    def test_strict_loading_rejects_unexpected_keys(self) -> None:
        """strict=True raises RuntimeError when unexpected keys exist."""
        model_c = ModelC()
        sd = model_c.state_dict()
        sd["unwanted_extra_layer.weight"] = torch.randn(10, 10)

        with pytest.raises(RuntimeError, match="Unexpected key\\(s\\) in state_dict"):
            model_c.load_state_dict(sd, strict=True)

    def test_strict_loading_rejects_missing_shared_keys(self) -> None:
        """strict=True raises RuntimeError when core shared keys are missing."""
        model_c = ModelC()
        sd = model_c.state_dict()
        del sd["g_a.conv1.weight"]

        with pytest.raises(RuntimeError, match="Missing key\\(s\\) in state_dict"):
            model_c.load_state_dict(sd, strict=True)

    def test_native_model_c_state_dict_roundtrip(self) -> None:
        """Native ModelC state dict saves and loads with strict=True without warnings."""
        model_c1 = ModelC()
        sd = model_c1.state_dict()

        model_c2 = ModelC()
        model_c2.load_state_dict(sd, strict=True)

        for k, v in sd.items():
            assert torch.allclose(v, model_c2.state_dict()[k])

    def test_load_pretrained_bmshj2018_wrapper(self) -> None:
        """load_pretrained_bmshj2018 loads into an existing ModelC instance."""
        model_a = ModelA()
        state_dict_a = model_a.state_dict()

        model_c = ModelC()
        loaded = load_pretrained_bmshj2018(state_dict_a, model=model_c)
        assert loaded is model_c
        assert torch.allclose(model_c.g_a.conv1.weight, model_a.g_a.conv1.weight)

    def test_from_pretrained_classmethod(self) -> None:
        """ModelC.from_pretrained constructs and loads weights."""
        model_a = ModelA()
        state_dict_a = model_a.state_dict()

        model_c = ModelC.from_pretrained(state_dict_a)
        assert isinstance(model_c, ModelC)
        assert torch.allclose(model_c.g_a.conv1.weight, model_a.g_a.conv1.weight)

    def test_cdf_compilation_gate_and_persistence(self) -> None:
        """ModelC compiles discrete CDFs and passes post-training CDF serialization gate."""
        model_c = ModelC()
        # Compile discrete CDFs
        res = model_c.update(force=True)
        assert res["entropy_bottleneck"] is True
        assert res["gaussian_conditional"] is True

        state_dict = model_c.state_dict()
        assert "entropy_bottleneck._quantized_cdf" in state_dict
        assert state_dict["entropy_bottleneck._quantized_cdf"].numel() > 0
        assert "gaussian_conditional._quantized_cdf" in state_dict
        assert state_dict["gaussian_conditional._quantized_cdf"].numel() > 0

        # Save checkpoint to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = Path(tmpdir) / "model_c_checkpoint.pt"
            saved = save_checkpoint(model_c, chk_path, compile_cdfs=True)
            assert saved.is_file()
            assert saved.stat().st_size > 0
