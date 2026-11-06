"""Unit test suite for Model B and pretrained weight loading utilities.

Verifies:
- ModelB subclassing of ModelA and parameter shape consistency (N=128, M=192).
- Forward passes in dual operational modes:
  * Continuous uniform noise injection in training mode (training=True).
  * Deterministic integer rounding in evaluation mode (training=False).
- Spatial dimension validation (must be divisible by 64; unpadded raises ValueError).
- Direct state dict loading from ModelA producing 100% bit-identical outputs.
- CompressAI Sequential layer indexing remapping with zero missing or unexpected keys.
- Checkpoint loading edge cases: wrapped dicts ({"state_dict": ...}, {"model": ...}),
  disk files (.pt/.pth), and missing file handling (FileNotFoundError).
- Clear reporting of unmapped keys and prefix cleaning (module., _orig_mod.).
- RateDistortionLoss integration and finite metric breakdown (loss, bpp, distortion).
- End-to-end gradient backpropagation across all network parameters.
- Mixed precision (BF16 autocast) execution and FP32 isolation boundaries.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.codec import (
    ModelA,
    ModelB,
    ModelOutput,
    load_pretrained_bmshj2018,
    remap_compressai_state_dict,
)
from src.training.loss import RateDistortionLoss

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

COMPRESSAI_RENAME_PAIRS: tuple[tuple[str, str], ...] = (
    ("g_a.conv1.", "g_a.0."),
    ("g_a.gdn1.", "g_a.1."),
    ("g_a.conv2.", "g_a.2."),
    ("g_a.gdn2.", "g_a.3."),
    ("g_a.conv3.", "g_a.4."),
    ("g_a.gdn3.", "g_a.5."),
    ("g_a.conv4.", "g_a.6."),
    ("g_s.deconv1.", "g_s.0."),
    ("g_s.igdn1.", "g_s.1."),
    ("g_s.deconv2.", "g_s.2."),
    ("g_s.igdn2.", "g_s.3."),
    ("g_s.deconv3.", "g_s.4."),
    ("g_s.igdn3.", "g_s.5."),
    ("g_s.deconv4.", "g_s.6."),
    ("h_a.conv1.", "h_a.0."),
    ("h_a.conv2.", "h_a.2."),
    ("h_a.conv3.", "h_a.4."),
    ("h_s.deconv1.", "h_s.0."),
    ("h_s.deconv2.", "h_s.2."),
    ("h_s.conv3.", "h_s.4."),
    ("entropy_bottleneck.weights.", "entropy_bottleneck._matrices."),
    ("entropy_bottleneck.biases.", "entropy_bottleneck._biases."),
)


def _convert_model_a_state_dict_to_compressai(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Converts a standard ModelA/ModelB state dict to CompressAI Sequential indexing format."""
    cai_dict: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        remapped = False
        for src_prefix, dst_prefix in COMPRESSAI_RENAME_PAIRS:
            if k.startswith(src_prefix):
                cai_dict[dst_prefix + k[len(src_prefix) :]] = v
                remapped = True
                break
        if not remapped:
            cai_dict[k] = v
    return cai_dict


# -----------------------------------------------------------------------------
# 1. Architecture and Subclassing Tests
# -----------------------------------------------------------------------------


class TestModelBArchitecture:
    """Verifies that ModelB properly subclasses ModelA and retains identical architecture."""

    def test_subclass_and_inheritance(self):
        assert issubclass(ModelB, ModelA)
        model = ModelB()
        assert isinstance(model, ModelA)
        assert isinstance(model, nn.Module)

    def test_default_parameter_shapes(self):
        model = ModelB()
        assert model.in_channels == 3
        assert model.n_channels == 128
        assert model.m_channels == 192

        # Verify transform submodule presence
        assert hasattr(model, "g_a")
        assert hasattr(model, "g_s")
        assert hasattr(model, "h_a")
        assert hasattr(model, "h_s")
        assert hasattr(model, "entropy_bottleneck")
        assert hasattr(model, "gaussian_conditional")

    def test_parameter_count_matches_model_a(self):
        model_a = ModelA()
        model_b = ModelB()

        params_a = list(model_a.parameters())
        params_b = list(model_b.parameters())
        assert len(params_a) == len(params_b)

        total_a = sum(p.numel() for p in params_a)
        total_b = sum(p.numel() for p in params_b)
        assert total_a == total_b

        keys_a = set(model_a.state_dict().keys())
        keys_b = set(model_b.state_dict().keys())
        assert keys_a == keys_b


# -----------------------------------------------------------------------------
# 2. Forward Execution and Operational Modes
# -----------------------------------------------------------------------------


class TestModelBForwardPass:
    """Verifies dual operational modes and dimension validation."""

    def test_forward_train_mode_noise_injection(self):
        torch.manual_seed(42)
        model = ModelB()
        model.train()

        x = torch.rand(2, 3, 64, 64)
        out = model(x, training=True)

        assert isinstance(out, ModelOutput)
        assert out.x_hat.shape == (2, 3, 64, 64)
        assert out.y.shape == (2, 192, 4, 4)
        assert out.z.shape == (2, 128, 1, 1)
        assert out.log_sigma.shape == (2, 192, 4, 4)
        assert out.y_hat.shape == (2, 192, 4, 4)
        assert out.z_hat.shape == (2, 128, 1, 1)
        assert out.y_likelihoods.shape == (2, 192, 4, 4)
        assert out.z_likelihoods.shape == (2, 128, 1, 1)

        # Under uniform noise injection, y_hat is continuous, not integer
        assert not torch.equal(out.y_hat, torch.round(out.y_hat))
        assert not torch.equal(out.z_hat, torch.round(out.z_hat))

    def test_forward_eval_mode_integer_rounding(self):
        torch.manual_seed(42)
        model = ModelB()
        model.eval()

        x = torch.rand(2, 3, 64, 64)
        out1 = model(x, training=False)
        out2 = model(x, training=False)

        assert isinstance(out1, ModelOutput)
        # In eval mode, y_hat and z_hat must be rounded integers
        assert torch.equal(out1.y_hat, torch.round(out1.y_hat))
        assert torch.equal(out1.z_hat, torch.round(out1.z_hat))

        # Output must be deterministic across calls
        assert torch.equal(out1.x_hat, out2.x_hat)
        assert torch.equal(out1.y_hat, out2.y_hat)
        assert torch.equal(out1.z_hat, out2.z_hat)
        assert torch.equal(out1.y_likelihoods, out2.y_likelihoods)
        assert torch.equal(out1.z_likelihoods, out2.z_likelihoods)

    def test_forward_dimension_divisible_by_64(self):
        model = ModelB()
        model.eval()

        # Multiples of 64
        for h, w in [(64, 64), (128, 64), (64, 128)]:
            x = torch.rand(1, 3, h, w)
            out = model(x)
            assert out.x_hat.shape == (1, 3, h, w)

    def test_forward_dimension_not_divisible_by_64_raises(self):
        model = ModelB()
        for h, w in [(63, 64), (64, 65), (100, 100), (32, 32)]:
            x = torch.rand(1, 3, h, w)
            with pytest.raises(ValueError, match="must be multiples of 64"):
                model(x)

    def test_forward_invalid_tensor_ndim_raises(self):
        model = ModelB()
        # 3D tensor
        x_3d = torch.rand(3, 64, 64)
        with pytest.raises(ValueError, match="Expected 4D input tensor"):
            model(x_3d)

        # 5D tensor
        x_5d = torch.rand(1, 3, 64, 64, 1)
        with pytest.raises(ValueError, match="Expected 4D input tensor"):
            model(x_5d)

    def test_forward_channel_mismatch_raises(self):
        model = ModelB(in_channels=3)
        x_1c = torch.rand(1, 1, 64, 64)
        with pytest.raises(ValueError, match="Channel mismatch"):
            model(x_1c)

        x_4c = torch.rand(1, 4, 64, 64)
        with pytest.raises(ValueError, match="Channel mismatch"):
            model(x_4c)


# -----------------------------------------------------------------------------
# 3. Pretrained Weight Loading and Remapping Tests
# -----------------------------------------------------------------------------


class TestPretrainedWeightLoading:
    """Verifies state dict remapping, loading from disk/memory, and bit-identity."""

    def test_direct_model_a_state_dict_bit_identical(self):
        torch.manual_seed(101)
        model_a = ModelA()
        model_b = ModelB()

        # Load ModelA weights directly into ModelB
        model_b.load_state_dict(model_a.state_dict(), strict=True)

        model_a.eval()
        model_b.eval()

        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            out_a = model_a(x, training=False)
            out_b = model_b(x, training=False)

        assert torch.equal(out_a.x_hat, out_b.x_hat)
        assert torch.equal(out_a.y_hat, out_b.y_hat)
        assert torch.equal(out_a.z_hat, out_b.z_hat)
        assert torch.equal(out_a.log_sigma, out_b.log_sigma)
        assert torch.equal(out_a.y_likelihoods, out_b.y_likelihoods)
        assert torch.equal(out_a.z_likelihoods, out_b.z_likelihoods)

    def test_compressai_format_remapping_zero_missing_unexpected(self):
        torch.manual_seed(202)
        model_ref = ModelA()
        ref_sd = model_ref.state_dict()

        # Format state dict as CompressAI Sequential
        cai_sd = _convert_model_a_state_dict_to_compressai(ref_sd)
        assert any(k.startswith("g_a.0.") for k in cai_sd)
        assert any(k.startswith("entropy_bottleneck._matrices.") for k in cai_sd)

        # Load via load_pretrained_bmshj2018 with strict=True
        model_b = load_pretrained_bmshj2018(cai_sd, strict=True)
        assert isinstance(model_b, ModelB)

        # Verify bit-identity with reference model
        model_ref.eval()
        model_b.eval()
        x = torch.rand(2, 3, 64, 64)
        with torch.no_grad():
            out_ref = model_ref(x, training=False)
            out_b = model_b(x, training=False)

        assert torch.equal(out_ref.x_hat, out_b.x_hat)
        assert torch.equal(out_ref.y_hat, out_b.y_hat)
        assert torch.equal(out_ref.z_hat, out_b.z_hat)
        assert torch.equal(out_ref.y_likelihoods, out_b.y_likelihoods)
        assert torch.equal(out_ref.z_likelihoods, out_b.z_likelihoods)

    def test_wrapped_checkpoint_dicts(self):
        model_ref = ModelA()
        ref_sd = model_ref.state_dict()
        cai_sd = _convert_model_a_state_dict_to_compressai(ref_sd)

        # Test wrapper variants
        for wrapper_key in ("state_dict", "model", "model_state_dict", "network", "net"):
            wrapped = {wrapper_key: cai_sd, "epoch": 50, "loss": 0.05}
            model_b = load_pretrained_bmshj2018(wrapped, strict=True)
            assert isinstance(model_b, ModelB)

        # Test doubly wrapped: {"model": {"state_dict": ...}}
        doubly_wrapped = {"model": {"state_dict": cai_sd}}
        model_b = load_pretrained_bmshj2018(doubly_wrapped, strict=True)
        assert isinstance(model_b, ModelB)

    def test_load_from_disk_checkpoint_file(self, tmp_path: Path):
        model_ref = ModelA()
        cai_sd = _convert_model_a_state_dict_to_compressai(model_ref.state_dict())

        # Save to temporary .pt file
        ckpt_path = tmp_path / "compressai_bmshj2018_checkpoint.pt"
        torch.save({"state_dict": cai_sd}, ckpt_path)

        model_b = load_pretrained_bmshj2018(str(ckpt_path), strict=True)
        assert isinstance(model_b, ModelB)

        # Save to temporary .pth file using Path object directly
        ckpt_path_pth = tmp_path / "model_weights.pth"
        torch.save(cai_sd, ckpt_path_pth)

        model_b2 = load_pretrained_bmshj2018(ckpt_path_pth, strict=True)
        assert isinstance(model_b2, ModelB)

    def test_missing_checkpoint_file_raises_file_not_found(self, tmp_path: Path):
        missing_file = tmp_path / "non_existent_file.pt"
        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            load_pretrained_bmshj2018(missing_file)

    def test_invalid_checkpoint_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Expected checkpoint path"):
            load_pretrained_bmshj2018(12345)  # type: ignore

    def test_model_b_load_state_dict_direct_call(self):
        model_ref = ModelA()
        cai_sd = _convert_model_a_state_dict_to_compressai(model_ref.state_dict())

        model_b = ModelB()
        # Direct call to model.load_state_dict should handle CompressAI keys
        incompatible = model_b.load_state_dict(cai_sd, strict=True)
        assert len(incompatible.missing_keys) == 0
        assert len(incompatible.unexpected_keys) == 0

    def test_from_pretrained_classmethod(self, tmp_path: Path):
        model_ref = ModelA()
        sd = model_ref.state_dict()
        ckpt_path = tmp_path / "weights.pt"
        torch.save(sd, ckpt_path)

        model_b = ModelB.from_pretrained(ckpt_path, strict=True)
        assert isinstance(model_b, ModelB)

    def test_flexible_invocation_signatures(self):
        model_ref = ModelA()
        sd = model_ref.state_dict()

        target_model = ModelB()
        # Signature: load_pretrained_bmshj2018(model, state_dict)
        res1 = load_pretrained_bmshj2018(target_model, sd)
        assert res1 is target_model

        # Signature: load_pretrained_bmshj2018(state_dict, model=target_model)
        res2 = load_pretrained_bmshj2018(sd, model=target_model)
        assert res2 is target_model

    def test_unexpected_second_positional_argument_raises(self):
        # Two non-ModelA positional arguments must raise ValueError
        with pytest.raises(ValueError, match="When two positional arguments are passed"):
            load_pretrained_bmshj2018("path1.pt", "path2.pt")

        with pytest.raises(ValueError, match="When two positional arguments are passed"):
            load_pretrained_bmshj2018({}, {})

        # Specifying model both positionally and via keyword must raise ValueError
        model1 = ModelB()
        model2 = ModelB()
        with pytest.raises(ValueError, match="specified both positionally and via 'model'"):
            load_pretrained_bmshj2018(model1, {}, model=model2)

    def test_distributed_prefix_stripping(self):
        model_ref = ModelA()
        sd = model_ref.state_dict()

        # Add module. and _orig_mod. prefixes
        ddp_sd = {f"module.{k}": v for k, v in sd.items()}
        remapped_ddp = remap_compressai_state_dict(ddp_sd)
        assert all(not k.startswith("module.") for k in remapped_ddp)

        compiled_sd = {f"_orig_mod.{k}": v for k, v in sd.items()}
        remapped_compiled = remap_compressai_state_dict(compiled_sd)
        assert all(not k.startswith("_orig_mod.") for k in remapped_compiled)

    def test_composite_prefix_stripping(self):
        # Composite prefixes: _orig_mod.module., module._orig_mod., repeated _orig_mod.
        raw_sd = {
            "_orig_mod.module.g_a.0.weight": torch.zeros(128, 3, 5, 5),
            "module._orig_mod.g_s.0.weight": torch.zeros(192, 128, 5, 5),
            "_orig_mod._orig_mod.h_a.0.weight": torch.zeros(128, 192, 3, 3),
        }
        remapped = remap_compressai_state_dict(raw_sd)
        assert "g_a.conv1.weight" in remapped
        assert "g_s.deconv1.weight" in remapped
        assert "h_a.conv1.weight" in remapped

    def test_compressai_factors_filtering_and_missing_buffer_population(self, caplog: pytest.LogCaptureFixture):
        model_ref = ModelA()
        cai_sd = _convert_model_a_state_dict_to_compressai(model_ref.state_dict())

        # Simulate authentic CompressAI checkpoint:
        # 1. Contains unneeded CompressAI-specific keys: factors.0, _factors.0, quantiles,
        #    target, and likelihood_lower_bound.bound
        cai_sd["entropy_bottleneck.factors.0"] = torch.zeros(128, 1, 1)
        cai_sd["entropy_bottleneck._factors.0"] = torch.zeros(128, 1, 1)
        cai_sd["entropy_bottleneck.quantiles"] = torch.zeros(128, 1, 3)
        cai_sd["entropy_bottleneck.target"] = torch.zeros(128, 1)
        cai_sd["entropy_bottleneck.likelihood_lower_bound.bound"] = torch.tensor(1e-9)
        cai_sd["gaussian_conditional.likelihood_lower_bound.bound"] = torch.tensor(1e-9)

        # 2. Lacks entropy_bottleneck.lower_bound.bound and gaussian_conditional scale bounds
        cai_sd.pop("entropy_bottleneck.lower_bound.bound", None)
        cai_sd.pop("gaussian_conditional.scale_bound", None)
        cai_sd.pop("gaussian_conditional.lower_bound_scale.bound", None)

        # Remap and verify CompressAI-specific keys are dropped without warnings
        with caplog.at_level(logging.WARNING):
            remapped = remap_compressai_state_dict(cai_sd)

        # Ensure no unmapped key warnings were emitted for authentic CompressAI keys
        assert not any("unmapped key" in record.message for record in caplog.records)

        # Verify ignored keys are dropped
        assert not any("factors" in k for k in remapped)
        assert not any("quantiles" in k for k in remapped)
        assert not any("target" in k for k in remapped)
        assert not any("likelihood_lower_bound" in k for k in remapped)

        # Verify default buffers are populated
        assert "entropy_bottleneck.lower_bound.bound" in remapped
        assert "gaussian_conditional.scale_bound" in remapped
        assert "gaussian_conditional.lower_bound_scale.bound" in remapped

        # Loading into ModelB with strict=True must succeed with zero missing/unexpected keys
        model_b = load_pretrained_bmshj2018(cai_sd, strict=True)
        assert isinstance(model_b, ModelB)

        # Forward pass runs deterministically in eval mode
        model_b.eval()
        x = torch.rand(1, 3, 64, 64)
        out = model_b(x)
        assert out.x_hat.shape == (1, 3, 64, 64)

    def test_load_pretrained_no_duplicate_warning(self, caplog: pytest.LogCaptureFixture):
        # When an unmapped key is present, load_pretrained_bmshj2018 must only log warning once
        sd = {"unmapped_extra_key.weight": torch.zeros(1), "g_a.0.weight": torch.zeros(128, 3, 5, 5)}
        with caplog.at_level(logging.WARNING):
            load_pretrained_bmshj2018(sd, strict=False)

        warning_count = sum(1 for record in caplog.records if "unmapped key" in record.message)
        assert warning_count == 1, f"Expected exactly 1 warning log, got {warning_count}"

    def test_unmapped_keys_reporting(self, caplog: pytest.LogCaptureFixture):
        sd = {"unmapped_conv.weight": torch.zeros(1), "g_a.0.weight": torch.zeros(128, 3, 5, 5)}
        with caplog.at_level(logging.WARNING):
            remapped = remap_compressai_state_dict(sd)

        assert "unmapped_conv.weight" in remapped
        assert any("unmapped key" in record.message for record in caplog.records)

    def test_entropy_and_gaussian_unmapped_keys_reporting(self, caplog: pytest.LogCaptureFixture):
        # Specific unmapped keys under entropy_bottleneck and gaussian_conditional must not bypass warning
        sd = {
            "entropy_bottleneck.unmapped_layer": torch.zeros(1),
            "gaussian_conditional.unknown_param": torch.zeros(1),
        }
        with caplog.at_level(logging.WARNING):
            remap_compressai_state_dict(sd)

        warning_messages = " ".join(record.message for record in caplog.records)
        assert "entropy_bottleneck.unmapped_layer" in warning_messages
        assert "gaussian_conditional.unknown_param" in warning_messages

    @pytest.mark.filterwarnings("ignore:.*copying from a non-meta parameter.*:UserWarning")
    def test_cpu_state_dict_loaded_to_target_device_synchronizes_cdf_buffers(self):
        # Build reference model on CPU and compile CDF tables
        ref_model = ModelA()
        ref_model.update()
        cpu_sd = ref_model.state_dict()

        target_device = torch.device("cuda" if torch.cuda.is_available() else "meta")

        # Test 1: model.load_state_dict on a model existing on target_device
        model1 = ModelB().to(target_device)
        model1.load_state_dict(cpu_sd)
        param_dev1 = next(model1.parameters()).device
        assert param_dev1 == target_device

        # All entropy model CDF buffers must match parameter device (not left desynchronized on CPU)
        assert model1.entropy_bottleneck._quantized_cdf.device == param_dev1
        assert model1.entropy_bottleneck._cdf_length.device == param_dev1
        assert model1.entropy_bottleneck._offset.device == param_dev1
        assert model1.gaussian_conditional.scale_table.device == param_dev1
        assert model1.gaussian_conditional._quantized_cdf.device == param_dev1
        assert model1.gaussian_conditional._cdf_length.device == param_dev1
        assert model1.gaussian_conditional._offset.device == param_dev1

        # Test 2: load_pretrained_bmshj2018 with existing target model
        model2 = ModelB().to(target_device)
        load_pretrained_bmshj2018(cpu_sd, model=model2)
        param_dev2 = next(model2.parameters()).device
        assert param_dev2 == target_device
        assert model2.entropy_bottleneck._quantized_cdf.device == param_dev2
        assert model2.entropy_bottleneck._cdf_length.device == param_dev2
        assert model2.entropy_bottleneck._offset.device == param_dev2
        assert model2.gaussian_conditional.scale_table.device == param_dev2
        assert model2.gaussian_conditional._quantized_cdf.device == param_dev2
        assert model2.gaussian_conditional._cdf_length.device == param_dev2
        assert model2.gaussian_conditional._offset.device == param_dev2

        # Test 3: load_pretrained_bmshj2018 with device argument
        model3 = load_pretrained_bmshj2018(cpu_sd, device=target_device)
        param_dev3 = next(model3.parameters()).device
        assert param_dev3 == target_device
        assert model3.entropy_bottleneck._quantized_cdf.device == target_device
        assert model3.entropy_bottleneck._cdf_length.device == target_device
        assert model3.entropy_bottleneck._offset.device == target_device
        assert model3.gaussian_conditional.scale_table.device == target_device
        assert model3.gaussian_conditional._quantized_cdf.device == target_device
        assert model3.gaussian_conditional._cdf_length.device == target_device
        assert model3.gaussian_conditional._offset.device == target_device


# -----------------------------------------------------------------------------
# 4. Rate-Distortion Loss Integration and Gradients
# -----------------------------------------------------------------------------


class TestModelBLossAndGradients:
    """Verifies integration with RateDistortionLoss and gradient backpropagation."""

    def test_rate_distortion_loss_computation(self):
        model = ModelB()
        loss_fn = RateDistortionLoss(lmbda=0.01)

        x = torch.rand(2, 3, 64, 64)
        s = torch.rand(2, 1, 64, 64)

        # Train mode evaluation
        model.train()
        out_train = model(x, training=True)
        res_train = loss_fn(out_train, x, s=s)

        for key in ("loss", "bpp_loss", "y_bpp", "z_bpp", "distortion", "mse_loss"):
            assert key in res_train
            val = res_train[key]
            assert not torch.isnan(val)
            assert not torch.isinf(val)
            assert val.item() >= 0.0

        # Eval mode evaluation
        model.eval()
        out_eval = model(x, training=False)
        res_eval = loss_fn(out_eval, x, s=s)

        for key in ("loss", "bpp_loss", "y_bpp", "z_bpp", "distortion", "mse_loss"):
            assert key in res_eval
            val = res_eval[key]
            assert not torch.isnan(val)
            assert not torch.isinf(val)
            assert val.item() >= 0.0

    def test_end_to_end_gradient_backprop(self):
        torch.manual_seed(42)
        model = ModelB()
        model.train()
        loss_fn = RateDistortionLoss(lmbda=0.01)

        x = torch.rand(1, 3, 64, 64, requires_grad=False)
        s = torch.rand(1, 1, 64, 64)

        out = model(x, training=True)
        loss_dict = loss_fn(out, x, s=s)
        loss_val = loss_dict["loss"]

        loss_val.backward()

        # Check that gradients flow to all network transform parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient missing for parameter {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    def test_entropy_bottleneck_aux_loss_and_update(self):
        model = ModelB()
        aux = model.aux_loss()
        assert isinstance(aux, torch.Tensor)
        assert not torch.isnan(aux)

        # Update CDF tables
        res = model.update()
        assert "entropy_bottleneck" in res
        assert "gaussian_conditional" in res
        assert model.entropy_bottleneck._quantized_cdf.numel() > 0
        assert model.gaussian_conditional._quantized_cdf.numel() > 0


# -----------------------------------------------------------------------------
# 5. Mixed Precision and Numerical Isolation
# -----------------------------------------------------------------------------


class TestModelBPrecisionBoundaries:
    """Verifies BF16 autocast compatibility and numerical stability."""

    def test_autocast_bfloat16_execution(self):
        model = ModelB()
        model.eval()

        x = torch.rand(1, 3, 64, 64)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = model(x, training=False)

        # Reconstruction and likelihoods must remain in strict FP32
        assert out.x_hat.dtype == torch.float32
        assert out.y_likelihoods.dtype == torch.float32
        assert out.z_likelihoods.dtype == torch.float32

    def test_zero_and_flat_inputs(self):
        model = ModelB()
        model.eval()
        loss_fn = RateDistortionLoss(lmbda=0.01)

        # All zeros
        x_zero = torch.zeros(1, 3, 64, 64)
        out_zero = model(x_zero)
        loss_zero = loss_fn(out_zero, x_zero)
        assert not torch.isnan(loss_zero["loss"])

        # All ones
        x_ones = torch.ones(1, 3, 64, 64)
        out_ones = model(x_ones)
        loss_ones = loss_fn(out_ones, x_ones)
        assert not torch.isnan(loss_ones["loss"])
