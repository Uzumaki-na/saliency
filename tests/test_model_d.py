"""Comprehensive unit test suite for Model D (mechanism-only architecture control variant).

Verifies Story 4.4 requirements:
- ModelD architecture and forward pass execution:
  * Subclasses ModelC and ModelA, maintaining exact parameter shapes and layer hierarchy.
  * Class attribute model_type == "D".
  * Dual operational modes (training noise vs evaluation rounding).
  * Explicit Delta allocation and effective scale sigma_eff computation.
  * Spatial shape validation (multiples of 64).
  * Valid mask isolation and Oracle ground truth saliency override.
- RateDistortionLoss uniform MSE control:
  * Accepts saliency_weighting boolean parameter (default True).
  * When saliency_weighting=False: evaluates reconstruction distortion as unweighted MSE
    (S'_{ij} = 1.0), while still computing and logging auxiliary saliency loss L_aux with
    constant lambda_aux = 0.1.
  * Returns saliency_weighting in dictionary.
- Detached Rate-Distortion gradient flow:
  * Reconstruction and rate losses produce zero gradients to Delta, s_pred, and conv_saliency.
  * Auxiliary loss trains conv_saliency, h_s trunk, and h_a, while g_a receives strictly zero
    gradients when detach_y_for_ha=True.
- Hyperprior bitrate overhead tracking:
  * compute_hyperprior_overhead accurately computes Delta BPP_z = BPP_z(Model) - BPP_z(Baseline).
  * Supports float and Tensor inputs, as well as as_tensor output flag.
- Pretrained checkpoint loading and CDF compilation:
  * Weight restoration from ModelA, ModelB, and ModelC checkpoints.
  * ModelD.from_pretrained and load_pretrained_bmshj2018.
  * CDF compilation gate (model.update()) and persistence with save_checkpoint / load_checkpoint.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from src.codec import (
    DualHeadHyperSynthesisTransform,
    ModelA,
    ModelB,
    ModelC,
    ModelCOutput,
)
from src.codec.model_d import (
    ModelD,
    compute_hyperprior_overhead,
    load_pretrained_bmshj2018,
)
from src.training import RateDistortionLoss, load_checkpoint, save_checkpoint
from src.training.loss import auxiliary_saliency_loss


class TestModelDArchitecture:
    """Tests for ModelD class structure, attributes, and forward pass."""

    def test_inheritance_and_class_attributes(self) -> None:
        """ModelD subclasses ModelC and ModelA with model_type == 'D'."""
        assert issubclass(ModelD, ModelC)
        assert issubclass(ModelD, ModelA)

        model = ModelD()
        assert model.model_type == "D"
        assert isinstance(model.h_s, DualHeadHyperSynthesisTransform)
        assert model.detach_y_for_ha is True

    def test_identical_parameter_structure_to_model_c(self) -> None:
        """ModelD shares identical state dict keys and parameter shapes with ModelC."""
        model_c = ModelC()
        model_d = ModelD()

        keys_c = set(model_c.state_dict().keys())
        keys_d = set(model_d.state_dict().keys())
        assert keys_c == keys_d

        for key in keys_c:
            shape_c = tuple(model_c.state_dict()[key].shape)
            shape_d = tuple(model_d.state_dict()[key].shape)
            assert shape_c == shape_d, f"Shape mismatch for key {key}: {shape_c} vs {shape_d}"

    def test_forward_pass_training_and_eval_modes(self) -> None:
        """ModelD forward pass executes in training mode (noise) and eval mode (rounding)."""
        model = ModelD()
        x = torch.rand(2, 3, 64, 64, dtype=torch.float32)

        # Training mode: uses uniform noise relaxation
        out_train = model(x, training=True)
        assert isinstance(out_train, ModelCOutput)
        assert out_train.x_hat.shape == (2, 3, 64, 64)
        assert out_train.y.shape == (2, 192, 4, 4)
        assert out_train.z.shape == (2, 128, 1, 1)
        assert out_train.log_sigma.shape == (2, 192, 4, 4)
        assert out_train.log_sigma_eff.shape == (2, 192, 4, 4)
        assert out_train.s_pred.shape == (2, 1, 4, 4)
        assert out_train.log_delta.shape == (2, 1, 4, 4)
        assert out_train.delta.shape == (2, 1, 4, 4)
        assert out_train.y_likelihoods.shape == (2, 192, 4, 4)
        assert out_train.z_likelihoods.shape == (2, 128, 1, 1)

        # Evaluation mode: uses integer rounding
        out_eval = model(x, training=False)
        assert isinstance(out_eval, ModelCOutput)
        assert out_eval.x_hat.shape == (2, 3, 64, 64)
        assert (out_eval.s_pred >= 0.0).all()
        assert (out_eval.s_pred <= 1.0).all()
        assert (out_eval.delta >= 0.25).all()
        assert (out_eval.delta <= 4.0).all()

    def test_oracle_saliency_override(self) -> None:
        """Supports ground-truth saliency map override (Oracle baseline)."""
        model = ModelD()
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        # Construct contrasting saliency map (upper half salient, lower half non-salient)
        oracle_s = torch.zeros(1, 1, 64, 64, dtype=torch.float32)
        oracle_s[:, :, :32, :] = 1.0

        out_normal = model(x, training=False)
        out_oracle = model(x, saliency_map=oracle_s, training=False)

        # S_pred is still synthesized by h_s
        assert out_oracle.s_pred.shape == (1, 1, 4, 4)
        # Delta maps must differ because Oracle allocation used oracle_s
        assert not torch.allclose(out_normal.delta, out_oracle.delta)
        # Upper half (salient) receives smaller delta, lower half (non-salient) receives larger delta
        assert torch.all(out_oracle.delta[:, :, :2, :] < 1.0)
        assert torch.all(out_oracle.delta[:, :, 2:, :] > 1.0)

    def test_valid_mask_isolation(self) -> None:
        """Forces log_delta == 0 and delta == 1 outside valid mask."""
        model = ModelD()
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        mask = torch.zeros(4, 4, dtype=torch.bool)
        mask[:2, :2] = True

        out = model(x, valid_mask=mask)
        padded_region = ~mask
        assert torch.all(out.log_delta[0, 0][padded_region] == 0.0)
        assert torch.all(out.delta[0, 0][padded_region] == 1.0)

    def test_input_validation(self) -> None:
        """Rejects non-4D tensors, channel mismatches, and dimensions not multiples of 64."""
        model = ModelD()
        with pytest.raises(ValueError, match="Expected 4D input tensor"):
            model(torch.rand(3, 64, 64))
        with pytest.raises(ValueError, match="Channel mismatch"):
            model(torch.rand(1, 1, 64, 64))
        with pytest.raises(ValueError, match="must be multiples of 64"):
            model(torch.rand(1, 3, 60, 60))


class TestRateDistortionLossControl:
    """Verifies RateDistortionLoss with saliency_weighting=False uniform MSE control."""

    def test_init_parameters(self) -> None:
        """Accepts saliency_weighting flag, defaulting to True."""
        loss_default = RateDistortionLoss(lmbda=0.01, lambda_aux=0.1)
        assert loss_default.saliency_weighting is True

        loss_unweighted = RateDistortionLoss(lmbda=0.01, lambda_aux=0.1, saliency_weighting=False)
        assert loss_unweighted.saliency_weighting is False

    def test_uniform_mse_distortion_matches_unweighted_mse(self) -> None:
        """When saliency_weighting=False, distortion evaluates to unweighted MSE."""
        loss_fn = RateDistortionLoss(lmbda=0.02, lambda_aux=0.1, saliency_weighting=False)
        target = torch.rand(2, 3, 64, 64, dtype=torch.float32)
        s_target = torch.rand(2, 1, 64, 64, dtype=torch.float32)

        model = ModelD()
        out = model(target, training=True)

        res = loss_fn(out, target, s=s_target)

        assert res["saliency_weighting"] is False
        # Distortion must equal unweighted MSE
        assert torch.allclose(res["distortion"], res["mse_loss"], atol=1e-7)

        # Direct MSE calculation over image
        expected_mse = torch.mean((out.x_hat - target) ** 2)
        assert torch.allclose(res["distortion"], expected_mse, atol=1e-6)

        # Auxiliary loss must still be computed and non-zero
        assert res["aux_loss"] > 0.0
        expected_aux = auxiliary_saliency_loss(out.s_pred, s_target)
        assert torch.allclose(res["aux_loss"], expected_aux, atol=1e-6)

        # Total loss calculation check
        expected_total = res["bpp_loss"] + 0.02 * (255.0**2) * res["distortion"] + 0.1 * res["aux_loss"]
        assert torch.allclose(res["loss"], expected_total, atol=1e-5)

    def test_comparison_between_weighted_and_unweighted_modes(self) -> None:
        """Saliency weighting creates measurable difference in distortion for non-uniform maps."""
        loss_weighted = RateDistortionLoss(lmbda=0.01, lambda_aux=0.1, saliency_weighting=True)
        loss_unweighted = RateDistortionLoss(lmbda=0.01, lambda_aux=0.1, saliency_weighting=False)

        target = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        # Concentrated saliency in center
        s_target = torch.zeros(1, 1, 64, 64, dtype=torch.float32)
        s_target[:, :, 20:44, 20:44] = 1.0

        model = ModelD()
        out = model(target, training=True)

        res_weighted = loss_weighted(out, target, s=s_target)
        res_unweighted = loss_unweighted(out, target, s=s_target)

        assert res_weighted["saliency_weighting"] is True
        assert res_unweighted["saliency_weighting"] is False

        # Unweighted distortion == mse_loss
        assert torch.allclose(res_unweighted["distortion"], res_unweighted["mse_loss"], atol=1e-7)

        # In weighted mode, distortion != mse_loss because S' != 1 everywhere
        assert not torch.allclose(res_weighted["distortion"], res_weighted["mse_loss"], atol=1e-4)

        # Both modes evaluate the exact same auxiliary saliency loss
        assert torch.allclose(res_weighted["aux_loss"], res_unweighted["aux_loss"], atol=1e-7)

    def test_per_call_saliency_weighting_override(self) -> None:
        """Supports per-call dynamic override of saliency_weighting in forward."""
        loss_fn = RateDistortionLoss(lmbda=0.01, lambda_aux=0.1, saliency_weighting=False)
        target = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        s_target = torch.zeros(1, 1, 64, 64, dtype=torch.float32)
        s_target[:, :, 16:48, 16:48] = 1.0

        model = ModelD()
        out = model(target, training=True)

        # Default is saliency_weighting=False
        res_default = loss_fn(out, target, s=s_target)
        assert res_default["saliency_weighting"] is False
        assert torch.allclose(res_default["distortion"], res_default["mse_loss"], atol=1e-7)

        # Override to saliency_weighting=True per-call
        res_override = loss_fn(out, target, s=s_target, saliency_weighting=True)
        assert res_override["saliency_weighting"] is True
        assert not torch.allclose(res_override["distortion"], res_override["mse_loss"], atol=1e-4)


class TestGradientFlowInModelD:
    """Verifies gradient flow characteristics in ModelD under mechanism-only training."""

    def test_detached_rd_gradients_with_unweighted_distortion(self) -> None:
        """Rate and uniform MSE distortion produce zero gradient to saliency head."""
        torch.manual_seed(42)
        model = ModelD()
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)

        out = model(x, training=True)

        # Pure rate-distortion loss without auxiliary loss
        distortion = torch.mean((out.x_hat - x) ** 2)
        rate = -torch.sum(torch.log2(out.y_likelihoods)) / (64 * 64)
        rd_loss = rate + 0.01 * (255.0**2) * distortion
        rd_loss.backward()

        # RD gradients into S_pred and conv_saliency MUST be None
        assert model.h_s.conv_saliency.weight.grad is None
        assert model.h_s.conv_saliency.bias.grad is None

        # Main transforms receive gradients normally
        assert model.g_a.conv1.weight.grad is not None
        assert model.g_s.deconv1.weight.grad is not None
        assert model.h_s.conv3.weight.grad is not None

    def test_auxiliary_loss_supervises_saliency_head_with_ga_cut(self) -> None:
        """Auxiliary loss trains conv_saliency and h_a, but cuts gradients before g_a."""
        torch.manual_seed(43)
        model = ModelD(detach_y_for_ha=True)
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        s_target = torch.rand(1, 1, 4, 4, dtype=torch.float32)

        out = model(x, training=True)
        aux_loss = auxiliary_saliency_loss(out.s_pred, s_target)
        aux_loss.backward()

        saliency_grad = model.h_s.conv_saliency.weight.grad
        assert saliency_grad is not None
        assert torch.isfinite(saliency_grad).all()
        assert not torch.allclose(saliency_grad, torch.zeros_like(saliency_grad))

        # Hyper-transforms receive gradients
        assert model.h_s.deconv1.weight.grad is not None
        assert model.h_s.deconv2.weight.grad is not None
        assert model.h_a.conv3.weight.grad is not None

        # Main analysis transform g_a receives ZERO gradient due to y.detach()
        assert model.g_a.conv1.weight.grad is None
        assert model.g_a.conv4.weight.grad is None

    def test_end_to_end_combined_loss_backward(self) -> None:
        """Simultaneous backpropagation of combined RD and auxiliary loss."""
        torch.manual_seed(44)
        model = ModelD(detach_y_for_ha=True)
        criterion = RateDistortionLoss(lmbda=0.01, lambda_aux=0.1, saliency_weighting=False)

        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        s_target = torch.rand(1, 1, 4, 4, dtype=torch.float32)

        out = model(x, training=True)
        loss_dict = criterion(out, x, s=s_target)
        loss_dict["loss"].backward()

        # h_s.conv_saliency receives gradients exclusively from aux_loss
        assert model.h_s.conv_saliency.weight.grad is not None
        assert torch.isfinite(model.h_s.conv_saliency.weight.grad).all()

        # g_a receives gradients from RD loss, but zero contribution from aux_loss
        assert model.g_a.conv1.weight.grad is not None
        assert torch.isfinite(model.g_a.conv1.weight.grad).all()


class TestHyperpriorBitrateOverhead:
    """Verifies compute_hyperprior_overhead utility function."""

    def test_scalar_float_inputs(self) -> None:
        """Computes Delta BPP_z from Python float inputs."""
        overhead = compute_hyperprior_overhead(0.045, 0.030)
        assert isinstance(overhead, float)
        assert overhead == pytest.approx(0.015)

    def test_tensor_inputs(self) -> None:
        """Computes Delta BPP_z from torch.Tensor inputs."""
        z_model = torch.tensor(0.052, dtype=torch.float32)
        z_baseline = torch.tensor(0.031, dtype=torch.float32)

        overhead = compute_hyperprior_overhead(z_model, z_baseline)
        assert isinstance(overhead, float)
        assert overhead == pytest.approx(0.021)

    def test_mixed_inputs_and_as_tensor_flag(self) -> None:
        """Supports mixed float/Tensor inputs and as_tensor=True output."""
        z_model = torch.tensor(0.060, dtype=torch.float32)
        z_baseline = 0.040

        overhead_t = compute_hyperprior_overhead(z_model, z_baseline, as_tensor=True)
        assert isinstance(overhead_t, torch.Tensor)
        assert overhead_t.dtype == torch.float32
        assert torch.allclose(overhead_t, torch.tensor(0.020, dtype=torch.float32))

        overhead_f = compute_hyperprior_overhead(0.060, 0.040, as_tensor=True)
        assert isinstance(overhead_f, torch.Tensor)
        assert torch.allclose(overhead_f, torch.tensor(0.020, dtype=torch.float32))

    def test_negative_overhead(self) -> None:
        """Handles negative overhead when model hyper-bitrate is smaller than baseline."""
        overhead = compute_hyperprior_overhead(0.025, 0.035)
        assert overhead == pytest.approx(-0.010)

    def test_asymmetric_types_and_device_alignment(self) -> None:
        """Correctly resolves device when first argument is float and second is Tensor."""
        z_model = 0.045
        z_baseline = torch.tensor(0.030, dtype=torch.float32)

        overhead = compute_hyperprior_overhead(z_model, z_baseline, as_tensor=True)
        assert isinstance(overhead, torch.Tensor)
        assert overhead.device == z_baseline.device
        assert torch.allclose(overhead, torch.tensor(0.015, dtype=torch.float32))

    def test_empty_tensor_raises_value_error(self) -> None:
        """Raises ValueError if either tensor argument is empty."""
        empty_t = torch.tensor([], dtype=torch.float32)
        valid_t = torch.tensor(0.030, dtype=torch.float32)

        with pytest.raises(ValueError, match="cannot be empty"):
            compute_hyperprior_overhead(empty_t, valid_t)

        with pytest.raises(ValueError, match="cannot be empty"):
            compute_hyperprior_overhead(valid_t, empty_t)


class TestCheckpointLoadingAndPersistence:
    """Verifies weight restoration from ModelA, ModelB, ModelC, and CDF updates."""

    def test_load_state_dict_from_model_a(self) -> None:
        """Loads ModelA state dict with strict=True (tolerating freshly initialized conv_saliency)."""
        model_a = ModelA()
        model_d = ModelD()

        # Load ModelA weights into ModelD
        model_d.load_state_dict(model_a.state_dict(), strict=True)

        # Verify weights match for shared layers
        assert torch.allclose(model_d.g_a.conv1.weight, model_a.g_a.conv1.weight)
        assert torch.allclose(model_d.h_s.conv3.weight, model_a.h_s.conv3.weight)

    def test_load_state_dict_from_model_b(self) -> None:
        """Loads ModelB state dict into ModelD."""
        model_b = ModelB()
        model_d = ModelD()

        model_d.load_state_dict(model_b.state_dict(), strict=True)
        assert torch.allclose(model_d.g_a.conv1.weight, model_b.g_a.conv1.weight)
        assert torch.allclose(model_d.h_s.conv3.weight, model_b.h_s.conv3.weight)

    def test_load_state_dict_from_model_c(self) -> None:
        """Loads ModelC state dict into ModelD with 100% parameter match."""
        model_c = ModelC()
        model_d = ModelD()

        model_d.load_state_dict(model_c.state_dict(), strict=True)
        assert torch.allclose(model_d.h_s.conv_saliency.weight, model_c.h_s.conv_saliency.weight)
        assert torch.allclose(model_d.h_s.conv3.weight, model_c.h_s.conv3.weight)

    def test_from_pretrained_and_load_pretrained_bmshj2018(self) -> None:
        """Tests from_pretrained and load_pretrained_bmshj2018 helper functions."""
        model_c = ModelC()
        state = model_c.state_dict()

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "model_c.pth"
            torch.save(state, ckpt_path)

            # Test ModelD.from_pretrained
            restored = ModelD.from_pretrained(ckpt_path)
            assert isinstance(restored, ModelD)
            assert restored.model_type == "D"
            assert torch.allclose(restored.h_s.conv_saliency.weight, model_c.h_s.conv_saliency.weight)

            # Test load_pretrained_bmshj2018 instantiating ModelD
            bmshj_loaded = load_pretrained_bmshj2018(ckpt_path)
            assert isinstance(bmshj_loaded, ModelD)
            assert bmshj_loaded.model_type == "D"

    def test_cdf_update_and_save_checkpoint(self) -> None:
        """Verifies CDF table compilation via update() and save_checkpoint / load_checkpoint."""
        model = ModelD()
        # Compile CDFs
        updated = model.update(force=True)
        assert updated["entropy_bottleneck"] is True
        assert updated["gaussian_conditional"] is True
        assert hasattr(model.entropy_bottleneck, "_quantized_cdf")
        assert model.entropy_bottleneck._quantized_cdf.numel() > 0

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "model_d_cdf.pt"
            save_checkpoint(model, ckpt_path, compile_cdfs=True, epoch=1, step=100)

            loaded_model = ModelD()
            meta = load_checkpoint(ckpt_path, loaded_model)

            assert meta["epoch"] == 1
            assert meta["step"] == 100
            assert hasattr(loaded_model.entropy_bottleneck, "_quantized_cdf")
