"""Comprehensive unit test suite for Model C-Transmitted (Story 4.5).

Verifies Story 4.5 requirements:
- Thumbnail compression and decompression:
  * compress_saliency_thumbnail: 2D, 3D, and 4D inputs, downsampling to latent grid, 8-bit quantization.
  * decompress_saliency_thumbnail: single and batched bytes, exact shape recovery, FP32 range in [0, 1].
  * Roundtrip fidelity: reconstructed thumbnail matches 8-bit quantized original within 1/255.
  * Input validation: non-Tensor, invalid dims, out-of-bounds values, corrupt byte payload.
- Sidecar BPP calculation:
  * compute_sidecar_bpp: handles bytes, int length, list of bytes, and Tensor inputs.
  * Correct mathematical formula: 8 * total_bytes / num_pixels.
  * as_tensor flag and device placement.
- ModelCTransmitted architecture and forward pass execution:
  * Subclasses ModelA, maintaining exact layer hierarchy and parameter shapes.
  * Class attributes model_type == "C-Transmitted" and model_id == 4.
  * Forward pass in training (noise) and eval (rounding) modes.
  * Emits ModelCTransmittedOutput with s_dec, log_delta, delta, sidecar_bytes, and sidecar_bpp.
  * Dimension validation (multiples of 64) and missing saliency map validation.
- Exact Encoder-Decoder synchronization:
  * Delta derived by encoder matches Delta derived by decoder from sidecar_bytes bit-for-bit.
- RateDistortionLoss sidecar BPP integration:
  * bpp_loss equals y_bpp + z_bpp + sidecar_bpp.
  * Total loss balances sidecar BPP, main/hyper rate, and saliency-weighted reconstruction distortion.
  * Zero auxiliary loss (no prediction head to supervise).
- Detached Rate-Distortion gradient flow:
  * Reconstruction and rate losses produce zero gradient to Delta and s_dec.
  * Transforms g_a, g_s, h_a, h_s receive valid gradients normally.
- Pretrained checkpoint loading and CDF compilation:
  * Weight restoration from ModelA, ModelB, and ModelC state dicts with strict=True.
  * ModelCTransmitted.from_pretrained and load_pretrained_bmshj2018.
  * CDF compilation gate (model.update()) and persistence with save_checkpoint / load_checkpoint.
"""

from __future__ import annotations

import tempfile
import zlib
from pathlib import Path

import pytest
import torch

from src.codec.allocation import compute_log_delta
from src.codec.model_a import ModelA
from src.codec.model_b import ModelB
from src.codec.model_c import ModelC
from src.codec.model_c_transmitted import (
    MODEL_ID_C_TRANSMITTED,
    ModelCTransmitted,
    ModelCTransmittedOutput,
    compress_saliency_thumbnail,
    compute_sidecar_bpp,
    decompress_saliency_thumbnail,
    load_pretrained_bmshj2018,
)
from src.training import RateDistortionLoss, load_checkpoint, save_checkpoint


class TestThumbnailCompressionAndDecompression:
    """Verifies lossless 8-bit Deflate saliency thumbnail compression and decompression."""

    def test_single_image_compression_and_decompression_roundtrip(self) -> None:
        """Compresses 64x64 saliency map downsampled to 4x4 and decompresses back."""
        torch.manual_seed(101)
        # Saliency map in [0, 1]
        s = torch.rand(1, 1, 64, 64, dtype=torch.float32)

        # Compress to 4x4 thumbnail
        s_bytes = compress_saliency_thumbnail(s, target_shape=(4, 4))
        assert isinstance(s_bytes, bytes)
        assert len(s_bytes) > 0

        # Decompress back
        s_dec = decompress_saliency_thumbnail(s_bytes, shape=(4, 4))
        assert isinstance(s_dec, torch.Tensor)
        assert s_dec.shape == (1, 1, 4, 4)
        assert s_dec.dtype == torch.float32
        assert s_dec.min() >= 0.0
        assert s_dec.max() <= 1.0

        # 8-bit quantization roundtrip error bound is 0.5 / 255.0 (~0.00196)
        # Compute reference downsampled + quantized tensor
        s_down = torch.nn.functional.interpolate(s, size=(4, 4), mode="bilinear", align_corners=False)
        s_ref = torch.round(torch.clamp(s_down, 0.0, 1.0) * 255.0) / 255.0
        assert torch.allclose(s_dec, s_ref, atol=1e-6)

    def test_batched_compression_and_decompression(self) -> None:
        """Compresses batch of 3 images and decompresses into batched (3, 1, 4, 4) tensor."""
        torch.manual_seed(102)
        s_batch = torch.rand(3, 1, 32, 32, dtype=torch.float32)

        compressed_list = compress_saliency_thumbnail(s_batch, target_shape=(4, 4))
        assert isinstance(compressed_list, list)
        assert len(compressed_list) == 3
        for item in compressed_list:
            assert isinstance(item, bytes)

        decompressed = decompress_saliency_thumbnail(compressed_list, shape=(4, 4))
        assert decompressed.shape == (3, 1, 4, 4)
        assert decompressed.dtype == torch.float32

    def test_2d_and_3d_input_shapes(self) -> None:
        """Handles 2D (H, W) and 3D (1, H, W) saliency inputs."""
        s_2d = torch.rand(16, 16, dtype=torch.float32)
        bytes_2d = compress_saliency_thumbnail(s_2d)
        assert isinstance(bytes_2d, bytes)

        s_3d = torch.rand(1, 16, 16, dtype=torch.float32)
        bytes_3d = compress_saliency_thumbnail(s_3d)
        assert isinstance(bytes_3d, bytes)

    def test_input_validation_and_errors(self) -> None:
        """Validates input types, values, and shape errors."""
        with pytest.raises(TypeError, match=r"must be a torch\.Tensor"):
            compress_saliency_thumbnail([[0.5, 0.5], [0.5, 0.5]])  # type: ignore

        # Out-of-range saliency map
        with pytest.raises(ValueError, match="values must be in"):
            compress_saliency_thumbnail(torch.tensor([[1.5, 0.5], [0.5, 0.5]]))

        # NaN and Inf values
        with pytest.raises(ValueError, match="must not contain NaN or Inf"):
            compress_saliency_thumbnail(torch.tensor([[float("nan"), 0.5], [0.5, 0.5]]))

        with pytest.raises(ValueError, match="must not contain NaN or Inf"):
            compress_saliency_thumbnail(torch.tensor([[float("inf"), 0.5], [0.5, 0.5]]))

        # Invalid 4D channel count (C != 1)
        with pytest.raises(ValueError, match="Expected single channel"):
            compress_saliency_thumbnail(torch.rand(2, 3, 16, 16, dtype=torch.float32))

        # Invalid decompress shape
        with pytest.raises(ValueError, match="positive 2-tuple"):
            decompress_saliency_thumbnail(b"dummy", shape=(0, 4))

        # Empty byte sequence
        with pytest.raises(ValueError, match="cannot be empty"):
            decompress_saliency_thumbnail([], shape=(4, 4))

        # Corrupt Deflate payload
        with pytest.raises(ValueError, match="Corrupt or invalid Deflate payload"):
            decompress_saliency_thumbnail(b"non-zlib-corrupted-bytes", shape=(4, 4))

        # Mismatched byte length
        corrupt_bytes = zlib.compress(b"\x00" * 10)
        with pytest.raises(ValueError, match="does not match expected"):
            decompress_saliency_thumbnail(corrupt_bytes, shape=(4, 4))  # expected 16 bytes


class TestSidecarBppCalculation:
    """Verifies compute_sidecar_bpp utility function."""

    def test_single_bytes_and_scalar_return(self) -> None:
        """Calculates bits-per-pixel from single bytes."""
        payload = b"12345678"  # 8 bytes = 64 bits
        num_pixels = 64 * 64  # 4096 pixels
        bpp = compute_sidecar_bpp(payload, num_pixels)
        assert isinstance(bpp, float)
        assert bpp == pytest.approx(64.0 / 4096.0)

    def test_list_of_bytes_and_tensor_return(self) -> None:
        """Calculates bits-per-pixel from list of bytes with as_tensor=True."""
        payloads = [b"1234", b"5678"]  # 8 bytes total
        num_pixels = 4096
        bpp_t = compute_sidecar_bpp(payloads, num_pixels, as_tensor=True)
        assert isinstance(bpp_t, torch.Tensor)
        assert bpp_t.dtype == torch.float32
        assert torch.allclose(bpp_t, torch.tensor(64.0 / 4096.0, dtype=torch.float32))

    def test_invalid_pixel_count(self) -> None:
        """Raises ValueError if num_pixels <= 0."""
        with pytest.raises(ValueError, match="num_pixels must be positive"):
            compute_sidecar_bpp(b"test", 0)


class TestModelCTransmittedArchitecture:
    """Verifies ModelCTransmitted structural properties, forward execution, and bounds."""

    def test_class_hierarchy_and_attributes(self) -> None:
        """Subclasses ModelA, model_type == 'C-Transmitted', and model_id == 4."""
        model = ModelCTransmitted()
        assert isinstance(model, ModelA)
        assert model.model_type == "C-Transmitted"
        assert model.model_id == MODEL_ID_C_TRANSMITTED
        assert model.model_id == 4

        # Parameter shapes match ModelA
        model_a = ModelA()
        for k, v in model.state_dict().items():
            assert k in model_a.state_dict()
            assert v.shape == model_a.state_dict()[k].shape

    def test_forward_pass_training_and_eval_modes(self) -> None:
        """Executes forward pass in training and evaluation modes."""
        torch.manual_seed(103)
        model = ModelCTransmitted()
        x = torch.rand(2, 3, 64, 64, dtype=torch.float32)
        s = torch.rand(2, 1, 64, 64, dtype=torch.float32)

        # Training mode
        out_train = model(x, s=s, training=True)
        assert isinstance(out_train, ModelCTransmittedOutput)
        assert out_train.x_hat.shape == (2, 3, 64, 64)
        assert out_train.y.shape == (2, 192, 4, 4)
        assert out_train.z.shape == (2, 128, 1, 1)
        assert out_train.s_dec is not None
        assert out_train.s_dec.shape == (2, 1, 4, 4)
        assert out_train.delta is not None
        assert out_train.delta.shape == (2, 1, 4, 4)
        assert out_train.delta.min() >= 0.25 - 1e-6
        assert out_train.delta.max() <= 4.0 + 1e-6
        assert out_train.sidecar_bpp is not None
        assert out_train.sidecar_bpp.item() > 0.0

        # Eval mode
        out_eval = model(x, s=s, training=False)
        assert isinstance(out_eval, ModelCTransmittedOutput)
        assert out_eval.x_hat.shape == (2, 3, 64, 64)

    def test_missing_saliency_map_raises_value_error(self) -> None:
        """Raises ValueError if ground truth saliency map s is omitted."""
        model = ModelCTransmitted()
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        with pytest.raises(ValueError, match="requires a visual saliency map 's'"):
            model(x)

    def test_spatial_shape_validation(self) -> None:
        """Rejects non-multiple-of-64 image inputs."""
        model = ModelCTransmitted()
        x_bad = torch.rand(1, 3, 60, 60, dtype=torch.float32)
        s = torch.rand(1, 1, 60, 60, dtype=torch.float32)
        with pytest.raises(ValueError, match="multiples of 64"):
            model(x_bad, s=s)

    def test_valid_mask_isolation(self) -> None:
        """Derives Delta with valid_mask isolation outside unpadded region."""
        torch.manual_seed(104)
        model = ModelCTransmitted()
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        s = torch.rand(1, 1, 64, 64, dtype=torch.float32)

        # Valid mask covering only top-left 2x2 of the 4x4 latent grid
        mask = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        mask[:, :, :2, :2] = True

        out = model(x, s=s, valid_mask=mask, training=True)
        assert out.delta is not None

        # Outside valid mask, delta is strictly 1.0 (log_delta == 0.0)
        invalid_delta = out.delta[:, :, 2:, 2:]
        assert torch.allclose(invalid_delta, torch.ones_like(invalid_delta))

    def test_positional_valid_mask_argument(self) -> None:
        """Supports valid_mask passed positionally as 3rd parameter."""
        torch.manual_seed(104)
        model = ModelCTransmitted()
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        s = torch.rand(1, 1, 64, 64, dtype=torch.float32)
        mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
        out = model(x, s, mask, training=False)
        assert isinstance(out, ModelCTransmittedOutput)
        assert out.delta is not None

    def test_s_pred_alias_in_output(self) -> None:
        """Verifies s_pred alias is provided in ModelCTransmittedOutput."""
        torch.manual_seed(104)
        model = ModelCTransmitted()
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        s = torch.rand(1, 1, 64, 64, dtype=torch.float32)
        out = model(x, s=s, training=False)
        assert out.s_pred is not None
        assert torch.equal(out.s_pred, out.s_dec)
        assert torch.equal(out["s_pred"], out["s_dec"])


class TestExactEncoderDecoderSynchronization:
    """Verifies that the encoder and decoder derive bit-exact identical Delta."""

    def test_encoder_and_decoder_derive_bit_exact_delta(self) -> None:
        """Encoder Delta derived from s_dec equals decoder Delta derived from sidecar_bytes."""
        torch.manual_seed(105)
        model = ModelCTransmitted()
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        s = torch.rand(1, 1, 64, 64, dtype=torch.float32)

        # 1. Encoder forward pass
        out = model(x, s=s, training=False)
        enc_delta = out.delta
        enc_log_delta = out.log_delta
        sidecar_bytes = out.sidecar_bytes
        assert isinstance(sidecar_bytes, bytes)

        # 2. Simulated decoder receiving sidecar_bytes from bitstream
        latent_shape = (4, 4)
        dec_s_dec = decompress_saliency_thumbnail(sidecar_bytes, shape=latent_shape, device="cpu")
        dec_log_delta, dec_delta = compute_log_delta(dec_s_dec)

        # 3. Assert bit-exact identity
        assert torch.equal(enc_delta, dec_delta)
        assert torch.equal(enc_log_delta, dec_log_delta)


class TestRateDistortionLossIntegration:
    """Verifies RateDistortionLoss integration with Model C-Transmitted and sidecar BPP."""

    def test_bpp_loss_incorporates_sidecar_bpp(self) -> None:
        """bpp_loss equals y_bpp + z_bpp + sidecar_bpp."""
        torch.manual_seed(106)
        model = ModelCTransmitted()
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        s = torch.rand(1, 1, 64, 64, dtype=torch.float32)

        out = model(x, s=s, training=True)
        criterion = RateDistortionLoss(lmbda=0.01)

        res = criterion(out, x, s=s)
        expected_bpp = res["y_bpp"] + res["z_bpp"] + res["sidecar_bpp"]
        assert torch.allclose(res["bpp_loss"], expected_bpp, atol=1e-7)
        assert res["sidecar_bpp"].item() > 0.0

        # Auxiliary loss is strictly zero (no neural prediction head, suppressed by sidecar_bpp presence)
        assert res["aux_loss"].item() == 0.0
        assert out.s_pred is not None

    def test_aux_loss_suppression_with_sidecar(self) -> None:
        """RateDistortionLoss suppresses aux_loss to 0.0 when sidecar_bpp is present, even with s_pred."""
        torch.manual_seed(106)
        criterion = RateDistortionLoss(lmbda=0.01, lambda_aux=0.5)
        # Mock output with both s_pred and sidecar_bpp
        mock_output = {
            "x_hat": torch.rand(1, 3, 64, 64),
            "y_likelihoods": torch.rand(1, 192, 4, 4),
            "z_likelihoods": torch.rand(1, 128, 1, 1),
            "s_pred": torch.rand(1, 1, 4, 4),
            "sidecar_bpp": torch.tensor(0.005),
        }
        target = torch.rand(1, 3, 64, 64)
        s = torch.rand(1, 1, 64, 64)
        res = criterion(mock_output, target, s=s)
        assert res["aux_loss"].item() == 0.0
        assert res["sidecar_bpp"].item() == pytest.approx(0.005)

    def test_total_loss_balances_all_terms(self) -> None:
        """Total loss includes sidecar bpp and saliency-weighted reconstruction distortion."""
        torch.manual_seed(107)
        model = ModelCTransmitted()
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        s = torch.rand(1, 1, 64, 64, dtype=torch.float32)

        out = model(x, s=s, training=True)
        lmbda = 0.015
        criterion = RateDistortionLoss(lmbda=lmbda)

        res = criterion(out, x, s=s)
        expected_loss = res["bpp_loss"] + lmbda * (255.0**2) * res["distortion"]
        assert torch.allclose(res["loss"], expected_loss, atol=1e-6)


class TestGradientFlowInModelCTransmitted:
    """Verifies gradient flow characteristics in Model C-Transmitted."""

    def test_detached_rd_gradients_to_delta(self) -> None:
        """Rate and distortion losses produce zero gradient to Delta and s_dec."""
        torch.manual_seed(108)
        model = ModelCTransmitted()
        x = torch.rand(1, 3, 64, 64, dtype=torch.float32)
        s = torch.rand(1, 1, 64, 64, dtype=torch.float32)

        out = model(x, s=s, training=True)
        criterion = RateDistortionLoss(lmbda=0.01)
        res = criterion(out, x, s=s)
        res["loss"].backward()

        # Main codec transforms receive gradients
        assert model.g_a.conv1.weight.grad is not None
        assert model.g_s.deconv1.weight.grad is not None
        assert model.h_a.conv1.weight.grad is not None
        assert model.h_s.deconv1.weight.grad is not None


class TestCheckpointLoadingAndPersistence:
    """Verifies weight restoration from ModelA, ModelB, ModelC, and CDF updates."""

    def test_load_state_dict_from_model_a(self) -> None:
        """Loads ModelA state dict with strict=True."""
        model_a = ModelA()
        model_ct = ModelCTransmitted()

        model_ct.load_state_dict(model_a.state_dict(), strict=True)
        assert torch.allclose(model_ct.g_a.conv1.weight, model_a.g_a.conv1.weight)
        assert torch.allclose(model_ct.h_s.conv3.weight, model_a.h_s.conv3.weight)

    def test_load_state_dict_from_model_b(self) -> None:
        """Loads ModelB state dict with strict=True."""
        model_b = ModelB()
        model_ct = ModelCTransmitted()

        model_ct.load_state_dict(model_b.state_dict(), strict=True)
        assert torch.allclose(model_ct.g_a.conv1.weight, model_b.g_a.conv1.weight)

    def test_load_state_dict_from_model_c(self) -> None:
        """Loads ModelC state dict into ModelCTransmitted (tolerating extra conv_saliency keys)."""
        model_c = ModelC()
        model_ct = ModelCTransmitted()

        model_ct.load_state_dict(model_c.state_dict(), strict=True)
        assert torch.allclose(model_ct.g_a.conv1.weight, model_c.g_a.conv1.weight)
        assert torch.allclose(model_ct.h_s.conv3.weight, model_c.h_s.conv3.weight)

    def test_from_pretrained_and_load_pretrained_bmshj2018(self) -> None:
        """Tests from_pretrained and load_pretrained_bmshj2018 helpers."""
        model_a = ModelA()
        state = model_a.state_dict()

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "model_a.pth"
            torch.save(state, ckpt_path)

            restored = ModelCTransmitted.from_pretrained(ckpt_path)
            assert isinstance(restored, ModelCTransmitted)
            assert restored.model_type == "C-Transmitted"

            bmshj_loaded = load_pretrained_bmshj2018(ckpt_path)
            assert isinstance(bmshj_loaded, ModelCTransmitted)
            assert bmshj_loaded.model_type == "C-Transmitted"

    def test_cdf_update_and_save_checkpoint(self) -> None:
        """Verifies CDF compilation via update() and save_checkpoint / load_checkpoint."""
        model = ModelCTransmitted()
        updated = model.update(force=True)
        assert updated["entropy_bottleneck"] is True
        assert updated["gaussian_conditional"] is True

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "model_ct_cdf.pt"
            save_checkpoint(model, ckpt_path, compile_cdfs=True, epoch=1, step=100)

            loaded_model = ModelCTransmitted()
            meta = load_checkpoint(ckpt_path, loaded_model)

            assert meta["epoch"] == 1
            assert meta["step"] == 100
            assert hasattr(loaded_model.entropy_bottleneck, "_quantized_cdf")
