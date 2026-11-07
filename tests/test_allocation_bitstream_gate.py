"""Extended Bitstream Acceptance Gate for Models B, C, D, and C-Transmitted (Story 4.6).

Verifies across the Kodak PhotoCD benchmark:
1. Exact integer latent symbol equality:
   max |q_z - hat_q_z| == 0 and max |q_y - hat_q_y| == 0.
2. Exact scale index equality:
   max |scale_idx_enc - scale_idx_dec| == 0.
3. Exact adaptive quantization step Delta derivation:
   max |delta_enc - delta_dec| == 0 and max |log_delta_enc - log_delta_dec| == 0.
4. Exact de-quantized latent identity:
   max |y_hat_enc - y_hat_dec| == 0.
5. Reconstruction consistency:
   PSNR > 60.0 dB and max |x_enc_ref - x_dec| < 1e-4.
6. Isolated process execution:
   GPU/CPU encode -> .sanc file on disk -> isolated CPU decode subprocess with checkpoints.
7. Multi-ISA verification asserting bit-identical reconstruction across ATEN_CPU_CAPABILITY ('default' vs 'avx2').
8. Non-multiple-of-64 dimension crop verification (500x375) with exact dimension recovery and PSNR > 60.0 dB.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.bitstream.container import (
    SancContainer,
    create_model,
    decode_image,
    encode_image,
    load_image_tensor,
)
from src.codec.allocation import compute_log_delta
from src.codec.model_a import ModelA
from src.codec.model_b import ModelB
from src.codec.model_c import ModelC
from src.codec.model_c_transmitted import (
    ModelCTransmitted,
    compress_saliency_thumbnail,
    decompress_saliency_thumbnail,
)
from src.codec.model_d import ModelD
from src.data.padding import compute_latent_valid_mask, pad_symmetric_centered, unpad_centered
from src.entropy.coder import decode_hyper_latents, decode_latents
from src.entropy.cpu_engine import DeterministicCPUEngine

KODAK_DIR = Path("data/raw/kodak")


def compute_psnr(x1: torch.Tensor, x2: torch.Tensor) -> float:
    r"""Computes Peak Signal-to-Noise Ratio (dB) between two image tensors in [0, 1]."""
    diff = (x1 - x2).float()
    mse = torch.mean(diff**2).item()
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


@pytest.fixture(scope="module")
def shared_model_b() -> ModelB:
    r"""Module-scoped updated ModelB instance."""
    torch.manual_seed(400)
    model = ModelB()
    model.update()
    model.eval()
    return model


@pytest.fixture(scope="module")
def shared_model_c() -> ModelC:
    r"""Module-scoped updated ModelC instance."""
    torch.manual_seed(401)
    model = ModelC()
    model.update()
    model.eval()
    return model


@pytest.fixture(scope="module")
def shared_model_d() -> ModelD:
    r"""Module-scoped updated ModelD instance."""
    torch.manual_seed(402)
    model = ModelD()
    model.update()
    model.eval()
    return model


@pytest.fixture(scope="module")
def shared_model_c_transmitted() -> ModelCTransmitted:
    r"""Module-scoped updated ModelCTransmitted instance."""
    torch.manual_seed(403)
    model = ModelCTransmitted()
    model.update()
    model.eval()
    return model


@pytest.fixture(scope="module")
def shared_checkpoints(
    shared_model_b: ModelB,
    shared_model_c: ModelC,
    shared_model_d: ModelD,
    shared_model_c_transmitted: ModelCTransmitted,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Path]:
    r"""Module-scoped checkpoint paths for isolated subprocess tests."""
    temp_dir = tmp_path_factory.mktemp("allocation_checkpoints")
    ckpt_b = temp_dir / "model_b.pt"
    ckpt_c = temp_dir / "model_c.pt"
    ckpt_d = temp_dir / "model_d.pt"
    ckpt_c_tran = temp_dir / "model_c_transmitted.pt"

    torch.save({"state_dict": shared_model_b.state_dict()}, ckpt_b)
    torch.save({"state_dict": shared_model_c.state_dict()}, ckpt_c)
    torch.save({"state_dict": shared_model_d.state_dict()}, ckpt_d)
    torch.save({"state_dict": shared_model_c_transmitted.state_dict()}, ckpt_c_tran)

    return {
        "B": ckpt_b,
        "C": ckpt_c,
        "D": ckpt_d,
        "C-Transmitted": ckpt_c_tran,
    }


class TestKodakAllocationBitstreamGate:
    r"""Extended Bitstream Acceptance Gate for Models B, C, D, and C-Transmitted on Kodak benchmark."""

    def test_kodak_dataset_availability(self) -> None:
        r"""Verifies all 24 Kodak benchmark images are present."""
        assert KODAK_DIR.is_dir(), f"Kodak directory not found: {KODAK_DIR}"
        images = sorted(KODAK_DIR.glob("kodim*.png"))
        assert len(images) == 24, f"Expected 24 Kodak images, found {len(images)}"

    @pytest.mark.parametrize("model_key", ["B", "C", "D", "C-Transmitted"])
    @pytest.mark.parametrize("img_idx", [1, 2, 3, 5, 8, 12])
    def test_kodak_symbol_scale_and_delta_equality(
        self,
        model_key: str,
        img_idx: int,
        shared_model_b: ModelB,
        shared_model_c: ModelC,
        shared_model_d: ModelD,
        shared_model_c_transmitted: ModelCTransmitted,
    ) -> None:
        r"""Validates exact symbol equality, scale index equality, Delta identity, and PSNR > 60 dB."""
        img_name = f"kodim{img_idx:02d}.png"
        img_path = KODAK_DIR / img_name
        assert img_path.is_file(), f"Image not found: {img_path}"

        if model_key == "B":
            model: ModelA = shared_model_b
        elif model_key == "C":
            model = shared_model_c
        elif model_key == "D":
            model = shared_model_d
        else:
            model = shared_model_c_transmitted

        device = "cuda" if torch.cuda.is_available() else "cpu"
        target_dev = torch.device(device)
        model.to(target_dev).eval()

        x = load_image_tensor(img_path)
        x_pad, bounds = pad_symmetric_centered(x, factor=64)
        x_pad_dev = x_pad.to(target_dev)

        # 1. Forward analysis transforms
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    y = model.g_a(x_pad_dev)
                    z = model.h_a(y)
            else:
                y = model.g_a(x_pad_dev)
                z = model.h_a(y)

        q_z_enc = torch.round(z).cpu()

        # 2. Derive reference encoder allocation and scales on CPU in FP32
        cpu_engine = DeterministicCPUEngine.from_model(model)
        valid_mask = compute_latent_valid_mask(orig_shape=(x.shape[-2], x.shape[-1]), pad_bounds=bounds, device="cpu")

        saliency_input: torch.Tensor | None = None
        if model_key == "B":
            _log_sigma_enc, scale_indexes_enc = cpu_engine.forward(q_z_enc)
            delta_enc = torch.ones((1, 1, y.shape[-2], y.shape[-1]), dtype=torch.float32)
            log_delta_enc = torch.zeros((1, 1, y.shape[-2], y.shape[-1]), dtype=torch.float32)
            delta_dev = torch.ones_like(y, device=target_dev, dtype=torch.float32)
            q_y_enc = torch.round(y.float()).cpu()
        elif model_key in ("C", "D"):
            (
                _log_sigma_enc,
                scale_indexes_enc,
                delta_enc,
                log_delta_enc,
                _s_pred_enc,
            ) = cpu_engine.forward_allocation(q_z_enc, valid_mask=valid_mask)
            delta_dev = delta_enc.to(device=target_dev, dtype=torch.float32)
            q_y_enc = torch.round(y.float() / delta_dev).cpu()
        else:
            # Model C-Transmitted: generate reference visual saliency map
            torch.manual_seed(img_idx * 17)
            saliency_input = torch.rand(1, 1, x.shape[-2], x.shape[-1], dtype=torch.float32)
            s_pad, _ = pad_symmetric_centered(saliency_input, pad_bounds=bounds)
            latent_shape = (y.shape[-2], y.shape[-1])
            comp_bytes = compress_saliency_thumbnail(s_pad, target_shape=latent_shape)
            ref_sidecar_bytes = comp_bytes[0] if isinstance(comp_bytes, list) else comp_bytes
            s_dec_enc = decompress_saliency_thumbnail(ref_sidecar_bytes, shape=latent_shape, device="cpu")
            log_delta_enc, delta_enc = compute_log_delta(s_dec_enc, valid_mask=valid_mask)
            _log_sigma_enc, scale_indexes_enc = cpu_engine.forward(q_z_enc, log_delta=log_delta_enc)
            delta_dev = delta_enc.to(device=target_dev, dtype=torch.float32)
            q_y_enc = torch.round(y.float() / delta_dev).cpu()

        # Encoder synthesis reference
        y_hat_enc_ref = q_y_enc.to(target_dev).float() * delta_dev
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    x_pad_enc_ref = model.g_s(y_hat_enc_ref).cpu()
            else:
                x_pad_enc_ref = model.g_s(y_hat_enc_ref).cpu()
        x_enc_ref = torch.clamp(unpad_centered(x_pad_enc_ref, bounds), 0.0, 1.0)

        # 3. Full encode to SancContainer
        container = encode_image(
            x,
            model,
            device=device,
            saliency_map=saliency_input,
        )

        assert container.model_id == model.model_id
        if model_key == "C-Transmitted":
            assert container.sidecar_bitstream is not None
            assert len(container.sidecar_bitstream) > 0
        else:
            assert container.sidecar_bitstream is None

        # 4. Independent decode via SancContainer bitstream
        h_pad = container.orig_height + container.pad_bounds.pad_top + container.pad_bounds.pad_bottom
        w_pad = container.orig_width + container.pad_bounds.pad_left + container.pad_bounds.pad_right
        h_z = h_pad // 64
        w_z = w_pad // 64
        h_y = h_pad // 16
        w_y = w_pad // 16

        # Hyper-latent entropy decoding
        q_z_dec = decode_hyper_latents(container.z_bitstream, (1, 128, h_z, w_z), model.entropy_bottleneck)

        # Decoder parameter and allocation derivation
        if model_key == "B":
            _log_sigma_dec, scale_indexes_dec = cpu_engine.forward(q_z_dec)
            delta_dec = torch.ones((1, 1, h_y, w_y), dtype=torch.float32)
            log_delta_dec = torch.zeros((1, 1, h_y, w_y), dtype=torch.float32)
        elif model_key in ("C", "D"):
            (
                _log_sigma_dec,
                scale_indexes_dec,
                delta_dec,
                log_delta_dec,
                _s_pred_dec,
            ) = cpu_engine.forward_allocation(q_z_dec, valid_mask=valid_mask)
        else:
            assert container.sidecar_bitstream is not None
            s_dec_dec = decompress_saliency_thumbnail(container.sidecar_bitstream, shape=(h_y, w_y), device="cpu")
            log_delta_dec, delta_dec = compute_log_delta(s_dec_dec, valid_mask=valid_mask)
            _log_sigma_dec, scale_indexes_dec = cpu_engine.forward(q_z_dec, log_delta=log_delta_dec)

        # Main latent entropy decoding
        q_y_dec = decode_latents(container.y_bitstream, scale_indexes_dec, model.gaussian_conditional)

        # Full image decode
        x_dec = decode_image(container, model, device=device).cpu()

        # --- Invariant Assertions ---
        # A. Exact hyper-latent symbol equality: max |q_z - hat_q_z| == 0
        diff_qz = torch.abs(q_z_enc - q_z_dec).max().item()
        assert diff_qz == 0.0, f"[{model_key} - {img_name}] Hyper-latent symbol mismatch: {diff_qz}"

        # B. Exact scale index equality: max |scale_idx_enc - scale_idx_dec| == 0
        diff_idx = torch.abs(scale_indexes_enc - scale_indexes_dec).max().item()
        assert diff_idx == 0, f"[{model_key} - {img_name}] Scale index mismatch: {diff_idx}"

        # C. Exact adaptive quantization step Delta derivation: max |delta_enc - delta_dec| == 0
        diff_delta = torch.abs(delta_enc - delta_dec).max().item()
        assert diff_delta == 0.0, f"[{model_key} - {img_name}] Delta mismatch: {diff_delta}"

        diff_log_delta = torch.abs(log_delta_enc - log_delta_dec).max().item()
        assert diff_log_delta == 0.0, f"[{model_key} - {img_name}] Log-Delta mismatch: {diff_log_delta}"

        # D. Exact main latent symbol equality: max |q_y - hat_q_y| == 0
        diff_qy = torch.abs(q_y_enc - q_y_dec).max().item()
        assert diff_qy == 0.0, f"[{model_key} - {img_name}] Main latent symbol mismatch: {diff_qy}"

        # D2. Exact de-quantized latent identity: max |y_hat_enc_ref - y_hat_dec| == 0
        y_hat_dec = q_y_dec.to(target_dev).float() * delta_dec.to(target_dev).float()
        diff_y_hat = torch.abs(y_hat_enc_ref.float() - y_hat_dec).max().item()
        assert diff_y_hat == 0.0, f"[{model_key} - {img_name}] De-quantized latent mismatch: {diff_y_hat}"

        # E. Reconstruction consistency: PSNR > 60.0 dB and max |x_enc_ref - x_dec| < 1e-4
        psnr = compute_psnr(x_enc_ref, x_dec)
        assert psnr > 60.0, f"[{model_key} - {img_name}] Insufficient PSNR {psnr:.2f} dB (expected > 60 dB)"
        max_diff = torch.abs(x_enc_ref - x_dec).max().item()
        assert max_diff < 1e-4, f"[{model_key} - {img_name}] Max reconstruction error {max_diff} exceeds 1e-4"


class TestIsolatedProcessGate:
    r"""Verifies CLI encode/decode subprocess execution across disk boundaries."""

    @pytest.mark.parametrize("model_key", ["B", "C", "D", "C-Transmitted"])
    def test_isolated_subprocess_encode_and_decode(
        self,
        model_key: str,
        tmp_path: Path,
        shared_checkpoints: dict[str, Path],
    ) -> None:
        r"""Encodes Kodak image to disk and decodes in isolated CPU Python subprocess, asserting PSNR > 60 dB."""
        img_in = KODAK_DIR / "kodim01.png"
        sanc_file = tmp_path / f"kodim01_{model_key}.sanc"
        img_out = tmp_path / f"kodim01_{model_key}_recon.png"
        ckpt_path = shared_checkpoints[model_key]

        model = create_model(model_key)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.update()
        model.eval()

        # Compute exact in-memory reference reconstruction
        x_in = load_image_tensor(img_in)
        x_pad, bounds = pad_symmetric_centered(x_in, factor=64)
        with torch.no_grad():
            y = model.g_a(x_pad)
            z = model.h_a(y)
        q_z = torch.round(z)
        cpu_engine = DeterministicCPUEngine.from_model(model)
        valid_mask = compute_latent_valid_mask(
            orig_shape=(x_in.shape[-2], x_in.shape[-1]), pad_bounds=bounds, device="cpu"
        )

        saliency_path = tmp_path / "saliency.png"
        if model_key in ("C", "D"):
            _, _, delta, _, _ = cpu_engine.forward_allocation(q_z, valid_mask=valid_mask)
            q_y = torch.round(y / delta)
            y_hat = q_y * delta
        elif model_key == "C-Transmitted":
            sal_arr = np.full((512, 768), 128, dtype=np.uint8)
            Image.fromarray(sal_arr).save(saliency_path)
            s_tensor = torch.full((1, 1, x_in.shape[-2], x_in.shape[-1]), 128.0 / 255.0, dtype=torch.float32)
            s_pad, _ = pad_symmetric_centered(s_tensor, pad_bounds=bounds)
            latent_shape = (y.shape[-2], y.shape[-1])
            comp_bytes = compress_saliency_thumbnail(s_pad, target_shape=latent_shape)
            s_bytes = comp_bytes[0] if isinstance(comp_bytes, list) else comp_bytes
            s_dec = decompress_saliency_thumbnail(s_bytes, shape=latent_shape, device="cpu")
            _, delta = compute_log_delta(s_dec, valid_mask=valid_mask)
            q_y = torch.round(y / delta)
            y_hat = q_y * delta
        else:  # Model B
            q_y = torch.round(y)
            y_hat = q_y

        with torch.no_grad():
            x_pad_ref = model.g_s(y_hat)
        x_ref = torch.clamp(unpad_centered(x_pad_ref, bounds), 0.0, 1.0)
        x_ref_quant = torch.round(x_ref * 255.0) / 255.0

        # 1. Encode via CLI
        encode_cmd = [
            sys.executable,
            "scripts/encode.py",
            "-i",
            str(img_in),
            "-o",
            str(sanc_file),
            "-c",
            str(ckpt_path),
            "-m",
            model_key,
            "--device",
            "cpu",
        ]
        if model_key == "C-Transmitted":
            encode_cmd.extend(["-s", str(saliency_path)])

        res_enc = subprocess.run(encode_cmd, capture_output=True, text=True, check=True)
        assert "Compressed" in res_enc.stdout
        assert sanc_file.is_file()

        # 2. Decode via CLI in isolated process (zero-config, model_id deduced from container)
        decode_cmd = [
            sys.executable,
            "scripts/decode.py",
            "-i",
            str(sanc_file),
            "-o",
            str(img_out),
            "-c",
            str(ckpt_path),
            "--device",
            "cpu",
        ]
        res_dec = subprocess.run(decode_cmd, capture_output=True, text=True, check=True)
        assert "Decompressed" in res_dec.stdout
        assert img_out.is_file()

        # 3. Verify dimensions and reconstruction fidelity
        recon_img = Image.open(img_out)
        orig_img = Image.open(img_in)
        assert recon_img.size == orig_img.size

        x_recon = load_image_tensor(img_out)
        psnr = compute_psnr(x_ref_quant, x_recon)
        assert psnr > 60.0, f"[{model_key}] Subprocess decode PSNR {psnr:.2f} dB (expected > 60 dB)"
        assert torch.abs(x_ref_quant - x_recon).max().item() <= (1.0 / 255.0 + 1e-4)

    @pytest.mark.parametrize("model_key", ["B", "C", "D", "C-Transmitted"])
    def test_multi_isa_isolated_decode_parity(
        self,
        model_key: str,
        tmp_path: Path,
        shared_checkpoints: dict[str, Path],
    ) -> None:
        r"""Verifies exact bit-identical reconstruction between ATEN_CPU_CAPABILITY=default and avx2."""
        img_in = KODAK_DIR / "kodim03.png"
        sanc_file = tmp_path / f"kodim03_{model_key}.sanc"
        img_out_default = tmp_path / f"kodim03_{model_key}_default.png"
        img_out_avx2 = tmp_path / f"kodim03_{model_key}_avx2.png"
        ckpt_path = shared_checkpoints[model_key]

        encode_cmd = [
            sys.executable,
            "scripts/encode.py",
            "-i",
            str(img_in),
            "-o",
            str(sanc_file),
            "-c",
            str(ckpt_path),
            "-m",
            model_key,
            "--device",
            "cpu",
        ]
        subprocess.run(encode_cmd, capture_output=True, text=True, check=True)

        # 1. Decode under default ISA
        env_default = os.environ.copy()
        env_default["ATEN_CPU_CAPABILITY"] = "default"
        decode_cmd_default = [
            sys.executable,
            "scripts/decode.py",
            "-i",
            str(sanc_file),
            "-o",
            str(img_out_default),
            "-c",
            str(ckpt_path),
            "--device",
            "cpu",
        ]
        res_default = subprocess.run(decode_cmd_default, capture_output=True, text=True, env=env_default, check=True)
        assert "Decompressed" in res_default.stdout
        assert img_out_default.is_file()

        # 2. Decode under AVX2 ISA
        env_avx2 = os.environ.copy()
        env_avx2["ATEN_CPU_CAPABILITY"] = "avx2"
        env_avx2["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"
        decode_cmd_avx2 = [
            sys.executable,
            "scripts/decode.py",
            "-i",
            str(sanc_file),
            "-o",
            str(img_out_avx2),
            "-c",
            str(ckpt_path),
            "--device",
            "cpu",
        ]
        res_avx2 = subprocess.run(decode_cmd_avx2, capture_output=True, text=True, env=env_avx2, check=True)
        assert "Decompressed" in res_avx2.stdout
        assert img_out_avx2.is_file()

        # 3. Exact bit-identical parity assertion
        img_bytes_default = img_out_default.read_bytes()
        img_bytes_avx2 = img_out_avx2.read_bytes()
        assert img_bytes_default == img_bytes_avx2, f"[{model_key}] ISA decode byte mismatch"

        x_default = load_image_tensor(img_out_default)
        x_avx2 = load_image_tensor(img_out_avx2)
        diff = torch.abs(x_default - x_avx2).max().item()
        assert diff == 0.0, f"[{model_key}] ISA decode tensor mismatch: {diff}"


class TestUnalignedCropGate:
    r"""Verifies non-multiple-of-64 crop recovery across Models B, C, D, and C-Transmitted."""

    @pytest.mark.parametrize("model_key", ["B", "C", "D", "C-Transmitted"])
    def test_unaligned_crop_bitstream_gate(
        self,
        model_key: str,
        tmp_path: Path,
        shared_model_b: ModelB,
        shared_model_c: ModelC,
        shared_model_d: ModelD,
        shared_model_c_transmitted: ModelCTransmitted,
    ) -> None:
        r"""Verifies non-multiple-of-64 crop (500x375) satisfies exact dimension recovery and PSNR > 60 dB."""
        img_in_path = KODAK_DIR / "kodim05.png"
        img = Image.open(img_in_path)
        crop = img.crop((0, 0, 375, 500))  # (width=375, height=500)
        crop_path = tmp_path / f"crop_{model_key}.png"
        crop.save(crop_path)

        sanc_path = tmp_path / f"crop_{model_key}.sanc"

        if model_key == "B":
            model: ModelA = shared_model_b
        elif model_key == "C":
            model = shared_model_c
        elif model_key == "D":
            model = shared_model_d
        else:
            model = shared_model_c_transmitted

        model.to("cpu").eval()
        x_crop = load_image_tensor(crop_path)
        x_pad, bounds = pad_symmetric_centered(x_crop, factor=64)

        # Reference encoder synthesis on crop
        with torch.no_grad():
            y = model.g_a(x_pad)
            z = model.h_a(y)

        q_z = torch.round(z)
        cpu_engine = DeterministicCPUEngine.from_model(model)
        valid_mask = compute_latent_valid_mask(orig_shape=(crop.size[1], crop.size[0]), pad_bounds=bounds, device="cpu")

        saliency_input: Path | None = None
        if model_key == "B":
            delta = torch.ones_like(y)
        elif model_key in ("C", "D"):
            _, _, delta, _, _ = cpu_engine.forward_allocation(q_z, valid_mask=valid_mask)
        else:
            # Generate non-uniform Gaussian saliency map centered at (150, 200)
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, 500), torch.linspace(-1, 1, 375), indexing="ij")
            s_nonuniform = torch.exp(-((xx - 0.2) ** 2 + (yy + 0.1) ** 2) / 0.3).unsqueeze(0).unsqueeze(0)
            s_crop_path = tmp_path / "crop_saliency.png"
            s_arr = (s_nonuniform[0, 0].numpy() * 255.0).astype(np.uint8)
            Image.fromarray(s_arr).save(s_crop_path)
            saliency_input = s_crop_path

            s_tensor = load_image_tensor(saliency_input)
            if s_tensor.shape[1] > 1:
                s_tensor = s_tensor.mean(dim=1, keepdim=True)
            s_pad, _ = pad_symmetric_centered(s_tensor, pad_bounds=bounds)
            latent_shape = (y.shape[-2], y.shape[-1])
            comp_bytes = compress_saliency_thumbnail(s_pad, target_shape=latent_shape)
            s_bytes = comp_bytes[0] if isinstance(comp_bytes, list) else comp_bytes
            s_dec = decompress_saliency_thumbnail(s_bytes, shape=latent_shape, device="cpu")
            _, delta = compute_log_delta(s_dec, valid_mask=valid_mask)
            assert (delta != 1.0).any(), "Non-uniform saliency must produce delta != 1.0"

        q_y = torch.round(y / delta)
        y_hat_ref = q_y * delta
        with torch.no_grad():
            x_pad_ref = model.g_s(y_hat_ref)
        x_crop_ref = torch.clamp(unpad_centered(x_pad_ref, bounds), 0.0, 1.0)

        # Encode and decode
        container = encode_image(crop_path, model, device="cpu", saliency_map=saliency_input)
        container.save(sanc_path)

        loaded_container = SancContainer.load(sanc_path)
        x_recon = decode_image(loaded_container, model, device="cpu")

        assert x_recon.shape == (1, 3, 500, 375)
        psnr = compute_psnr(x_crop_ref, x_recon)
        assert psnr > 60.0, f"[{model_key}] Unaligned crop PSNR {psnr:.2f} dB (expected > 60 dB)"
        assert torch.abs(x_crop_ref - x_recon).max().item() < 1e-4
