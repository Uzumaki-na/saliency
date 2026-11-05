"""Extended Bitstream Acceptance Gate test suite (Story 2.6).

Verifies across the Kodak PhotoCD benchmark:
1. Exact integer latent symbol equality: max |q_z - hat_q_z| == 0 and max |q_y - hat_q_y| == 0.
2. Exact scale index equality: max |scale_idx_enc - scale_idx_dec| == 0.
3. Reconstruction consistency: PSNR > 60.0 dB and max |x_enc - x_dec| < 1e-4.
4. Isolated process execution: GPU/CPU encode -> .sanc file on disk -> isolated CPU decode subprocess with checkpoints.
5. Multi-ISA verification asserting bit-identical reconstruction across ATEN_CPU_CAPABILITY ('default' vs 'avx2').
6. Non-multiple-of-64 dimension crop verification (500x375) with PSNR > 60.0 dB.
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
    decode_image,
    encode_image,
    load_image_tensor,
)
from src.codec.model_a import ModelA
from src.data.padding import pad_symmetric_centered, unpad_centered
from src.entropy.coder import decode_hyper_latents, decode_latents
from src.entropy.cpu_engine import DeterministicCPUEngine

KODAK_DIR = Path("data/raw/kodak")


@pytest.fixture(scope="module")
def shared_model() -> ModelA:
    r"""Module-scoped updated ModelA instance for the bitstream gate."""
    model = ModelA()
    model.update()
    model.eval()
    return model


@pytest.fixture(scope="module")
def shared_checkpoint_path(
    shared_model: ModelA,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    r"""Module-scoped checkpoint path for isolated subprocess tests."""
    temp_dir = tmp_path_factory.mktemp("gate_checkpoint")
    ckpt_path = temp_dir / "model_a_gate.pt"
    torch.save({"state_dict": shared_model.state_dict()}, ckpt_path)
    return ckpt_path


def compute_psnr(x1: torch.Tensor, x2: torch.Tensor) -> float:
    r"""Computes Peak Signal-to-Noise Ratio (dB) between two image tensors in [0, 1]."""
    diff = (x1 - x2).float()
    mse = torch.mean(diff ** 2).item()
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * np.log10(1.0 / mse))


class TestKodakBitstreamGate:
    r"""Extended Bitstream Acceptance Gate across all 24 Kodak PhotoCD benchmark images."""

    def test_kodak_dataset_availability(self) -> None:
        r"""Verifies all 24 Kodak benchmark images are present."""
        assert KODAK_DIR.is_dir(), f"Kodak directory not found: {KODAK_DIR}"
        images = sorted(KODAK_DIR.glob("kodim*.png"))
        assert len(images) == 24, f"Expected 24 Kodak images, found {len(images)}"

    @pytest.mark.parametrize("img_idx", list(range(1, 25)))
    def test_kodak_symbol_and_scale_equality(
        self,
        shared_model: ModelA,
        img_idx: int,
    ) -> None:
        r"""Validates exact symbol equality, scale index equality, and PSNR > 60 dB on Kodak."""
        img_name = f"kodim{img_idx:02d}.png"
        img_path = KODAK_DIR / img_name
        assert img_path.is_file(), f"Image not found: {img_path}"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = load_image_tensor(img_path)

        # 1. Manual inspection of encoder intermediate quantized latents
        model = shared_model.to(device).eval()
        x_dev = x.to(device)

        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    y = model.g_a(x_dev)
                    z = model.h_a(y)
            else:
                y = model.g_a(x_dev)
                z = model.h_a(y)

        q_z_enc = torch.round(z).cpu()
        cpu_engine = DeterministicCPUEngine.from_model(model)
        _, scale_indexes_enc = cpu_engine.forward(q_z_enc)
        q_y_enc = torch.round(y).cpu()

        # Encoder synthesis reference (clamped to [0, 1] matching decoder image domain)
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    x_enc_ref = model.g_s(q_y_enc.to(device)).cpu()
            else:
                x_enc_ref = model.g_s(q_y_enc.to(device)).cpu()
        x_enc_ref = torch.clamp(x_enc_ref, 0.0, 1.0)

        # 2. Encode to SancContainer
        container = encode_image(x, model, device=device)

        # 3. Decode hyper-latents and derive decoder scale indexes
        h_z = container.orig_height // 64
        w_z = container.orig_width // 64
        q_z_dec = decode_hyper_latents(
            container.z_bitstream, (1, 128, h_z, w_z), model.entropy_bottleneck
        )
        _, scale_indexes_dec = cpu_engine.forward(q_z_dec)

        # 4. Decode main latents
        q_y_dec = decode_latents(
            container.y_bitstream, scale_indexes_dec, model.gaussian_conditional
        )

        # 5. Full image decode targeting device to verify exact reconstruction
        x_dec = decode_image(container, model, device=device).cpu()

        # --- Invariant Assertions ---
        # A. Exact hyper-latent symbol equality: max |q_z - hat_q_z| == 0
        diff_qz = torch.abs(q_z_enc - q_z_dec).max().item()
        assert diff_qz == 0.0, f"[{img_name}] Hyper-latent symbol mismatch: max diff {diff_qz}"

        # B. Exact scale index equality: max |scale_idx_enc - scale_idx_dec| == 0
        diff_idx = torch.abs(scale_indexes_enc - scale_indexes_dec).max().item()
        assert diff_idx == 0, f"[{img_name}] Scale index mismatch: max diff {diff_idx}"

        # C. Exact main latent symbol equality: max |q_y - hat_q_y| == 0
        diff_qy = torch.abs(q_y_enc - q_y_dec).max().item()
        assert diff_qy == 0.0, f"[{img_name}] Main latent symbol mismatch: max diff {diff_qy}"

        # D. Reconstruction consistency: PSNR > 60.0 dB and max |x_enc - x_dec| < 1e-4
        psnr = compute_psnr(x_enc_ref, x_dec)
        assert psnr > 60.0, f"[{img_name}] Insufficient reconstruction PSNR: {psnr:.2f} dB (expected > 60 dB)"
        max_diff = torch.abs(x_enc_ref - x_dec).max().item()
        assert max_diff < 1e-4, f"[{img_name}] Max reconstruction error {max_diff} exceeds 1e-4"


class TestIsolatedProcessGate:
    r"""Tests for isolated process execution and multi-ISA environment sweeps."""

    def test_isolated_subprocess_decode_kodak(
        self,
        tmp_path: Path,
        shared_model: ModelA,
        shared_checkpoint_path: Path,
    ) -> None:
        r"""Encodes Kodak image to disk and decodes in an isolated CPU Python subprocess with verified PSNR > 60 dB."""
        img_in = KODAK_DIR / "kodim01.png"
        sanc_file = tmp_path / "kodim01.sanc"
        img_out = tmp_path / "kodim01_recon.png"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Reference synthesis calculation matching device execution mode
        model = shared_model.to(device).eval()
        x_in = load_image_tensor(img_in)
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    y = model.g_a(x_in.to(device))
                    q_y = torch.round(y)
                    x_enc_ref = model.g_s(q_y.to(device)).cpu()
            else:
                y = model.g_a(x_in.to(device))
                q_y = torch.round(y)
                x_enc_ref = model.g_s(q_y.to(device)).cpu()
        x_enc_ref = torch.clamp(x_enc_ref, 0.0, 1.0)

        # 2. Encode via CLI with shared checkpoint
        encode_cmd = [
            sys.executable,
            "scripts/encode.py",
            "-i",
            str(img_in),
            "-o",
            str(sanc_file),
            "-c",
            str(shared_checkpoint_path),
            "--device",
            device,
        ]
        res_enc = subprocess.run(encode_cmd, capture_output=True, text=True, check=True)
        assert "Compressed" in res_enc.stdout
        assert sanc_file.is_file()

        # 3. Decode in isolated CPU process with shared checkpoint
        decode_cmd = [
            sys.executable,
            "scripts/decode.py",
            "-i",
            str(sanc_file),
            "-o",
            str(img_out),
            "-c",
            str(shared_checkpoint_path),
            "--device",
            "cpu",
        ]
        res_dec = subprocess.run(decode_cmd, capture_output=True, text=True, check=True)
        assert "Decompressed" in res_dec.stdout
        assert img_out.is_file()

        # 4. Verify dimensions and reconstruction fidelity
        recon = Image.open(img_out)
        orig = Image.open(img_in)
        assert recon.size == orig.size

        x_recon = load_image_tensor(img_out)
        psnr = compute_psnr(x_enc_ref, x_recon)
        assert psnr > 60.0, f"Isolated decode PSNR {psnr:.2f} dB (expected > 60 dB)"
        # 8-bit PNG file quantization bound allows up to 1 LSB (1.0 / 255.0) under cross-device execution
        assert torch.abs(x_enc_ref - x_recon).max().item() <= (1.0 / 255.0 + 1e-4)

    def test_multi_isa_isolated_decode_parity(
        self,
        tmp_path: Path,
        shared_checkpoint_path: Path,
    ) -> None:
        r"""Verifies exact bit-identical reconstruction between ATEN_CPU_CAPABILITY=default and avx2."""
        img_in = KODAK_DIR / "kodim03.png"
        sanc_file = tmp_path / "kodim03.sanc"
        img_out_default = tmp_path / "kodim03_default.png"
        img_out_avx2 = tmp_path / "kodim03_avx2.png"

        # Encode once
        encode_cmd = [
            sys.executable,
            "scripts/encode.py",
            "-i",
            str(img_in),
            "-o",
            str(sanc_file),
            "-c",
            str(shared_checkpoint_path),
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
            str(shared_checkpoint_path),
            "--device",
            "cpu",
        ]
        res_default = subprocess.run(
            decode_cmd_default, capture_output=True, text=True, env=env_default, check=True
        )
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
            str(shared_checkpoint_path),
            "--device",
            "cpu",
        ]
        res_avx2 = subprocess.run(
            decode_cmd_avx2, capture_output=True, text=True, env=env_avx2, check=True
        )
        assert "Decompressed" in res_avx2.stdout
        assert img_out_avx2.is_file()

        # 3. Assert bit-identical parity between default and avx2
        t_default = load_image_tensor(img_out_default)
        t_avx2 = load_image_tensor(img_out_avx2)
        diff = torch.abs(t_default - t_avx2).max().item()
        assert diff == 0.0, f"Multi-ISA discrepancy: max diff {diff}"

    def test_unaligned_crop_bitstream_gate(
        self,
        tmp_path: Path,
        shared_model: ModelA,
        shared_checkpoint_path: Path,
    ) -> None:
        r"""Verifies non-multiple-of-64 crop (500x375) satisfies exact dimension recovery and PSNR > 60 dB."""
        img_in_path = KODAK_DIR / "kodim05.png"
        img = Image.open(img_in_path)
        # Crop to 500x375 (height=500, width=375)
        crop = img.crop((0, 0, 375, 500))
        crop_path = tmp_path / "crop_500x375.png"
        crop.save(crop_path)

        sanc_path = tmp_path / "crop.sanc"
        recon_path = tmp_path / "crop_recon.png"

        # 1. Compute reference synthesis on crop with centered padding on CPU
        model = shared_model.to("cpu").eval()
        x_crop = load_image_tensor(crop_path)
        x_pad, bounds = pad_symmetric_centered(x_crop, factor=64)
        with torch.no_grad():
            y = model.g_a(x_pad)
            q_y = torch.round(y)
            x_pad_ref = model.g_s(q_y)
            x_crop_ref = torch.clamp(unpad_centered(x_pad_ref, bounds), 0.0, 1.0)

        # 2. Encode via CLI with shared checkpoint
        encode_cmd = [
            sys.executable,
            "scripts/encode.py",
            "-i",
            str(crop_path),
            "-o",
            str(sanc_path),
            "-c",
            str(shared_checkpoint_path),
            "--device",
            "cpu",
        ]
        subprocess.run(encode_cmd, capture_output=True, text=True, check=True)

        # 3. Decode via CLI with shared checkpoint
        decode_cmd = [
            sys.executable,
            "scripts/decode.py",
            "-i",
            str(sanc_path),
            "-o",
            str(recon_path),
            "-c",
            str(shared_checkpoint_path),
            "--device",
            "cpu",
        ]
        subprocess.run(decode_cmd, capture_output=True, text=True, check=True)

        # 4. Verify exact dimensions and PSNR > 60 dB
        recon_img = Image.open(recon_path)
        assert recon_img.size == (375, 500)

        x_recon = load_image_tensor(recon_path)
        psnr = compute_psnr(x_crop_ref, x_recon)
        assert psnr > 60.0, f"Unaligned crop PSNR {psnr:.2f} dB (expected > 60 dB)"
        # 8-bit PNG file quantization bound is 0.5 / 255 (~0.00196)
        assert torch.abs(x_crop_ref - x_recon).max().item() <= (0.5 / 255.0 + 1e-5)
