"""Unit tests for compact binary container (.sanc) and codec CLI (Story 2.5).

Verifies:
1. Header serialization, little-endian packing, and magic byte validation.
2. Error handling on corrupted, truncated, or invalid containers.
3. Image loading helper supporting tensors, numpy arrays, PIL images, and file paths.
4. End-to-end encode/decode roundtrips on divisible dimensions (e.g. 256x256).
5. Exact unaligned dimension recovery on non-multiple-of-64 dimensions (e.g. 500x375).
6. Cross-backend roundtrip parity (constriction and python).
7. Standalone CLI execution via scripts/encode.py and scripts/decode.py.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.bitstream.container import (
    CURRENT_VERSION,
    MAGIC_BYTES,
    SancContainer,
    decode_image,
    encode_image,
    load_image_tensor,
)
from src.codec.model_a import ModelA
from src.data.padding import PadBounds
from src.entropy.coder import HAS_CONSTRICTION


@pytest.fixture
def dummy_model() -> ModelA:
    r"""Fixture providing an updated ModelA instance for end-to-end codec tests."""
    model = ModelA()
    model.update()
    model.eval()
    return model


class TestSancContainerSerialization:
    r"""Tests for binary header serialization, unpacking, and file I/O."""

    def test_container_byte_roundtrip(self) -> None:
        r"""Verifies exact metadata and payload preservation through to_bytes / from_bytes."""
        pad = PadBounds(pad_top=6, pad_bottom=6, pad_left=4, pad_right=5)
        z_payload = b"\x01\x02\x03\x04\x05"
        y_payload = b"\x10\x20\x30\x40\x50\x60"

        container = SancContainer(
            orig_height=500,
            orig_width=375,
            pad_bounds=pad,
            z_bitstream=z_payload,
            y_bitstream=y_payload,
            version=CURRENT_VERSION,
            model_id=0,
            flags=0,
        )

        raw_bytes = container.to_bytes()
        assert raw_bytes.startswith(MAGIC_BYTES)
        assert len(raw_bytes) == container.total_bytes

        restored = SancContainer.from_bytes(raw_bytes)
        assert restored.orig_height == 500
        assert restored.orig_width == 375
        assert restored.pad_bounds == pad
        assert restored.z_bitstream == z_payload
        assert restored.y_bitstream == y_payload
        assert restored.version == CURRENT_VERSION
        assert restored.model_id == 0

    def test_container_file_save_and_load(self, tmp_path: Path) -> None:
        r"""Verifies disk file save and load integrity."""
        file_path = tmp_path / "test_file.sanc"
        pad = PadBounds(0, 0, 0, 0)
        container = SancContainer(
            orig_height=768,
            orig_width=512,
            pad_bounds=pad,
            z_bitstream=b"hyper_data",
            y_bitstream=b"main_latent_data",
        )

        bytes_written = container.save(file_path)
        assert file_path.is_file()
        assert bytes_written == file_path.stat().st_size

        loaded = SancContainer.load(file_path)
        assert loaded.orig_height == 768
        assert loaded.orig_width == 512
        assert loaded.pad_bounds == pad
        assert loaded.z_bitstream == b"hyper_data"
        assert loaded.y_bitstream == b"main_latent_data"

    def test_bpp_calculation(self) -> None:
        r"""Verifies bits per pixel calculation property."""
        pad = PadBounds(0, 0, 0, 0)
        container = SancContainer(
            orig_height=64,
            orig_width=64,
            pad_bounds=pad,
            z_bitstream=b"x" * 50,
            y_bitstream=b"y" * 150,
        )
        # Total bytes = 24 (header) + 4 (z_len) + 50 + 4 (y_len) + 150 = 232 bytes
        # Pixels = 4,096
        # Expected bpp = 232 * 8 / 4096 = 0.453125
        assert container.total_bytes == 232
        assert pytest.approx(container.bpp, 1e-4) == 0.453125


class TestContainerValidationErrors:
    r"""Tests for defensive exception handling on malformed container bytes."""

    def test_invalid_magic_bytes_raises(self) -> None:
        r"""Asserts that wrong magic bytes raise ValueError."""
        pad = PadBounds(0, 0, 0, 0)
        valid = SancContainer(64, 64, pad, b"z", b"y").to_bytes()
        corrupted = b"NOPE" + valid[4:]

        with pytest.raises(ValueError, match="Invalid magic signature"):
            SancContainer.from_bytes(corrupted)

    def test_unsupported_version_raises(self) -> None:
        r"""Asserts that non-supported version number raises ValueError."""
        pad = PadBounds(0, 0, 0, 0)
        valid = bytearray(SancContainer(64, 64, pad, b"z", b"y").to_bytes())
        # Set version byte at index 4 to 99
        valid[4] = 99

        with pytest.raises(ValueError, match="Unsupported container version"):
            SancContainer.from_bytes(bytes(valid))

    def test_non_divisible_dimensions_raises(self) -> None:
        r"""Asserts that padded dimensions not divisible by 64 raise ValueError."""
        pad = PadBounds(0, 0, 0, 0)
        raw = SancContainer(100, 100, pad, b"z", b"y").to_bytes()

        with pytest.raises(ValueError, match="not divisible by 64"):
            SancContainer.from_bytes(raw)

    def test_truncated_header_raises(self) -> None:
        r"""Asserts that byte streams shorter than minimum header size raise ValueError."""
        with pytest.raises(ValueError, match="Container data truncated"):
            SancContainer.from_bytes(b"SANC\x01\x00")

    def test_truncated_payload_raises(self) -> None:
        r"""Asserts that declared payload lengths exceeding actual data raise ValueError."""
        pad = PadBounds(0, 0, 0, 0)
        valid = SancContainer(64, 64, pad, b"z_payload_12345", b"y_payload_67890").to_bytes()
        # Truncate final 5 bytes
        truncated = valid[:-5]

        with pytest.raises(ValueError, match="Corrupted or truncated container payload"):
            SancContainer.from_bytes(truncated)

    def test_nonexistent_file_load_raises(self, tmp_path: Path) -> None:
        r"""Asserts that loading a non-existent file path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Container file not found"):
            SancContainer.load(tmp_path / "missing.sanc")


class TestImageLoadingHelper:
    r"""Tests for load_image_tensor utility."""

    def test_load_from_pil(self) -> None:
        r"""Verifies loading from PIL Image."""
        img = Image.new("RGB", (64, 48), color=(128, 64, 32))
        tensor = load_image_tensor(img)
        assert tensor.shape == (1, 3, 48, 64)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0

    def test_load_from_numpy(self) -> None:
        r"""Verifies loading from uint8 NumPy array."""
        arr = np.random.randint(0, 256, size=(48, 64, 3), dtype=np.uint8)
        tensor = load_image_tensor(arr)
        assert tensor.shape == (1, 3, 48, 64)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0

    def test_load_from_tensor(self) -> None:
        r"""Verifies loading from 3D and 4D PyTorch tensors."""
        t3d = torch.rand(3, 48, 64)
        out3d = load_image_tensor(t3d)
        assert out3d.shape == (1, 3, 48, 64)

        t4d = torch.rand(1, 3, 48, 64)
        out4d = load_image_tensor(t4d)
        assert out4d.shape == (1, 3, 48, 64)

    def test_load_from_file_path(self, tmp_path: Path) -> None:
        r"""Verifies loading from image file saved on disk."""
        img_path = tmp_path / "test_sample.png"
        img = Image.new("RGB", (80, 60), color=(255, 128, 0))
        img.save(img_path)

        tensor = load_image_tensor(img_path)
        assert tensor.shape == (1, 3, 60, 80)


class TestEndToEndCodecPipeline:
    r"""Tests for full image compression and decompression roundtrip."""

    @pytest.mark.parametrize("backend", ["python"] + (["constriction"] if HAS_CONSTRICTION else []))
    def test_roundtrip_divisible_dimensions(self, dummy_model: ModelA, backend: str) -> None:
        r"""Tests full encode/decode on 128x128 image with zero padding required."""
        # Use simple gradient pattern
        x = torch.zeros(1, 3, 128, 128, dtype=torch.float32)
        x[:, 0, :, :] = torch.linspace(0, 1, 128).unsqueeze(0)
        x[:, 1, :, :] = torch.linspace(0, 1, 128).unsqueeze(1)
        x[:, 2, :, :] = 0.5

        container = encode_image(x, dummy_model, device="cpu", backend=backend)
        assert container.orig_height == 128
        assert container.orig_width == 128
        assert container.pad_bounds == (0, 0, 0, 0)
        assert len(container.z_bitstream) > 0
        assert len(container.y_bitstream) > 0

        x_hat = decode_image(container, dummy_model, device="cpu", backend=backend)
        assert x_hat.shape == (1, 3, 128, 128)
        assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0

    @pytest.mark.parametrize("backend", ["python"] + (["constriction"] if HAS_CONSTRICTION else []))
    def test_roundtrip_unaligned_dimensions(self, dummy_model: ModelA, backend: str) -> None:
        r"""Tests full encode/decode on non-multiple-of-64 image (100x80) with centered padding."""
        h_orig, w_orig = 100, 80
        x = torch.rand(1, 3, h_orig, w_orig, dtype=torch.float32)

        container = encode_image(x, dummy_model, device="cpu", backend=backend)
        assert container.orig_height == h_orig
        assert container.orig_width == w_orig
        # Padded to 128x128 -> delta_h = 28 (14, 14), delta_w = 48 (24, 24)
        assert container.pad_bounds.pad_top == 14
        assert container.pad_bounds.pad_bottom == 14
        assert container.pad_bounds.pad_left == 24
        assert container.pad_bounds.pad_right == 24

        x_hat = decode_image(container, dummy_model, device="cpu", backend=backend)
        assert x_hat.shape == (1, 3, h_orig, w_orig)
        assert x_hat.min() >= 0.0 and x_hat.max() <= 1.0


class TestCodecCLIExecution:
    r"""Tests for standalone scripts/encode.py and scripts/decode.py CLI invocations."""

    def test_cli_encode_decode_roundtrip(self, tmp_path: Path) -> None:
        r"""Executes scripts/encode.py and scripts/decode.py via subprocess CLI."""
        img_in = tmp_path / "cli_input.png"
        sanc_out = tmp_path / "compressed.sanc"
        img_out = tmp_path / "cli_output.png"

        # Create 128x128 test image
        img = Image.new("RGB", (128, 128), color=(200, 100, 50))
        img.save(img_in)

        # 1. Run encode CLI
        encode_cmd = [
            sys.executable,
            "scripts/encode.py",
            "-i",
            str(img_in),
            "-o",
            str(sanc_out),
            "--device",
            "cpu",
            "--backend",
            "python",
        ]
        res_encode = subprocess.run(encode_cmd, capture_output=True, text=True, check=True)
        assert "Compressed" in res_encode.stdout
        assert sanc_out.is_file()
        assert sanc_out.stat().st_size > 0

        # 2. Run decode CLI
        decode_cmd = [
            sys.executable,
            "scripts/decode.py",
            "-i",
            str(sanc_out),
            "-o",
            str(img_out),
            "--device",
            "cpu",
            "--backend",
            "python",
        ]
        res_decode = subprocess.run(decode_cmd, capture_output=True, text=True, check=True)
        assert "Decompressed" in res_decode.stdout
        assert img_out.is_file()

        # Check recovered image
        recon_img = Image.open(img_out)
        assert recon_img.size == (128, 128)

