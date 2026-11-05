"""Binary container (.sanc) serialization and end-to-end codec pipelines.

Implements:
1. `SancContainer`: Lightweight binary bitstream container with header validation.
2. `encode_image`: Full compression pipeline (symmetric padding -> g_a -> h_a -> rANS hyper -> hs CPU -> rANS latents).
3. `decode_image`: Full decompression pipeline (rANS hyper -> hs CPU -> rANS latents -> g_s -> centered unpadding).
4. `load_image_tensor`: Utility to load images from disk, PIL, or arrays into normalized tensors.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from src.data.padding import PadBounds, pad_symmetric_centered, unpad_centered
from src.entropy.coder import (
    decode_hyper_latents,
    decode_latents,
    encode_hyper_latents,
    encode_latents,
)
from src.entropy.cpu_engine import DeterministicCPUEngine

if TYPE_CHECKING:
    from src.codec.model_a import ModelA

MAGIC_BYTES: bytes = b"SANC"
CURRENT_VERSION: int = 1
DEFAULT_MODEL_ID: int = 0

# Binary Header Format:
# - magic: 4 bytes (b"SANC")
# - version: uint8 (1 byte)
# - model_id: uint8 (1 byte)
# - flags: uint16 (2 bytes)
# - orig_height: uint32 (4 bytes)
# - orig_width: uint32 (4 bytes)
# - pad_top: uint16 (2 bytes)
# - pad_bottom: uint16 (2 bytes)
# - pad_left: uint16 (2 bytes)
# - pad_right: uint16 (2 bytes)
# Total: 24 bytes
HEADER_STRUCT: struct.Struct = struct.Struct("<4sBBHIIHHHH")
PAYLOAD_LEN_STRUCT: struct.Struct = struct.Struct("<I")


@dataclass(frozen=True)
class SancContainer:
    r"""In-memory representation of a .sanc compressed bitstream container.

    Attributes:
        orig_height: Original spatial height of the unpadded image.
        orig_width: Original spatial width of the unpadded image.
        pad_bounds: Centered padding margins applied during encoding.
        z_bitstream: Serialized rANS hyper-latent bitstream bytes.
        y_bitstream: Serialized rANS main latent bitstream bytes.
        version: Container format version (default 1).
        model_id: Model architecture identifier (default 0 for Model A).
        flags: Reserved metadata flags.
    """

    orig_height: int
    orig_width: int
    pad_bounds: PadBounds
    z_bitstream: bytes
    y_bitstream: bytes
    version: int = CURRENT_VERSION
    model_id: int = DEFAULT_MODEL_ID
    flags: int = 0

    @property
    def total_bytes(self) -> int:
        r"""Total byte length of the container on disk."""
        return (
            HEADER_STRUCT.size
            + PAYLOAD_LEN_STRUCT.size
            + len(self.z_bitstream)
            + PAYLOAD_LEN_STRUCT.size
            + len(self.y_bitstream)
        )

    @property
    def bpp(self) -> float:
        r"""Bits per pixel normalized by the unpadded original resolution."""
        if self.orig_height <= 0 or self.orig_width <= 0:
            return 0.0
        return (self.total_bytes * 8.0) / (self.orig_height * self.orig_width)

    def to_bytes(self) -> bytes:
        r"""Serializes container into immutable binary bytes."""
        header = HEADER_STRUCT.pack(
            MAGIC_BYTES,
            self.version,
            self.model_id,
            self.flags,
            int(self.orig_height),
            int(self.orig_width),
            int(self.pad_bounds.pad_top),
            int(self.pad_bounds.pad_bottom),
            int(self.pad_bounds.pad_left),
            int(self.pad_bounds.pad_right),
        )
        return (
            header
            + PAYLOAD_LEN_STRUCT.pack(len(self.z_bitstream))
            + self.z_bitstream
            + PAYLOAD_LEN_STRUCT.pack(len(self.y_bitstream))
            + self.y_bitstream
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> SancContainer:
        r"""Deserializes container from binary bytes with strict header validation."""
        min_size = HEADER_STRUCT.size + PAYLOAD_LEN_STRUCT.size * 2
        if len(data) < min_size:
            raise ValueError(f"Container data truncated: expected at least {min_size} bytes, got {len(data)}")

        (
            magic,
            version,
            model_id,
            flags,
            orig_h,
            orig_w,
            pt,
            pb,
            pl,
            pr,
        ) = HEADER_STRUCT.unpack_from(data, 0)

        if magic != MAGIC_BYTES:
            raise ValueError(f"Invalid magic signature: expected {MAGIC_BYTES!r}, got {magic!r}")

        if version != CURRENT_VERSION:
            raise ValueError(f"Unsupported container version: {version}. Expected version {CURRENT_VERSION}")

        if orig_h <= 0 or orig_w <= 0:
            raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")

        h_pad = orig_h + pt + pb
        w_pad = orig_w + pl + pr
        if h_pad % 64 != 0 or w_pad % 64 != 0:
            raise ValueError(f"Padded dimensions {w_pad}x{h_pad} not divisible by 64 (margins: {pt},{pb},{pl},{pr})")

        offset = HEADER_STRUCT.size
        (z_len,) = PAYLOAD_LEN_STRUCT.unpack_from(data, offset)
        offset += PAYLOAD_LEN_STRUCT.size

        if len(data) < offset + z_len + PAYLOAD_LEN_STRUCT.size:
            raise ValueError("Corrupted or truncated container payload for hyper-latents")

        z_bitstream = data[offset : offset + z_len]
        offset += z_len

        (y_len,) = PAYLOAD_LEN_STRUCT.unpack_from(data, offset)
        offset += PAYLOAD_LEN_STRUCT.size

        if len(data) < offset + y_len:
            raise ValueError("Corrupted or truncated container payload for main latents")

        y_bitstream = data[offset : offset + y_len]

        return cls(
            orig_height=orig_h,
            orig_width=orig_w,
            pad_bounds=PadBounds(pad_top=pt, pad_bottom=pb, pad_left=pl, pad_right=pr),
            z_bitstream=z_bitstream,
            y_bitstream=y_bitstream,
            version=version,
            model_id=model_id,
            flags=flags,
        )

    def save(self, path: str | Path) -> int:
        r"""Writes container to a .sanc binary file on disk. Returns total bytes written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_bytes()
        path.write_bytes(data)
        return len(data)

    @classmethod
    def load(cls, path: str | Path) -> SancContainer:
        r"""Loads and parses a .sanc binary container file from disk."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Container file not found: {path}")
        return cls.from_bytes(path.read_bytes())


def load_image_tensor(
    image_input: str | Path | Image.Image | np.ndarray | torch.Tensor,
) -> torch.Tensor:
    r"""Converts input image source into a normalized 4D float32 tensor (1, 3, H, W) in [0, 1]."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    if isinstance(image_input, np.ndarray):
        if image_input.dtype == np.uint8:
            arr = image_input.astype(np.float32) / 255.0
        else:
            arr = image_input.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
        if arr.ndim == 3:
            # (H, W, C) -> (C, H, W)
            if arr.shape[2] in (1, 3):
                arr = np.transpose(arr, (2, 0, 1))
            t = torch.from_numpy(arr).unsqueeze(0)
        elif arr.ndim == 4:
            if arr.shape[3] in (1, 3):
                arr = np.transpose(arr, (0, 3, 1, 2))
            t = torch.from_numpy(arr)
        else:
            raise ValueError(f"Unsupported numpy array dimensions: {arr.shape}")
        return t

    if isinstance(image_input, torch.Tensor):
        t = image_input.float()
        if t.max() > 1.0:
            t = t / 255.0
        if t.ndim == 3:
            t = t.unsqueeze(0)
        elif t.ndim != 4:
            raise ValueError(f"Expected 3D or 4D tensor, got {t.ndim}D tensor of shape {tuple(t.shape)}")
        return t

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def encode_image(
    image: str | Path | Image.Image | np.ndarray | torch.Tensor,
    model: ModelA,
    device: torch.device | str = "cpu",
    backend: str = "auto",
) -> SancContainer:
    r"""Compresses an input image into a SancContainer.

    Pipeline:
    1. Symmetric reflection padding to multiples of 64.
    2. Analysis transforms (g_a, h_a) on specified device.
    3. Integer rounding of hyper-latents q_z and entropy coding via EntropyBottleneck CDFs.
    4. Deterministic single-threaded CPU hyper-synthesis (h_s) to derive discrete scale indices.
    5. Integer rounding of main latents q_y and entropy coding via GaussianConditional CDFs.
    6. Packaging into SancContainer with unaligned dimension recovery metadata.
    """
    x = load_image_tensor(image)
    orig_h, orig_w = x.shape[-2], x.shape[-1]

    # Ensure CDF tables are compiled
    if model.entropy_bottleneck._quantized_cdf.numel() == 0:
        model.entropy_bottleneck.update()
    if model.gaussian_conditional._quantized_cdf.numel() == 0:
        model.gaussian_conditional.update()

    # Align spatial dimensions to multiples of 64
    x_pad, bounds = pad_symmetric_centered(x, factor=64)

    target_device = torch.device(device)
    model.eval()
    model.to(target_device)
    x_pad_dev = x_pad.to(target_device)

    with torch.no_grad():
        if target_device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = model.g_a(x_pad_dev)
                z = model.h_a(y)
        else:
            y = model.g_a(x_pad_dev)
            z = model.h_a(y)

    q_z = torch.round(z).cpu()
    z_bitstream = encode_hyper_latents(q_z, model.entropy_bottleneck, backend=backend)

    # Deterministic CPU parameter derivation
    cpu_engine = DeterministicCPUEngine.from_model(model)
    _, scale_indexes = cpu_engine.forward(q_z)

    q_y = torch.round(y).cpu()
    y_bitstream = encode_latents(q_y, scale_indexes, model.gaussian_conditional, backend=backend)

    return SancContainer(
        orig_height=orig_h,
        orig_width=orig_w,
        pad_bounds=bounds,
        z_bitstream=z_bitstream,
        y_bitstream=y_bitstream,
    )


def decode_image(
    container: SancContainer,
    model: ModelA,
    device: torch.device | str = "cpu",
    backend: str = "auto",
) -> torch.Tensor:
    r"""Decompresses a SancContainer back into a reconstructed image tensor (1, 3, H, W) in [0, 1].

    Pipeline:
    1. Recover padded latent spatial dimensions from header metadata.
    2. rANS decoding of hyper-latents q_z via EntropyBottleneck CDFs.
    3. Deterministic single-threaded CPU hyper-synthesis (h_s) to derive identical scale indices.
    4. rANS decoding of main latents q_y via GaussianConditional CDFs.
    5. Synthesis transform (g_s) on specified device.
    6. Centered unpadding back to exact original image dimensions.
    """
    # Ensure CDF tables are compiled
    if model.entropy_bottleneck._quantized_cdf.numel() == 0:
        model.entropy_bottleneck.update()
    if model.gaussian_conditional._quantized_cdf.numel() == 0:
        model.gaussian_conditional.update()

    h_pad = container.orig_height + container.pad_bounds.pad_top + container.pad_bounds.pad_bottom
    w_pad = container.orig_width + container.pad_bounds.pad_left + container.pad_bounds.pad_right
    h_z = h_pad // 64
    w_z = w_pad // 64

    # Hyper-latent entropy decoding
    z_channels = getattr(model, "n_channels", getattr(model.entropy_bottleneck, "channels", 128))
    q_z = decode_hyper_latents(
        container.z_bitstream, (1, z_channels, h_z, w_z), model.entropy_bottleneck, backend=backend
    )

    # Deterministic CPU parameter derivation
    cpu_engine = DeterministicCPUEngine.from_model(model)
    _, scale_indexes = cpu_engine.forward(q_z)

    # Main latent entropy decoding
    q_y = decode_latents(container.y_bitstream, scale_indexes, model.gaussian_conditional, backend=backend)

    target_device = torch.device(device)
    model.eval()
    model.to(target_device)
    q_y_dev = q_y.to(target_device)

    with torch.no_grad():
        if target_device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x_hat_pad = model.g_s(q_y_dev)
        else:
            x_hat_pad = model.g_s(q_y_dev)

    # Centered unpadding and range clamp
    x_hat = unpad_centered(x_hat_pad, container.pad_bounds)
    return torch.clamp(x_hat, 0.0, 1.0)
