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
from typing import TYPE_CHECKING, Any

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
FLAG_HAS_SIDECAR: int = 0x0001

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
        model_id: Model architecture identifier (0=A, 1=B, 2=C, 3=D, 4=C-Transmitted).
        flags: Reserved metadata flags (0x0001 = sidecar bitstream present).
        sidecar_bitstream: Optional serialized sidecar bitstream bytes (e.g. Model C-Transmitted thumbnail).
    """

    orig_height: int
    orig_width: int
    pad_bounds: PadBounds
    z_bitstream: bytes
    y_bitstream: bytes
    version: int = CURRENT_VERSION
    model_id: int = DEFAULT_MODEL_ID
    flags: int = 0
    sidecar_bitstream: bytes | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pad_bounds, PadBounds):
            if isinstance(self.pad_bounds, (tuple, list)) and len(self.pad_bounds) == 4:
                object.__setattr__(
                    self,
                    "pad_bounds",
                    PadBounds(
                        pad_top=int(self.pad_bounds[0]),
                        pad_bottom=int(self.pad_bounds[1]),
                        pad_left=int(self.pad_bounds[2]),
                        pad_right=int(self.pad_bounds[3]),
                    ),
                )
            else:
                raise TypeError(f"pad_bounds must be PadBounds, got {type(self.pad_bounds)}")
        if self.model_id == 4 and (self.sidecar_bitstream is None or len(self.sidecar_bitstream) == 0):
            raise ValueError("Model C-Transmitted (model_id=4) requires non-empty sidecar_bitstream")

    @property
    def total_bytes(self) -> int:
        r"""Total byte length of the container on disk."""
        size = (
            HEADER_STRUCT.size
            + PAYLOAD_LEN_STRUCT.size
            + len(self.z_bitstream)
            + PAYLOAD_LEN_STRUCT.size
            + len(self.y_bitstream)
        )
        if self.sidecar_bitstream is not None and len(self.sidecar_bitstream) > 0:
            size += PAYLOAD_LEN_STRUCT.size + len(self.sidecar_bitstream)
        elif self.model_id == 4:
            size += PAYLOAD_LEN_STRUCT.size
        return size

    @property
    def bpp(self) -> float:
        r"""Bits per pixel normalized by the unpadded original resolution."""
        if self.orig_height <= 0 or self.orig_width <= 0:
            return 0.0
        return (self.total_bytes * 8.0) / (self.orig_height * self.orig_width)

    def to_bytes(self) -> bytes:
        r"""Serializes container into immutable binary bytes."""
        has_sidecar = self.sidecar_bitstream is not None and len(self.sidecar_bitstream) > 0
        effective_flags = (self.flags | FLAG_HAS_SIDECAR) if has_sidecar else (self.flags & ~FLAG_HAS_SIDECAR)

        header = HEADER_STRUCT.pack(
            MAGIC_BYTES,
            self.version,
            self.model_id,
            effective_flags,
            int(self.orig_height),
            int(self.orig_width),
            int(self.pad_bounds.pad_top),
            int(self.pad_bounds.pad_bottom),
            int(self.pad_bounds.pad_left),
            int(self.pad_bounds.pad_right),
        )

        payloads: list[bytes] = []
        if has_sidecar:
            assert self.sidecar_bitstream is not None
            payloads.append(PAYLOAD_LEN_STRUCT.pack(len(self.sidecar_bitstream)))
            payloads.append(self.sidecar_bitstream)

        payloads.append(PAYLOAD_LEN_STRUCT.pack(len(self.z_bitstream)))
        payloads.append(self.z_bitstream)
        payloads.append(PAYLOAD_LEN_STRUCT.pack(len(self.y_bitstream)))
        payloads.append(self.y_bitstream)

        return header + b"".join(payloads)

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

        if model_id not in (0, 1, 2, 3, 4):
            raise ValueError(f"Unsupported or unknown model_id: {model_id}. Expected one of (0, 1, 2, 3, 4)")

        if orig_h <= 0 or orig_w <= 0:
            raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")

        h_pad = orig_h + pt + pb
        w_pad = orig_w + pl + pr
        if h_pad % 64 != 0 or w_pad % 64 != 0:
            raise ValueError(f"Padded dimensions {w_pad}x{h_pad} not divisible by 64 (margins: {pt},{pb},{pl},{pr})")

        offset = HEADER_STRUCT.size
        has_sidecar = bool(flags & FLAG_HAS_SIDECAR)
        sidecar_bitstream: bytes | None = None

        if has_sidecar:
            if len(data) < offset + PAYLOAD_LEN_STRUCT.size:
                raise ValueError("Corrupted or truncated container payload for sidecar length")
            (s_len,) = PAYLOAD_LEN_STRUCT.unpack_from(data, offset)
            offset += PAYLOAD_LEN_STRUCT.size
            if len(data) < offset + s_len:
                raise ValueError("Corrupted or truncated container payload for sidecar")
            sidecar_bitstream = data[offset : offset + s_len] if s_len > 0 else None
            offset += s_len

        if len(data) < offset + PAYLOAD_LEN_STRUCT.size:
            raise ValueError("Corrupted or truncated container payload for hyper-latents length")
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
        if len(data) != offset + y_len:
            raise ValueError(
                f"Unexpected trailing data in container: total length {len(data)} exceeds expected {offset + y_len}"
            )

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
            sidecar_bitstream=sidecar_bitstream,
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
    saliency_map: str | Path | Image.Image | np.ndarray | torch.Tensor | None = None,
) -> SancContainer:
    r"""Compresses an input image into a SancContainer.

    Supports Model A, Model B, Model C, Model D, and Model C-Transmitted.

    Pipeline:
    1. Symmetric reflection padding to multiples of 64.
    2. Analysis transforms (g_a, h_a) on specified device.
    3. Integer rounding of hyper-latents q_z and entropy coding via EntropyBottleneck CDFs.
    4. Deterministic single-threaded CPU hyper-synthesis (h_s) to derive discrete scale indices
       and adaptive quantization step Delta.
    5. Adaptive latent scaling and integer rounding of main latents q_y, encoded via GaussianConditional CDFs.
    6. Packaging into SancContainer with unaligned dimension recovery metadata and optional sidecar payload.
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

    m_id = getattr(model, "model_id", 0)
    m_type = getattr(model, "model_type", "A")
    sidecar_bytes: bytes | None = None

    if m_id in (2, 3) or m_type in ("C", "D"):
        from src.data.padding import compute_valid_latent_mask

        valid_mask = compute_valid_latent_mask(orig_shape=(orig_h, orig_w), pad_bounds=bounds, device="cpu")
        _, scale_indexes, delta, _, _ = cpu_engine.forward_allocation(q_z, valid_mask=valid_mask)
        delta_dev = delta.to(device=target_device, dtype=torch.float32)
        q_y = torch.round(y.float() / delta_dev).cpu()
    elif m_id == 4 or m_type == "C-Transmitted":
        from src.codec.allocation import compute_log_delta
        from src.codec.model_c_transmitted import compress_saliency_thumbnail, decompress_saliency_thumbnail
        from src.data.padding import compute_valid_latent_mask

        if saliency_map is None:
            s_tensor = torch.full((1, 1, orig_h, orig_w), 0.5, dtype=torch.float32)
        else:
            s_tensor = load_image_tensor(saliency_map)
            if s_tensor.shape[1] > 1:
                s_tensor = s_tensor.mean(dim=1, keepdim=True)
            if s_tensor.shape[-2:] != (orig_h, orig_w):
                s_tensor = torch.nn.functional.interpolate(
                    s_tensor, size=(orig_h, orig_w), mode="bilinear", align_corners=False
                )

        s_pad, _ = pad_symmetric_centered(s_tensor, pad_bounds=bounds)
        latent_shape = (y.shape[-2], y.shape[-1])
        comp_bytes = compress_saliency_thumbnail(s_pad, target_shape=latent_shape)
        sidecar_bytes = comp_bytes[0] if isinstance(comp_bytes, list) else comp_bytes

        s_dec = decompress_saliency_thumbnail(sidecar_bytes, shape=latent_shape, device="cpu")
        valid_mask = compute_valid_latent_mask(orig_shape=(orig_h, orig_w), pad_bounds=bounds, device="cpu")
        log_delta, delta = compute_log_delta(s_dec, valid_mask=valid_mask)
        _, scale_indexes = cpu_engine.forward(q_z, log_delta=log_delta)
        delta_dev = delta.to(device=target_device, dtype=torch.float32)
        q_y = torch.round(y.float() / delta_dev).cpu()
    else:
        _, scale_indexes = cpu_engine.forward(q_z)
        q_y = torch.round(y.float()).cpu()

    y_bitstream = encode_latents(q_y, scale_indexes, model.gaussian_conditional, backend=backend)

    return SancContainer(
        orig_height=orig_h,
        orig_width=orig_w,
        pad_bounds=bounds,
        z_bitstream=z_bitstream,
        y_bitstream=y_bitstream,
        version=CURRENT_VERSION,
        model_id=m_id,
        sidecar_bitstream=sidecar_bytes,
    )


def decode_image(
    container: SancContainer,
    model: ModelA,
    device: torch.device | str = "cpu",
    backend: str = "auto",
) -> torch.Tensor:
    r"""Decompresses a SancContainer back into a reconstructed image tensor (1, 3, H, W) in [0, 1].

    Supports Model A, Model B, Model C, Model D, and Model C-Transmitted.

    Pipeline:
    1. Recover padded latent spatial dimensions from header metadata.
    2. rANS decoding of hyper-latents q_z via EntropyBottleneck CDFs.
    3. Deterministic single-threaded CPU hyper-synthesis (h_s) to derive identical scale indices
       and adaptive quantization step Delta.
    4. rANS decoding of main latents q_y via GaussianConditional CDFs.
    5. Adaptive de-quantization (q_y * Delta) and synthesis transform (g_s) on specified device.
    6. Centered unpadding back to exact original image dimensions.
    """
    # Ensure CDF tables are compiled
    if model.entropy_bottleneck._quantized_cdf.numel() == 0:
        model.entropy_bottleneck.update()
    if model.gaussian_conditional._quantized_cdf.numel() == 0:
        model.gaussian_conditional.update()

    expected_id = getattr(model, "model_id", None)
    if expected_id is not None and container.model_id != expected_id:
        raise ValueError(f"Container model_id ({container.model_id}) does not match model.model_id ({expected_id})")

    h_pad = container.orig_height + container.pad_bounds.pad_top + container.pad_bounds.pad_bottom
    w_pad = container.orig_width + container.pad_bounds.pad_left + container.pad_bounds.pad_right
    h_z = h_pad // 64
    w_z = w_pad // 64
    h_y = h_pad // 16
    w_y = w_pad // 16

    # Hyper-latent entropy decoding
    z_channels = getattr(model, "n_channels", getattr(model.entropy_bottleneck, "channels", 128))
    q_z = decode_hyper_latents(
        container.z_bitstream, (1, z_channels, h_z, w_z), model.entropy_bottleneck, backend=backend
    )

    # Deterministic CPU parameter derivation
    cpu_engine = DeterministicCPUEngine.from_model(model)

    m_id = getattr(container, "model_id", getattr(model, "model_id", 0))
    m_type = getattr(model, "model_type", "A")
    target_device = torch.device(device)

    if m_id in (2, 3) or m_type in ("C", "D"):
        from src.data.padding import compute_valid_latent_mask

        valid_mask = compute_valid_latent_mask(
            orig_shape=(container.orig_height, container.orig_width), pad_bounds=container.pad_bounds, device="cpu"
        )
        _, scale_indexes, delta, _, _ = cpu_engine.forward_allocation(q_z, valid_mask=valid_mask)
        q_y = decode_latents(container.y_bitstream, scale_indexes, model.gaussian_conditional, backend=backend)
        delta_dev = delta.to(device=target_device, dtype=torch.float32)
        q_y_dev = q_y.to(target_device).float() * delta_dev
    elif m_id == 4 or m_type == "C-Transmitted":
        from src.codec.allocation import compute_log_delta
        from src.codec.model_c_transmitted import decompress_saliency_thumbnail
        from src.data.padding import compute_valid_latent_mask

        if container.sidecar_bitstream is None:
            raise ValueError("Model C-Transmitted container is missing sidecar_bitstream payload")

        s_dec = decompress_saliency_thumbnail(container.sidecar_bitstream, shape=(h_y, w_y), device="cpu")
        valid_mask = compute_valid_latent_mask(
            orig_shape=(container.orig_height, container.orig_width), pad_bounds=container.pad_bounds, device="cpu"
        )
        log_delta, delta = compute_log_delta(s_dec, valid_mask=valid_mask)
        _, scale_indexes = cpu_engine.forward(q_z, log_delta=log_delta)
        q_y = decode_latents(container.y_bitstream, scale_indexes, model.gaussian_conditional, backend=backend)
        delta_dev = delta.to(device=target_device, dtype=torch.float32)
        q_y_dev = q_y.to(target_device).float() * delta_dev
    else:
        _, scale_indexes = cpu_engine.forward(q_z)
        q_y = decode_latents(container.y_bitstream, scale_indexes, model.gaussian_conditional, backend=backend)
        q_y_dev = q_y.to(target_device).float()

    model.eval()
    model.to(target_device)

    with torch.no_grad():
        if target_device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x_hat_pad = model.g_s(q_y_dev)
        else:
            x_hat_pad = model.g_s(q_y_dev)

    # Centered unpadding and range clamp
    x_hat = unpad_centered(x_hat_pad, container.pad_bounds)
    return torch.clamp(x_hat, 0.0, 1.0)


def create_model(model_type_or_id: str | int = 0, **kwargs: Any) -> ModelA:
    r"""Instantiates the appropriate model architecture given a model type string or integer ID."""
    from src.codec.model_a import ModelA
    from src.codec.model_b import ModelB
    from src.codec.model_c import ModelC
    from src.codec.model_c_transmitted import ModelCTransmitted
    from src.codec.model_d import ModelD

    if isinstance(model_type_or_id, int):
        id_map = {
            0: ModelA,
            1: ModelB,
            2: ModelC,
            3: ModelD,
            4: ModelCTransmitted,
        }
        if model_type_or_id not in id_map:
            raise ValueError(f"Unknown model_id: {model_type_or_id}. Supported: {list(id_map.keys())}")
        return id_map[model_type_or_id](**kwargs)

    type_str = str(model_type_or_id).strip().upper()
    str_map = {
        "A": ModelA,
        "0": ModelA,
        "B": ModelB,
        "1": ModelB,
        "C": ModelC,
        "2": ModelC,
        "D": ModelD,
        "3": ModelD,
        "C-TRANSMITTED": ModelCTransmitted,
        "CTRANSMITTED": ModelCTransmitted,
        "4": ModelCTransmitted,
    }
    if type_str not in str_map:
        raise ValueError(f"Unknown model_type: {model_type_or_id}. Supported: {list(str_map.keys())}")
    return str_map[type_str](**kwargs)
