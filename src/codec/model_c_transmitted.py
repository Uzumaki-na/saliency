"""Model C-Transmitted (transmitted saliency side-information control codec).

This module implements:
- compress_saliency_thumbnail: Losslessly compresses continuous visual saliency maps into
  8-bit Deflate (zlib) thumbnail bitstreams downsampled to the latent grid (1/16 scale).
- decompress_saliency_thumbnail: Decompresses Deflate-compressed 8-bit thumbnails into [0, 1]
  FP32 saliency tensors, guaranteeing exact encoder-decoder synchrony without drift.
- compute_sidecar_bpp: Calculates empirical side-channel bits-per-pixel overhead.
- ModelCTransmittedOutput: Output container holding reconstructed image, latents, scales,
  saliency step delta, sidecar bytes, and sidecar BPP.
- ModelCTransmitted: Control variant codec transmitting compressed visual saliency maps as
  side information to empirically benchmark the bitstream overhead against hyperprior-derived Model C.
- load_pretrained_bmshj2018: Pretrained weight loading utility for ModelCTransmitted instances.
"""

from __future__ import annotations

import logging
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
import torch.nn.functional as F

from src.codec.allocation import (
    compute_effective_log_scale,
    compute_log_delta,
)
from src.codec.model_a import ModelA, ModelOutput
from src.codec.model_b import (
    _unwrap_state_dict,
    remap_compressai_state_dict,
)
from src.codec.model_b import (
    load_pretrained_bmshj2018 as _load_pretrained_bmshj2018,
)

logger = logging.getLogger(__name__)

# Model ID constant for Model C-Transmitted (aligned with container spec 0x04)
MODEL_ID_C_TRANSMITTED: int = 4


def compress_saliency_thumbnail(
    s: torch.Tensor,
    target_shape: tuple[int, int] | None = None,
    compression_level: int = 9,
) -> bytes | list[bytes]:
    r"""Losslessly compresses continuous visual saliency maps into 8-bit Deflate thumbnail bitstreams.

    Protocol:
    1. Downsamples ``s`` to ``target_shape`` (e.g. latent grid :math:`H/16 \times W/16`) via bilinear interpolation.
    2. Clamps values to :math:`[0, 1]` and quantizes to 8-bit unsigned integers:
       :math:`S_8 = \operatorname{round}(255.0 \cdot S)`.
    3. Serializes uint8 byte buffer and applies standard Deflate (zlib level 9) lossless compression.

    Args:
        s: Saliency map tensor of shape (H, W), (1, H, W), (B, 1, H, W), or (B, H, W).
        target_shape: Optional target spatial shape (H_latent, W_latent). If None, retains current resolution.
        compression_level: zlib compression level in [1, 9] (default 9, maximum compression).

    Returns:
        bytes (for single sample) or list[bytes] (for batch size B > 1).

    Raises:
        TypeError: If ``s`` is not a torch.Tensor.
        ValueError: If ``s`` has invalid dimensions or values outside [-1e-2, 1 + 1e-2].
    """
    if not isinstance(s, torch.Tensor):
        raise TypeError(f"s must be a torch.Tensor, got {type(s)}")

    if s.ndim not in (2, 3, 4):
        raise ValueError(f"s must be 2D, 3D, or 4D, got ndim={s.ndim} with shape {tuple(s.shape)}")

    if s.ndim == 4 and s.shape[1] != 1:
        raise ValueError(f"Expected single channel (C=1) for 4D saliency tensor, got shape {tuple(s.shape)}")

    if torch.isnan(s).any() or torch.isinf(s).any():
        raise ValueError("s must not contain NaN or Inf values")

    # Range validation
    if (s < -1e-2).any() or (s > 1.0 + 1e-2).any():
        raise ValueError(f"s values must be in [0, 1], got min {s.min().item():.4f} and max {s.max().item():.4f}")

    # Standardize to 4D: (B, 1, H, W)
    is_batched = s.ndim == 4 and s.shape[0] > 1
    if s.ndim == 2:
        s_4d = s.unsqueeze(0).unsqueeze(0)
    elif s.ndim == 3:
        if s.shape[0] == 1:
            s_4d = s.unsqueeze(0)
        else:
            s_4d = s.unsqueeze(1)
    else:
        s_4d = s

    # Spatial downsampling to latent grid via bilinear interpolation
    if target_shape is not None and s_4d.shape[-2:] != target_shape:
        s_4d = F.interpolate(
            s_4d.float(),
            size=target_shape,
            mode="bilinear",
            align_corners=False,
        )

    # 8-bit quantization: round(255 * s)
    s_clamped = torch.clamp(s_4d.float(), 0.0, 1.0)
    s_uint8 = torch.round(s_clamped * 255.0).to(torch.uint8)

    # Deflate byte compression: batch host transfer to avoid per-sample overhead
    s_numpy = s_uint8[:, 0].detach().cpu().contiguous().numpy()
    compressed_list: list[bytes] = []
    b = s_numpy.shape[0]
    for i in range(b):
        sample_bytes = s_numpy[i].tobytes()
        compressed = zlib.compress(sample_bytes, level=compression_level)
        compressed_list.append(compressed)

    if not is_batched and len(compressed_list) == 1:
        return compressed_list[0]
    return compressed_list


def decompress_saliency_thumbnail(
    s_bytes: bytes | list[bytes],
    shape: tuple[int, int],
    device: str | torch.device | None = None,
) -> torch.Tensor:
    r"""Decompresses Deflate-compressed 8-bit saliency thumbnail into continuous [0, 1] FP32 tensor.

    Protocol:
    1. Decompresses raw bytes with ``zlib.decompress``.
    2. Converts byte buffer to uint8 array and reshapes to ``(B, 1, H, W)``.
    3. Normalizes to FP32 in :math:`[0, 1]`: :math:`S_{\text{dec}} = S_8 / 255.0`.

    Args:
        s_bytes: Compressed Deflate byte payload (single bytes or list of bytes for batch).
        shape: Expected spatial shape (H_latent, W_latent) of the uncompressed thumbnail.
        device: Device to map the decompressed tensor to.

    Returns:
        torch.Tensor: Decompressed visual saliency tensor of shape (B, 1, H, W) in [0, 1] FP32.

    Raises:
        TypeError: If s_bytes is neither bytes nor list of bytes.
        ValueError: If shape is invalid or decompressed byte count does not match expected size.
    """
    if isinstance(s_bytes, bytes):
        byte_list = [s_bytes]
    elif isinstance(s_bytes, (list, tuple)):
        byte_list = list(s_bytes)
    else:
        raise TypeError(f"s_bytes must be bytes or list of bytes, got {type(s_bytes)}")

    if not byte_list:
        raise ValueError("s_bytes cannot be empty")

    if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(f"shape must be a positive 2-tuple (H, W), got {shape}")

    expected_sample_bytes = shape[0] * shape[1]
    tensors: list[torch.Tensor] = []

    for idx, raw_comp in enumerate(byte_list):
        if not isinstance(raw_comp, bytes):
            raise TypeError(f"Element at index {idx} must be bytes, got {type(raw_comp)}")
        try:
            raw_decomp = zlib.decompress(raw_comp)
        except zlib.error as err:
            raise ValueError(f"Corrupt or invalid Deflate payload at index {idx}: {err}") from err
        if len(raw_decomp) != expected_sample_bytes:
            raise ValueError(
                f"Decompressed byte length {len(raw_decomp)} does not match expected {expected_sample_bytes} "
                f"for spatial shape {shape}"
            )
        arr = np.frombuffer(raw_decomp, dtype=np.uint8).reshape((1, 1, shape[0], shape[1]))
        tensors.append(torch.from_numpy(arr.copy()))

    stacked_uint8 = torch.cat(tensors, dim=0)
    target_dev = torch.device(device) if device is not None else torch.device("cpu")
    # Strict FP32 scaling: S_dec in [0, 1]
    return (stacked_uint8.to(device=target_dev, dtype=torch.float32) / 255.0).contiguous()


def compute_sidecar_bpp(
    s_bytes_or_len: bytes | int | list[bytes] | list[int] | torch.Tensor,
    num_pixels: int | float,
    *,
    device: str | torch.device | None = None,
    as_tensor: bool = False,
) -> float | torch.Tensor:
    r"""Computes sidecar bits-per-pixel overhead from compressed thumbnail bytes.

    .. math::
        \text{BPP}_{\text{side}} = \frac{8 \cdot L_s}{\text{num\_pixels}}

    Args:
        s_bytes_or_len: Single bytes, byte length, list of bytes, list of lengths, or tensor.
        num_pixels: Total original image pixels (B * H * W) for bitrate normalization.
        device: Device to place the returned tensor on if as_tensor=True.
        as_tensor: If True, returns a scalar torch.Tensor; otherwise returns a Python float.

    Returns:
        float | torch.Tensor: Bits per pixel allocated to the transmitted saliency sidecar.
    """
    if num_pixels <= 0:
        raise ValueError(f"num_pixels must be positive, got {num_pixels}")

    total_bytes = 0
    if isinstance(s_bytes_or_len, bytes):
        total_bytes = len(s_bytes_or_len)
    elif isinstance(s_bytes_or_len, int):
        total_bytes = s_bytes_or_len
    elif isinstance(s_bytes_or_len, (list, tuple)):
        for item in s_bytes_or_len:
            if isinstance(item, bytes):
                total_bytes += len(item)
            elif isinstance(item, int):
                total_bytes += item
            else:
                raise TypeError(f"Unexpected item in byte sequence: {type(item)}")
    elif isinstance(s_bytes_or_len, torch.Tensor):
        total_bytes = int(s_bytes_or_len.sum().item())
    else:
        raise TypeError(f"Unsupported type for s_bytes_or_len: {type(s_bytes_or_len)}")

    bpp_val = (float(total_bytes) * 8.0) / float(num_pixels)
    if as_tensor:
        target_dev = torch.device(device) if device is not None else torch.device("cpu")
        return torch.tensor(bpp_val, dtype=torch.float32, device=target_dev)
    return bpp_val


class ModelCTransmittedOutput(ModelOutput):
    r"""Output container for Model C-Transmitted forward execution.

    Inherits base compression fields from :class:`ModelOutput`:
    - ``x_hat``: Reconstructed image tensor.
    - ``y``: Continuous main latents from :math:`g_a`.
    - ``z``: Continuous hyper-latents from :math:`h_a`.
    - ``log_sigma``: Predicted conditional log-scale from :math:`h_s`.
    - ``y_hat``: Quantized main latents.
    - ``z_hat``: Quantized hyper-latents.
    - ``y_likelihoods``: Main latent likelihoods under Gaussian conditional.
    - ``z_likelihoods``: Hyper-latent likelihoods under EntropyBottleneck.

    Model C-Transmitted specific attributes:
    - ``s_dec``: Decoded post-quantization saliency map in :math:`[0, 1]`.
    - ``log_delta``: Log-quantization step on 1/32 grid.
    - ``delta``: Quantization step multiplier :math:`\Delta \in [0.25, 4.0]`.
    - ``log_sigma_eff``: Effective log-scale :math:`\log \sigma - \log \Delta`.
    - ``sigma_eff``: Effective scale :math:`\sigma / \Delta`.
    - ``sidecar_bytes``: Compressed Deflate byte payload(s).
    - ``sidecar_bpp``: Empirical sidecar bits-per-pixel.
    - ``s_pred``: Alias for ``s_dec`` for compatibility with generic saliency tracking.
    """

    def __init__(
        self,
        x_hat: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        log_sigma: torch.Tensor,
        log_sigma_eff: torch.Tensor,
        sigma_eff: torch.Tensor,
        s_dec: torch.Tensor,
        log_delta: torch.Tensor,
        delta: torch.Tensor,
        sidecar_bytes: bytes | list[bytes],
        sidecar_bpp: torch.Tensor,
        y_hat: torch.Tensor | None = None,
        z_hat: torch.Tensor | None = None,
        y_likelihoods: torch.Tensor | None = None,
        z_likelihoods: torch.Tensor | None = None,
        s_pred: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            x_hat=x_hat,
            y=y,
            z=z,
            log_sigma=log_sigma,
            y_hat=y_hat if y_hat is not None else y,
            z_hat=z_hat if z_hat is not None else z,
            y_likelihoods=y_likelihoods,
            z_likelihoods=z_likelihoods,
        )
        self["log_sigma_eff"] = log_sigma_eff
        self["sigma_eff"] = sigma_eff
        self["s_dec"] = s_dec
        self["log_delta"] = log_delta
        self["delta"] = delta
        self["sidecar_bytes"] = sidecar_bytes
        self["sidecar_bpp"] = sidecar_bpp
        self["s_pred"] = s_pred if s_pred is not None else s_dec


class ModelCTransmitted(ModelA):
    r"""Model C-Transmitted: explicit saliency side-information control codec (R7).

    Protocol:
    1. Encodes input image :math:`x` to continuous latents :math:`y = g_a(x)` and hyper-latents :math:`z = h_a(y)`.
    2. Quantizes hyper-latents :math:`\hat{z}` and predicts conditional scale :math:`\log \sigma = h_s(\hat{z})`.
    3. Downsamples input ground truth saliency map :math:`S` to latent grid :math:`H/16 \times W/16`, quantizes to
       8-bit integers, and compresses losslessly via Deflate (zlib level 9).
    4. Decompresses thumbnail to :math:`S_{\text{dec}} = S_8 / 255.0` at both encoder and decoder, deriving identical
       quantization step :math:`\Delta(S_{\text{dec}})` without drift.
    5. Scales continuous latents :math:`y / \Delta` and scales :math:`\sigma / \Delta` on the CompressAI CDF tables.
    6. Reconstructs image :math:`\hat{x} = g_s(\hat{y})` and reports explicit side-channel bitrate overhead
       :math:`\text{BPP}_{\text{side}} = \frac{8 \cdot L_s}{B \cdot H \cdot W}`.

    During rate-distortion optimization, total bitrate accounts for the side-information stream:

    .. math::
        \mathcal{L} = R_{\text{bpp}} + \lambda \cdot 255^2 \cdot \text{Distortion}(S')

    where :math:`R_{\text{bpp}} = \text{BPP}_y + \text{BPP}_z + \text{BPP}_{\text{side}}`.
    """

    model_type: str = "C-Transmitted"
    model_id: int = MODEL_ID_C_TRANSMITTED

    def __init__(
        self,
        in_channels: int = 3,
        n_channels: int = 128,
        m_channels: int = 192,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            n_channels=n_channels,
            m_channels=m_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        *,
        training: bool | None = None,
        saliency_map: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        target_shape: tuple[int, int] | None = None,
    ) -> ModelCTransmittedOutput:
        r"""Forward execution with explicit saliency thumbnail compression and adaptive scaling.

        Args:
            x: Input image tensor (B, C, H, W) where H and W are multiples of 64.
            s: Ground truth visual saliency map in [0, 1]. Required for Model C-Transmitted.
            valid_mask: Optional spatial valid mask for unpadded region evaluation.
            training: If True, uses uniform noise relaxation; if False, uses discrete rounding.
            saliency_map: Keyword alias for s.
            mask: Keyword alias for valid_mask.
            target_shape: Optional explicit latent grid shape override.

        Returns:
            ModelCTransmittedOutput container with reconstructions, intermediate representations,
            and sidecar bitrate.
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (B, C, H, W), got {x.dim()}D tensor of shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Channel mismatch: expected {self.in_channels}, got {x.shape[1]}")

        _, _, h, w = x.shape
        if h % 64 != 0 or w % 64 != 0:
            raise ValueError(
                f"Input spatial dimensions ({h}, {w}) must be multiples of 64 for consistent downsampling/upsampling. "
                f"Pad the input using pad_symmetric_centered before model execution."
            )

        resolved_s = s if s is not None else saliency_map
        resolved_mask = valid_mask if valid_mask is not None else mask

        if resolved_s is None:
            raise ValueError(
                "Model C-Transmitted requires a visual saliency map 's' to compress and transmit as side information."
            )

        # 1. Main analysis transform
        y = self.g_a(x)

        # 2. Hyper analysis transform
        z = self.h_a(y)

        # 3. Hyperprior bottleneck
        z_hat, z_likelihoods = self.entropy_bottleneck(z, training=training)

        # 4. Hyper synthesis transform predicting conditional log_sigma
        log_sigma = self.h_s(z_hat)

        # Determine latent grid spatial shape (H/16, W/16)
        latent_shape = target_shape if target_shape is not None else (y.shape[-2], y.shape[-1])

        # 5. Compress saliency map into 8-bit Deflate thumbnail bitstream
        sidecar_bytes = compress_saliency_thumbnail(resolved_s, target_shape=latent_shape)

        # 6. Decoded post-quantization thumbnail S_dec in [0, 1]
        # In forward pass, computing directly from the downsampled 8-bit quantization on device yields
        # the exact bit-level equivalent of decompress_saliency_thumbnail while eliminating redundant CPU roundtrip.
        s_4d = resolved_s
        if s_4d.ndim == 2:
            s_4d = s_4d.unsqueeze(0).unsqueeze(0)
        elif s_4d.ndim == 3:
            s_4d = s_4d.unsqueeze(0) if s_4d.shape[0] == 1 else s_4d.unsqueeze(1)
        if s_4d.shape[-2:] != latent_shape:
            s_down = F.interpolate(s_4d.float(), size=latent_shape, mode="bilinear", align_corners=False)
        else:
            s_down = s_4d.float()
        s_clamped = torch.clamp(s_down, 0.0, 1.0)
        s_uint8 = torch.round(s_clamped * 255.0).to(torch.uint8)
        s_dec = (s_uint8.to(device=y.device, dtype=torch.float32) / 255.0).contiguous()

        # 7. Compute sidecar BPP
        total_orig_pixels = x.shape[0] * x.shape[-2] * x.shape[-1]
        sidecar_bpp = compute_sidecar_bpp(
            sidecar_bytes,
            num_pixels=total_orig_pixels,
            device=x.device,
            as_tensor=True,
        )

        # 8-12. Compute explicit quantization step Delta and execute adaptive scaling in FP32 outside autocast
        # Synchronous with decoder; gradients are detached from rate-distortion loss
        device_type = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Align mask if needed
            vm = resolved_mask
            if vm is not None:
                if vm.ndim == 2:
                    vm = vm.unsqueeze(0).unsqueeze(0)
                elif vm.ndim == 3:
                    vm = vm.unsqueeze(1)
                if vm.shape[-2:] != s_dec.shape[-2:]:
                    vm = F.interpolate(vm.float(), size=s_dec.shape[-2:], mode="nearest").bool()

            log_delta, delta = compute_log_delta(s_dec, valid_mask=vm)
            log_delta_detached = log_delta.detach()
            delta_detached = delta.detach()

            # 9. Compute effective conditional scale: log_sigma_eff = log_sigma - log_delta
            log_sigma_eff = compute_effective_log_scale(log_sigma, log_delta_detached)
            sigma_eff = torch.exp(log_sigma_eff)

            # 10. Adaptive latent scaling: q = round(y / delta)
            y_scaled = y / delta_detached

            # 11. Quantization and likelihood modeling outside autocast
            y_hat_scaled, y_likelihoods = self.gaussian_conditional(y_scaled, sigma_eff, training=training)

            # 12. De-quantization: y_hat = y_hat_scaled * delta
            y_hat = y_hat_scaled * delta_detached

        # 13. Synthesis transform
        x_hat = self.g_s(y_hat)

        return ModelCTransmittedOutput(
            x_hat=x_hat,
            y=y,
            z=z,
            log_sigma=log_sigma,
            log_sigma_eff=log_sigma_eff,
            sigma_eff=sigma_eff,
            s_dec=s_dec,
            log_delta=log_delta,
            delta=delta,
            sidecar_bytes=sidecar_bytes,
            sidecar_bpp=sidecar_bpp,
            y_hat=y_hat,
            z_hat=z_hat,
            y_likelihoods=y_likelihoods,
            z_likelihoods=z_likelihoods,
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        r"""Loads state dict with automatic unnesting, key remapping, and Model C compatibility.

        Tolerates extra ``h_s.conv_saliency`` weights from Model C / Model D checkpoints without failing strict check.
        """
        raw_dict = _unwrap_state_dict(state_dict)
        target_device = next(self.parameters()).device if list(self.parameters()) else None
        remapped = remap_compressai_state_dict(raw_dict)

        if strict:
            current_keys = set(self.state_dict().keys())
            provided_keys = set(remapped.keys())

            # Filter out known extra keys (such as conv_saliency from Model C checkpoints)
            filtered_provided = {k for k in provided_keys if not k.startswith("h_s.conv_saliency.")}

            missing_keys = current_keys - filtered_provided
            unexpected_keys = filtered_provided - current_keys

            if missing_keys or unexpected_keys:
                error_msgs = []
                if missing_keys:
                    error_msgs.append(f"Missing key(s) in state_dict: {', '.join(sorted(missing_keys))}.")
                if unexpected_keys:
                    error_msgs.append(f"Unexpected key(s) in state_dict: {', '.join(sorted(unexpected_keys))}.")
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t" + "\n\t".join(error_msgs)
                )

        result = super().load_state_dict(remapped, strict=False, assign=assign)
        if target_device is not None:
            self.to(target_device)
        return result

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path | Mapping[str, Any],
        strict: bool = True,
        device: str | torch.device | None = None,
        **kwargs: Any,
    ) -> Self:
        r"""Constructs a ModelCTransmitted instance and loads weights from a checkpoint or state dict."""
        model = cls(**kwargs)
        if isinstance(checkpoint, (str, Path)):
            path = Path(checkpoint)
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
            map_loc = device if device is not None else "cpu"
            raw_checkpoint = torch.load(path, map_location=map_loc, weights_only=False)
        elif isinstance(checkpoint, (dict, Mapping)):
            raw_checkpoint = checkpoint
        else:
            raise TypeError(
                f"Expected checkpoint path (str, Path) or state dict mapping, got {type(checkpoint).__name__}"
            )
        model.load_state_dict(raw_checkpoint, strict=strict)
        if device is not None:
            model.to(torch.device(device))
        return model


def load_pretrained_bmshj2018(
    checkpoint: str | Path | Mapping[str, Any] | ModelA,
    target_or_checkpoint: str | Path | Mapping[str, Any] | ModelA | None = None,
    *,
    model: ModelA | None = None,
    strict: bool = True,
    device: str | torch.device | None = None,
    **model_kwargs: Any,
) -> ModelCTransmitted | ModelA:
    r"""Loads pretrained bmshj2018 weights into a ModelCTransmitted (or target model) instance."""
    if model is None and not isinstance(checkpoint, ModelA) and not isinstance(target_or_checkpoint, ModelA):
        model = ModelCTransmitted(**model_kwargs)
    return _load_pretrained_bmshj2018(
        checkpoint,
        target_or_checkpoint,
        model=model,
        strict=strict,
        device=device,
        **model_kwargs,
    )


__all__ = [
    "MODEL_ID_C_TRANSMITTED",
    "ModelCTransmitted",
    "ModelCTransmittedOutput",
    "compress_saliency_thumbnail",
    "compute_sidecar_bpp",
    "decompress_saliency_thumbnail",
    "load_pretrained_bmshj2018",
]
