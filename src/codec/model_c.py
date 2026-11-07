"""Model C (variational compression with explicit adaptive quantization step allocation).

This module implements:
- DualHeadHyperSynthesisTransform: Dual-head hyper-synthesis network upsampling quantized
  hyper-latents z_hat by 4x to synchronously predict conditional log-scales log_sigma and
  visual saliency maps S_pred in [0, 1].
- ModelCOutput: Output container holding reconstructed image, continuous/quantized latents,
  predicted scale, visual saliency, discrete log-step log_delta, step delta, and likelihoods.
- ModelC: Variational image compression model subclassing ModelA. Dynamically scales latents y
  and conditional scales sigma using explicit log-domain allocation delta(S_pred) with detached
  rate-distortion gradients and auxiliary gradient isolation.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.codec.allocation import (
    compute_effective_log_scale,
    compute_log_delta,
)
from src.codec.model_a import ModelA, ModelOutput
from src.codec.model_b import (
    _unwrap_state_dict,
    load_pretrained_bmshj2018,
    remap_compressai_state_dict,
)

logger = logging.getLogger(__name__)


class DualHeadHyperSynthesisTransform(nn.Module):
    r"""Dual-head hyper-synthesis transform :math:`h_s` mapping quantized hyper-latents :math:`\hat{z}`
    to conditional log-scale :math:`\log \sigma` and predicted visual saliency :math:`S_{\text{pred}} \in [0, 1]`.

    Upsamples spatial resolution by 4x through a shared 2-stage transposed convolution trunk
    (from 1/64 to 1/16 grid), then branches into:
    1. Scale head: conv3 (128 -> 192) emitting conditional scale representation log_sigma.
    2. Saliency head: conv_saliency (128 -> 1) + Sigmoid emitting predicted visual saliency S_pred.

    Maintains exact naming compatibility with bmshj2018 / ModelA:
    - deconv1
    - deconv2
    - conv3
    """

    def __init__(
        self,
        in_channels: int = 128,
        n_channels: int = 128,
        out_channels: int = 192,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_channels = int(n_channels)
        self.out_channels = int(out_channels)

        # Shared 4x upsampling feature trunk (matches ModelA h_s structure)
        self.deconv1 = nn.ConvTranspose2d(
            self.in_channels,
            self.n_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(
            self.n_channels,
            self.n_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        self.relu2 = nn.ReLU(inplace=True)

        # Head 1: Conditional scale head (named conv3 for exact ModelA weight compatibility)
        self.conv3 = nn.Conv2d(
            self.n_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Head 2: Visual saliency prediction head emitting S_pred in [0, 1]
        self.conv_saliency = nn.Conv2d(
            self.n_channels,
            1,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.sigmoid_saliency = nn.Sigmoid()

        # Initialize saliency head weights
        nn.init.xavier_uniform_(self.conv_saliency.weight)
        if self.conv_saliency.bias is not None:
            nn.init.zeros_(self.conv_saliency.bias)

    @property
    def conv_scale(self) -> nn.Conv2d:
        """Alias for conv3 scale head."""
        return self.conv3

    def forward(self, z_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Synthesizes log_sigma and s_pred synchronously in strict FP32.

        Args:
            z_hat: Quantized hyper-latent tensor (B, 128, H/64, W/64).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (log_sigma, s_pred) on 1/16 grid.
        """
        if z_hat.dim() != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), got {z_hat.dim()}D tensor of shape {tuple(z_hat.shape)}"
            )
        if z_hat.shape[1] != self.in_channels:
            raise ValueError(f"Channel mismatch: expected {self.in_channels}, got {z_hat.shape[1]}")

        device_type = z_hat.device.type if z_hat.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            z_f = z_hat.float()
            feat = self.relu1(self.deconv1(z_f))
            feat = self.relu2(self.deconv2(feat))
            log_sigma = self.conv3(feat)
            s_pred = self.sigmoid_saliency(self.conv_saliency(feat))

        return log_sigma, s_pred


class ModelCOutput(ModelOutput):
    """Container for Model C outputs supporting dictionary and attribute access."""

    def __init__(
        self,
        x_hat: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        log_sigma: torch.Tensor,
        log_sigma_eff: torch.Tensor,
        s_pred: torch.Tensor,
        log_delta: torch.Tensor,
        delta: torch.Tensor,
        y_hat: torch.Tensor | None = None,
        z_hat: torch.Tensor | None = None,
        y_likelihoods: torch.Tensor | None = None,
        z_likelihoods: torch.Tensor | None = None,
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
        self["s_pred"] = s_pred
        self["log_delta"] = log_delta
        self["delta"] = delta

    def to_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns primary outputs as a 5-tuple: (x_hat, y, z, log_sigma, s_pred)."""
        return self["x_hat"], self["y"], self["z"], self["log_sigma"], self["s_pred"]


class ModelC(ModelA):
    r"""Model C variational image compression architecture with explicit adaptive quantization step allocation.

    Combines:
    - g_a: Analysis transform (16x downsampling)
    - g_s: Synthesis transform (16x upsampling)
    - h_a: Hyper-analysis transform (4x downsampling on latents y, total 64x)
    - h_s: DualHeadHyperSynthesisTransform (4x upsampling on hyper-latents z, emitting log_sigma and s_pred)
    - entropy_bottleneck: Factorized EntropyBottleneck for hyper-latents z
    - gaussian_conditional: GaussianConditional for adaptive latents y / delta conditioned on effective scale sigma_eff

    Allocation & Gradient Isolation:
    - Saliency prediction head h_s.conv_saliency emits S_pred in [0, 1].
    - Discrete log-step log_delta and step delta in [0.25, 4.0] are derived via compute_log_delta(s_pred.detach()).
    - Detached RD gradients: Reconstruction and rate losses have zero gradient with respect to delta and s_pred.
    - Detached y for h_a: Auxiliary saliency gradients from s_pred flow into h_a but are cut before g_a.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_channels: int = 128,
        m_channels: int = 192,
        detach_y_for_ha: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            n_channels=n_channels,
            m_channels=m_channels,
        )
        self.detach_y_for_ha = bool(detach_y_for_ha)
        self.h_s = DualHeadHyperSynthesisTransform(
            in_channels=self.n_channels,
            n_channels=self.n_channels,
            out_channels=self.m_channels,
        )

    def synthesize_scales_and_saliency(
        self,
        z_hat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Synthesizes log-scales log_sigma and visual saliency s_pred."""
        return self.h_s(z_hat)

    def synthesize_scales(self, z_hat: torch.Tensor) -> torch.Tensor:
        r"""Synthesizes log-scales log_sigma (ModelA compatibility)."""
        return self.h_s(z_hat)[0]

    def forward(
        self,
        x: torch.Tensor,
        saliency_map: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        training: bool | None = None,
    ) -> ModelCOutput:
        r"""Executes forward pass through Model C adaptive transforms.

        Args:
            x: Input image tensor (B, 3, H, W) with spatial dimensions multiples of 64.
            saliency_map: Optional ground-truth saliency map (reserved for interface parity).
            valid_mask: Optional spatial valid mask M_valid indicating unpadded image cells.
            training: Explicit override for training mode (noise vs rounding). Defaults to self.training.

        Returns:
            ModelCOutput containing all intermediate representations, steps, and reconstructions.
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

        # Step 1: Analysis transform g_a
        y = self.g_a(x)

        # Step 2: Hyper-analysis transform h_a (with optional gradient cutting)
        y_for_ha = y.detach() if self.detach_y_for_ha else y
        z = self.h_a(y_for_ha)

        # Step 3: Hyper-latent entropy bottleneck
        z_hat, z_likelihoods = self.entropy_bottleneck(z, training=training)

        # Step 4: Dual-head hyper-synthesis h_s emitting log_sigma and s_pred
        log_sigma, s_pred = self.h_s(z_hat)

        # Step 5: Saliency-based adaptive quantization step allocation
        # Support ground-truth saliency override (Oracle O1 baseline) when explicitly provided
        if saliency_map is not None:
            s_alloc = saliency_map.float()
            if s_alloc.ndim == 2:
                s_alloc = s_alloc.unsqueeze(0).unsqueeze(0)
            elif s_alloc.ndim == 3:
                s_alloc = s_alloc.unsqueeze(1)
            if s_alloc.shape[-2:] != s_pred.shape[-2:]:
                s_alloc = nn.functional.interpolate(
                    s_alloc,
                    size=s_pred.shape[-2:],
                    mode="area",
                )
        else:
            s_alloc = s_pred

        # Standardize valid_mask to (B, 1, H_lat, W_lat) if provided
        vm = valid_mask
        if vm is not None:
            if vm.ndim == 2:
                vm = vm.unsqueeze(0).unsqueeze(0)
            elif vm.ndim == 3:
                if vm.shape[0] == x.shape[0]:
                    vm = vm.unsqueeze(1)
                else:
                    vm = vm.unsqueeze(0)
            if vm.shape[-2:] != s_pred.shape[-2:]:
                vm = nn.functional.interpolate(vm.float(), size=s_pred.shape[-2:], mode="nearest").bool()

        device_type = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Detach s_alloc from rate-distortion loss to eliminate circular gradient dependencies
            log_delta, delta = compute_log_delta(s_alloc.detach(), valid_mask=vm)
            log_delta = log_delta.detach()
            delta = delta.detach()

            # Step 6: Effective log-scale computation in log-domain
            log_sigma_eff = compute_effective_log_scale(log_sigma, log_delta)
            sigma_eff = torch.exp(log_sigma_eff)

            # Step 7: Quantization and likelihood evaluation with aligned delta
            delta_aligned = delta.to(device=y.device, dtype=y.dtype)
            if delta_aligned.ndim == 2:
                delta_aligned = delta_aligned.unsqueeze(0).unsqueeze(0)
            elif delta_aligned.ndim == 3:
                delta_aligned = delta_aligned.unsqueeze(1)

            scaled_y = y / delta_aligned
            scaled_y_hat, y_likelihoods = self.gaussian_conditional(scaled_y, sigma_eff, training=training)
            y_hat = scaled_y_hat * delta_aligned

        # Step 8: Synthesis transform g_s
        x_hat = self.g_s(y_hat)

        return ModelCOutput(
            x_hat=x_hat,
            y=y,
            z=z,
            log_sigma=log_sigma,
            log_sigma_eff=log_sigma_eff,
            s_pred=s_pred,
            log_delta=log_delta,
            delta=delta,
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
        r"""Loads state dict with automatic unwrap, CompressAI key remapping, and saliency head tolerance."""
        try:
            target_device = next(self.parameters()).device
        except StopIteration:
            target_device = None

        unwrapped = _unwrap_state_dict(state_dict)
        remapped = remap_compressai_state_dict(unwrapped)

        current_keys = set(self.state_dict().keys())
        provided_keys = set(remapped.keys())
        missing_keys = current_keys - provided_keys
        unexpected_keys = provided_keys - current_keys
        saliency_keys = {
            "h_s.conv_saliency.weight",
            "h_s.conv_saliency.bias",
        }

        if strict:
            real_missing = missing_keys - saliency_keys
            if real_missing or unexpected_keys:
                error_msgs = []
                if real_missing:
                    error_msgs.append(f"Missing key(s) in state_dict: {', '.join(sorted(real_missing))}.")
                if unexpected_keys:
                    error_msgs.append(f"Unexpected key(s) in state_dict: {', '.join(sorted(unexpected_keys))}.")
                raise RuntimeError("Error(s) in loading state_dict for ModelC:\n\t" + "\n\t".join(error_msgs))

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
    ) -> ModelC:
        r"""Constructs a ModelC instance and loads weights from a checkpoint or state dict."""
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


__all__ = [
    "DualHeadHyperSynthesisTransform",
    "ModelC",
    "ModelCOutput",
    "load_pretrained_bmshj2018",
]
