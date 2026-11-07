"""Model A (bmshj2018) architecture and network transforms.

This module implements:
- AnalysisTransform (g_a): 4-stage convolutional analysis with GDN activations.
- SynthesisTransform (g_s): 4-stage transposed convolutional synthesis with IGDN activations.
- HyperAnalysisTransform (h_a): 3-stage convolutional hyper-analysis mapping |y| to z.
- HyperSynthesisTransform (h_s): 3-stage hyper-synthesis mapping z_hat to log_sigma in strict FP32.
- ModelA: Baseline variational image compression model supporting training noise injection
  and evaluation integer rounding.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.codec.entropy_models import EntropyBottleneck, GaussianConditional
from src.codec.layers import GDN, IGDN

MODEL_ID_A: int = 0


class ModelOutput(dict):
    """Container for model outputs supporting dictionary, attribute access, and tuple conversion."""

    def __init__(
        self,
        x_hat: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        log_sigma: torch.Tensor,
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

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(f"ModelOutput has no attribute '{name}'") from err

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns primary outputs as a 4-tuple: (x_hat, y, z, log_sigma)."""
        return self["x_hat"], self["y"], self["z"], self["log_sigma"]


class AnalysisTransform(nn.Module):
    r"""Analysis transform :math:`g_a` mapping input image :math:`x` to latents :math:`y`.

    Downsamples spatial resolution by a factor of 16 through 4 convolutional stages with stride 2
    and GDN non-linearities. The final layer executes in strict FP32 to prevent quantizer coarsening.
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_channels: int = 128,
        m_channels: int = 192,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_channels = int(n_channels)
        self.m_channels = int(m_channels)

        self.conv1 = nn.Conv2d(self.in_channels, self.n_channels, kernel_size=5, stride=2, padding=2)
        self.gdn1 = GDN(self.n_channels)

        self.conv2 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=5, stride=2, padding=2)
        self.gdn2 = GDN(self.n_channels)

        self.conv3 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=5, stride=2, padding=2)
        self.gdn3 = GDN(self.n_channels)

        self.conv4 = nn.Conv2d(self.n_channels, self.m_channels, kernel_size=5, stride=2, padding=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (B, C, H, W), got {x.dim()}D tensor of shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Channel mismatch: expected {self.in_channels}, got {x.shape[1]}")

        _, _, h, w = x.shape
        if h % 16 != 0 or w % 16 != 0:
            raise ValueError(
                f"Spatial dimensions ({h}, {w}) must be multiples of 16 for AnalysisTransform downsampling."
            )

        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.gdn3(self.conv3(x))

        # Final layer executes in strict FP32 to prevent quantizer coarsening
        device_type = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            y = self.conv4(x.float())

        return y


class SynthesisTransform(nn.Module):
    r"""Synthesis transform :math:`g_s` mapping quantized latents :math:`\hat{y}` to image :math:`\hat{x}`.

    Upsamples spatial resolution by a factor of 16 through 4 transposed convolutional stages with stride 2
    and IGDN non-linearities.
    """

    def __init__(
        self,
        in_channels: int = 192,
        n_channels: int = 128,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_channels = int(n_channels)
        self.out_channels = int(out_channels)

        self.deconv1 = nn.ConvTranspose2d(
            self.in_channels, self.n_channels, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.igdn1 = IGDN(self.n_channels)

        self.deconv2 = nn.ConvTranspose2d(
            self.n_channels, self.n_channels, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.igdn2 = IGDN(self.n_channels)

        self.deconv3 = nn.ConvTranspose2d(
            self.n_channels, self.n_channels, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.igdn3 = IGDN(self.n_channels)

        self.deconv4 = nn.ConvTranspose2d(
            self.n_channels, self.out_channels, kernel_size=5, stride=2, padding=2, output_padding=1
        )

    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        if y_hat.dim() != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), got {y_hat.dim()}D tensor of shape {tuple(y_hat.shape)}"
            )
        if y_hat.shape[1] != self.in_channels:
            raise ValueError(f"Channel mismatch: expected {self.in_channels}, got {y_hat.shape[1]}")

        x = self.igdn1(self.deconv1(y_hat))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x_hat = self.deconv4(x)
        return x_hat.float()


class HyperAnalysisTransform(nn.Module):
    r"""Hyper-analysis transform :math:`h_a` mapping latents :math:`|y|` to hyper-latents :math:`z`.

    Downsamples spatial resolution by a factor of 4 through 3 convolutional stages (3x3 stride 1,
    followed by two 5x5 stride 2 convolutions with ReLU activations).
    """

    def __init__(
        self,
        in_channels: int = 192,
        n_channels: int = 128,
        out_channels: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_channels = int(n_channels)
        self.out_channels = int(out_channels)

        self.conv1 = nn.Conv2d(self.in_channels, self.n_channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.n_channels, self.n_channels, kernel_size=5, stride=2, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(self.n_channels, self.out_channels, kernel_size=5, stride=2, padding=2)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (B, C, H, W), got {y.dim()}D tensor of shape {tuple(y.shape)}")
        if y.shape[1] != self.in_channels:
            raise ValueError(f"Channel mismatch: expected {self.in_channels}, got {y.shape[1]}")

        _, _, h, w = y.shape
        if h % 4 != 0 or w % 4 != 0:
            raise ValueError(
                f"Spatial dimensions ({h}, {w}) must be multiples of 4 for HyperAnalysisTransform downsampling."
            )

        y_abs = torch.abs(y)
        x = self.relu1(self.conv1(y_abs))
        x = self.relu2(self.conv2(x))
        z = self.conv3(x)
        return z


class HyperSynthesisTransform(nn.Module):
    r"""Hyper-synthesis transform :math:`h_s` mapping quantized hyper-latents :math:`\hat{z}`
    to predicted scale representations :math:`\log \sigma`.

    Upsamples spatial resolution by a factor of 4 through two 5x5 stride 2 transposed convolutions
    with ReLU activations, followed by a 3x3 stride 1 convolution.
    Executes strictly in FP32 outside autocast.
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

        self.deconv1 = nn.ConvTranspose2d(
            self.in_channels, self.n_channels, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(
            self.n_channels, self.n_channels, kernel_size=5, stride=2, padding=2, output_padding=1
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(self.n_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        if z_hat.dim() != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), got {z_hat.dim()}D tensor of shape {tuple(z_hat.shape)}"
            )
        if z_hat.shape[1] != self.in_channels:
            raise ValueError(f"Channel mismatch: expected {self.in_channels}, got {z_hat.shape[1]}")

        device_type = z_hat.device.type if z_hat.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            z_f = z_hat.float()
            x = self.relu1(self.deconv1(z_f))
            x = self.relu2(self.deconv2(x))
            log_sigma = self.conv3(x)

        return log_sigma


class ModelA(nn.Module):
    r"""Model A baseline variational image compression architecture (bmshj2018).

    Combines:
    - g_a: Analysis transform (16x downsampling)
    - g_s: Synthesis transform (16x upsampling)
    - h_a: Hyper-analysis transform (4x downsampling on latents y, total 64x)
    - h_s: Hyper-synthesis transform (4x upsampling on hyper-latents z)

    Dual operational modes:
    - training=True: Additive uniform noise relaxation U(-0.5, 0.5) applied to y and z.
    - training=False: Integer rounding torch.round applied to y and z for deterministic evaluation.
    """

    model_type: str = "A"
    model_id: int = MODEL_ID_A

    def __init__(
        self,
        in_channels: int = 3,
        n_channels: int = 128,
        m_channels: int = 192,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_channels = int(n_channels)
        self.m_channels = int(m_channels)

        self.g_a = AnalysisTransform(
            in_channels=self.in_channels, n_channels=self.n_channels, m_channels=self.m_channels
        )
        self.g_s = SynthesisTransform(
            in_channels=self.m_channels, n_channels=self.n_channels, out_channels=self.in_channels
        )
        self.h_a = HyperAnalysisTransform(
            in_channels=self.m_channels, n_channels=self.n_channels, out_channels=self.n_channels
        )
        self.h_s = HyperSynthesisTransform(
            in_channels=self.n_channels, n_channels=self.n_channels, out_channels=self.m_channels
        )
        self.entropy_bottleneck = EntropyBottleneck(channels=self.n_channels)
        self.gaussian_conditional = GaussianConditional()

    def quantize(self, tensor: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        r"""Quantizes a latent tensor via continuous uniform noise (training) or rounding (eval)."""
        is_training = self.training if training is None else bool(training)
        if is_training:
            noise = torch.empty_like(tensor).uniform_(-0.5, 0.5)
            return tensor + noise
        return torch.round(tensor)

    def encode_transforms(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Extracts unquantized latents y = g_a(x) and hyper-latents z = h_a(y)."""
        y = self.g_a(x)
        z = self.h_a(y)
        return y, z

    def decode_transforms(self, y_hat: torch.Tensor) -> torch.Tensor:
        r"""Synthesizes reconstructed image x_hat = g_s(y_hat)."""
        return self.g_s(y_hat)

    def synthesize_scales(self, z_hat: torch.Tensor) -> torch.Tensor:
        r"""Synthesizes log-scales log_sigma = h_s(z_hat)."""
        return self.h_s(z_hat)

    def forward(self, x: torch.Tensor, training: bool | None = None) -> ModelOutput:
        r"""Executes forward pass through Model A transforms.

        Args:
            x: Input image tensor (B, 3, H, W) with spatial dimensions multiples of 64.
            training: Explicit override for training mode (noise vs rounding). Defaults to self.training.

        Returns:
            ModelOutput containing:
                x_hat: Reconstructed image tensor (B, 3, H, W).
                y: Continuous analysis latents (B, 192, H/16, W/16).
                z: Continuous hyper-analysis latents (B, 128, H/64, W/64).
                log_sigma: Predicted log-scale representation (B, 192, H/16, W/16).
                y_hat: Relaxed / quantized latents.
                z_hat: Relaxed / quantized hyper-latents.
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

        y = self.g_a(x)
        z = self.h_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z, training=training)
        log_sigma = self.h_s(z_hat)
        sigma = torch.exp(log_sigma)

        y_hat, y_likelihoods = self.gaussian_conditional(y, sigma, training=training)
        x_hat = self.g_s(y_hat)

        return ModelOutput(
            x_hat=x_hat,
            y=y,
            z=z,
            log_sigma=log_sigma,
            y_hat=y_hat,
            z_hat=z_hat,
            y_likelihoods=y_likelihoods,
            z_likelihoods=z_likelihoods,
        )

    def update(self, force: bool = False) -> dict[str, bool]:
        r"""Compiles discrete CDF tables for both entropy models."""
        eb_res = self.entropy_bottleneck.update(force=force)
        gc_res = self.gaussian_conditional.update_scale_table()
        return {"entropy_bottleneck": eb_res, "gaussian_conditional": gc_res}

    def aux_loss(self) -> torch.Tensor:
        r"""Returns auxiliary parameter loss for entropy bottleneck."""
        return self.entropy_bottleneck.loss()
