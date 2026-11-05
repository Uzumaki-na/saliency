"""Loss functions for saliency-weighted learned image compression.

This module implements:
- saliency_weighted_distortion: Functional saliency-weighted reconstruction distortion
  with an explicit 0.2 background floor (S' = 0.2 + 0.8 * S), valid masking, and strict FP32 precision.
- RateDistortionLoss: CompressAI-compatible rate-distortion loss module balancing
  bitrate (bits per pixel) and scaled distortion (loss = bpp_loss + lambda * 255^2 * distortion).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


def saliency_weighted_distortion(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    s: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    *,
    saliency_map: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Computes saliency-weighted reconstruction distortion in strict FP32.

    Implements:
        .. math::
            S'_{ij} = 0.2 + 0.8 \cdot S_{ij}

            W_{ij} = S'_{ij} \cdot M_{\text{valid}, ij}

            D_{\text{saliency}} = \frac{\sum_{i,j} W_{ij} \cdot \frac{1}{C}\sum_c (x_{c,ij} - \hat{x}_{c,ij})^2}
                                       {\sum_{i,j} W_{ij}}

    Guarantees:
    - Background floor: :math:`S'_{ij} \ge 0.2` everywhere, ensuring non-salient regions never receive zero weight.
    - Equivalence: When :math:`S` is uniform (:math:`S=1.0` or :math:`S=0.0`) or None, :math:`D_{\text{saliency}}`
      is mathematically equivalent to standard unweighted mean squared error.
    - Numerical precision: Operations are strictly executed in FP32 outside autocast to prevent quantization errors.
    - Gradient isolation: Saliency map tensor :math:`S` is detached inside the loss; gradients flow only through
      the reconstruction :math:`\hat{x}`.
    - Valid mask isolation: Errors in padded regions outside :math:`M_{\text{valid}}` do not affect the loss value
      or gradients.

    Args:
        x: Ground truth target image tensor of shape (B, C, H, W), (C, H, W), or (H, W) in [0, 1].
        x_hat: Reconstructed image tensor of identical shape to x.
        s: Optional visual saliency map in [0, 1]. Supported shapes: (B, 1, H, W), (B, H, W),
            (1, H, W), or (H, W).
        valid_mask: Optional boolean or float mask indicating valid unpadded regions. Supported
            shapes: (B, 1, H, W), (B, H, W), (1, H, W), or (H, W).
        saliency_map: Keyword-only alias for s.
        mask: Keyword-only alias for valid_mask.

    Returns:
        torch.Tensor: Scalar distortion value in [0, 1] in strict FP32.

    Raises:
        TypeError: If x or x_hat is not a torch.Tensor, or if s/valid_mask is not a torch.Tensor when provided.
        ValueError: If x and x_hat shapes do not match, or if spatial dimensions of s or valid_mask
            do not match x.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
    if not isinstance(x_hat, torch.Tensor):
        raise TypeError(f"x_hat must be a torch.Tensor, got {type(x_hat)}")

    if x.shape != x_hat.shape:
        raise ValueError(
            f"Dimension mismatch between target x of shape {tuple(x.shape)} and "
            f"reconstruction x_hat of shape {tuple(x_hat.shape)}"
        )

    if x.ndim < 2:
        raise ValueError(f"Expected at least 2D tensor, got shape {tuple(x.shape)}")

    if s is None and saliency_map is not None:
        s = saliency_map
    if valid_mask is None and mask is not None:
        valid_mask = mask

    device_type = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        x_f = x.float()
        x_hat_f = x_hat.float()

        # Reshape / treat as 4D (B, C, H, W) for uniform processing
        orig_ndim = x_f.ndim
        if orig_ndim == 4:
            b, _, h, w = x_f.shape
            sq_err = torch.mean((x_f - x_hat_f) ** 2, dim=1, keepdim=True)
        elif orig_ndim == 3:
            _, h, w = x_f.shape
            sq_err = torch.mean((x_f - x_hat_f) ** 2, dim=0, keepdim=True).unsqueeze(0)
            b = 1
        else:  # 2D
            h, w = x_f.shape
            sq_err = ((x_f - x_hat_f) ** 2).unsqueeze(0).unsqueeze(0)
            b = 1

        # Process saliency map S
        if s is not None:
            if not isinstance(s, torch.Tensor):
                raise TypeError(f"saliency map must be a torch.Tensor, got {type(s)}")
            s_det = s.detach().float()
            if s_det.shape[-2:] != (h, w):
                raise ValueError(
                    f"Spatial dimension mismatch between target ({h}, {w}) and saliency map {tuple(s_det.shape[-2:])}"
                )
            if s_det.ndim == 2:
                s_det = s_det.unsqueeze(0).unsqueeze(0)
            elif s_det.ndim == 3:
                if s_det.shape[0] == b and b > 1:
                    s_det = s_det.unsqueeze(1)
                else:
                    s_det = s_det.unsqueeze(0) if s_det.shape[0] == 1 else s_det.unsqueeze(1)
            elif s_det.ndim == 4:
                if s_det.shape[1] != 1:
                    s_det = torch.mean(s_det, dim=1, keepdim=True)
            else:
                raise ValueError(f"Unsupported saliency tensor ndim: {s_det.ndim}")

            s_clamped = torch.clamp(s_det, 0.0, 1.0)
            s_prime = 0.2 + 0.8 * s_clamped
        else:
            s_prime = torch.ones_like(sq_err)

        # Process valid mask M_valid
        if valid_mask is not None:
            if not isinstance(valid_mask, torch.Tensor):
                raise TypeError(f"valid_mask must be a torch.Tensor, got {type(valid_mask)}")
            m_det = valid_mask.detach().float()
            if m_det.shape[-2:] != (h, w):
                raise ValueError(
                    f"Spatial dimension mismatch between target ({h}, {w}) and valid_mask {tuple(m_det.shape[-2:])}"
                )
            if m_det.ndim == 2:
                m_det = m_det.unsqueeze(0).unsqueeze(0)
            elif m_det.ndim == 3:
                if m_det.shape[0] == b and b > 1:
                    m_det = m_det.unsqueeze(1)
                else:
                    m_det = m_det.unsqueeze(0) if m_det.shape[0] == 1 else m_det.unsqueeze(1)
            elif m_det.ndim == 4:
                if m_det.shape[1] != 1:
                    m_det = m_det[:, :1, :, :]
            else:
                raise ValueError(f"Unsupported valid_mask tensor ndim: {m_det.ndim}")

            weights = s_prime * m_det
        else:
            weights = s_prime

        if weights.shape[0] == 1 and b > 1:
            weights = weights.expand(b, -1, -1, -1)
        elif weights.shape[0] != b and weights.shape[0] != 1:
            raise ValueError(f"Batch dimension mismatch between target ({b}) and weights ({weights.shape[0]})")

        num = torch.sum(weights * sq_err, dim=(-2, -1))
        denom = torch.sum(weights, dim=(-2, -1))
        denom_safe = torch.clamp(denom, min=1e-12)
        d_per_sample = num / denom_safe

        return torch.mean(d_per_sample)


class RateDistortionLoss(nn.Module):
    r"""Rate-distortion loss module with saliency-weighted reconstruction distortion.

    Optimizes the CompressAI-compatible objective:
        .. math::
            \mathcal{L} = R_{\text{bpp}} + \lambda \cdot 255^2 \cdot D_{\text{saliency}}

    Where:
        - :math:`R_{\text{bpp}} = \text{y\_bpp} + \text{z\_bpp}`
        - :math:`\text{y\_bpp} = -\frac{\sum \log_2(p_y)}{B \cdot H \cdot W}`
        - :math:`\text{z\_bpp} = -\frac{\sum \log_2(p_z)}{B \cdot H \cdot W}`
        - :math:`D_{\text{saliency}}`: Saliency-weighted distortion with 0.2 background floor.
        - :math:`\text{mse\_loss}`: Unweighted mean squared error over the valid evaluation area.

    Operates in strict FP32 outside autocast to prevent CDF underflow or loss quantization.
    """

    def __init__(self, lmbda: float = 0.01) -> None:
        super().__init__()
        if lmbda < 0.0:
            raise ValueError(f"lmbda must be non-negative, got {lmbda}")
        self.lmbda = float(lmbda)

    def forward(
        self,
        output: dict[str, Any] | Any,
        target: torch.Tensor,
        s: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        *,
        saliency_map: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        target_num_pixels: int | None = None,
        lmbda: float | None = None,
    ) -> dict[str, torch.Tensor]:
        r"""Evaluates rate-distortion loss and metric breakdown.

        Args:
            output: ModelOutput container, dict, or object containing 'x_hat', 'y_likelihoods',
                and 'z_likelihoods' (or 'likelihoods' dictionary).
            target: Ground truth target image tensor (B, C, H, W) or (C, H, W).
            s: Optional visual saliency map in [0, 1].
            valid_mask: Optional spatial valid mask for unpadded region evaluation.
            saliency_map: Keyword-only alias for s.
            mask: Keyword-only alias for valid_mask.
            target_num_pixels: Optional explicit pixel count for bpp normalization. Defaults
                to B * H * W of target.
            lmbda: Optional per-call rate-distortion trade-off parameter override.

        Returns:
            Dict containing:
                - 'loss': Total rate-distortion loss.
                - 'bpp_loss': Total bits-per-pixel (y_bpp + z_bpp).
                - 'y_bpp': Main latent bitrate.
                - 'z_bpp': Hyper-latent bitrate.
                - 'distortion': Saliency-weighted distortion in [0, 1].
                - 'mse_loss': Unweighted mean squared error in [0, 1].
        """
        # Support swapped argument order if target is passed first
        if isinstance(output, torch.Tensor) and not isinstance(target, torch.Tensor):
            output, target = target, output

        if not isinstance(target, torch.Tensor):
            raise TypeError(f"target must be a torch.Tensor, got {type(target)}")

        if isinstance(output, dict):
            x_hat = output.get("x_hat")
            y_likelihoods = output.get("y_likelihoods")
            z_likelihoods = output.get("z_likelihoods")
            lik_container = output.get("likelihoods")
        else:
            x_hat = getattr(output, "x_hat", None)
            y_likelihoods = getattr(output, "y_likelihoods", None)
            z_likelihoods = getattr(output, "z_likelihoods", None)
            lik_container = getattr(output, "likelihoods", None)

        if x_hat is None:
            raise ValueError("output must contain 'x_hat' tensor")

        # Fallback to nested likelihoods dict if y/z not at top level
        if y_likelihoods is None and z_likelihoods is None and isinstance(lik_container, dict):
            y_likelihoods = lik_container.get("y")
            z_likelihoods = lik_container.get("z")

        device_type = target.device.type if target.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Compute normalization pixel count: B * H * W
            if target_num_pixels is not None:
                num_pixels = float(target_num_pixels)
            else:
                if target.ndim >= 3:
                    b = target.shape[0] if target.ndim == 4 else 1
                    h = target.shape[-2]
                    w = target.shape[-1]
                    num_pixels = float(b * h * w)
                else:
                    num_pixels = float(target.numel())

            log2_factor = math.log(2.0)

            # Likelihood to bpp computation
            if y_likelihoods is not None:
                y_lik_f = y_likelihoods.float()
                y_lik_safe = torch.clamp(y_lik_f, min=1e-12)
                y_bpp = -torch.sum(torch.log(y_lik_safe)) / (log2_factor * num_pixels)
            else:
                y_bpp = torch.tensor(0.0, device=target.device, dtype=torch.float32)

            if z_likelihoods is not None:
                z_lik_f = z_likelihoods.float()
                z_lik_safe = torch.clamp(z_lik_f, min=1e-12)
                z_bpp = -torch.sum(torch.log(z_lik_safe)) / (log2_factor * num_pixels)
            else:
                z_bpp = torch.tensor(0.0, device=target.device, dtype=torch.float32)

            # Handle case where lik_container is provided without y/z keys
            if y_likelihoods is None and z_likelihoods is None and isinstance(lik_container, dict):
                total_bpp = torch.tensor(0.0, device=target.device, dtype=torch.float32)
                for val in lik_container.values():
                    if isinstance(val, torch.Tensor):
                        val_safe = torch.clamp(val.float(), min=1e-12)
                        bpp_k = -torch.sum(torch.log(val_safe)) / (log2_factor * num_pixels)
                        total_bpp = total_bpp + bpp_k
                bpp_loss = total_bpp
            else:
                bpp_loss = y_bpp + z_bpp

            distortion = saliency_weighted_distortion(
                target,
                x_hat,
                s=s,
                valid_mask=valid_mask,
                saliency_map=saliency_map,
                mask=mask,
            )

            mse_loss = saliency_weighted_distortion(
                target,
                x_hat,
                s=None,
                valid_mask=valid_mask,
                mask=mask,
            )

            eff_lmbda = self.lmbda if lmbda is None else float(lmbda)
            loss = bpp_loss + eff_lmbda * (255.0**2) * distortion

            return {
                "loss": loss,
                "bpp_loss": bpp_loss,
                "y_bpp": y_bpp,
                "z_bpp": z_bpp,
                "distortion": distortion,
                "mse_loss": mse_loss,
            }
