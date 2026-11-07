"""Loss functions for saliency-weighted learned image compression.

This module implements:
- auxiliary_saliency_loss: Masked binary cross-entropy loss over valid cells with strict FP32 precision.
- saliency_weighted_distortion: Functional saliency-weighted reconstruction distortion
  with an explicit 0.2 background floor (S' = 0.2 + 0.8 * S), valid masking, and strict FP32 precision.
- RateDistortionLoss: CompressAI-compatible rate-distortion loss module balancing
  bitrate (bits per pixel), scaled distortion (loss = bpp_loss + lambda * 255^2 * distortion),
  and auxiliary saliency loss (lambda_aux * aux_loss), with optional uniform MSE distortion control.
- compute_hyperprior_overhead: Calculates hyperprior bitrate overhead Delta BPP_z between models.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def auxiliary_saliency_loss(
    s_pred: torch.Tensor,
    s_target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-7,
) -> torch.Tensor:
    r"""Computes auxiliary saliency loss (masked binary cross-entropy) over valid cells.

    .. math::
        \mathcal{L}_{\text{aux}} = \frac{1}{|M_{\text{valid}}|} \sum_{(i,j) \in M_{\text{valid}}}
            \Bigl[ -S_{ij} \log(S_{\text{pred}, ij} + \epsilon)
                   - (1 - S_{ij}) \log(1 - S_{\text{pred}, ij} + \epsilon) \Bigr]

    Operates in strict FP32 outside autocast. Automatically aligns spatial dimensions of
    ``s_target`` and ``valid_mask`` to ``s_pred`` if higher resolution (e.g. image vs latent grid).
    Target ``s_target`` is detached so gradients flow exclusively into ``s_pred``.

    Args:
        s_pred: Predicted visual saliency map tensor in [0, 1].
        s_target: Ground truth visual saliency map tensor in [0, 1].
        valid_mask: Optional boolean or float mask of valid unpadded regions. Cells with 0/False
            contribute exactly zero loss and zero gradient.
        eps: Small numerical constant for log stability (default 1e-7).

    Returns:
        torch.Tensor: Scalar auxiliary loss in strict FP32.

    Raises:
        TypeError: If inputs are not torch.Tensor.
        ValueError: If inputs have invalid dimensions, contain NaN/Inf, or valid_mask is empty.
    """
    if not isinstance(s_pred, torch.Tensor):
        raise TypeError(f"s_pred must be a torch.Tensor, got {type(s_pred)}")
    if not isinstance(s_target, torch.Tensor):
        raise TypeError(f"s_target must be a torch.Tensor, got {type(s_target)}")
    if valid_mask is not None and not isinstance(valid_mask, torch.Tensor):
        raise TypeError(f"valid_mask must be a torch.Tensor, got {type(valid_mask)}")

    if s_pred.ndim not in (2, 3, 4):
        raise ValueError(f"s_pred must be a 2D, 3D, or 4D tensor, got shape {tuple(s_pred.shape)}")
    if s_target.ndim not in (2, 3, 4):
        raise ValueError(f"s_target must be a 2D, 3D, or 4D tensor, got shape {tuple(s_target.shape)}")

    if not s_pred.is_cuda or not torch.is_grad_enabled():
        if torch.isnan(s_pred).any() or torch.isinf(s_pred).any():
            raise ValueError("s_pred contains NaN or Inf values")
        if torch.isnan(s_target).any() or torch.isinf(s_target).any():
            raise ValueError("s_target contains NaN or Inf values")
        if valid_mask is not None and (torch.isnan(valid_mask).any() or torch.isinf(valid_mask).any()):
            raise ValueError("valid_mask contains NaN or Inf values")

    device_type = s_pred.device.type if s_pred.device.type in ("cuda", "cpu") else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        p = s_pred.float()
        t = s_target.detach().float().to(device=s_pred.device)

        # Standardize p to 4D (B, 1, H, W)
        if p.ndim == 2:
            p = p.unsqueeze(0).unsqueeze(0)
        elif p.ndim == 3:
            p = p.unsqueeze(1)
        elif p.shape[1] != 1:
            p = torch.mean(p, dim=1, keepdim=True)

        b, _, h, w = p.shape

        # Standardize t to 4D (B, 1, H_t, W_t)
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            t = t.unsqueeze(1)
        elif t.shape[1] != 1:
            t = torch.mean(t, dim=1, keepdim=True)

        if t.shape[0] == 1 and b > 1:
            t = t.expand(b, -1, -1, -1)
        elif t.shape[0] != b:
            raise ValueError(f"Batch size mismatch: s_pred batch={b} vs s_target batch={t.shape[0]}")

        # Spatially pool/interpolate s_target if higher resolution
        if t.shape[-2:] != (h, w):
            if t.shape[-2] >= h and t.shape[-1] >= w:
                t = F.interpolate(t, size=(h, w), mode="area")
            else:
                t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)

        # Clamp inputs safely into [0, 1]
        p_clamped = torch.clamp(p, min=0.0, max=1.0)
        t_clamped = torch.clamp(t, min=0.0, max=1.0)

        # Masked BCE elementwise:
        bce = -t_clamped * torch.log(p_clamped + eps) - (1.0 - t_clamped) * torch.log(1.0 - p_clamped + eps)

        if valid_mask is not None:
            m = valid_mask.to(device=s_pred.device)
            if m.ndim == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.ndim == 3:
                m = m.unsqueeze(1)
            elif m.shape[1] != 1:
                m = m[:, :1, :, :]

            if m.shape[0] == 1 and b > 1:
                m = m.expand(b, -1, -1, -1)
            elif m.shape[0] != b:
                raise ValueError(f"Batch size mismatch: s_pred batch={b} vs valid_mask batch={m.shape[0]}")

            if m.shape[-2:] != (h, w):
                m = F.interpolate(m.float(), size=(h, w), mode="nearest")

            m_bool = m.bool()
            m_float = m_bool.float()

            if not s_pred.is_cuda or not torch.is_grad_enabled():
                if m_float.sum().item() <= 0:
                    raise ValueError("valid_mask contains no valid cells")

            n_valid = torch.clamp(m_float.sum(), min=1.0)
            masked_bce = bce * m_float
            loss = masked_bce.sum() / n_valid
        else:
            loss = bce.mean()

        return torch.clamp(loss, min=0.0)


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
                          + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}

    Where:
        - :math:`R_{\text{bpp}} = \text{y\_bpp} + \text{z\_bpp} + \text{sidecar\_bpp}`
        - :math:`\text{y\_bpp} = -\frac{\sum \log_2(p_y)}{B \cdot H \cdot W}`
        - :math:`\text{z\_bpp} = -\frac{\sum \log_2(p_z)}{B \cdot H \cdot W}`
        - :math:`\text{sidecar\_bpp}`: Transmitted sidecar bitrate (e.g. Model C-Transmitted).
        - :math:`D_{\text{saliency}}`: Saliency-weighted distortion with 0.2 background floor.
        - :math:`\text{mse\_loss}`: Unweighted mean squared error over the valid evaluation area.
        - :math:`\mathcal{L}_{\text{aux}}`: Auxiliary saliency loss (masked BCE over valid cells; suppressed to 0.0
          when sidecar_bpp is present since saliency is transmitted rather than predicted).

    Operates in strict FP32 outside autocast to prevent CDF underflow or loss quantization.
    """

    def __init__(
        self,
        lmbda: float = 0.01,
        lambda_aux: float = 0.1,
        saliency_weighting: bool = True,
    ) -> None:
        super().__init__()
        if lmbda < 0.0:
            raise ValueError(f"lmbda must be non-negative, got {lmbda}")
        if lambda_aux < 0.0:
            raise ValueError(f"lambda_aux must be non-negative, got {lambda_aux}")
        self.lmbda = float(lmbda)
        self.lambda_aux = float(lambda_aux)
        self.saliency_weighting = bool(saliency_weighting)

    def forward(
        self,
        output: dict[str, Any] | Any,
        target: torch.Tensor,
        s: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        *,
        s_pred: torch.Tensor | None = None,
        saliency_map: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        target_num_pixels: int | None = None,
        lmbda: float | None = None,
        saliency_weighting: bool | None = None,
        sidecar_bpp: float | torch.Tensor | None = None,
    ) -> dict[str, Any]:
        r"""Evaluates rate-distortion loss and metric breakdown.

        Args:
            output: ModelOutput container, dict, or object containing 'x_hat', 'y_likelihoods',
                and 'z_likelihoods' (or 'likelihoods' dictionary). Optionally contains 's_pred' or 'sidecar_bpp'.
            target: Ground truth target image tensor (B, C, H, W) or (C, H, W).
            s: Optional visual saliency map in [0, 1].
            valid_mask: Optional spatial valid mask for unpadded region evaluation.
            s_pred: Optional predicted saliency map override.
            saliency_map: Keyword-only alias for s.
            mask: Keyword-only alias for valid_mask.
            target_num_pixels: Optional explicit pixel count for bpp normalization. Defaults
                to B * H * W of target.
            lmbda: Optional per-call rate-distortion trade-off parameter override.
            saliency_weighting: Optional per-call override for whether saliency weighting is applied.
            sidecar_bpp: Optional sidecar bits-per-pixel override (e.g. for Model C-Transmitted).

        Returns:
            Dict containing:
                - 'loss': Total rate-distortion loss.
                - 'bpp_loss': Total bits-per-pixel (y_bpp + z_bpp + sidecar_bpp).
                - 'y_bpp': Main latent bitrate.
                - 'z_bpp': Hyper-latent bitrate.
                - 'sidecar_bpp': Transmitted sidecar bitrate (tensor or 0.0).
                - 'distortion': Saliency-weighted distortion in [0, 1] (or unweighted MSE if saliency_weighting=False).
                - 'mse_loss': Unweighted mean squared error in [0, 1].
                - 'aux_loss': Auxiliary saliency loss (or 0.0 if s_pred is None or sidecar_bpp is present).
                - 'lambda_aux': Weight of the auxiliary loss term.
                - 'saliency_weighting': Boolean flag indicating whether saliency weighting was enabled.
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
            pred_from_output = output.get("s_pred")
            pred_sidecar_bpp = output.get("sidecar_bpp")
        else:
            x_hat = getattr(output, "x_hat", None)
            y_likelihoods = getattr(output, "y_likelihoods", None)
            z_likelihoods = getattr(output, "z_likelihoods", None)
            lik_container = getattr(output, "likelihoods", None)
            pred_from_output = getattr(output, "s_pred", None)
            pred_sidecar_bpp = getattr(output, "sidecar_bpp", None)

        if x_hat is None:
            raise ValueError("output must contain 'x_hat' tensor")

        resolved_s_pred = s_pred if s_pred is not None else pred_from_output
        resolved_sidecar_bpp = sidecar_bpp if sidecar_bpp is not None else pred_sidecar_bpp
        target_s = s if s is not None else saliency_map
        target_mask = valid_mask if valid_mask is not None else mask

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

            if resolved_sidecar_bpp is not None:
                if isinstance(resolved_sidecar_bpp, torch.Tensor):
                    sidecar_bpp_t = resolved_sidecar_bpp.to(device=target.device, dtype=torch.float32)
                else:
                    sidecar_bpp_t = torch.tensor(float(resolved_sidecar_bpp), device=target.device, dtype=torch.float32)
            else:
                sidecar_bpp_t = torch.tensor(0.0, device=target.device, dtype=torch.float32)

            bpp_loss = bpp_loss + sidecar_bpp_t

            eff_saliency_weighting = self.saliency_weighting if saliency_weighting is None else bool(saliency_weighting)

            if eff_saliency_weighting:
                distortion = saliency_weighted_distortion(
                    target,
                    x_hat,
                    s=target_s,
                    valid_mask=target_mask,
                )
                mse_loss = saliency_weighted_distortion(
                    target,
                    x_hat,
                    s=None,
                    valid_mask=target_mask,
                )
            else:
                distortion = mse_loss = saliency_weighted_distortion(
                    target,
                    x_hat,
                    s=None,
                    valid_mask=target_mask,
                )

            # Evaluate auxiliary saliency loss when s_pred is present and not a sidecar model
            if resolved_s_pred is not None and resolved_sidecar_bpp is None:
                if target_s is None:
                    raise ValueError(
                        "Ground truth saliency map 's' must be provided when 's_pred' is present in output"
                    )
                aux_loss = auxiliary_saliency_loss(
                    resolved_s_pred,
                    target_s,
                    valid_mask=target_mask,
                )
            else:
                aux_loss = torch.tensor(0.0, device=target.device, dtype=torch.float32)

            eff_lmbda = self.lmbda if lmbda is None else float(lmbda)
            loss = bpp_loss + eff_lmbda * (255.0**2) * distortion + self.lambda_aux * aux_loss

            return {
                "loss": loss,
                "bpp_loss": bpp_loss,
                "y_bpp": y_bpp,
                "z_bpp": z_bpp,
                "sidecar_bpp": sidecar_bpp_t,
                "distortion": distortion,
                "mse_loss": mse_loss,
                "aux_loss": aux_loss,
                "lambda_aux": torch.tensor(self.lambda_aux, device=target.device, dtype=torch.float32),
                "saliency_weighting": eff_saliency_weighting,
            }


def compute_hyperprior_overhead(
    z_bpp_model: float | torch.Tensor,
    z_bpp_baseline: float | torch.Tensor,
    *,
    as_tensor: bool = False,
) -> float | torch.Tensor:
    r"""Computes hyperprior bitrate overhead of an adaptive model relative to a baseline.

    .. math::
        \Delta \text{BPP}_z = \text{BPP}_z(\text{Model}) - \text{BPP}_z(\text{Baseline})

    Quantifies the bitrate overhead incurred by transmitting hyperprior latents in adaptive
    models (e.g. Model D or Model C) relative to standard uniform baselines (e.g. Model A).

    Args:
        z_bpp_model: Hyperprior bits-per-pixel of the evaluated model (float or tensor).
        z_bpp_baseline: Hyperprior bits-per-pixel of the baseline model (float or tensor).
        as_tensor: If True, returns a scalar FP32 torch.Tensor; otherwise returns a Python float.

    Returns:
        float | torch.Tensor: Bitrate overhead Delta BPP_z.
    """
    if isinstance(z_bpp_model, torch.Tensor) or isinstance(z_bpp_baseline, torch.Tensor):
        target_device = z_bpp_model.device if isinstance(z_bpp_model, torch.Tensor) else z_bpp_baseline.device
        m_t = (
            z_bpp_model.to(device=target_device, dtype=torch.float32)
            if isinstance(z_bpp_model, torch.Tensor)
            else torch.tensor(float(z_bpp_model), dtype=torch.float32, device=target_device)
        )
        b_t = (
            z_bpp_baseline.to(device=target_device, dtype=torch.float32)
            if isinstance(z_bpp_baseline, torch.Tensor)
            else torch.tensor(float(z_bpp_baseline), dtype=torch.float32, device=target_device)
        )
        if m_t.numel() == 0 or b_t.numel() == 0:
            raise ValueError("Hyperprior BPP tensors cannot be empty")

        diff_tensor = m_t - b_t
        if as_tensor:
            return diff_tensor
        return float(diff_tensor.item() if diff_tensor.numel() == 1 else diff_tensor.mean().item())

    diff_float = float(z_bpp_model) - float(z_bpp_baseline)
    if as_tensor:
        return torch.tensor(diff_float, dtype=torch.float32)
    return diff_float


__all__ = [
    "RateDistortionLoss",
    "auxiliary_saliency_loss",
    "compute_hyperprior_overhead",
    "saliency_weighted_distortion",
]
