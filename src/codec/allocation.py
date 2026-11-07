"""Log-domain adaptive quantization step module for visual saliency neural compression.

Implements the allocation engine:
- Raw log-importance: w = -0.5 * log(S_pred + 0.01)
- Valid mask M_valid centering: w_bar = mean over valid cells, u = (w - w_bar) * M_valid
- Peak deviation and dynamic linear contraction:
    M = max_{M_valid} |u|
    s = min(1.0, log(4.0) / (M + 1e-7))
    log_delta_cont = s * u
- Discrete 1/32 log-grid quantization via Straight-Through Estimator (STE)
- Static 129-entry offline constant table T_Delta spanning [-log 4.0, +log 4.0]
- Effective log-scale computation: log_sigma_eff = log_sigma - log_delta
- Spatially varying adaptive latent quantization: y_tilde = quantize(y / delta) * delta
- Strict FP32 precision execution outside autocast
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_delta_table(
    num_entries: int = 129,
    max_factor: float = 4.0,
) -> torch.Tensor:
    r"""Precomputes static 1D CPU tensor :math:`T_\Delta` spanning :math:`[1 / \text{max\_factor}, \text{max\_factor}]`.

    For default parameters (num_entries=129, max_factor=4.0):
    - Uses step resolution :math:`\Delta_{\text{grid}} = 1/32`.
    - Spans :math:`k \in \{-64, \dots, +64\}` clamped strictly to :math:`[0.25, 4.0]`.
    - Guarantees :math:`T_\Delta[64] = 1.0` exactly.
    - For all :math:`|k| \le 44` (:math:`\ln 4.0 \approx 1.3863`), matches :math:`\exp(k / 32)` with error < 1e-7.
    - Clamped to 0.25 for :math:`k \le -45` and 4.0 for :math:`k \ge 45`.
    - Guarantees :math:`T_\Delta[0] = 0.25` and :math:`T_\Delta[128] = 4.0` exactly.

    For custom parameters:
    - Sets :math:`\Delta_{\text{grid}} = \ln(\text{max\_factor}) / \text{half}`.
    - Clamps strictly to :math:`[1 / \text{max\_factor}, \text{max\_factor}]`.

    Args:
        num_entries: Number of discrete table entries (default 129). Must be an odd positive integer >= 3.
        max_factor: Maximum allocation scale multiplier bound (default 4.0). Must be > 1.0.

    Returns:
        torch.Tensor: 1D float32 tensor of length ``num_entries`` on CPU.

    Raises:
        ValueError: If num_entries is not an odd positive integer >= 3 or max_factor <= 1.0.
    """
    if not isinstance(num_entries, int) or num_entries < 3 or num_entries % 2 == 0:
        raise ValueError(f"num_entries must be an odd positive integer >= 3, got {num_entries}")
    if max_factor <= 1.0:
        raise ValueError(f"max_factor must be greater than 1.0, got {max_factor}")

    half = (num_entries - 1) // 2  # 64 for 129
    if num_entries == 129 and math.isclose(max_factor, 4.0):
        delta_grid = 1.0 / 32.0
    else:
        delta_grid = math.log(max_factor) / float(half)

    k = torch.arange(-half, half + 1, dtype=torch.float32)
    log_deltas = k * delta_grid
    raw_table = torch.exp(log_deltas)
    table = torch.clamp(raw_table, min=1.0 / max_factor, max=max_factor)
    return table.cpu()


# Module-level static 129-entry offline constant lookup table
DELTA_TABLE_129: torch.Tensor = build_delta_table(num_entries=129, max_factor=4.0)


def quantize_ste_1_32(log_delta_cont: torch.Tensor) -> torch.Tensor:
    r"""Straight-Through Estimator (STE) quantizer rounding to the 1/32 grid.

    Forward pass applies discrete rounding:
    .. math::
        \log \Delta = \frac{\operatorname{round}(32 \cdot \log \Delta^{\text{cont}})}{32}

    Backward pass passes gradients through with identity Jacobian:
    .. math::
        \frac{\partial \log \Delta}{\partial \log \Delta^{\text{cont}}} = 1.0

    Args:
        log_delta_cont: Continuous log-step tensor.

    Returns:
        torch.Tensor: Discrete 1/32-grid quantized tensor with STE backward pass.

    Raises:
        TypeError: If log_delta_cont is not a torch.Tensor.
    """
    if not isinstance(log_delta_cont, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(log_delta_cont)}")

    device_type = log_delta_cont.device.type if log_delta_cont.device.type in ("cuda", "cpu") else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        x = log_delta_cont.float()
        rounded = torch.round(x * 32.0) / 32.0
        return x + (rounded - x).detach()


class STE1Over32Quantizer(nn.Module):
    r"""PyTorch Module wrapper for :func:`quantize_ste_1_32` Straight-Through Estimator."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return quantize_ste_1_32(x)


def compute_log_delta(
    s_pred: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    max_factor: float = 4.0,
    delta_grid: float = 1.0 / 32.0,
    apply_ste: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Computes log-domain adaptive quantization step :math:`\log \Delta` and step :math:`\Delta`.

    Formulation:
    1. Raw log-importance: :math:`w = -0.5 \log(S_{\text{pred}} + 0.01)`
    2. Valid mask :math:`M_{\text{valid}}` centering:
       :math:`\bar{w} = \frac{1}{|M_{\text{valid}}|} \sum_{(k,l) \in M_{\text{valid}}} w_{kl}`
       :math:`u = (w - \bar{w}) \cdot M_{\text{valid}}`
    3. Peak deviation and dynamic linear contraction:
       :math:`M = \max_{(k,l) \in M_{\text{valid}}} |u_{kl}|`
       :math:`s = \min(1.0, \frac{\log(\text{max\_factor})}{M + 10^{-7}})`
       :math:`\log \Delta^{\text{cont}} = s \cdot u`
    4. Fixed-grid quantization with Straight-Through Estimator (STE):
       :math:`\log \Delta = \operatorname{quantize\_ste\_1\_32}(\log \Delta^{\text{cont}})`
    5. Mask isolation:
       For cells outside :math:`M_{\text{valid}}`, :math:`\log \Delta = 0.0` and :math:`\Delta = 1.0`.

    Args:
        s_pred: Visual saliency map tensor in [0, 1] of shape (H, W), (1, H, W),
            (B, 1, H, W), or (B, C, H, W).
        valid_mask: Optional boolean or float validity mask M_valid of spatial shape (H, W).
            Cells with 1/True are valid; 0/False are reflection padded. If None, all cells are valid.
        max_factor: Maximum step multiplier factor (default 4.0). Step is bounded in [1/4.0, 4.0].
        delta_grid: Discrete log-step resolution (default 1/32).
        apply_ste: If True (default), rounds log_delta to discrete delta_grid via STE.
            If False, returns continuous log_delta_cont.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (log_delta, delta) in strict FP32.

    Raises:
        TypeError: If inputs are not torch.Tensor.
        ValueError: If input dimensions, shapes, or value ranges are invalid.
    """
    if not isinstance(s_pred, torch.Tensor):
        raise TypeError(f"s_pred must be a torch.Tensor, got {type(s_pred)}")

    if s_pred.ndim not in (2, 3, 4):
        raise ValueError(
            f"s_pred must be a 2D, 3D, or 4D tensor, got ndim={s_pred.ndim} with shape {tuple(s_pred.shape)}"
        )

    h, w = s_pred.shape[-2], s_pred.shape[-1]
    if h <= 0 or w <= 0:
        raise ValueError(f"Spatial dimensions must be positive, got height={h}, width={w}")

    if torch.isnan(s_pred).any() or torch.isinf(s_pred).any():
        raise ValueError("s_pred contains NaN or Inf values")

    # Range validation: saliency maps are defined in [0, 1]
    if (s_pred < -1e-2).any() or (s_pred > 1.0 + 1e-2).any():
        raise ValueError(
            f"s_pred values must be in [0, 1], got min {s_pred.min().item():.4f} and max {s_pred.max().item():.4f}"
        )

    if max_factor <= 1.0:
        raise ValueError(f"max_factor must be greater than 1.0, got {max_factor}")

    if delta_grid <= 0.0:
        raise ValueError(f"delta_grid must be positive, got {delta_grid}")

    if valid_mask is not None:
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError(f"valid_mask must be a torch.Tensor, got {type(valid_mask)}")
        if valid_mask.ndim > s_pred.ndim:
            raise ValueError(f"valid_mask ndim ({valid_mask.ndim}) cannot exceed s_pred ndim ({s_pred.ndim})")
        if valid_mask.shape[-2:] != s_pred.shape[-2:]:
            raise ValueError(
                f"Spatial dimensions of valid_mask {tuple(valid_mask.shape[-2:])} must match "
                f"s_pred {tuple(s_pred.shape[-2:])}"
            )
        if torch.isnan(valid_mask).any() or torch.isinf(valid_mask).any():
            raise ValueError("valid_mask contains NaN or Inf values")

    device_type = s_pred.device.type if s_pred.device.type in ("cuda", "cpu") else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        # Ensure strict FP32 computation
        s_clamped = torch.clamp(s_pred.float(), min=0.0, max=1.0)

        # Step 1: Raw log-importance
        w_raw = -0.5 * torch.log(s_clamped + 0.01)

        # Prepare valid mask
        if valid_mask is None:
            mask_bool = torch.ones_like(s_clamped, dtype=torch.bool)
            m_float = torch.ones_like(s_clamped, dtype=torch.float32)
        else:
            mb = valid_mask.bool().to(device=s_pred.device)
            mf = mb.float()
            # Align leading dimensions if needed
            if mb.ndim == 3 and s_clamped.ndim == 4 and mb.shape[0] == s_clamped.shape[0]:
                mb = mb.unsqueeze(1)
                mf = mf.unsqueeze(1)
            else:
                while mb.ndim < s_clamped.ndim:
                    mb = mb.unsqueeze(0)
                    mf = mf.unsqueeze(0)
            mask_bool = mb.expand_as(s_clamped)
            m_float = mf.expand_as(s_clamped)

        # Count valid cells per spatial slice
        n_valid = m_float.sum(dim=(-2, -1), keepdim=True)
        if (n_valid <= 0).any():
            raise ValueError("valid_mask contains no valid cells in at least one spatial slice")

        # Step 2: Valid mask centering
        w_bar = (w_raw * m_float).sum(dim=(-2, -1), keepdim=True) / n_valid
        u = (w_raw - w_bar) * m_float

        # Step 3: Peak deviation & dynamic linear contraction
        abs_u = torch.abs(u)
        m_peak = abs_u.amax(dim=(-2, -1), keepdim=True)
        max_log = math.log(max_factor)
        s_contraction = torch.minimum(
            torch.tensor(1.0, device=s_pred.device, dtype=torch.float32),
            max_log / (m_peak + 1e-7),
        )
        log_delta_cont = s_contraction * u

        # Step 4: Fixed-grid quantization (STE) & discrete grid clamping
        if apply_ste:
            if abs(delta_grid - (1.0 / 32.0)) < 1e-9:
                log_delta = quantize_ste_1_32(log_delta_cont)
            else:
                inv_grid = 1.0 / delta_grid
                rounded = torch.round(log_delta_cont * inv_grid) / inv_grid
                log_delta = log_delta_cont + (rounded - log_delta_cont).detach()

            max_k = math.floor(max_log / delta_grid + 1e-7)
            grid_max = max_k * delta_grid
            log_delta = torch.clamp(log_delta, min=-grid_max, max=grid_max)
        else:
            log_delta = torch.clamp(log_delta_cont, min=-max_log, max=max_log)

        # Step 5: Enforce exact boundary constraints and mask isolation
        # Outside M_valid, log_delta = 0.0 and delta = 1.0
        log_delta = torch.where(mask_bool, log_delta, torch.zeros_like(log_delta))

        # Step 6: Step delta computation
        if apply_ste:
            if abs(delta_grid - (1.0 / 32.0)) < 1e-9 and math.isclose(max_factor, 4.0):
                delta_quantized = lookup_delta_from_table(log_delta)
            else:
                delta_quantized = torch.exp(log_delta)
            delta = torch.exp(log_delta_cont) + (delta_quantized - torch.exp(log_delta_cont)).detach()
        else:
            delta = torch.exp(log_delta)

        delta = torch.where(mask_bool, delta, torch.ones_like(delta))
        delta = torch.clamp(delta, min=1.0 / max_factor, max=max_factor)

        return log_delta, delta


def lookup_delta_from_table(
    log_delta: torch.Tensor,
    table: torch.Tensor | None = None,
    inv_grid: float = 32.0,
) -> torch.Tensor:
    r"""Retrieves :math:`\Delta` from precomputed table by integer index mapping.

    Mapping:
    .. math::
        idx = \operatorname{clamp}(
            \operatorname{round}(\text{inv\_grid} \cdot \log \Delta) + \text{half}, 0, N - 1
        )
        \Delta = T_\Delta[idx]

    Eliminates all runtime transcendental operations (``exp`` / ``pow``) on CPU.

    Args:
        log_delta: Quantized log-step tensor on 1/32 grid.
        table: Optional 1D table tensor with odd length >= 3. Defaults to module-level DELTA_TABLE_129.
        inv_grid: Inverse log-grid resolution multiplier (default 32.0).

    Returns:
        torch.Tensor: Quantized step :math:`\Delta` with identical shape to ``log_delta``.

    Raises:
        TypeError: If log_delta or table is not a torch.Tensor.
        ValueError: If table is not 1D, has fewer than 3 elements, has an even number of elements, or inv_grid <= 0.
    """
    if not isinstance(log_delta, torch.Tensor):
        raise TypeError(f"log_delta must be a torch.Tensor, got {type(log_delta)}")

    if inv_grid <= 0.0:
        raise ValueError(f"inv_grid must be greater than 0, got {inv_grid}")

    if table is None:
        table_tensor = DELTA_TABLE_129
    else:
        if not isinstance(table, torch.Tensor):
            raise TypeError(f"table must be a torch.Tensor, got {type(table)}")
        if table.ndim != 1 or table.shape[0] < 3 or table.shape[0] % 2 == 0:
            raise ValueError(
                f"table must be a 1D tensor with an odd number of elements (>= 3), got shape {tuple(table.shape)}"
            )
        table_tensor = table

    table_device = table_tensor.to(device=log_delta.device, dtype=torch.float32)
    num_entries = table_device.shape[0]
    half = (num_entries - 1) // 2

    idx = torch.clamp(
        (torch.round(inv_grid * log_delta.float()).to(torch.long) + half),
        0,
        num_entries - 1,
    )
    return table_device[idx]


def compute_effective_log_scale(
    log_sigma: torch.Tensor,
    log_delta: torch.Tensor,
) -> torch.Tensor:
    r"""Computes effective log-scale in the log domain: :math:`\log \sigma_{\text{eff}} = \log \sigma - \log \Delta`.

    Allows downstream discrete entropy parameter lookup without runtime divisions.

    Args:
        log_sigma: Predicted log-scale tensor from hyper-synthesis transform.
        log_delta: Log-domain quantization step tensor.

    Returns:
        torch.Tensor: Effective log-scale tensor in strict FP32.

    Raises:
        TypeError: If inputs are not torch.Tensor.
        ValueError: If shapes are incompatible or device mismatch occurs.
    """
    if not isinstance(log_sigma, torch.Tensor) or not isinstance(log_delta, torch.Tensor):
        raise TypeError(f"Inputs must be torch.Tensor instances, got {type(log_sigma)} and {type(log_delta)}")

    if log_sigma.device != log_delta.device:
        raise ValueError(f"Device mismatch: log_sigma ({log_sigma.device}) vs log_delta ({log_delta.device})")

    if log_sigma.ndim < 2 or log_delta.ndim < 2:
        raise ValueError(
            f"Both tensors must have at least 2 dimensions, got log_sigma {tuple(log_sigma.shape)}, "
            f"log_delta {tuple(log_delta.shape)}"
        )

    if log_sigma.shape[-2:] != log_delta.shape[-2:]:
        raise ValueError(
            f"Spatial dimensions must match: log_sigma {tuple(log_sigma.shape[-2:])} vs "
            f"log_delta {tuple(log_delta.shape[-2:])}"
        )

    device_type = log_sigma.device.type if log_sigma.device.type in ("cuda", "cpu") else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        sigma_f = log_sigma.float()
        delta_f = log_delta.float()

        # Align leading dimensions if needed for broadcasting
        if delta_f.ndim == 2 and sigma_f.ndim >= 3:
            delta_aligned = delta_f.reshape((1,) * (sigma_f.ndim - 2) + delta_f.shape)
        elif delta_f.ndim == 3 and sigma_f.ndim == 4:
            if delta_f.shape[0] == sigma_f.shape[1]:
                delta_aligned = delta_f.unsqueeze(0)
            elif delta_f.shape[0] == sigma_f.shape[0]:
                delta_aligned = delta_f.unsqueeze(1)
            elif delta_f.shape[0] == 1:
                delta_aligned = delta_f.unsqueeze(0)
            else:
                delta_aligned = delta_f.unsqueeze(1)
        else:
            delta_aligned = delta_f

        try:
            return sigma_f - delta_aligned
        except RuntimeError as err:
            raise ValueError(
                f"Cannot broadcast log_delta shape {tuple(log_delta.shape)} to "
                f"log_sigma shape {tuple(log_sigma.shape)}: {err}"
            ) from err


def quantize_latents_adaptive(
    y: torch.Tensor,
    delta: torch.Tensor,
    training: bool | None = None,
) -> torch.Tensor:
    r"""Quantizes continuous latents with spatially varying step :math:`\Delta`:

    .. math::
        \tilde{y} = \operatorname{quantize}(y / \Delta) \cdot \Delta

    Operational modes:
    - training=True: Additive uniform noise relaxation :math:`\mathcal{U}(-0.5, 0.5)` applied to :math:`y / \Delta`.
    - training=False or None: Deterministic integer rounding :math:`\operatorname{round}(y / \Delta) \cdot \Delta`.

    Args:
        y: Continuous analysis latents tensor of shape (B, C, H, W) or (C, H, W).
        delta: Spatially varying step tensor of shape (B, 1, H, W), (1, H, W), (H, W), or matching y.
        training: Mode flag. If True, applies uniform noise; if False or None, applies rounding.

    Returns:
        torch.Tensor: Quantized latents with identical shape to ``y``.

    Raises:
        TypeError: If y or delta is not a torch.Tensor.
        ValueError: If spatial dimensions do not match or device mismatch occurs.
    """
    if not isinstance(y, torch.Tensor) or not isinstance(delta, torch.Tensor):
        raise TypeError(f"y and delta must be torch.Tensor instances, got {type(y)} and {type(delta)}")

    if y.device != delta.device:
        raise ValueError(f"Device mismatch: y ({y.device}) vs delta ({delta.device})")

    if y.ndim < 2 or delta.ndim < 2:
        raise ValueError(
            f"Both y and delta must have at least 2 dimensions, got y {tuple(y.shape)}, delta {tuple(delta.shape)}"
        )

    if y.shape[-2:] != delta.shape[-2:]:
        raise ValueError(f"Spatial dimensions must match: y {tuple(y.shape[-2:])} vs delta {tuple(delta.shape[-2:])}")

    device_type = y.device.type if y.device.type in ("cuda", "cpu") else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        y_f = y.float()
        delta_f = delta.float()

        # Align leading dimensions if needed for broadcasting
        if delta_f.ndim == 2 and y_f.ndim >= 3:
            delta_aligned = delta_f.reshape((1,) * (y_f.ndim - 2) + delta_f.shape)
        elif delta_f.ndim == 3 and y_f.ndim == 4:
            if delta_f.shape[0] == y_f.shape[1]:
                delta_aligned = delta_f.unsqueeze(0)
            elif delta_f.shape[0] == y_f.shape[0]:
                delta_aligned = delta_f.unsqueeze(1)
            elif delta_f.shape[0] == 1:
                delta_aligned = delta_f.unsqueeze(0)
            else:
                delta_aligned = delta_f.unsqueeze(1)
        else:
            delta_aligned = delta_f

        scaled_y = y_f / delta_aligned
        is_training = bool(training) if training is not None else False

        if is_training:
            noise = torch.empty_like(scaled_y).uniform_(-0.5, 0.5)
            quantized = scaled_y + noise
        else:
            quantized = torch.round(scaled_y)

        return quantized * delta_aligned


__all__ = [
    "DELTA_TABLE_129",
    "STE1Over32Quantizer",
    "build_delta_table",
    "compute_effective_log_scale",
    "compute_log_delta",
    "lookup_delta_from_table",
    "quantize_latents_adaptive",
    "quantize_ste_1_32",
]
