"""Spatial bit density mapping and area-normalized Relative Allocation Ratio (RAR).

Provides tools to:
1. Extract per-cell spatial bit consumption at 1/16 latent grid resolution from
   learned compression model likelihoods (y latents and z hyper-latents).
2. Segment top-percentile salient regions (ROI) and background (BG) constrained to
   valid unpadded latent cells (M_valid).
3. Compute area-normalized AllocationRatio = (B_ROI / N_ROI) / (B_BG / N_BG).
4. Compute RelativeAllocationRatio (RAR) = AllocationRatio(Model) / AllocationRatio(Baseline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SpatialBitMap:
    """Per-cell spatial bit consumption mapping at 1/16 latent resolution.

    Attributes:
        y_bits: Tensor of shape (B, 1, H_latent, W_latent) with per-cell bits for y latents.
        z_bits: Tensor of shape (B, 1, H_latent, W_latent) with per-cell bits for z hyper-latents.
        total_bits: Tensor of shape (B, 1, H_latent, W_latent) with total bits (y + z + optional sidecar).
        valid_mask: Optional boolean tensor of shape (B, 1, H_latent, W_latent) indicating valid cells.
        sidecar_bits: Optional tensor of shape (B, 1, H_latent, W_latent) with sidecar bits.
    """

    y_bits: torch.Tensor
    z_bits: torch.Tensor
    total_bits: torch.Tensor
    valid_mask: torch.Tensor | None = None
    sidecar_bits: torch.Tensor | None = None

    def total_valid_bits(self) -> float:
        """Returns total bits across all valid unpadded cells."""
        if self.valid_mask is not None:
            return float(self.total_bits[self.valid_mask].sum().item())
        return float(self.total_bits.sum().item())

    def total_y_bits(self) -> float:
        """Returns total y bits across valid unpadded cells."""
        if self.valid_mask is not None:
            return float(self.y_bits[self.valid_mask].sum().item())
        return float(self.y_bits.sum().item())

    def total_z_bits(self) -> float:
        """Returns total z bits across valid unpadded cells."""
        if self.valid_mask is not None:
            return float(self.z_bits[self.valid_mask].sum().item())
        return float(self.z_bits.sum().item())


def compute_spatial_bit_map(
    output: Any,
    *,
    valid_mask: torch.Tensor | None = None,
    sidecar_total_bits: float | None = None,
    num_orig_pixels: int | None = None,
) -> SpatialBitMap:
    """Extracts per-cell spatial bit consumption mapping at 1/16 latent resolution.

    Protocol:
    1. Extracts y_likelihoods (B, M, H_latent, W_latent) and computes per-cell
       Shannon bit consumption: bits_y = -sum_c log2(clamp(p_y, min=1e-12)).
    2. Extracts z_likelihoods (B, N, H_hyper, W_hyper) and computes hyper-latent
       bits: bits_z_hyper = -sum_c log2(clamp(p_z, min=1e-12)).
    3. Spatially upsamples z bits to 1/16 latent grid via nearest-neighbor interpolation,
       dividing by 16 so that the integral of bits across each 4x4 block exactly conserves
       the total hyper-latent bit consumption.
    4. Distributes any sidecar bit overhead uniformly across valid latent cells.

    Args:
        output: Model forward output object or dict containing 'y_likelihoods' and 'z_likelihoods'.
        valid_mask: Optional boolean or float mask indicating valid unpadded latent cells.
        sidecar_total_bits: Optional explicit sidecar bit count (e.g. len(sidecar_bytes) * 8).
        num_orig_pixels: Optional original unpadded pixel count for verification.

    Returns:
        SpatialBitMap instance containing y_bits, z_bits, total_bits, and valid_mask.

    Raises:
        TypeError: If output is invalid or likelihoods are missing.
        ValueError: If spatial dimensions are inconsistent.
    """
    if isinstance(output, dict):
        y_lik = output.get("y_likelihoods")
        z_lik = output.get("z_likelihoods")
        if y_lik is None and z_lik is None and "likelihoods" in output:
            lik_container = output["likelihoods"]
            if isinstance(lik_container, dict):
                y_lik = lik_container.get("y")
                z_lik = lik_container.get("z")
        sidecar_bpp = output.get("sidecar_bpp")
        sidecar_bytes = output.get("sidecar_bytes")
    else:
        y_lik = getattr(output, "y_likelihoods", None)
        z_lik = getattr(output, "z_likelihoods", None)
        sidecar_bpp = getattr(output, "sidecar_bpp", None)
        sidecar_bytes = getattr(output, "sidecar_bytes", None)

    if y_lik is None or not isinstance(y_lik, torch.Tensor):
        raise TypeError("output must provide 'y_likelihoods' as a torch.Tensor")
    if z_lik is None or not isinstance(z_lik, torch.Tensor):
        raise TypeError("output must provide 'z_likelihoods' as a torch.Tensor")

    device = y_lik.device
    dev_type = device.type if device.type in ("cuda", "cpu") else "cpu"

    with torch.autocast(device_type=dev_type, enabled=False):
        y_f = y_lik.float()
        z_f = z_lik.float()

        if y_f.ndim != 4:
            raise ValueError(f"y_likelihoods must be a 4D tensor (B, M, H, W), got shape {tuple(y_f.shape)}")
        if z_f.ndim != 4:
            raise ValueError(f"z_likelihoods must be a 4D tensor (B, N, H_z, W_z), got shape {tuple(z_f.shape)}")

        b, _, h_lat, w_lat = y_f.shape
        _, _, h_hyp, w_hyp = z_f.shape

        if h_lat % h_hyp != 0 or w_lat % w_hyp != 0:
            raise ValueError(
                f"Latent spatial dimensions ({h_lat}, {w_lat}) must be multiples of "
                f"hyper-latent dimensions ({h_hyp}, {w_hyp})"
            )

        stride_factor = (h_lat // h_hyp) * (w_lat // w_hyp)

        # 1. Main latent bits per cell: (B, 1, H_lat, W_lat)
        y_safe = torch.clamp(y_f, min=1e-12, max=1.0)
        y_bits = -torch.sum(torch.log2(y_safe), dim=1, keepdim=True)

        # 2. Hyper-latent bits per hyper-cell: (B, 1, H_hyp, W_hyp)
        z_safe = torch.clamp(z_f, min=1e-12, max=1.0)
        z_bits_hyper = -torch.sum(torch.log2(z_safe), dim=1, keepdim=True)

        # 3. Conserved spatial projection to 1/16 latent grid
        z_bits_scaled = z_bits_hyper / float(stride_factor)
        z_bits = F.interpolate(z_bits_scaled, size=(h_lat, w_lat), mode="nearest")

        # 4. Standardize valid mask to (B, 1, H_lat, W_lat)
        v_mask: torch.Tensor | None = None
        if valid_mask is not None:
            if not isinstance(valid_mask, torch.Tensor):
                raise TypeError(f"valid_mask must be a torch.Tensor, got {type(valid_mask)}")
            v = valid_mask.to(device=device)
            if v.ndim == 2:
                v = v.unsqueeze(0).unsqueeze(0)
            elif v.ndim == 3:
                if v.shape[0] == b:
                    v = v.unsqueeze(1)
                elif b == 1:
                    v = v[:1, :, :].unsqueeze(0)
                else:
                    v = v.unsqueeze(0)
            elif v.shape[1] != 1:
                v = v[:, :1, :, :]

            if v.shape[-2:] != (h_lat, w_lat):
                v = F.interpolate(v.float(), size=(h_lat, w_lat), mode="nearest")
            if v.shape[0] == 1 and b > 1:
                v = v.expand(b, -1, -1, -1)
            v_mask = v.bool()

        # 5. Sidecar bits if present
        sidecar_bits: torch.Tensor | None = None
        has_sidecar = (
            sidecar_total_bits is not None
            or sidecar_bytes is not None
            or (sidecar_bpp is not None and num_orig_pixels is not None)
        )
        if has_sidecar:
            sidecar_map = torch.zeros((b, 1, h_lat, w_lat), device=device, dtype=torch.float32)
            if isinstance(sidecar_bytes, (list, tuple)) and len(sidecar_bytes) == b:
                for idx in range(b):
                    s_b = sidecar_bytes[idx]
                    sample_bits = float(len(s_b) * 8) if isinstance(s_b, bytes) else 0.0
                    if v_mask is not None:
                        nv = float(v_mask[idx, 0].sum().item())
                        if nv > 0:
                            sidecar_map[idx, 0][v_mask[idx, 0]] = sample_bits / nv
                    else:
                        sidecar_map[idx].fill_(sample_bits / float(h_lat * w_lat))
            else:
                resolved_bits = 0.0
                if sidecar_total_bits is not None:
                    resolved_bits = float(sidecar_total_bits)
                elif sidecar_bytes is not None:
                    if isinstance(sidecar_bytes, bytes):
                        resolved_bits = float(len(sidecar_bytes) * 8)
                    elif isinstance(sidecar_bytes, (list, tuple)):
                        resolved_bits = float(sum(len(item) * 8 for item in sidecar_bytes if isinstance(item, bytes)))
                elif sidecar_bpp is not None and num_orig_pixels is not None:
                    bpp_val = float(sidecar_bpp.item() if isinstance(sidecar_bpp, torch.Tensor) else sidecar_bpp)
                    resolved_bits = bpp_val * float(num_orig_pixels)

                if resolved_bits > 0.0:
                    per_sample_bits = resolved_bits / float(b)
                    for idx in range(b):
                        if v_mask is not None:
                            nv = float(v_mask[idx, 0].sum().item())
                            if nv > 0:
                                sidecar_map[idx, 0][v_mask[idx, 0]] = per_sample_bits / nv
                        else:
                            sidecar_map[idx].fill_(per_sample_bits / float(h_lat * w_lat))
            sidecar_bits = sidecar_map
            total_bits = y_bits + z_bits + sidecar_bits
        else:
            total_bits = y_bits + z_bits

        return SpatialBitMap(
            y_bits=y_bits,
            z_bits=z_bits,
            total_bits=total_bits,
            valid_mask=v_mask,
            sidecar_bits=sidecar_bits,
        )


def extract_top_percentile_mask(
    saliency: torch.Tensor | None = None,
    *,
    s: torch.Tensor | None = None,
    percentile: float = 80.0,
    target_shape: tuple[int, int] | None = None,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Segments binary top-percentile ROI and background masks on latent grid.

    Protocol:
    1. Resizes saliency tensor to target_shape (H_latent, W_latent) using area interpolation.
    2. Constrains evaluation to unpadded valid cells in valid_mask (M_valid).
    3. Computes the threshold tau at the given percentile (default 80.0 -> top 20% most salient).
    4. Evaluates:
       M_ROI = (S_latent >= tau) & M_valid
       M_BG  = (S_latent < tau) & M_valid
    5. Includes tie-breaking logic so uniform saliency inputs produce valid non-empty partitions.

    Args:
        saliency: Saliency map tensor in [0, 1]. Shapes: (B, 1, H, W), (B, H, W), (1, H, W), (H, W).
        s: Keyword-only alias for saliency.
        percentile: Percentile cutoff in [0, 100] (default 80.0 -> top 20% ROI).
        target_shape: Expected spatial shape (H_latent, W_latent). Inferred if None.
        valid_mask: Optional boolean or float mask of valid unpadded cells.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (roi_mask, bg_mask) boolean tensors of shape (B, 1, H, W).

    Raises:
        TypeError: If saliency is not a torch.Tensor.
        ValueError: If percentile is not in [0, 100] or spatial dimensions are invalid.
    """
    resolved_s = saliency if saliency is not None else s
    if resolved_s is None:
        raise ValueError("saliency tensor must be provided via 'saliency' or 's'")
    if not isinstance(resolved_s, torch.Tensor):
        raise TypeError(f"saliency must be a torch.Tensor, got {type(resolved_s)}")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")

    s = resolved_s.float()
    if s.ndim == 2:
        s = s.unsqueeze(0).unsqueeze(0)
    elif s.ndim == 3:
        s = s.unsqueeze(0) if s.shape[0] == 1 else s.unsqueeze(1)
    elif s.shape[1] != 1:
        s = torch.mean(s, dim=1, keepdim=True)

    b = s.shape[0]

    # Target spatial resolution
    if target_shape is not None:
        if len(target_shape) != 2 or target_shape[0] <= 0 or target_shape[1] <= 0:
            raise ValueError(f"target_shape must be positive 2-tuple (H, W), got {target_shape}")
        if s.shape[-2:] != target_shape:
            s = F.interpolate(s, size=target_shape, mode="area")

    h, w = s.shape[-2], s.shape[-1]
    s = torch.clamp(s, 0.0, 1.0)

    # Standardize valid_mask to (B, 1, H, W)
    if valid_mask is not None:
        v = valid_mask.to(device=s.device)
        if v.ndim == 2:
            v = v.unsqueeze(0).unsqueeze(0)
        elif v.ndim == 3:
            v = v.unsqueeze(0) if v.shape[0] == 1 else v.unsqueeze(1)
        elif v.shape[1] != 1:
            v = v[:, :1, :, :]

        if v.shape[-2:] != (h, w):
            v = F.interpolate(v.float(), size=(h, w), mode="nearest")
        if v.shape[0] == 1 and b > 1:
            v = v.expand(b, -1, -1, -1)
        v_mask = v.bool()
    else:
        v_mask = torch.ones((b, 1, h, w), device=s.device, dtype=torch.bool)

    roi_masks: list[torch.Tensor] = []
    bg_masks: list[torch.Tensor] = []

    q_fraction = percentile / 100.0

    for sample_idx in range(b):
        s_sample = s[sample_idx, 0]
        v_sample = v_mask[sample_idx, 0]

        valid_values = s_sample[v_sample]
        n_valid = int(valid_values.numel())

        if n_valid == 0:
            roi_masks.append(torch.zeros((1, 1, h, w), device=s.device, dtype=torch.bool))
            bg_masks.append(torch.zeros((1, 1, h, w), device=s.device, dtype=torch.bool))
            continue

        tau = float(torch.quantile(valid_values, q=q_fraction).item())

        sample_roi = (s_sample >= tau) & v_sample
        sample_bg = (s_sample < tau) & v_sample

        # Handle uniform or near-uniform tie-breaking: ensure non-empty partitions
        if sample_roi.sum().item() == n_valid and n_valid > 1:
            # All values are equal; partition deterministically by linear index
            valid_coords = torch.nonzero(v_sample, as_tuple=False)
            k_roi = max(1, round(n_valid * (1.0 - q_fraction)))
            sample_roi = torch.zeros_like(v_sample, dtype=torch.bool)
            selected = valid_coords[:k_roi]
            sample_roi[selected[:, 0], selected[:, 1]] = True
            sample_bg = v_sample & (~sample_roi)
        elif sample_roi.sum().item() == 0 and n_valid > 0:
            # Fallback if strict rounding excluded all elements
            max_val = valid_values.max()
            sample_roi = (s_sample >= max_val) & v_sample
            sample_bg = v_sample & (~sample_roi)

        roi_masks.append(sample_roi.unsqueeze(0).unsqueeze(0))
        bg_masks.append(sample_bg.unsqueeze(0).unsqueeze(0))

    final_roi = torch.cat(roi_masks, dim=0)
    final_bg = torch.cat(bg_masks, dim=0)
    return final_roi, final_bg


def compute_allocation_ratio(
    bit_tensor: torch.Tensor,
    roi_mask: torch.Tensor,
    bg_mask: torch.Tensor | None = None,
    *,
    valid_mask: torch.Tensor | None = None,
) -> float:
    """Computes area-normalized spatial bit allocation ratio.

    Implements:
        AllocationRatio = (B_ROI / N_ROI) / (B_BG / N_BG)

    Args:
        bit_tensor: Tensor of per-cell bit counts (B, 1, H, W) or (H, W).
        roi_mask: Boolean tensor indicating salient region of interest (ROI).
        bg_mask: Optional boolean tensor indicating non-salient background (BG).
        valid_mask: Optional boolean tensor indicating valid unpadded cells.
            If bg_mask is None and valid_mask is provided, bg_mask is (~roi_mask) & valid_mask.

    Returns:
        float: Scalar allocation ratio (>= 0.0). Returns 1.0 if either partition is empty.
    """
    if not isinstance(bit_tensor, torch.Tensor):
        raise TypeError(f"bit_tensor must be a torch.Tensor, got {type(bit_tensor)}")
    if not isinstance(roi_mask, torch.Tensor):
        raise TypeError(f"roi_mask must be a torch.Tensor, got {type(roi_mask)}")

    b_t = bit_tensor.float()
    r_m = roi_mask.bool()

    if bg_mask is not None:
        if not isinstance(bg_mask, torch.Tensor):
            raise TypeError(f"bg_mask must be a torch.Tensor, got {type(bg_mask)}")
        b_m = bg_mask.bool()
    elif valid_mask is not None:
        b_m = (~r_m) & valid_mask.bool()
    else:
        b_m = ~r_m

    n_roi = float(r_m.sum().item())
    n_bg = float(b_m.sum().item())

    if n_roi <= 0 or n_bg <= 0:
        return 1.0

    b_roi = float(b_t[r_m].sum().item())
    b_bg = float(b_t[b_m].sum().item())

    density_roi = b_roi / n_roi
    density_bg = b_bg / n_bg

    if density_bg <= 1e-12:
        return float("inf") if density_roi > 1e-12 else 1.0

    return density_roi / density_bg


def compute_relative_allocation_ratio(
    model_bit_map: SpatialBitMap | torch.Tensor,
    baseline_bit_map: SpatialBitMap | torch.Tensor,
    roi_mask: torch.Tensor,
    bg_mask: torch.Tensor | None = None,
    *,
    metric: str = "total",
) -> float:
    """Computes the Relative Allocation Ratio (RAR) against Model A baseline.

    Implements:
        RAR = AllocationRatio(Model) / AllocationRatio(Baseline)

    Args:
        model_bit_map: SpatialBitMap or bit tensor for evaluated model.
        baseline_bit_map: SpatialBitMap or bit tensor for baseline model (Model A).
        roi_mask: Boolean tensor indicating salient region of interest (ROI).
        bg_mask: Optional boolean tensor indicating non-salient background (BG).
        metric: Metric component to evaluate ('total', 'y', or 'z').

    Returns:
        float: Scalar Relative Allocation Ratio.
    """
    if isinstance(model_bit_map, SpatialBitMap):
        if metric == "y":
            t_model = model_bit_map.y_bits
        elif metric == "z":
            t_model = model_bit_map.z_bits
        else:
            t_model = model_bit_map.total_bits
    else:
        t_model = model_bit_map

    if isinstance(baseline_bit_map, SpatialBitMap):
        if metric == "y":
            t_base = baseline_bit_map.y_bits
        elif metric == "z":
            t_base = baseline_bit_map.z_bits
        else:
            t_base = baseline_bit_map.total_bits
    else:
        t_base = baseline_bit_map

    ar_model = compute_allocation_ratio(t_model, roi_mask, bg_mask)
    ar_base = compute_allocation_ratio(t_base, roi_mask, bg_mask)

    if math.isinf(ar_base) and math.isinf(ar_model):
        return 1.0
    if ar_base <= 1e-12:
        return 1.0 if ar_model <= 1e-12 else float("inf")

    return ar_model / ar_base


def evaluate_rar_across_models(
    models_outputs: dict[str, Any],
    saliency: torch.Tensor | None = None,
    *,
    s: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    percentile: float = 80.0,
    baseline_key: str = "A",
    num_orig_pixels: int | None = None,
    sidecar_total_bits: float | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluates Allocation Ratio and Relative Allocation Ratio across multiple models.

    Args:
        models_outputs: Mapping of model_name -> model forward output or dict.
        saliency: Ground truth visual saliency map.
        s: Keyword-only alias for saliency.
        valid_mask: Optional validity mask for unpadded latent cells.
        percentile: Percentile threshold for ROI (default 80.0 -> top 20%).
        baseline_key: Key for reference baseline model (default 'A').
        num_orig_pixels: Optional original pixel count for sidecar BPP derivation.
        sidecar_total_bits: Optional explicit sidecar bit count override.

    Returns:
        dict[str, dict[str, float]]: Nested mapping containing allocation metrics for each model:
            - 'allocation_ratio_y'
            - 'allocation_ratio_z'
            - 'allocation_ratio_total'
            - 'rar_y'
            - 'rar_z'
            - 'rar_total'
            - 'n_roi'
            - 'n_bg'
    """
    if not models_outputs:
        raise ValueError("models_outputs dictionary cannot be empty")

    resolved_s = saliency if saliency is not None else s
    if resolved_s is None:
        raise ValueError("saliency tensor must be provided via 'saliency' or 's'")

    # Compute spatial bit maps for all models
    bit_maps: dict[str, SpatialBitMap] = {}
    target_shape: tuple[int, int] | None = None

    for name, out in models_outputs.items():
        bm = compute_spatial_bit_map(
            out,
            valid_mask=valid_mask,
            num_orig_pixels=num_orig_pixels,
            sidecar_total_bits=sidecar_total_bits,
        )
        bit_maps[name] = bm
        if target_shape is None:
            target_shape = (bm.total_bits.shape[-2], bm.total_bits.shape[-1])

    if target_shape is None:
        raise ValueError("Could not determine spatial latent dimensions from model outputs")

    # Extract common ROI and BG masks
    roi_mask, bg_mask = extract_top_percentile_mask(
        resolved_s,
        percentile=percentile,
        target_shape=target_shape,
        valid_mask=valid_mask,
    )

    n_roi = float(roi_mask.sum().item())
    n_bg = float(bg_mask.sum().item())

    # Reference baseline bit map
    if baseline_key not in bit_maps:
        raise KeyError(f"baseline_key '{baseline_key}' not found in models_outputs keys: {list(bit_maps.keys())}")
    base_bm = bit_maps[baseline_key]

    results: dict[str, dict[str, float]] = {}

    for name, bm in bit_maps.items():
        ar_y = compute_allocation_ratio(bm.y_bits, roi_mask, bg_mask)
        ar_z = compute_allocation_ratio(bm.z_bits, roi_mask, bg_mask)
        ar_tot = compute_allocation_ratio(bm.total_bits, roi_mask, bg_mask)

        rar_y = compute_relative_allocation_ratio(bm, base_bm, roi_mask, bg_mask, metric="y")
        rar_z = compute_relative_allocation_ratio(bm, base_bm, roi_mask, bg_mask, metric="z")
        rar_tot = compute_relative_allocation_ratio(bm, base_bm, roi_mask, bg_mask, metric="total")

        results[name] = {
            "allocation_ratio_y": ar_y,
            "allocation_ratio_z": ar_z,
            "allocation_ratio_total": ar_tot,
            "rar_y": rar_y,
            "rar_z": rar_z,
            "rar_total": rar_tot,
            "n_roi": n_roi,
            "n_bg": n_bg,
        }

    return results
