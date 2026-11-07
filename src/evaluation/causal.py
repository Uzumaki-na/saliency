"""Causal intervention protocol via latent block-shuffling and transplants.

Implements causal analysis experiments to demonstrate that visual saliency-conditioned
quantization (Model C and Model D) causally drives local bit allocation rather than
merely tracking underlying visual texture complexity.

Protocol:
1. Spatial block-shuffling: Permutes non-overlapping KxK spatial blocks (e.g. 4x4 or 8x8)
   of saliency maps S_pred / S, re-encodes the image, and verifies that bit consumption
   follows the permuted spatial locations.
2. Invertible block un-shuffling: Mathematically inverts the spatial permutation to restore
   the original spatial tile arrangement, proving round-trip conservation.
3. Targeted tile swap: Swaps a salient ROI block and a background block, confirming that
   local bit consumption flips between the two tiles.
4. Cross-image saliency transplanting: Overlays a donor image's saliency map onto a target
   image, proving that bit density shifts to the donor's ROI regardless of target content.
5. Control validation on Model A: Verifies that interventions produce zero bit redistribution
   on the unconditioned standard hyperprior baseline.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluation.spatial_allocation import (
    SpatialBitMap,
    compute_allocation_ratio,
    compute_spatial_bit_map,
    extract_top_percentile_mask,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlockShuffleResult:
    """Result of spatial block shuffling with detailed metadata.

    Attributes:
        shuffled_tensor: Shuffled output tensor of same shape and dtype as input.
        permutation: 1D or 2D tensor of block indices used for shuffling.
        inverse_permutation: Tensor of inverse block indices restoring original order.
        block_size: Spatial size K of each square tile.
        num_blocks_h: Number of blocks along height.
        num_blocks_w: Number of blocks along width.
        num_blocks_total: Total number of blocks (num_blocks_h * num_blocks_w).
    """

    shuffled_tensor: torch.Tensor
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    block_size: int
    num_blocks_h: int
    num_blocks_w: int
    num_blocks_total: int


@dataclass(frozen=True)
class CausalInterventionResult:
    """Result of causal block-shuffling intervention experiment.

    Attributes:
        model_name: Name or class identifier of evaluated model.
        block_size: Spatial tile size K used for shuffling.
        baseline_bit_map: SpatialBitMap from unpermuted baseline encoding.
        intervened_bit_map: SpatialBitMap from intervened (shuffled saliency) encoding.
        unshuffled_bit_map: SpatialBitMap obtained by inverting block permutation on intervened bits.
        original_saliency: Reference unpermuted saliency map.
        shuffled_saliency: Intervened shuffled saliency map.
        permutation: Permutation index tensor.
        corr_interv_vs_shuffled: Spatial correlation between intervened y-bits and shuffled saliency.
        corr_interv_vs_original: Spatial correlation between intervened y-bits and original saliency.
        corr_unshuffled_vs_baseline: Spatial correlation between unshuffled intervened bits and baseline bits.
        causal_shift: Difference (corr_interv_vs_shuffled - corr_interv_vs_original).
    """

    model_name: str
    block_size: int
    baseline_bit_map: SpatialBitMap
    intervened_bit_map: SpatialBitMap
    unshuffled_bit_map: SpatialBitMap
    original_saliency: torch.Tensor
    shuffled_saliency: torch.Tensor
    permutation: torch.Tensor
    corr_interv_vs_shuffled: float
    corr_interv_vs_original: float
    corr_unshuffled_vs_baseline: float
    causal_shift: float


@dataclass(frozen=True)
class TileSwapResult:
    """Result of targeted tile swap micro-intervention experiment.

    Attributes:
        model_name: Evaluated model identifier.
        block_idx_1: Index of first block (e.g., ROI).
        block_idx_2: Index of second block (e.g., background).
        block_size: Tile dimension K.
        baseline_bits_tile_1: Mean bits per cell in tile 1 under baseline encoding.
        baseline_bits_tile_2: Mean bits per cell in tile 2 under baseline encoding.
        swapped_bits_tile_1: Mean bits per cell in tile 1 under swapped encoding.
        swapped_bits_tile_2: Mean bits per cell in tile 2 under swapped encoding.
        baseline_bit_diff: Difference (baseline_tile_1 - baseline_tile_2).
        swapped_bit_diff: Difference (swapped_tile_1 - swapped_tile_2).
        bit_redistribution_ratio: Ratio quantifying the reversal of bit consumption.
    """

    model_name: str
    block_idx_1: int
    block_idx_2: int
    block_size: int
    baseline_bits_tile_1: float
    baseline_bits_tile_2: float
    swapped_bits_tile_1: float
    swapped_bits_tile_2: float
    baseline_bit_diff: float
    swapped_bit_diff: float
    bit_redistribution_ratio: float


@dataclass(frozen=True)
class CausalTransplantResult:
    """Result of cross-image saliency transplant intervention experiment.

    Attributes:
        model_name: Evaluated model identifier.
        target_baseline_bit_map: SpatialBitMap of target image with natural saliency.
        transplanted_bit_map: SpatialBitMap of target image encoded with donor saliency.
        donor_saliency: Resampled donor saliency map.
        target_saliency: Reference target image saliency map (or predicted).
        target_roi_mask: Top-percentile ROI mask of target image.
        donor_roi_mask: Top-percentile ROI mask of donor image.
        baseline_ar_on_donor_roi: Target image Allocation Ratio on donor ROI before transplant.
        transplanted_ar_on_donor_roi: Target image Allocation Ratio on donor ROI after transplant.
        ar_gain_on_donor_roi: Gain (transplanted_ar - baseline_ar).
        corr_baseline_vs_donor_saliency: Spatial correlation between target baseline bits and donor saliency.
        corr_transplanted_vs_donor_saliency: Spatial correlation between transplanted bits and donor saliency.
    """

    model_name: str
    target_baseline_bit_map: SpatialBitMap
    transplanted_bit_map: SpatialBitMap
    donor_saliency: torch.Tensor
    target_saliency: torch.Tensor | None
    target_roi_mask: torch.Tensor
    donor_roi_mask: torch.Tensor
    baseline_ar_on_donor_roi: float
    transplanted_ar_on_donor_roi: float
    ar_gain_on_donor_roi: float
    corr_baseline_vs_donor_saliency: float
    corr_transplanted_vs_donor_saliency: float


def shuffle_spatial_blocks(
    tensor: torch.Tensor,
    block_size: int,
    *,
    permutation: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partitions a 2D, 3D, or 4D spatial tensor into non-overlapping KxK blocks and permutes them.

    Args:
        tensor: Spatial tensor of shape (H, W), (C, H, W), or (B, C, H, W).
        block_size: Integer size K of each square spatial block (e.g. 4 or 8).
        permutation: Optional pre-defined 1D tensor of block indices (length N_blocks)
            or 2D tensor of shape (B, N_blocks).
        generator: Optional PyTorch generator for reproducible permutations.
        seed: Optional integer seed for reproducibility (used if generator is None).

    Returns:
        tuple (shuffled_tensor, permutation) where shuffled_tensor has identical
        shape and dtype, and permutation contains the block ordering.

    Raises:
        ValueError: If tensor has invalid dimensions or spatial dimensions are not divisible by block_size.
    """
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    orig_ndim = tensor.ndim
    if orig_ndim == 2:
        t = tensor.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        t = tensor.unsqueeze(0)
    elif orig_ndim == 4:
        t = tensor
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {orig_ndim}D tensor of shape {tuple(tensor.shape)}")

    b, c, h, w = t.shape
    if h % block_size != 0 or w % block_size != 0:
        raise ValueError(
            f"Spatial dimensions ({h}, {w}) must be divisible by block_size={block_size}. "
            f"Pad tensor or select a compatible block size."
        )

    n_h = h // block_size
    n_w = w // block_size
    num_blocks = n_h * n_w

    if num_blocks == 1:
        perm = torch.tensor([0], device=tensor.device, dtype=torch.long)
        return tensor.clone(), perm

    # Generate or validate permutation
    if permutation is not None:
        if isinstance(permutation, (list, tuple)):
            perm = torch.tensor(permutation, dtype=torch.long, device=t.device)
        else:
            perm = permutation.to(device=t.device, dtype=torch.long)

        if perm.ndim == 1:
            if perm.shape[0] != num_blocks:
                raise ValueError(f"Permutation length ({perm.shape[0]}) must equal total blocks ({num_blocks})")
        elif perm.ndim == 2:
            if perm.shape != (b, num_blocks):
                raise ValueError(
                    f"2D permutation shape {tuple(perm.shape)} must match (batch_size={b}, num_blocks={num_blocks})"
                )
        else:
            raise ValueError(f"Permutation must be 1D or 2D, got {perm.ndim}D")
    else:
        gen = generator
        if gen is None and seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
        perm = torch.randperm(num_blocks, generator=gen, device="cpu").to(device=t.device)

    # Decompose into blocks: (B, C, n_h, K, n_w, K)
    t_blocks = t.view(b, c, n_h, block_size, n_w, block_size)
    # Permute to: (B, n_h, n_w, C, K, K)
    t_blocks = t_blocks.permute(0, 2, 4, 1, 3, 5).contiguous()
    # Flatten spatial block grid: (B, num_blocks, C, K, K)
    t_blocks = t_blocks.view(b, num_blocks, c, block_size, block_size)

    # Re-index blocks by permutation
    if perm.ndim == 1:
        shuffled_blocks = t_blocks[:, perm]
    else:
        idx = perm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, c, block_size, block_size)
        shuffled_blocks = torch.gather(t_blocks, 1, idx)

    # Reconstruct 2D spatial grid: (B, n_h, n_w, C, K, K)
    shuffled_blocks = shuffled_blocks.view(b, n_h, n_w, c, block_size, block_size)
    # Permute back: (B, C, n_h, K, n_w, K)
    shuffled_blocks = shuffled_blocks.permute(0, 3, 1, 4, 2, 5).contiguous()
    # Reshape to: (B, C, H, W)
    shuffled_t = shuffled_blocks.view(b, c, h, w)

    # Restore original tensor rank
    if orig_ndim == 2:
        out = shuffled_t.squeeze(0).squeeze(0)
    elif orig_ndim == 3:
        out = shuffled_t.squeeze(0)
    else:
        out = shuffled_t

    return out, perm


def shuffle_spatial_blocks_detailed(
    tensor: torch.Tensor,
    block_size: int,
    *,
    permutation: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    seed: int | None = None,
) -> BlockShuffleResult:
    """Partitions spatial blocks and returns detailed BlockShuffleResult container."""
    shuffled, perm = shuffle_spatial_blocks(
        tensor,
        block_size=block_size,
        permutation=permutation,
        generator=generator,
        seed=seed,
    )
    if perm.ndim == 1:
        inv_perm = torch.argsort(perm)
    else:
        inv_perm = torch.argsort(perm, dim=1)

    h, w = tensor.shape[-2:]
    n_h = h // block_size
    n_w = w // block_size

    return BlockShuffleResult(
        shuffled_tensor=shuffled,
        permutation=perm,
        inverse_permutation=inv_perm,
        block_size=block_size,
        num_blocks_h=n_h,
        num_blocks_w=n_w,
        num_blocks_total=n_h * n_w,
    )


def unshuffle_spatial_blocks(
    shuffled_tensor: torch.Tensor,
    permutation: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Inverts spatial block shuffling, restoring the original spatial tile arrangement.

    Args:
        shuffled_tensor: Shuffled tensor of shape (H, W), (C, H, W), or (B, C, H, W).
        permutation: 1D or 2D permutation index tensor returned by shuffle_spatial_blocks.
        block_size: Tile dimension K.

    Returns:
        Tensor of identical shape and dtype restored to its original spatial arrangement.
    """
    if permutation.ndim == 1:
        inv_perm = torch.argsort(permutation)
    elif permutation.ndim == 2:
        inv_perm = torch.argsort(permutation, dim=1)
    else:
        raise ValueError(f"Permutation must be 1D or 2D, got {permutation.ndim}D")

    restored, _ = shuffle_spatial_blocks(
        shuffled_tensor,
        block_size=block_size,
        permutation=inv_perm,
    )
    return restored


def swap_spatial_blocks(
    tensor: torch.Tensor,
    block_idx_1: int,
    block_idx_2: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Swaps two distinct spatial blocks in a tensor.

    Args:
        tensor: Spatial tensor (H, W), (C, H, W), or (B, C, H, W).
        block_idx_1: First block linear index in [0, num_blocks - 1].
        block_idx_2: Second block linear index in [0, num_blocks - 1].
        block_size: Tile dimension K.

    Returns:
        tuple (swapped_tensor, permutation).
    """
    orig_ndim = tensor.ndim
    if orig_ndim == 2:
        h, w = tensor.shape
    elif orig_ndim in (3, 4):
        h, w = tensor.shape[-2:]
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {orig_ndim}D")

    n_h = h // block_size
    n_w = w // block_size
    num_blocks = n_h * n_w

    if block_idx_1 == block_idx_2:
        raise ValueError(f"block_idx_1 and block_idx_2 must be distinct, got {block_idx_1}")

    if not (0 <= block_idx_1 < num_blocks and 0 <= block_idx_2 < num_blocks):
        raise ValueError(f"Block indices ({block_idx_1}, {block_idx_2}) must be in range [0, {num_blocks - 1}]")

    perm = torch.arange(num_blocks, device=tensor.device, dtype=torch.long)
    perm[block_idx_1] = block_idx_2
    perm[block_idx_2] = block_idx_1

    return shuffle_spatial_blocks(tensor, block_size=block_size, permutation=perm)


def transplant_saliency_map(
    donor_saliency: torch.Tensor,
    target_shape: tuple[int, int] | torch.Tensor,
    *,
    mode: str = "area",
) -> torch.Tensor:
    """Resamples a donor saliency map to align with a target spatial latent resolution.

    Args:
        donor_saliency: Donor saliency map of shape (H, W), (1, H, W), or (B, 1, H, W).
        target_shape: Target spatial dimensions (target_h, target_w) or a reference tensor.
        mode: Interpolation mode ('area', 'bilinear', or 'nearest'). Defaults to 'area'.

    Returns:
        Resampled and clamped [0, 1] saliency map of shape (B, 1, target_h, target_w).
    """
    if isinstance(target_shape, torch.Tensor):
        target_hw = (target_shape.shape[-2], target_shape.shape[-1])
    else:
        target_hw = (int(target_shape[0]), int(target_shape[1]))

    s = donor_saliency
    if s.ndim == 2:
        s = s.unsqueeze(0).unsqueeze(0)
    elif s.ndim == 3:
        if s.shape[0] > 1:
            s = torch.mean(s, dim=0, keepdim=True).unsqueeze(0)
        else:
            s = s.unsqueeze(0)
    elif s.ndim == 4:
        if s.shape[1] > 1:
            s = torch.mean(s, dim=1, keepdim=True)
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D saliency tensor, got {s.ndim}D")

    if s.shape[-2:] != target_hw:
        if mode in ("bilinear", "bicubic"):
            aligned = F.interpolate(s.float(), size=target_hw, mode=mode, align_corners=False)
        else:
            aligned = F.interpolate(s.float(), size=target_hw, mode=mode)
    else:
        aligned = s.float()

    return torch.clamp(aligned, 0.0, 1.0)


def compute_spatial_correlation(
    map1: torch.Tensor,
    map2: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    """Computes Pearson spatial correlation between two maps over valid cells.

    Supports batch size B >= 1, computing per-sample correlation to prevent
    between-image bitrate differences from confounding spatial correlation.

    Args:
        map1: First spatial map tensor.
        map2: Second spatial map tensor.
        valid_mask: Optional boolean valid mask.

    Returns:
        Pearson correlation coefficient r in [-1.0, 1.0]. Returns 0.0 if either map
        is degenerate/constant, or 1.0 if both are identical constants.
    """
    m1 = map1.float()
    m2 = map2.float().to(device=m1.device)

    if m1.ndim == 2:
        m1 = m1.unsqueeze(0).unsqueeze(0)
    elif m1.ndim == 3:
        m1 = m1.unsqueeze(1) if m1.shape[0] != 1 else m1.unsqueeze(0)

    if m2.ndim == 2:
        m2 = m2.unsqueeze(0).unsqueeze(0)
    elif m2.ndim == 3:
        m2 = m2.unsqueeze(1) if m2.shape[0] != 1 else m2.unsqueeze(0)

    # Align spatial dimensions if they differ
    if m1.shape[-2:] != m2.shape[-2:]:
        m2 = F.interpolate(m2, size=m1.shape[-2:], mode="area")

    vm = valid_mask
    if vm is not None:
        vm = vm.to(device=m1.device)
        if vm.ndim == 2:
            vm = vm.unsqueeze(0).unsqueeze(0)
        elif vm.ndim == 3:
            vm = vm.unsqueeze(1) if vm.shape[0] != 1 else vm.unsqueeze(0)
        if vm.shape[-2:] != m1.shape[-2:]:
            vm = F.interpolate(vm.float(), size=m1.shape[-2:], mode="nearest").bool()

    # If batch size B > 1, evaluate per-sample to avoid conflating image-level means
    if m1.shape[0] > 1:
        corrs = [
            compute_spatial_correlation(
                m1[i : i + 1],
                m2[i : i + 1] if m2.shape[0] > 1 else m2,
                vm[i : i + 1] if vm is not None and vm.shape[0] > 1 else vm,
            )
            for i in range(m1.shape[0])
        ]
        return float(sum(corrs) / len(corrs))

    # Single sample evaluation
    if vm is not None:
        if vm.shape[0] != m1.shape[0] and vm.shape[0] == 1:
            vm = vm.expand(m1.shape[0], -1, -1, -1)
        v1 = m1[vm].flatten()
        v2 = m2[vm].flatten()
    else:
        v1 = m1.flatten()
        v2 = m2.flatten()

    if v1.numel() < 2 or v2.numel() < 2:
        return 0.0

    d1 = v1 - v1.mean()
    d2 = v2 - v2.mean()

    var1 = float((d1**2).sum().item())
    var2 = float((d2**2).sum().item())

    # Handle zero/constant variance
    if var1 < 1e-12 and var2 < 1e-12:
        return 1.0 if float(torch.abs(v1.mean() - v2.mean()).item()) < 1e-6 else 0.0
    if var1 < 1e-12 or var2 < 1e-12:
        return 0.0

    denom = (var1 * var2) ** 0.5
    cov = float((d1 * d2).sum().item())
    r = cov / denom
    return max(-1.0, min(1.0, r))


def _invoke_model_forward(
    model: nn.Module,
    x: torch.Tensor,
    *,
    saliency_map: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    training: bool = False,
) -> Any:
    """Safely executes model forward pass, inspecting accepted keyword arguments."""
    sig = inspect.signature(model.forward)
    params = sig.parameters
    kwargs: dict[str, Any] = {}
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    if "training" in params or has_var_kw:
        kwargs["training"] = training

    if saliency_map is not None:
        if "saliency_map" in params or has_var_kw:
            kwargs["saliency_map"] = saliency_map
        elif "s" in params:
            kwargs["s"] = saliency_map

    if valid_mask is not None:
        if "valid_mask" in params or has_var_kw:
            kwargs["valid_mask"] = valid_mask
        elif "mask" in params:
            kwargs["mask"] = valid_mask

    return model(x, **kwargs)


def run_causal_block_intervention(
    model: nn.Module,
    x: torch.Tensor,
    block_size: int,
    *,
    saliency_map: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    permutation: torch.Tensor | None = None,
    seed: int | None = 42,
) -> CausalInterventionResult:
    """Executes spatial block-shuffling causal intervention protocol on a model.

    Protocol:
    1. Runs baseline encoding on image x with reference saliency to obtain B_base.
    2. Divides saliency into KxK blocks and applies spatial permutation pi -> S_shuffled.
    3. Runs intervened encoding on image x with S_shuffled to obtain B_intervened.
    4. Inverts permutation on B_intervened -> B_unshuffled.
    5. Computes correlation metrics:
       - corr(B_intervened, S_shuffled) vs corr(B_intervened, S_original)
       - corr(B_unshuffled, B_base)

    Args:
        model: Compression neural network (Model A, B, C, D, or Model C-Transmitted).
        x: Input image tensor of shape (B, 3, H, W) where H, W are multiples of 64.
        block_size: Tile dimension K for shuffling (e.g. 4 or 8).
        saliency_map: Optional ground-truth visual saliency map in [0, 1].
        valid_mask: Optional unpadded latent validity mask M_valid.
        permutation: Optional explicit permutation indices.
        seed: Random seed for reproducible block permutation. Defaults to 42.

    Returns:
        CausalInterventionResult containing all bit maps, correlations, and causal shift metrics.
    """
    was_training = model.training
    model.eval()

    # Align devices
    if saliency_map is not None:
        saliency_map = saliency_map.to(device=x.device)
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=x.device)

    try:
        with torch.no_grad():
            # Step 1: Baseline pass
            out_base = _invoke_model_forward(
                model,
                x,
                saliency_map=saliency_map,
                valid_mask=valid_mask,
                training=False,
            )
            bit_map_base = compute_spatial_bit_map(out_base, valid_mask=valid_mask)

            # Step 2: Extract reference saliency
            if saliency_map is not None:
                ref_s = saliency_map
            elif hasattr(out_base, "s_pred") and out_base.s_pred is not None:
                ref_s = out_base.s_pred
            else:
                raise ValueError("saliency_map must be provided when model does not predict s_pred.")

            latent_hw = bit_map_base.y_bits.shape[-2:]
            ref_s_aligned = transplant_saliency_map(ref_s, latent_hw)

            # Match batch size if single saliency map provided for B > 1
            if ref_s_aligned.shape[0] == 1 and x.shape[0] > 1:
                ref_s_aligned = ref_s_aligned.expand(x.shape[0], -1, -1, -1)

            # Step 3: Spatial block shuffling on saliency map
            shuffled_s, perm = shuffle_spatial_blocks(
                ref_s_aligned,
                block_size=block_size,
                permutation=permutation,
                seed=seed,
            )

            # Step 4: Intervened pass
            out_interv = _invoke_model_forward(
                model,
                x,
                saliency_map=shuffled_s,
                valid_mask=valid_mask,
                training=False,
            )
            bit_map_interv = compute_spatial_bit_map(out_interv, valid_mask=valid_mask)

            # Step 5: Un-shuffle intervened bit maps
            unshuffled_y = unshuffle_spatial_blocks(bit_map_interv.y_bits, perm, block_size=block_size)
            unshuffled_z = unshuffle_spatial_blocks(bit_map_interv.z_bits, perm, block_size=block_size)
            unshuffled_total = unshuffle_spatial_blocks(bit_map_interv.total_bits, perm, block_size=block_size)

            unshuffled_bit_map = SpatialBitMap(
                y_bits=unshuffled_y,
                z_bits=unshuffled_z,
                total_bits=unshuffled_total,
                valid_mask=valid_mask,
                sidecar_bits=bit_map_interv.sidecar_bits,
            )

            # Step 6: Correlations and causal shift metrics
            corr_interv_shuffled = compute_spatial_correlation(
                bit_map_interv.y_bits,
                shuffled_s,
                valid_mask=valid_mask,
            )
            corr_interv_orig = compute_spatial_correlation(
                bit_map_interv.y_bits,
                ref_s_aligned,
                valid_mask=valid_mask,
            )
            corr_unshuffled_base = compute_spatial_correlation(
                unshuffled_bit_map.y_bits,
                bit_map_base.y_bits,
                valid_mask=valid_mask,
            )
            causal_shift = corr_interv_shuffled - corr_interv_orig

            model_name = getattr(model, "model_type", model.__class__.__name__)

            return CausalInterventionResult(
                model_name=str(model_name),
                block_size=block_size,
                baseline_bit_map=bit_map_base,
                intervened_bit_map=bit_map_interv,
                unshuffled_bit_map=unshuffled_bit_map,
                original_saliency=ref_s_aligned,
                shuffled_saliency=shuffled_s,
                permutation=perm,
                corr_interv_vs_shuffled=corr_interv_shuffled,
                corr_interv_vs_original=corr_interv_orig,
                corr_unshuffled_vs_baseline=corr_unshuffled_base,
                causal_shift=causal_shift,
            )
    finally:
        model.train(was_training)


def run_causal_tile_swap_intervention(
    model: nn.Module,
    x: torch.Tensor,
    block_idx_1: int,
    block_idx_2: int,
    block_size: int,
    *,
    saliency_map: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> TileSwapResult:
    """Executes a targeted tile-swap micro-intervention between two blocks.

    Args:
        model: Model under evaluation.
        x: Input image tensor.
        block_idx_1: Linear index of first block.
        block_idx_2: Linear index of second block.
        block_size: Block dimension K.
        saliency_map: Optional ground-truth visual saliency map.
        valid_mask: Optional valid latent mask.

    Returns:
        TileSwapResult with baseline and swapped bit densities.
    """
    was_training = model.training
    model.eval()

    if saliency_map is not None:
        saliency_map = saliency_map.to(device=x.device)
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=x.device)

    try:
        with torch.no_grad():
            out_base = _invoke_model_forward(
                model,
                x,
                saliency_map=saliency_map,
                valid_mask=valid_mask,
                training=False,
            )
            bit_map_base = compute_spatial_bit_map(out_base, valid_mask=valid_mask)

            if saliency_map is not None:
                ref_s = saliency_map
            elif hasattr(out_base, "s_pred") and out_base.s_pred is not None:
                ref_s = out_base.s_pred
            else:
                raise ValueError("saliency_map must be provided when model does not predict s_pred.")

            latent_hw = bit_map_base.y_bits.shape[-2:]
            ref_s_aligned = transplant_saliency_map(ref_s, latent_hw)

            if ref_s_aligned.shape[0] == 1 and x.shape[0] > 1:
                ref_s_aligned = ref_s_aligned.expand(x.shape[0], -1, -1, -1)

            swapped_s, _ = swap_spatial_blocks(
                ref_s_aligned,
                block_idx_1=block_idx_1,
                block_idx_2=block_idx_2,
                block_size=block_size,
            )

            out_swapped = _invoke_model_forward(
                model,
                x,
                saliency_map=swapped_s,
                valid_mask=valid_mask,
                training=False,
            )
            bit_map_swapped = compute_spatial_bit_map(out_swapped, valid_mask=valid_mask)

            # Helper to extract mean bits for a specific block respecting valid mask
            def _get_block_bits(bit_tensor: torch.Tensor, b_idx: int) -> float:
                _h_lat, w_lat = bit_tensor.shape[-2:]
                n_w = w_lat // block_size
                b_r = b_idx // n_w
                b_c = b_idx % n_w
                r_start, r_end = b_r * block_size, (b_r + 1) * block_size
                c_start, c_end = b_c * block_size, (b_c + 1) * block_size
                tile = bit_tensor[..., r_start:r_end, c_start:c_end]
                if valid_mask is not None:
                    vm_tile = valid_mask[..., r_start:r_end, c_start:c_end]
                    if vm_tile.any():
                        return float(tile[vm_tile].mean().item())
                return float(tile.mean().item())

            base_b1 = _get_block_bits(bit_map_base.y_bits, block_idx_1)
            base_b2 = _get_block_bits(bit_map_base.y_bits, block_idx_2)
            swap_b1 = _get_block_bits(bit_map_swapped.y_bits, block_idx_1)
            swap_b2 = _get_block_bits(bit_map_swapped.y_bits, block_idx_2)

            base_diff = base_b1 - base_b2
            swap_diff = swap_b1 - swap_b2

            denom_base = base_b2 / max(1e-8, base_b1)
            denom_swap = swap_b2 / max(1e-8, swap_b1)
            redistribution_ratio = denom_swap / max(1e-8, denom_base)

            model_name = getattr(model, "model_type", model.__class__.__name__)

            return TileSwapResult(
                model_name=str(model_name),
                block_idx_1=block_idx_1,
                block_idx_2=block_idx_2,
                block_size=block_size,
                baseline_bits_tile_1=base_b1,
                baseline_bits_tile_2=base_b2,
                swapped_bits_tile_1=swap_b1,
                swapped_bits_tile_2=swap_b2,
                baseline_bit_diff=base_diff,
                swapped_bit_diff=swap_diff,
                bit_redistribution_ratio=redistribution_ratio,
            )
    finally:
        model.train(was_training)


def run_causal_transplant_intervention(
    model: nn.Module,
    target_image: torch.Tensor,
    donor_saliency: torch.Tensor,
    *,
    target_saliency: torch.Tensor | None = None,
    target_valid_mask: torch.Tensor | None = None,
    percentile: float = 0.20,
) -> CausalTransplantResult:
    """Executes cross-image saliency transplant intervention experiment.

    Protocol:
    1. Encodes target_image with its natural saliency to obtain baseline bit map.
    2. Overlays donor_saliency onto target_image.
    3. Re-encodes target_image with donor_saliency.
    4. Evaluates whether target_image's Allocation Ratio and bit correlation on donor ROI increase.

    Args:
        model: Compression neural network.
        target_image: Target image tensor (B, 3, H, W).
        donor_saliency: Saliency map from an independent donor image.
        target_saliency: Optional ground-truth saliency of target image.
        target_valid_mask: Optional valid latent mask of target image.
        percentile: Top-percentile defining ROI (default 0.20 = top-20%).

    Returns:
        CausalTransplantResult containing ROI allocation ratios, gains, and correlations.
    """
    was_training = model.training
    model.eval()

    donor_saliency = donor_saliency.to(device=target_image.device)
    if target_saliency is not None:
        target_saliency = target_saliency.to(device=target_image.device)
    if target_valid_mask is not None:
        target_valid_mask = target_valid_mask.to(device=target_image.device)

    try:
        with torch.no_grad():
            out_base = _invoke_model_forward(
                model,
                target_image,
                saliency_map=target_saliency,
                valid_mask=target_valid_mask,
                training=False,
            )
            bit_map_base = compute_spatial_bit_map(out_base, valid_mask=target_valid_mask)

            latent_hw = bit_map_base.y_bits.shape[-2:]
            s_donor_aligned = transplant_saliency_map(donor_saliency, latent_hw)

            # Match target image batch size
            if s_donor_aligned.shape[0] == 1 and target_image.shape[0] > 1:
                s_donor_aligned = s_donor_aligned.expand(target_image.shape[0], -1, -1, -1)

            # Extract target reference saliency and ROI masks
            if target_saliency is not None:
                s_target_ref = transplant_saliency_map(target_saliency, latent_hw)
            elif hasattr(out_base, "s_pred") and out_base.s_pred is not None:
                s_target_ref = out_base.s_pred
            else:
                s_target_ref = None

            pct = (1.0 - percentile) * 100.0 if 0.0 < percentile <= 1.0 else percentile

            if s_target_ref is not None:
                roi_target, _ = extract_top_percentile_mask(
                    s_target_ref,
                    percentile=pct,
                    valid_mask=target_valid_mask,
                )
            else:
                roi_target, _ = extract_top_percentile_mask(
                    bit_map_base.y_bits,
                    percentile=pct,
                    valid_mask=target_valid_mask,
                )

            roi_donor, _ = extract_top_percentile_mask(
                s_donor_aligned,
                percentile=pct,
                valid_mask=target_valid_mask,
            )

            # Baseline allocation ratio on donor ROI
            base_ar_on_donor = compute_allocation_ratio(
                bit_map_base.y_bits,
                roi_donor,
                valid_mask=target_valid_mask,
            )

            # Transplanted forward pass
            out_trans = _invoke_model_forward(
                model,
                target_image,
                saliency_map=s_donor_aligned,
                valid_mask=target_valid_mask,
                training=False,
            )
            bit_map_trans = compute_spatial_bit_map(out_trans, valid_mask=target_valid_mask)

            # Transplanted allocation ratio on donor ROI
            trans_ar_on_donor = compute_allocation_ratio(
                bit_map_trans.y_bits,
                roi_donor,
                valid_mask=target_valid_mask,
            )
            ar_gain = trans_ar_on_donor - base_ar_on_donor

            # Correlations
            corr_base_donor = compute_spatial_correlation(
                bit_map_base.y_bits,
                s_donor_aligned,
                valid_mask=target_valid_mask,
            )
            corr_trans_donor = compute_spatial_correlation(
                bit_map_trans.y_bits,
                s_donor_aligned,
                valid_mask=target_valid_mask,
            )

            model_name = getattr(model, "model_type", model.__class__.__name__)

            return CausalTransplantResult(
                model_name=str(model_name),
                target_baseline_bit_map=bit_map_base,
                transplanted_bit_map=bit_map_trans,
                donor_saliency=s_donor_aligned,
                target_saliency=s_target_ref,
                target_roi_mask=roi_target,
                donor_roi_mask=roi_donor,
                baseline_ar_on_donor_roi=base_ar_on_donor,
                transplanted_ar_on_donor_roi=trans_ar_on_donor,
                ar_gain_on_donor_roi=ar_gain,
                corr_baseline_vs_donor_saliency=corr_base_donor,
                corr_transplanted_vs_donor_saliency=corr_trans_donor,
            )
    finally:
        model.train(was_training)
