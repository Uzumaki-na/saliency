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
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy import stats
except ImportError:
    stats = None

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


@dataclass(frozen=True)
class CausalRegressionResult:
    """Results of image-fixed-effects spatial bit regression with clustered standard errors.

    Model:
        (bits_target - bits_baseline)_kij = alpha_k + beta_S * S_kij + beta_base * bits_baseline_kij + eps_kij

    Attributes:
        beta_s: Estimated regression coefficient for visual saliency map S.
        beta_base: Estimated regression coefficient for baseline bits.
        se_s: Cluster-robust standard error for beta_s (clustered by image).
        se_base: Cluster-robust standard error for beta_base.
        t_s: Student t-statistic for beta_s.
        t_base: Student t-statistic for beta_base.
        p_s: Two-tailed p-value for beta_s (testing H0: beta_s = 0).
        p_base: Two-tailed p-value for beta_base (testing H0: beta_base = 0).
        alpha_images: Estimated image-specific fixed effects alpha_k.
        r2_within: Within-image coefficient of determination (R-squared).
        r2_overall: Overall coefficient of determination across all valid cells.
        num_images: Number of image clusters K.
        num_cells_total: Total valid unpadded spatial cells evaluated N.
        degrees_of_freedom: Residual degrees of freedom (N - K - 2).
        hypothesis_confirmed: True if beta_s > 0.0 and p_s < 0.001.
    """

    beta_s: float
    beta_base: float
    se_s: float
    se_base: float
    t_s: float
    t_base: float
    p_s: float
    p_base: float
    alpha_images: list[float]
    r2_within: float
    r2_overall: float
    num_images: int
    num_cells_total: int
    degrees_of_freedom: int
    hypothesis_confirmed: bool


@dataclass(frozen=True)
class PartialCorrelationResult:
    """Results of partial correlation analysis controlling for baseline bits.

    Computes:
        PartialCorr(S_pred, S | bits_A) = (r_XY - r_XZ * r_YZ) / sqrt((1 - r_XZ^2) * (1 - r_YZ^2))

    Attributes:
        partial_corr: First-order partial correlation in [-1.0, 1.0].
        raw_corr_xy: Marginal Pearson correlation between X (S_pred) and Y (S).
        raw_corr_xz: Marginal Pearson correlation between X (S_pred) and Z (bits_A).
        raw_corr_yz: Marginal Pearson correlation between Y (S) and Z (bits_A).
        num_cells: Number of valid unpadded spatial cells evaluated.
    """

    partial_corr: float
    raw_corr_xy: float
    raw_corr_xz: float
    raw_corr_yz: float
    num_cells: int


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


def compute_partial_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> PartialCorrelationResult:
    """Computes first-order partial correlation PartialCorr(X, Y | Z) over valid cells.

    Formula:
        r_XY.Z = (r_XY - r_XZ * r_YZ) / sqrt((1 - r_XZ^2) * (1 - r_YZ^2))

    Args:
        x: First variable tensor (e.g., S_pred).
        y: Second variable tensor (e.g., S).
        z: Conditioning variable tensor (e.g., bits_A).
        valid_mask: Optional boolean valid mask M_valid.

    Returns:
        PartialCorrelationResult containing partial and marginal correlations.
    """
    x_f = x.float().cpu()
    y_f = y.float().cpu()
    z_f = z.float().cpu()

    # If inputs are already 1D vectors (e.g. from flattened cells across dataset)
    if x_f.ndim == 1 and y_f.ndim == 1 and z_f.ndim == 1:
        if not (x_f.numel() == y_f.numel() == z_f.numel()):
            raise ValueError(f"1D vector length mismatch: x({x_f.numel()}), y({y_f.numel()}), z({z_f.numel()})")
        if valid_mask is not None:
            mask_flat = valid_mask.cpu().bool().flatten()
            if mask_flat.numel() == x_f.numel():
                vx = x_f[mask_flat]
                vy = y_f[mask_flat]
                vz = z_f[mask_flat]
            else:
                vx = x_f
                vy = y_f
                vz = z_f
        else:
            vx = x_f
            vy = y_f
            vz = z_f
    else:
        # Reduce multi-channel inputs (C > 1) to single channel
        if x_f.ndim == 4 and x_f.shape[1] > 1:
            x_f = x_f.mean(dim=1, keepdim=True)
        elif x_f.ndim == 3 and x_f.shape[0] > 1:
            x_f = x_f.mean(dim=0, keepdim=True).unsqueeze(0)
        elif x_f.ndim == 2:
            x_f = x_f.unsqueeze(0).unsqueeze(0)

        if y_f.ndim == 4 and y_f.shape[1] > 1:
            y_f = y_f.mean(dim=1, keepdim=True)
        elif y_f.ndim == 3 and y_f.shape[0] > 1:
            y_f = y_f.mean(dim=0, keepdim=True).unsqueeze(0)
        elif y_f.ndim == 2:
            y_f = y_f.unsqueeze(0).unsqueeze(0)

        if z_f.ndim == 4 and z_f.shape[1] > 1:
            z_f = z_f.mean(dim=1, keepdim=True)
        elif z_f.ndim == 3 and z_f.shape[0] > 1:
            z_f = z_f.mean(dim=0, keepdim=True).unsqueeze(0)
        elif z_f.ndim == 2:
            z_f = z_f.unsqueeze(0).unsqueeze(0)

        # Align spatial dimensions to x if needed
        target_hw = x_f.shape[-2:]
        if y_f.shape[-2:] != target_hw:
            y_f = F.interpolate(y_f, size=target_hw, mode="area")
        if z_f.shape[-2:] != target_hw:
            z_f = F.interpolate(z_f, size=target_hw, mode="area")

        vm = valid_mask
        if vm is not None:
            vm = vm.cpu()
            if vm.ndim == 2:
                vm = vm.unsqueeze(0).unsqueeze(0)
            elif vm.ndim == 3:
                vm = vm.unsqueeze(1) if vm.shape[0] != 1 else vm.unsqueeze(0)
            if vm.shape[-2:] != target_hw:
                vm = F.interpolate(vm.float(), size=target_hw, mode="nearest").bool()
            if vm.shape[0] == 1 and x_f.shape[0] > 1:
                vm = vm.expand(x_f.shape[0], -1, -1, -1)

            mask_bool = vm.bool()
            vx = x_f[mask_bool].flatten()
            vy = y_f[mask_bool].flatten()
            vz = z_f[mask_bool].flatten()
        else:
            vx = x_f.flatten()
            vy = y_f.flatten()
            vz = z_f.flatten()

    n_cells = vx.numel()
    if n_cells < 3:
        return PartialCorrelationResult(
            partial_corr=0.0,
            raw_corr_xy=0.0,
            raw_corr_xz=0.0,
            raw_corr_yz=0.0,
            num_cells=n_cells,
        )

    def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
        da = a - a.mean()
        db = b - b.mean()
        va = float((da**2).sum().item())
        vb = float((db**2).sum().item())
        if va < 1e-12 or vb < 1e-12:
            return 0.0
        r = float((da * db).sum().item()) / ((va * vb) ** 0.5)
        return max(-1.0, min(1.0, r))

    r_xy = _pearson(vx, vy)
    r_xz = _pearson(vx, vz)
    r_yz = _pearson(vy, vz)

    # Compute partial correlation
    denom_sq = (1.0 - r_xz**2) * (1.0 - r_yz**2)
    if denom_sq <= 1e-12:
        partial_r = 0.0
    else:
        denom = denom_sq**0.5
        numer = r_xy - r_xz * r_yz
        partial_r = max(-1.0, min(1.0, numer / denom))

    return PartialCorrelationResult(
        partial_corr=partial_r,
        raw_corr_xy=r_xy,
        raw_corr_xz=r_xz,
        raw_corr_yz=r_yz,
        num_cells=n_cells,
    )


def fit_image_fixed_effects_regression(
    y_list: list[torch.Tensor],
    x_mat_list: list[torch.Tensor],
) -> CausalRegressionResult:
    """Estimates panel regression with absorbed image fixed effects and clustered covariance.

    Model:
        y_kij = alpha_k + X_kij^T beta + eps_kij

    Estimates beta using Frisch-Waugh-Lovell within-transformation (demeaning per image k),
    absorbs alpha_k, and calculates cluster-robust covariance (Liang-Zeger sandwich) clustered
    by image. Falls back to cell-level HC1 robust covariance when K=1.

    Args:
        y_list: List of 1D tensors of dependent variable (e.g. bits_target - bits_baseline), one per image.
        x_mat_list: List of 2D tensors of regressors (N_k, P), one per image (e.g. columns [S, bits_baseline]).

    Returns:
        CausalRegressionResult containing beta estimates, clustered SEs, t-stats, p-values, and R^2.
    """
    if len(y_list) != len(x_mat_list):
        raise ValueError(f"Mismatch: len(y_list)={len(y_list)} vs len(x_mat_list)={len(x_mat_list)}")
    if len(y_list) == 0:
        raise ValueError("Cannot fit regression on empty list of images.")

    # Filter out empty images
    valid_pairs = []
    for y_k, x_k in zip(y_list, x_mat_list, strict=True):
        y_t = y_k.cpu().to(dtype=torch.float64).flatten()
        x_t = x_k.cpu().to(dtype=torch.float64)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(1)
        if y_t.numel() > 0 and x_t.shape[0] == y_t.numel():
            valid_pairs.append((y_t, x_t))

    if len(valid_pairs) == 0:
        raise ValueError("All provided images have 0 valid spatial cells.")

    num_images = len(valid_pairs)
    p_vars = valid_pairs[0][1].shape[1]
    num_cells_total = sum(y_t.numel() for y_t, _ in valid_pairs)

    # Degrees of freedom
    if num_images > 1:
        df_cluster = num_images - 1
        df_resid = max(1, num_cells_total - num_images - p_vars)
        c_adj = (num_images / (num_images - 1)) * (max(1, num_cells_total - 1) / df_resid)
    else:
        df_cluster = max(1, num_cells_total - 1 - p_vars)
        df_resid = df_cluster
        c_adj = num_cells_total / df_resid

    # Step 1: Compute image-level means and demean (Frisch-Waugh-Lovell within transformation)
    demeaned_data = []
    a_mat = torch.zeros((p_vars, p_vars), dtype=torch.float64)
    b_vec = torch.zeros(p_vars, dtype=torch.float64)
    col_within_var = torch.zeros(p_vars, dtype=torch.float64)

    total_y_sum = 0.0

    for y_t, x_t in valid_pairs:
        total_y_sum += float(y_t.sum().item())
        y_mean = y_t.mean()
        x_mean = x_t.mean(dim=0)

        tilde_y = y_t - y_mean
        tilde_x = x_t - x_mean

        a_mat += tilde_x.T @ tilde_x
        b_vec += tilde_x.T @ tilde_y
        col_within_var += torch.sum(tilde_x**2, dim=0)

        demeaned_data.append((y_t, x_t, y_mean, x_mean, tilde_y, tilde_x))

    # Step 2: Solve within OLS normal equations
    # Use pseudo-inverse for numerical robustness against near-collinear regressors
    a_inv = torch.linalg.pinv(a_mat)
    beta_hat = a_inv @ b_vec

    # Step 3: Recover image fixed effects alpha_k, calculate residuals and covariance meat B
    alpha_images: list[float] = []
    rss = 0.0
    tss_within = 0.0
    tss_overall = 0.0

    y_global_mean = total_y_sum / max(1, num_cells_total)

    if num_images == 1:
        # Cell-level HC1 robust covariance fallback when only 1 image cluster exists
        y_t, _x_t, y_mean, x_mean, tilde_y, tilde_x = demeaned_data[0]
        alpha_k = float((y_mean - x_mean @ beta_hat).item())
        alpha_images.append(alpha_k)

        eps_1 = tilde_y - tilde_x @ beta_hat
        rss = float((eps_1**2).sum().item())
        tss_within = float((tilde_y**2).sum().item())
        tss_overall = float(((y_t - y_global_mean) ** 2).sum().item())

        weighted_x = tilde_x * eps_1.unsqueeze(1)
        b_meat = weighted_x.T @ weighted_x
    else:
        # Multi-cluster Liang-Zeger sandwich
        b_meat = torch.zeros((p_vars, p_vars), dtype=torch.float64)
        for y_t, _x_t, y_mean, x_mean, tilde_y, tilde_x in demeaned_data:
            alpha_k = float((y_mean - x_mean @ beta_hat).item())
            alpha_images.append(alpha_k)

            # Residuals
            eps_k = tilde_y - tilde_x @ beta_hat
            rss += float((eps_k**2).sum().item())
            tss_within += float((tilde_y**2).sum().item())
            tss_overall += float(((y_t - y_global_mean) ** 2).sum().item())

            # Cluster score vector s_k = tilde_x^T eps_k
            score_k = tilde_x.T @ eps_k
            b_meat += torch.outer(score_k, score_k)

    # Step 4: Cluster-robust variance-covariance matrix (Liang-Zeger sandwich)
    v_cluster = c_adj * (a_inv @ b_meat @ a_inv)
    diag_v = torch.diagonal(v_cluster)
    se_vec = torch.sqrt(torch.clamp(diag_v, min=0.0))

    t_stats = torch.zeros(p_vars, dtype=torch.float64)
    p_values: list[float] = []

    for j in range(p_vars):
        if float(col_within_var[j].item()) < 1e-12:
            # Unidentifiable regressor (no within-image variation)
            se_vec[j] = float("nan")
            t_stats[j] = 0.0
            p_values.append(1.0)
        else:
            se_j = float(se_vec[j].item())
            if se_j > 1e-12:
                t_val = float((beta_hat[j] / se_j).item())
                t_stats[j] = t_val
                if stats is not None:
                    pval = 2.0 * float(stats.t.sf(abs(t_val), df=df_cluster))
                else:
                    pval = math.erfc(abs(t_val) / math.sqrt(2.0))
                p_values.append(max(0.0, min(1.0, pval)))
            else:
                se_vec[j] = float("nan")
                t_stats[j] = 0.0
                p_values.append(1.0)

    # Step 5: Compute R-squared
    r2_within = max(0.0, min(1.0, 1.0 - rss / max(1e-12, tss_within)))
    r2_overall = max(0.0, min(1.0, 1.0 - rss / max(1e-12, tss_overall)))

    beta_s = float(beta_hat[0].item())
    beta_base = float(beta_hat[1].item()) if p_vars > 1 else 0.0
    se_s = float(se_vec[0].item())
    se_base = float(se_vec[1].item()) if p_vars > 1 else 0.0
    t_s = float(t_stats[0].item())
    t_base = float(t_stats[1].item()) if p_vars > 1 else 0.0
    p_s = p_values[0]
    p_base = p_values[1] if p_vars > 1 else 1.0

    hypothesis_confirmed = bool(not math.isnan(se_s) and beta_s > 0.0 and p_s < 0.001)

    return CausalRegressionResult(
        beta_s=beta_s,
        beta_base=beta_base,
        se_s=se_s,
        se_base=se_base,
        t_s=t_s,
        t_base=t_base,
        p_s=p_s,
        p_base=p_base,
        alpha_images=alpha_images,
        r2_within=r2_within,
        r2_overall=r2_overall,
        num_images=num_images,
        num_cells_total=num_cells_total,
        degrees_of_freedom=df_resid,
        hypothesis_confirmed=hypothesis_confirmed,
    )


def run_causal_regression(
    bits_target: torch.Tensor | list[torch.Tensor] | SpatialBitMap,
    bits_baseline: torch.Tensor | list[torch.Tensor] | SpatialBitMap,
    saliency: torch.Tensor | list[torch.Tensor],
    *,
    valid_mask: torch.Tensor | list[torch.Tensor] | None = None,
) -> CausalRegressionResult:
    """Executes image-fixed-effects spatial bit regression: (bits_T - bits_B) = alpha + beta_S * S + beta_B * bits_B.

    Args:
        bits_target: Spatial bits tensor for target model (e.g. Model C or D) or SpatialBitMap.
        bits_baseline: Spatial bits tensor for baseline model (Model A) or SpatialBitMap.
        saliency: Saliency map tensor(s) in [0, 1].
        valid_mask: Optional valid latent mask(s).

    Returns:
        CausalRegressionResult containing regression coefficients, clustered SEs, and hypothesis status.
    """
    resolved_mask = valid_mask
    if isinstance(bits_target, SpatialBitMap):
        if resolved_mask is None and bits_target.valid_mask is not None:
            resolved_mask = bits_target.valid_mask
        target_t = bits_target.y_bits
    else:
        target_t = bits_target

    if isinstance(bits_baseline, SpatialBitMap):
        if resolved_mask is None and bits_baseline.valid_mask is not None:
            resolved_mask = bits_baseline.valid_mask
        baseline_t = bits_baseline.y_bits
    else:
        baseline_t = bits_baseline

    def _to_2d_list(item: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(item, list):
            res = []
            for sub in item:
                res.extend(_to_2d_list(sub))
            return res
        if item.ndim == 2:
            return [item]
        if item.ndim == 3:
            return [item[i] for i in range(item.shape[0])]
        if item.ndim == 4:
            return [item[i, 0] if item.shape[1] == 1 else item[i].mean(dim=0) for i in range(item.shape[0])]
        raise ValueError(f"Unsupported tensor dimension {item.ndim}D")

    t_list = _to_2d_list(target_t)
    b_list = _to_2d_list(baseline_t)
    s_list = _to_2d_list(saliency)
    vm_list = _to_2d_list(resolved_mask) if resolved_mask is not None else None

    num_images = len(t_list)
    if len(b_list) == 1 and num_images > 1:
        b_list = [b_list[0] for _ in range(num_images)]
    if len(s_list) == 1 and num_images > 1:
        s_list = [s_list[0] for _ in range(num_images)]
    if vm_list is not None and len(vm_list) == 1 and num_images > 1:
        vm_list = [vm_list[0] for _ in range(num_images)]

    if not (len(t_list) == len(b_list) == len(s_list)):
        raise ValueError(
            f"Image count mismatch: target ({len(t_list)}), baseline ({len(b_list)}), saliency ({len(s_list)})"
        )

    y_list: list[torch.Tensor] = []
    x_mat_list: list[torch.Tensor] = []

    for k in range(num_images):
        t_k = t_list[k].float().cpu()
        b_k = b_list[k].float().cpu()
        s_k = s_list[k].float().cpu()

        target_hw = t_k.shape[-2:]
        if b_k.shape[-2:] != target_hw:
            b_k = F.interpolate(b_k.view(1, 1, *b_k.shape[-2:]), size=target_hw, mode="area").view(target_hw)
        if s_k.shape[-2:] != target_hw:
            s_k = F.interpolate(s_k.view(1, 1, *s_k.shape[-2:]), size=target_hw, mode="area").view(target_hw)

        if vm_list is not None:
            vm_k = vm_list[k].cpu()
            if vm_k.shape[-2:] != target_hw:
                vm_k = (
                    F.interpolate(vm_k.float().view(1, 1, *vm_k.shape[-2:]), size=target_hw, mode="nearest")
                    .bool()
                    .view(target_hw)
                )
            mask = vm_k.bool()
            y_k = (t_k - b_k)[mask].flatten()
            s_vals = s_k[mask].flatten()
            b_vals = b_k[mask].flatten()
        else:
            y_k = (t_k - b_k).flatten()
            s_vals = s_k.flatten()
            b_vals = b_k.flatten()

        if y_k.numel() > 0:
            x_k = torch.stack([s_vals, b_vals], dim=1)
            y_list.append(y_k)
            x_mat_list.append(x_k)

    return fit_image_fixed_effects_regression(y_list, x_mat_list)


def run_causal_partial_correlation(
    s_pred: torch.Tensor | list[torch.Tensor],
    s_ref: torch.Tensor | list[torch.Tensor],
    bits_baseline: torch.Tensor | list[torch.Tensor] | SpatialBitMap,
    *,
    valid_mask: torch.Tensor | list[torch.Tensor] | None = None,
    demean_per_image: bool = True,
) -> PartialCorrelationResult:
    """Executes partial correlation analysis: PartialCorr(S_pred, S_ref | bits_baseline).

    Controls for baseline bit density to verify that saliency representation fidelity is not
    confounded by texture complexity.

    Args:
        s_pred: Predicted visual saliency map(s).
        s_ref: Reference ground-truth visual saliency map(s).
        bits_baseline: Baseline model spatial bits or SpatialBitMap.
        valid_mask: Optional valid latent mask.
        demean_per_image: If True, centers variables per-image before concatenation to remove
            cross-image mean offsets.

    Returns:
        PartialCorrelationResult containing partial and marginal correlation coefficients.
    """
    resolved_mask = valid_mask
    if isinstance(bits_baseline, SpatialBitMap):
        if resolved_mask is None and bits_baseline.valid_mask is not None:
            resolved_mask = bits_baseline.valid_mask
        base_t = bits_baseline.y_bits
    else:
        base_t = bits_baseline

    def _to_2d_list(item: torch.Tensor | list[torch.Tensor]) -> list[torch.Tensor]:
        if isinstance(item, list):
            res = []
            for sub in item:
                res.extend(_to_2d_list(sub))
            return res
        if item.ndim == 2:
            return [item]
        if item.ndim == 3:
            return [item[i] for i in range(item.shape[0])]
        if item.ndim == 4:
            return [item[i, 0] if item.shape[1] == 1 else item[i].mean(dim=0) for i in range(item.shape[0])]
        raise ValueError(f"Unsupported tensor dimension {item.ndim}D")

    sp_list = _to_2d_list(s_pred)
    sr_list = _to_2d_list(s_ref)
    bb_list = _to_2d_list(base_t)
    vm_list = _to_2d_list(resolved_mask) if resolved_mask is not None else None

    num_images = len(sp_list)
    if len(sr_list) == 1 and num_images > 1:
        sr_list = [sr_list[0] for _ in range(num_images)]
    if len(bb_list) == 1 and num_images > 1:
        bb_list = [bb_list[0] for _ in range(num_images)]
    if vm_list is not None and len(vm_list) == 1 and num_images > 1:
        vm_list = [vm_list[0] for _ in range(num_images)]

    all_x: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    all_z: list[torch.Tensor] = []

    for k in range(num_images):
        sp_k = sp_list[k].float().cpu()
        sr_k = sr_list[k].float().cpu()
        bb_k = bb_list[k].float().cpu()

        target_hw = sp_k.shape[-2:]
        if sr_k.shape[-2:] != target_hw:
            sr_k = F.interpolate(sr_k.view(1, 1, *sr_k.shape[-2:]), size=target_hw, mode="area").view(target_hw)
        if bb_k.shape[-2:] != target_hw:
            bb_k = F.interpolate(bb_k.view(1, 1, *bb_k.shape[-2:]), size=target_hw, mode="area").view(target_hw)

        if vm_list is not None:
            vm_k = vm_list[k].cpu()
            if vm_k.shape[-2:] != target_hw:
                vm_k = (
                    F.interpolate(vm_k.float().view(1, 1, *vm_k.shape[-2:]), size=target_hw, mode="nearest")
                    .bool()
                    .view(target_hw)
                )
            mask = vm_k.bool()
            x_vals = sp_k[mask].flatten()
            y_vals = sr_k[mask].flatten()
            z_vals = bb_k[mask].flatten()
        else:
            x_vals = sp_k.flatten()
            y_vals = sr_k.flatten()
            z_vals = bb_k.flatten()

        if x_vals.numel() > 0:
            if demean_per_image and x_vals.numel() > 1:
                all_x.append(x_vals - x_vals.mean())
                all_y.append(y_vals - y_vals.mean())
                all_z.append(z_vals - z_vals.mean())
            else:
                all_x.append(x_vals)
                all_y.append(y_vals)
                all_z.append(z_vals)

    if len(all_x) == 0:
        return PartialCorrelationResult(
            partial_corr=0.0,
            raw_corr_xy=0.0,
            raw_corr_xz=0.0,
            raw_corr_yz=0.0,
            num_cells=0,
        )

    cat_x = torch.cat(all_x)
    cat_y = torch.cat(all_y)
    cat_z = torch.cat(all_z)

    return compute_partial_correlation(cat_x, cat_y, cat_z)
