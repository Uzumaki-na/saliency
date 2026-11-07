"""Evaluation, causal intervention, and benchmark metrics package."""

from __future__ import annotations

from src.evaluation.causal import (
    BlockShuffleResult,
    CausalInterventionResult,
    CausalTransplantResult,
    TileSwapResult,
    compute_spatial_correlation,
    run_causal_block_intervention,
    run_causal_tile_swap_intervention,
    run_causal_transplant_intervention,
    shuffle_spatial_blocks,
    shuffle_spatial_blocks_detailed,
    swap_spatial_blocks,
    transplant_saliency_map,
    unshuffle_spatial_blocks,
)
from src.evaluation.spatial_allocation import (
    SpatialBitMap,
    compute_allocation_ratio,
    compute_relative_allocation_ratio,
    compute_spatial_bit_map,
    evaluate_rar_across_models,
    extract_top_percentile_mask,
)

__all__ = [
    "BlockShuffleResult",
    "CausalInterventionResult",
    "CausalTransplantResult",
    "SpatialBitMap",
    "TileSwapResult",
    "compute_allocation_ratio",
    "compute_relative_allocation_ratio",
    "compute_spatial_bit_map",
    "compute_spatial_correlation",
    "evaluate_rar_across_models",
    "extract_top_percentile_mask",
    "run_causal_block_intervention",
    "run_causal_tile_swap_intervention",
    "run_causal_transplant_intervention",
    "shuffle_spatial_blocks",
    "shuffle_spatial_blocks_detailed",
    "swap_spatial_blocks",
    "transplant_saliency_map",
    "unshuffle_spatial_blocks",
]
