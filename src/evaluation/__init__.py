"""Evaluation, causal intervention, and benchmark metrics package."""

from __future__ import annotations

from src.evaluation.spatial_allocation import (
    SpatialBitMap,
    compute_allocation_ratio,
    compute_relative_allocation_ratio,
    compute_spatial_bit_map,
    evaluate_rar_across_models,
    extract_top_percentile_mask,
)

__all__ = [
    "SpatialBitMap",
    "compute_allocation_ratio",
    "compute_relative_allocation_ratio",
    "compute_spatial_bit_map",
    "evaluate_rar_across_models",
    "extract_top_percentile_mask",
]
