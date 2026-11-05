"""Training package providing loss functions and training utilities.

Exports:
- saliency_weighted_distortion: Functional implementation of saliency-weighted reconstruction distortion.
- RateDistortionLoss: CompressAI-compatible rate-distortion loss module with saliency weighting.
"""

from __future__ import annotations

from src.training.loss import (
    RateDistortionLoss,
    saliency_weighted_distortion,
)

__all__ = [
    "RateDistortionLoss",
    "saliency_weighted_distortion",
]
