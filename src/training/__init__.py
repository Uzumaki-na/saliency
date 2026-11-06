"""Training package providing loss functions, optimization harnesses, and training utilities.

Exports:
- saliency_weighted_distortion: Functional implementation of saliency-weighted reconstruction distortion.
- RateDistortionLoss: CompressAI-compatible rate-distortion loss module with saliency weighting.
- ParameterPartition: NamedTuple container holding partitioned model parameters.
- partition_parameters: Isolates auxiliary entropy model parameters from main transform parameters.
- CosineAnnealingWithWarmupLR: Cosine learning rate scheduler with linear warmup support.
- DualOptimizerHarness: Dual optimizer coordination harness with gradient clipping and scheduling.
- create_dual_optimizer: Factory for configuring and creating DualOptimizerHarness.
"""

from __future__ import annotations

from src.training.loss import (
    RateDistortionLoss,
    saliency_weighted_distortion,
)
from src.training.optimizer import (
    CosineAnnealingWithWarmupLR,
    DualOptimizerHarness,
    ParameterPartition,
    create_dual_optimizer,
    partition_parameters,
)

__all__ = [
    "CosineAnnealingWithWarmupLR",
    "DualOptimizerHarness",
    "ParameterPartition",
    "RateDistortionLoss",
    "create_dual_optimizer",
    "partition_parameters",
    "saliency_weighted_distortion",
]
