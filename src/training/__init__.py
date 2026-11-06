"""Training package providing loss functions, optimization harnesses, and training utilities.

Exports:
- saliency_weighted_distortion: Functional implementation of saliency-weighted reconstruction distortion.
- RateDistortionLoss: CompressAI-compatible rate-distortion loss module with saliency weighting.
- ParameterPartition: NamedTuple container holding partitioned model parameters.
- partition_parameters: Isolates auxiliary entropy model parameters from main transform parameters.
- CosineAnnealingWithWarmupLR: Cosine learning rate scheduler with linear warmup support.
- DualOptimizerHarness: Dual optimizer coordination harness with gradient clipping and scheduling.
- create_dual_optimizer: Factory for configuring and creating DualOptimizerHarness.
- MLflowTracker: MLflow experiment tracking and run lifecycle manager.
- save_checkpoint: Checkpoint persistence enforcing post-training CDF table compilation gate.
- load_checkpoint: Restores model state and DualOptimizerHarness from checkpoint.
- compute_psnr: Pure PyTorch PSNR evaluation in dB.
- compute_msssim: Wang et al. 5-scale MS-SSIM evaluation in pure PyTorch.
- compute_bpp: Bits-per-pixel evaluation from likelihoods or byte counts.
- compute_mse: Mean squared error evaluation.
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
from src.training.tracker import (
    MLflowTracker,
    compute_bpp,
    compute_mse,
    compute_msssim,
    compute_psnr,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "CosineAnnealingWithWarmupLR",
    "DualOptimizerHarness",
    "MLflowTracker",
    "ParameterPartition",
    "RateDistortionLoss",
    "compute_bpp",
    "compute_mse",
    "compute_msssim",
    "compute_psnr",
    "create_dual_optimizer",
    "load_checkpoint",
    "partition_parameters",
    "saliency_weighted_distortion",
    "save_checkpoint",
]
