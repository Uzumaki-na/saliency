"""Training package providing loss functions, optimization harnesses, and training utilities.

Exports:
- auxiliary_saliency_loss: Masked binary cross-entropy loss over valid cells with strict FP32 precision.
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
- compute_correlation_coefficient: Pearson CC between predicted and target saliency.
- compute_hyperprior_overhead: Calculates hyperprior bitrate overhead Delta BPP_z between models.
- compute_kl_divergence: Spatial distribution KL divergence over valid mask.
- compute_auc_judd: AUC-Judd ROC evaluation.
- evaluate_representation_quality: Representation quality gate evaluation.
"""

from __future__ import annotations

from src.training.loss import (
    RateDistortionLoss,
    auxiliary_saliency_loss,
    compute_hyperprior_overhead,
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
    compute_auc_judd,
    compute_bpp,
    compute_correlation_coefficient,
    compute_kl_divergence,
    compute_mse,
    compute_msssim,
    compute_psnr,
    evaluate_representation_quality,
    load_checkpoint,
    save_checkpoint,
)

__all__ = [
    "CosineAnnealingWithWarmupLR",
    "DualOptimizerHarness",
    "MLflowTracker",
    "ParameterPartition",
    "RateDistortionLoss",
    "auxiliary_saliency_loss",
    "compute_auc_judd",
    "compute_bpp",
    "compute_correlation_coefficient",
    "compute_hyperprior_overhead",
    "compute_kl_divergence",
    "compute_mse",
    "compute_msssim",
    "compute_psnr",
    "create_dual_optimizer",
    "evaluate_representation_quality",
    "load_checkpoint",
    "partition_parameters",
    "saliency_weighted_distortion",
    "save_checkpoint",
]
