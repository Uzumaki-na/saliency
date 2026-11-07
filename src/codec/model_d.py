"""Model D (mechanism-only architecture control variant).

This module implements:
- ModelD: Subclasses ModelC with identical parameter dimensions, dual-head hyper-synthesis h_s,
  and explicit quantization step allocation delta(S_pred) with detached rate-distortion gradients,
  serving as the isolated mechanism-only control variant evaluated with uniform unweighted MSE distortion.
- compute_hyperprior_overhead: Calculates hyperprior bitrate overhead Delta BPP_z between models.
- load_pretrained_bmshj2018: Pretrained weight loading utility for ModelD instances.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from src.codec.model_a import ModelA
from src.codec.model_b import load_pretrained_bmshj2018 as _load_pretrained_bmshj2018
from src.codec.model_c import ModelC, ModelCOutput
from src.training.loss import compute_hyperprior_overhead

logger = logging.getLogger(__name__)


class ModelD(ModelC):
    r"""Model D mechanism-only variational image compression architecture.

    Model D is the isolated mechanism-only control variant for the causal experimental framework.
    It has the exact same network architecture, parameter dimensions (N=128, M=192), dual-head
    hyper-synthesis transform :math:`h_s`, and explicit quantization step allocation
    :math:`\Delta(S_{\text{pred}})` as Model C, along with detached rate-distortion gradients
    (:math:`\Delta = \operatorname{stop\_gradient}(\Delta(S_{\text{pred}}))`) and
    auxiliary gradient cutting before :math:`g_a` (``detach_y_for_ha=True``).

    During rate-distortion optimization, Model D is trained with uniform unweighted MSE
    distortion (``saliency_weighting=False`` in :class:`RateDistortionLoss`), while the auxiliary
    loss supervises :math:`S_{\text{pred}}` with constant weight :math:`\lambda_{\text{aux}} = 0.1`:

    .. math::
        \mathcal{L}_D = R_{\text{bpp}} + \lambda \cdot 255^2 \cdot \text{MSE}
                        + \lambda_{\text{aux}} \mathcal{L}_{\text{aux}}
    """

    model_type: str = "D"

    def __init__(
        self,
        in_channels: int = 3,
        n_channels: int = 128,
        m_channels: int = 192,
        detach_y_for_ha: bool = True,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            n_channels=n_channels,
            m_channels=m_channels,
            detach_y_for_ha=detach_y_for_ha,
        )


def load_pretrained_bmshj2018(
    checkpoint: str | Path | Mapping[str, Any] | ModelA,
    target_or_checkpoint: str | Path | Mapping[str, Any] | ModelA | None = None,
    *,
    model: ModelA | None = None,
    strict: bool = True,
    device: str | torch.device | None = None,
    **model_kwargs: Any,
) -> ModelD | ModelA:
    r"""Loads pretrained bmshj2018 weights into a ModelD (or target model) instance.

    If neither ``model`` nor an existing model instance is provided in positional arguments,
    a new :class:`ModelD` instance is created and populated with weights.
    """
    if model is None and not isinstance(checkpoint, ModelA) and not isinstance(target_or_checkpoint, ModelA):
        model = ModelD(**model_kwargs)
    return _load_pretrained_bmshj2018(
        checkpoint,
        target_or_checkpoint,
        model=model,
        strict=strict,
        device=device,
        **model_kwargs,
    )


__all__ = [
    "ModelCOutput",
    "ModelD",
    "compute_hyperprior_overhead",
    "load_pretrained_bmshj2018",
]
