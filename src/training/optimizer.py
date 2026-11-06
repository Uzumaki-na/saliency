"""Dual optimizer and learning rate scheduling harness for variational learned image compression.

Decouples optimization of primary transform parameters (rate-distortion objective with gradient clipping)
from non-parametric entropy bottleneck CDF table parameters (auxiliary loss without clipping).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, NamedTuple

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.codec.entropy_models import EntropyBottleneck


class ParameterPartition(NamedTuple):
    """Container holding partitioned model parameters."""

    main_params: list[nn.Parameter]
    aux_params: list[nn.Parameter]


def partition_parameters(
    model: nn.Module,
    aux_param_names: tuple[str, ...] = ("entropy_bottleneck", "quantiles"),
) -> ParameterPartition:
    """Partitions model parameters into main transform parameters and auxiliary entropy bottleneck parameters.

    Excludes parameters with ``requires_grad=False``.
    Ensures disjointness (zero intersection) and completeness (100% coverage of trainable parameters).

    Args:
        model: Target neural network module.
        aux_param_names: Submodule or parameter name tokens identified as auxiliary entropy parameters.

    Returns:
        ParameterPartition containing `main_params` and `aux_params`.
    """
    aux_param_ids: set[int] = set()

    # Identify auxiliary parameters by submodule type (EntropyBottleneck)
    for _, module in model.named_modules():
        if isinstance(module, EntropyBottleneck) or module.__class__.__name__ == "EntropyBottleneck":
            for param in module.parameters():
                if param.requires_grad:
                    aux_param_ids.add(id(param))

    # Identify auxiliary parameters by parameter naming convention
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        parts = name.split(".")
        if any(aux_token in parts or name.endswith(f".{aux_token}") for aux_token in aux_param_names):
            aux_param_ids.add(id(param))

    main_params: list[nn.Parameter] = []
    aux_params: list[nn.Parameter] = []
    seen_ids: set[int] = set()

    for param in model.parameters():
        if not param.requires_grad or id(param) in seen_ids:
            continue
        seen_ids.add(id(param))

        if id(param) in aux_param_ids:
            aux_params.append(param)
        else:
            main_params.append(param)

    # Invariant: Disjoint and exhaustive coverage of trainable parameters
    assert set(map(id, main_params)).isdisjoint(set(map(id, aux_params))), (
        "Internal invariant violated: overlap between main and aux parameters"
    )

    return ParameterPartition(main_params=main_params, aux_params=aux_params)


class CosineAnnealingWithWarmupLR(LRScheduler):
    """Cosine annealing learning rate schedule with optional linear warmup.

    Smoothly ramps learning rate linearly from `warmup_start_lr` to `base_lr` over `warmup_epochs`,
    then follows a cosine decay curve from `base_lr` to `eta_min` up to `T_max`.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 1e-5,
        warmup_epochs: int = 0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
        min_lr: float | None = None,
        **kwargs: Any,
    ) -> None:
        if T_max <= 0:
            raise ValueError(f"T_max must be a positive integer, got {T_max}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be non-negative, got {warmup_epochs}")
        if warmup_epochs >= T_max:
            raise ValueError(f"warmup_epochs ({warmup_epochs}) must be strictly less than T_max ({T_max})")

        resolved_eta_min = min_lr if min_lr is not None else eta_min
        if resolved_eta_min < 0:
            raise ValueError(f"eta_min must be non-negative, got {resolved_eta_min}")

        self.T_max = int(T_max)
        self.eta_min = float(resolved_eta_min)
        self.warmup_epochs = int(warmup_epochs)
        self.warmup_start_lr = float(warmup_start_lr)

        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        """Calculates scheduled learning rate for each parameter group."""
        if self.warmup_epochs > 0 and self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / float(self.warmup_epochs)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]

        if self.last_epoch >= self.T_max:
            return [self.eta_min for _ in self.base_lrs]

        decay_steps = self.T_max - self.warmup_epochs
        curr_step = self.last_epoch - self.warmup_epochs
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * curr_step / float(decay_steps)))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine_factor
            for base_lr in self.base_lrs
        ]

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores scheduler state and synchronizes optimizer parameter group learning rates."""
        super().load_state_dict(state_dict)
        if self.optimizer is not None:
            if "_last_lr" in state_dict:
                for group, lr in zip(self.optimizer.param_groups, state_dict["_last_lr"]):
                    group["lr"] = lr
            else:
                for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                    group["lr"] = lr


class DualOptimizerHarness:
    """Coordinates dual Adam optimizers, schedulers, and gradient clipping for variational compression.

    Decouples:
    - `main_optimizer`: Updates convolutional kernels, biases, and GDN parameters with gradient clipping.
    - `aux_optimizer`: Updates cumulative probability tables in entropy bottlenecks unclipped.
    """

    def __init__(
        self,
        main_optimizer: Optimizer | None = None,
        aux_optimizer: Optimizer | None = None,
        main_scheduler: LRScheduler | None = None,
        aux_scheduler: LRScheduler | None = None,
        clip_max_norm: float | None = 1.0,
        main_params: Iterable[nn.Parameter] | None = None,
        aux_params: Iterable[nn.Parameter] | None = None,
    ) -> None:
        if clip_max_norm is not None and clip_max_norm < 0:
            raise ValueError(f"clip_max_norm must be non-negative or None, got {clip_max_norm}")

        self.main_optimizer = main_optimizer
        self.aux_optimizer = aux_optimizer
        self.main_scheduler = main_scheduler
        self.aux_scheduler = aux_scheduler
        self.clip_max_norm = clip_max_norm

        if main_params is not None:
            self.main_params = list(main_params)
        elif self.main_optimizer is not None:
            self.main_params = [p for group in self.main_optimizer.param_groups for p in group["params"]]
        else:
            self.main_params = []

        if aux_params is not None:
            self.aux_params = list(aux_params)
        elif self.aux_optimizer is not None:
            self.aux_params = [p for group in self.aux_optimizer.param_groups for p in group["params"]]
        else:
            self.aux_params = []

    def step_main(self, clip_max_norm: float | None = ...) -> float | None:  # type: ignore[assignment]
        """Applies gradient clipping to main parameters and steps main optimizer.

        Args:
            clip_max_norm: Optional per-step override for gradient clipping threshold.

        Returns:
            Computed total gradient norm before clipping, or None if main optimizer is absent.
        """
        if self.main_optimizer is None:
            return None

        if clip_max_norm is not ...:
            if clip_max_norm is not None and clip_max_norm < 0:
                raise ValueError(f"clip_max_norm must be non-negative or None, got {clip_max_norm}")
            effective_clip = clip_max_norm
        else:
            effective_clip = self.clip_max_norm

        if self.main_params:
            norm_threshold = effective_clip if effective_clip is not None else float("inf")
            total_norm_t = torch.nn.utils.clip_grad_norm_(self.main_params, max_norm=norm_threshold)
            grad_norm = float(total_norm_t.item() if isinstance(total_norm_t, torch.Tensor) else total_norm_t)
        else:
            grad_norm = 0.0

        self.main_optimizer.step()
        return grad_norm

    def step_aux(self) -> None:
        """Steps auxiliary optimizer without gradient clipping. Acts as a safe no-op if aux params are absent."""
        if self.aux_optimizer is None or not self.aux_params:
            return
        self.aux_optimizer.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Resets gradients for all active optimizers."""
        if self.main_optimizer is not None:
            self.main_optimizer.zero_grad(set_to_none=set_to_none)
        if self.aux_optimizer is not None:
            self.aux_optimizer.zero_grad(set_to_none=set_to_none)

    def zero_grad_main(self, set_to_none: bool = True) -> None:
        """Resets gradients for main optimizer."""
        if self.main_optimizer is not None:
            self.main_optimizer.zero_grad(set_to_none=set_to_none)

    def zero_grad_aux(self, set_to_none: bool = True) -> None:
        """Resets gradients for auxiliary optimizer."""
        if self.aux_optimizer is not None:
            self.aux_optimizer.zero_grad(set_to_none=set_to_none)

    def step_schedulers(self, epoch: int | None = None) -> None:
        """Steps all attached learning rate schedulers."""
        if self.main_scheduler is not None:
            if epoch is not None:
                self.main_scheduler.step(epoch)
            else:
                self.main_scheduler.step()
        if self.aux_scheduler is not None:
            if epoch is not None:
                self.aux_scheduler.step(epoch)
            else:
                self.aux_scheduler.step()

    step_scheduler = step_schedulers

    @property
    def main_lr(self) -> float | None:
        """Current learning rate of main optimizer."""
        if self.main_optimizer is None or not self.main_optimizer.param_groups:
            return None
        return float(self.main_optimizer.param_groups[0]["lr"])

    @property
    def aux_lr(self) -> float | None:
        """Current learning rate of auxiliary optimizer."""
        if self.aux_optimizer is None or not self.aux_optimizer.param_groups:
            return None
        return float(self.aux_optimizer.param_groups[0]["lr"])

    def get_lr(self) -> dict[str, float | None]:
        """Returns dictionary of active learning rates."""
        return {"main": self.main_lr, "aux": self.aux_lr}

    @property
    def optimizers(self) -> list[Optimizer]:
        """Returns list of active optimizers."""
        return [opt for opt in (self.main_optimizer, self.aux_optimizer) if opt is not None]

    @property
    def schedulers(self) -> list[LRScheduler]:
        """Returns list of active schedulers."""
        return [sched for sched in (self.main_scheduler, self.aux_scheduler) if sched is not None]

    def state_dict(self) -> dict[str, Any]:
        """Serializes internal state of optimizers, schedulers, and clipping parameters."""
        return {
            "main_optimizer": self.main_optimizer.state_dict() if self.main_optimizer is not None else None,
            "aux_optimizer": self.aux_optimizer.state_dict() if self.aux_optimizer is not None else None,
            "main_scheduler": self.main_scheduler.state_dict() if self.main_scheduler is not None else None,
            "aux_scheduler": self.aux_scheduler.state_dict() if self.aux_scheduler is not None else None,
            "clip_max_norm": self.clip_max_norm,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores state from serialized checkpoint dictionary."""
        required_keys = ("main_optimizer", "aux_optimizer")
        for key in required_keys:
            if key not in state_dict:
                raise KeyError(
                    f"Missing required key '{key}' in state_dict. "
                    f"Required keys: {required_keys}. Present keys: {list(state_dict.keys())}"
                )

        # Validate consistency between harness optimizers and state_dict
        if self.main_optimizer is not None and state_dict["main_optimizer"] is None:
            raise ValueError("Harness has active main_optimizer, but state_dict['main_optimizer'] is None")
        if self.main_optimizer is None and state_dict["main_optimizer"] is not None:
            raise ValueError("Harness has no main_optimizer, but state_dict['main_optimizer'] is not None")

        if self.aux_optimizer is not None and state_dict["aux_optimizer"] is None:
            raise ValueError("Harness has active aux_optimizer, but state_dict['aux_optimizer'] is None")
        if self.aux_optimizer is None and state_dict["aux_optimizer"] is not None:
            raise ValueError("Harness has no aux_optimizer, but state_dict['aux_optimizer'] is not None")

        if self.main_optimizer is not None:
            self.main_optimizer.load_state_dict(state_dict["main_optimizer"])

        if self.aux_optimizer is not None:
            self.aux_optimizer.load_state_dict(state_dict["aux_optimizer"])

        if self.main_scheduler is not None and state_dict.get("main_scheduler") is not None:
            self.main_scheduler.load_state_dict(state_dict["main_scheduler"])

        if self.aux_scheduler is not None and state_dict.get("aux_scheduler") is not None:
            self.aux_scheduler.load_state_dict(state_dict["aux_scheduler"])

        if "clip_max_norm" in state_dict:
            self.clip_max_norm = state_dict["clip_max_norm"]

    def __repr__(self) -> str:
        main_name = self.main_optimizer.__class__.__name__ if self.main_optimizer else "None"
        aux_name = self.aux_optimizer.__class__.__name__ if self.aux_optimizer else "None"
        return (
            f"DualOptimizerHarness("
            f"main_optimizer={main_name}, "
            f"aux_optimizer={aux_name}, "
            f"main_lr={self.main_lr}, "
            f"aux_lr={self.aux_lr}, "
            f"clip_max_norm={self.clip_max_norm})"
        )


def create_dual_optimizer(
    model: nn.Module,
    main_lr: float = 5e-5,
    aux_lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    clip_max_norm: float | None = 1.0,
    weight_decay: float = 0.0,
    lr_scheduler: str | None = None,
    total_epochs: int | None = None,
    T_max: int | None = None,
    min_lr: float = 1e-5,
    warmup_epochs: int = 0,
    warmup_start_lr: float = 0.0,
    main_scheduler: LRScheduler | None = None,
    aux_scheduler: LRScheduler | None = None,
) -> DualOptimizerHarness:
    """Factory creating a configured DualOptimizerHarness for a compression model.

    Partitions model parameters into main transforms and auxiliary entropy parameters.
    Safely handles models with zero auxiliary parameters without raising empty optimizer errors.

    Args:
        model: Compression neural network (e.g., ModelA or ModelB).
        main_lr: Learning rate for primary transforms Adam optimizer (default 5e-5).
        aux_lr: Learning rate for auxiliary entropy bottleneck Adam optimizer (default 1e-3).
        betas: Adam beta coefficients (default (0.9, 0.999)).
        eps: Adam numerical stability epsilon (default 1e-8).
        clip_max_norm: Maximum gradient norm for main transforms (default 1.0).
        weight_decay: Adam weight decay (default 0.0).
        lr_scheduler: Optional scheduler type (e.g., 'cosine').
        total_epochs: Total number of training epochs for scheduler.
        T_max: Total steps/epochs for CosineAnnealingWithWarmupLR (alias for total_epochs).
        min_lr: Minimum learning rate for cosine scheduler decay (default 1e-5).
        warmup_epochs: Warmup epoch count for cosine scheduler (default 0).
        warmup_start_lr: Warmup starting learning rate (default 0.0).
        main_scheduler: Optional pre-configured scheduler for main optimizer.
        aux_scheduler: Optional pre-configured scheduler for auxiliary optimizer.

    Returns:
        DualOptimizerHarness instance ready for training.
    """
    partition = partition_parameters(model)

    main_optimizer: Optimizer | None = None
    if partition.main_params:
        main_optimizer = Adam(
            partition.main_params,
            lr=main_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    aux_optimizer: Optimizer | None = None
    if partition.aux_params:
        aux_optimizer = Adam(
            partition.aux_params,
            lr=aux_lr,
            betas=betas,
            eps=eps,
        )

    # Resolve scheduler
    resolved_main_scheduler = main_scheduler
    effective_t_max = total_epochs if total_epochs is not None else T_max

    if resolved_main_scheduler is None and main_optimizer is not None:
        if lr_scheduler is not None and lr_scheduler.lower() != "cosine":
            raise ValueError(f"Unsupported lr_scheduler '{lr_scheduler}'. Supported: 'cosine', None.")

        if lr_scheduler is not None or effective_t_max is not None:
            epochs = effective_t_max if effective_t_max is not None else 100
            resolved_main_scheduler = CosineAnnealingWithWarmupLR(
                main_optimizer,
                T_max=epochs,
                eta_min=min_lr,
                warmup_epochs=warmup_epochs,
                warmup_start_lr=warmup_start_lr,
            )

    return DualOptimizerHarness(
        main_optimizer=main_optimizer,
        aux_optimizer=aux_optimizer,
        main_scheduler=resolved_main_scheduler,
        aux_scheduler=aux_scheduler,
        clip_max_norm=clip_max_norm,
        main_params=partition.main_params,
        aux_params=partition.aux_params,
    )
