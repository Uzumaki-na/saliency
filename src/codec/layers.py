"""Generalized Divisive Normalization (GDN) and Inverse GDN (IGDN) layers.

This module implements:
- LowerBoundFunction & LowerBound: Custom autograd function and module enforcing a lower bound.
- NonNegativeParametrizer: Stability reparameterization for non-negative parameter matrices.
- GDN: Non-linear divisive normalization with strict FP32 execution.
- IGDN: Non-linear multiplicative synthesis normalizer.

Precision Boundary:
All normalization operations execute strictly in FP32 outside autocast to prevent quantizer
coarsening and numerical underflow/overflow.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowerBoundFunction(torch.autograd.Function):
    r"""Autograd function enforcing a lower bound on tensors during forward pass,
    with custom gradient routing during backward pass.

    Forward:
        y = max(x, bound)

    Backward:
        dy/dx = grad_output if (x >= bound or grad_output < 0) else 0
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, bound: torch.Tensor) -> torch.Tensor:
        b = torch.as_tensor(bound, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, b)
        return torch.max(x, b)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, b = ctx.saved_tensors
        pass_through_if = (x >= b) | (grad_output < 0)
        grad_x = pass_through_if.to(dtype=grad_output.dtype) * grad_output
        return grad_x, None


class LowerBound(nn.Module):
    r"""Lower bound module wrapping `LowerBoundFunction`."""

    def __init__(self, bound: float | torch.Tensor) -> None:
        super().__init__()
        bound_tensor = torch.as_tensor(bound, dtype=torch.float32)
        self.register_buffer("bound", bound_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LowerBoundFunction.apply(x, self.bound)


class NonNegativeParametrizer(nn.Module):
    r"""Non-negative parameter reparametrization.

    Reparameterizes parameter p such that p >= minimum via:
        p = lower_bound(x)^2 - pedestal
    where pedestal = reparam_offset^2, bound = sqrt(minimum + pedestal).
    """

    def __init__(self, minimum: float = 0.0, reparam_offset: float = 2**-18) -> None:
        super().__init__()
        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)
        pedestal = self.reparam_offset**2
        self.register_buffer("pedestal", torch.tensor([pedestal], dtype=torch.float32))
        bound = (self.minimum + pedestal) ** 0.5
        self.lower_bound = LowerBound(bound)

    def init(self, x: torch.Tensor) -> torch.Tensor:
        """Computes internal parameter representation from target initial value."""
        return torch.sqrt(torch.max(x + self.pedestal, self.pedestal))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects internal parameter to bounded non-negative parameter space."""
        out = self.lower_bound(x)
        out = out**2 - self.pedestal
        return out


class GDN(nn.Module):
    r"""Generalized Divisive Normalization (GDN) layer.

    Introduced by Ballé et al. (ICLR 2016):
    "Density Modeling of Images Using a Generalized Normalization Transformation".

    Forward transformation (inverse=False):
        y_i = x_i / sqrt(beta_i + sum_j (gamma_{ij} * x_j^2))
    Inverse transformation (inverse=True):
        y_i = x_i * sqrt(beta_i + sum_j (gamma_{ij} * x_j^2))

    Parameter constraints:
        beta_i >= beta_min (default 1e-6)
        gamma_{ij} >= 0

    Precision policy:
        Executes strictly in FP32 outside autocast to prevent quantizer coarsening
        and numerical instability.
    """

    def __init__(
        self,
        in_channels: int,
        inverse: bool = False,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.inverse = bool(inverse)
        self.beta_min = float(beta_min)
        self.gamma_init = float(gamma_init)

        self.beta_reparam = NonNegativeParametrizer(minimum=self.beta_min)
        beta_init = torch.ones(self.in_channels, dtype=torch.float32)
        self.beta = nn.Parameter(self.beta_reparam.init(beta_init))

        self.gamma_reparam = NonNegativeParametrizer(minimum=0.0)
        gamma_init_mat = self.gamma_init * torch.eye(self.in_channels, dtype=torch.float32)
        self.gamma = nn.Parameter(self.gamma_reparam.init(gamma_init_mat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D input tensor (B, C, H, W), got {x.dim()}D tensor of shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Channel dimension mismatch: expected {self.in_channels}, got {x.shape[1]}")

        # Strict FP32 execution outside autocast
        device_type = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            x_f = x.float()
            beta = self.beta_reparam(self.beta.float())
            gamma = self.gamma_reparam(self.gamma.float())
            gamma_conv = gamma.reshape(self.in_channels, self.in_channels, 1, 1)

            norm = F.conv2d(x_f**2, gamma_conv, beta)
            if self.inverse:
                norm = torch.sqrt(norm)
            else:
                norm = torch.rsqrt(norm)

            out = x_f * norm
        return out


class IGDN(GDN):
    r"""Inverse Generalized Divisive Normalization (IGDN) layer.

    Equivalent to GDN(in_channels, inverse=True).
    """

    def __init__(
        self,
        in_channels: int,
        beta_min: float = 1e-6,
        gamma_init: float = 0.1,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            inverse=True,
            beta_min=beta_min,
            gamma_init=gamma_init,
        )
