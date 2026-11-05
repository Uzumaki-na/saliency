"""Deterministic single-threaded CPU entropy parameter execution engine.

Ensures that floating-point entropy parameters (scale parameters sigma and log_sigma)
and discrete scale table indices are derived identically across encoder and decoder,
eliminating ulp drift between GPU execution and CPU decoding.
"""

from __future__ import annotations

import copy
import math
import os
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from src.codec.model_a import ModelA


def configure_cpu_determinism() -> None:
    r"""Configures host runtime environment for bit-exact deterministic CPU inference.

    Enforces single-threaded execution, disables MKL-DNN and TF32, and configures
    MKL instruction set dispatch to AVX2.
    """
    os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # May fail if inter-op parallelism was already initialized

    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False

    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


class DeterministicCPUEngine:
    r"""Single-threaded strict FP32 CPU entropy parameter execution engine.

    Wraps the hyper-synthesis network :math:`h_s` and discrete scale table quantization,
    guaranteeing bit-identical discrete scale index derivation between encoder and decoder:
    :math:`\max |\text{scale\_idx}_{\text{enc}} - \text{scale\_idx}_{\text{dec}}| == 0`.

    Args:
        hyper_synthesis: Optional hyper-synthesis transform (h_s). If None, defaults to
            HyperSynthesisTransform(128, 128, 192).
        num_scales: Number of discrete scale table entries (default 64).
        scale_min: Minimum scale bound (default 0.11).
        scale_max: Maximum scale bound (default 256.0).
    """

    def __init__(
        self,
        hyper_synthesis: nn.Module | None = None,
        num_scales: int = 64,
        scale_min: float = 0.11,
        scale_max: float = 256.0,
    ) -> None:
        self.num_scales = int(num_scales)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)

        self.log_scale_min = math.log(self.scale_min)
        self.log_scale_max = math.log(self.scale_max)
        self.log_step = (self.log_scale_max - self.log_scale_min) / float(self.num_scales - 1)

        # Pre-compute static 1D CPU scale table
        log_scales = torch.linspace(
            self.log_scale_min,
            self.log_scale_max,
            self.num_scales,
            dtype=torch.float32,
        )
        self.scale_table = torch.exp(log_scales)

        # Clone and freeze hyper-synthesis module on CPU in strict FP32
        if hyper_synthesis is not None:
            self.h_s = copy.deepcopy(hyper_synthesis).cpu().float().eval()
        else:
            from src.codec.model_a import HyperSynthesisTransform

            self.h_s = HyperSynthesisTransform(in_channels=128, n_channels=128, out_channels=192).cpu().float().eval()

        self.h_s.requires_grad_(False)

    @classmethod
    def from_model(cls, model: ModelA) -> DeterministicCPUEngine:
        r"""Constructs engine instance directly from a trained ModelA instance."""
        return cls(
            hyper_synthesis=model.h_s,
            num_scales=getattr(model.gaussian_conditional, "num_scales", 64),
            scale_min=getattr(model.gaussian_conditional, "scale_min", 0.11),
            scale_max=getattr(model.gaussian_conditional, "scale_max", 256.0),
        )

    def sync_parameters(self, hyper_synthesis: nn.Module) -> None:
        r"""Synchronizes CPU hyper-synthesis weights from an updated source model."""
        self.h_s.load_state_dict(hyper_synthesis.state_dict())
        self.h_s.cpu().float().eval()
        self.h_s.requires_grad_(False)

    def compute_scale_indexes(
        self,
        log_sigma: torch.Tensor,
        log_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Derives discrete scale indices from continuous log-scale representation.

        Executes in strict single-threaded FP32 on CPU with log-domain quantization.

        Args:
            log_sigma: Continuous log-scale tensor.
            log_delta: Optional adaptive log-quantization step tensor.

        Returns:
            torch.Tensor of dtype int32 containing discrete scale indices in [0, num_scales - 1].
        """
        configure_cpu_determinism()

        with torch.no_grad():
            eff_log_sigma = log_sigma.detach().cpu().float()
            if log_delta is not None:
                eff_log_sigma = eff_log_sigma - log_delta.detach().cpu().float()

            indexes = torch.clamp(
                torch.round((eff_log_sigma - self.log_scale_min) / self.log_step).to(torch.int32),
                0,
                self.num_scales - 1,
            )
        return indexes

    def forward(
        self,
        z_hat: torch.Tensor,
        log_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Executes deterministic parameter derivation from quantized hyper-latents z_hat.

        Args:
            z_hat: Quantized hyper-latents tensor (B, 128, H, W) or (128, H, W).
            log_delta: Optional adaptive log-quantization step tensor.

        Returns:
            Tuple of:
                log_sigma: Predicted continuous log-scale tensor on CPU in FP32.
                scale_indexes: Discrete scale index tensor (int32) in [0, num_scales - 1].
        """
        configure_cpu_determinism()

        with torch.no_grad():
            orig_dim = z_hat.dim()
            if orig_dim == 3:
                z_input = z_hat.unsqueeze(0)
            elif orig_dim == 4:
                z_input = z_hat
            else:
                msg = f"Expected 3D or 4D tensor, got {orig_dim}D tensor of shape {tuple(z_hat.shape)}"
                raise ValueError(msg)

            z_cpu = z_input.detach().cpu().float()
            log_sigma = self.h_s(z_cpu)
            scale_indexes = self.compute_scale_indexes(log_sigma, log_delta=log_delta)

            if orig_dim == 3:
                log_sigma = log_sigma.squeeze(0)
                scale_indexes = scale_indexes.squeeze(0)

        return log_sigma, scale_indexes

    def __call__(
        self,
        z_hat: torch.Tensor,
        log_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Callable shortcut to forward()."""
        return self.forward(z_hat, log_delta=log_delta)
