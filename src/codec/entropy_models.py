"""Entropy models for variational learned image compression.

Implements:
- GaussianConditional: Zero-mean conditionally Gaussian entropy model for main latents y.
- EntropyBottleneck: Channel-wise non-parametric factorized entropy model for hyper-latents z.

Both models enforce strict FP32 numerical isolation outside autocast to prevent CDF underflow,
and construct 16-bit integer precision cumulative distribution function (CDF) tables
guaranteeing strict monotonicity for cross-device deterministic integer rANS entropy coding.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from src.codec.layers import LowerBound


def _standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    r"""Computes standard normal cumulative distribution function :math:`\Phi(x)`.

    Executes in strict FP32 using PyTorch erf:
    :math:`\Phi(x) = \frac{1}{2}\left[1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right]`.
    """
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    return 0.5 * (1.0 + torch.erf(x * inv_sqrt2))


def _normalize_frequencies(
    probs: torch.Tensor,
    total_freq: int = 65536,
) -> torch.Tensor:
    r"""Quantizes continuous probability mass vector into integer frequencies summing to total_freq.

    Guarantees that every symbol has frequency >= 1 (strictly non-zero), ensuring
    monotonicity in the resulting cumulative distribution function.

    Args:
        probs: 1D continuous probability mass tensor (sums to ~1.0).
        total_freq: Total frequency count (default 65536 = 2^16).

    Returns:
        1D integer tensor of frequencies summing to total_freq.
    """
    num_symbols = probs.numel()
    if num_symbols > total_freq:
        raise ValueError(
            f"Alphabet size ({num_symbols}) cannot exceed total frequency budget ({total_freq})"
        )

    # Initial rounding
    probs_clamped = torch.clamp(probs, min=1e-12)
    probs_norm = probs_clamped / probs_clamped.sum()
    freqs = torch.clamp(torch.round(probs_norm * float(total_freq)).to(torch.int32), min=1)

    # Reconcile sum to exactly total_freq
    curr_sum = int(freqs.sum().item())
    diff = total_freq - curr_sum

    if diff != 0:
        # Adjust the mode (largest frequency symbol)
        mode_idx = torch.argmax(freqs).item()
        freqs[mode_idx] += diff
        # If adjustment resulted in <= 0 (extreme edge case), distribute evenly
        if freqs[mode_idx] < 1:
            freqs[mode_idx] = 1
            remainder = total_freq - int(freqs.sum().item())
            sorted_indices = torch.argsort(freqs, descending=True)
            for idx in sorted_indices:
                if remainder == 0:
                    break
                step = 1 if remainder > 0 else -1
                if freqs[idx] + step >= 1:
                    freqs[idx] += step
                    remainder -= step

    return freqs


class GaussianConditional(nn.Module):
    r"""Conditional Gaussian entropy model :math:`p(y \mid \sigma) = \mathcal{N}(0, \sigma^2)`.

    Evaluates continuous likelihoods during training:
    :math:`p(\tilde{y} \mid \sigma) = \Phi((\tilde{y} + 0.5)/\sigma) - \Phi((\tilde{y} - 0.5)/\sigma)`.

    Provides offline discrete scale table compilation (`update_scale_table()`) producing
    16-bit integer CDF tables for discrete scale index lookups during inference.
    """

    def __init__(
        self,
        scale_bound: float = 0.11,
        num_scales: int = 64,
        scale_min: float = 0.11,
        scale_max: float = 256.0,
    ) -> None:
        super().__init__()
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.num_scales = int(num_scales)

        self.lower_bound_scale = LowerBound(scale_bound)
        self.register_buffer("scale_bound", torch.tensor(scale_bound, dtype=torch.float32))

        # Default scale table with logarithmic spacing
        log_scales = torch.linspace(
            math.log(self.scale_min),
            math.log(self.scale_max),
            self.num_scales,
            dtype=torch.float32,
        )
        scales = torch.exp(log_scales)
        self.register_buffer("scale_table", scales)

        # Pre-allocated CDF buffers (populated via update_scale_table)
        self.register_buffer("_quantized_cdf", torch.empty((0, 0), dtype=torch.int32))
        self.register_buffer("_cdf_length", torch.empty(0, dtype=torch.int32))
        self.register_buffer("_offset", torch.empty(0, dtype=torch.int32))

    def build_indexes(self, scales: torch.Tensor) -> torch.Tensor:
        r"""Maps continuous scales to discrete scale table indices in [0, num_scales - 1]."""
        scales_f = scales.float()
        log_scale_min = math.log(self.scale_min)
        log_scale_max = math.log(self.scale_max)
        log_step = (log_scale_max - log_scale_min) / float(self.num_scales - 1)

        log_s = torch.log(torch.clamp(scales_f, min=self.scale_min))
        indexes = torch.clamp(
            torch.round((log_s - log_scale_min) / log_step).long(),
            0,
            self.num_scales - 1,
        )
        return indexes

    def quantize(
        self,
        inputs: torch.Tensor,
        mode: str = "noise",
        means: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Applies quantization (continuous uniform noise or rounding) to inputs."""
        inputs_f = inputs.float()
        if means is not None:
            inputs_f = inputs_f - means.float()

        if mode == "noise":
            noise = torch.empty_like(inputs_f).uniform_(-0.5, 0.5)
            quantized = inputs_f + noise
        elif mode == "round":
            quantized = torch.round(inputs_f)
        else:
            raise ValueError(f"Unknown quantization mode: {mode}. Expected 'noise' or 'round'.")

        if means is not None:
            quantized = quantized + means.float()
        return quantized

    def dequantize(
        self,
        inputs: torch.Tensor,
        means: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Returns dequantized representation."""
        out = inputs.float()
        if means is not None:
            out = out + means.float()
        return out

    def forward(
        self,
        inputs: torch.Tensor,
        scales: torch.Tensor,
        means: torch.Tensor | None = None,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Evaluates quantized latents and Gaussian likelihoods in strict FP32.

        Args:
            inputs: Continuous latents y.
            scales: Predicted standard deviation scales sigma (positive).
            means: Optional mean offset mu (default 0).
            training: Mode override (True for noise relaxation, False for rounding).

        Returns:
            Tuple of (quantized_inputs, likelihoods).
        """
        device_type = inputs.device.type if inputs.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            inputs_f = inputs.float()
            scales_f = self.lower_bound_scale(scales.float())

            if means is not None:
                means_f = means.float()
                centered = inputs_f - means_f
            else:
                means_f = None
                centered = inputs_f

            is_training = self.training if training is None else bool(training)
            if is_training:
                noise = torch.empty_like(centered).uniform_(-0.5, 0.5)
                values = centered + noise
            else:
                values = torch.round(centered)

            # Likelihood integral across [-0.5, 0.5] bin
            inv_sqrt2 = 1.0 / math.sqrt(2.0)
            upper = 0.5 * (1.0 + torch.erf((values + 0.5) / scales_f * inv_sqrt2))
            lower = 0.5 * (1.0 + torch.erf((values - 0.5) / scales_f * inv_sqrt2))
            likelihoods = torch.clamp(upper - lower, min=1e-9, max=1.0)

            if means_f is not None:
                outputs = values + means_f
            else:
                outputs = values

        return outputs, likelihoods

    @torch.no_grad()
    def update_scale_table(self, scales: torch.Tensor | None = None) -> bool:
        r"""Compiles 16-bit integer CDF tables for all discrete scales.

        Args:
            scales: Optional 1D tensor of scales. If None, uses internal scale_table.

        Returns:
            True upon successful compilation.
        """
        if scales is not None:
            self.scale_table = scales.float().clone()
            self.num_scales = self.scale_table.numel()

        device = self.scale_table.device
        all_cdfs: list[torch.Tensor] = []
        all_offsets: list[int] = []
        all_lengths: list[int] = []

        total_freq = 65536

        for s in self.scale_table.cpu():
            scale_val = float(s.item())
            # For Gaussian tail probability < 1e-5, support bound is approx 4.5 * scale
            max_val = max(1, math.ceil(4.5 * scale_val))
            min_val = -max_val
            symbols = torch.arange(min_val, max_val + 1, dtype=torch.float32)

            # Evaluate probability mass
            inv_sqrt2 = 1.0 / math.sqrt(2.0)
            upper = 0.5 * (1.0 + torch.erf((symbols + 0.5) / scale_val * inv_sqrt2))
            lower = 0.5 * (1.0 + torch.erf((symbols - 0.5) / scale_val * inv_sqrt2))
            probs = torch.clamp(upper - lower, min=1e-12)

            # Integer frequency normalization
            freqs = _normalize_frequencies(probs, total_freq=total_freq)

            # Cumulative sum: CDF starts at 0, ends at total_freq
            cdf = torch.zeros(freqs.numel() + 1, dtype=torch.int32)
            cdf[1:] = torch.cumsum(freqs, dim=0)

            all_cdfs.append(cdf)
            all_offsets.append(min_val)
            all_lengths.append(cdf.numel())

        max_len = max(all_lengths)
        quantized_cdf = torch.zeros((self.num_scales, max_len), dtype=torch.int32)
        for i, cdf in enumerate(all_cdfs):
            quantized_cdf[i, : cdf.numel()] = cdf
            # Pad remainder with the total frequency to maintain non-decreasing invariant
            if cdf.numel() < max_len:
                quantized_cdf[i, cdf.numel() :] = total_freq

        self._quantized_cdf = quantized_cdf.to(device)
        self._cdf_length = torch.tensor(all_lengths, dtype=torch.int32, device=device)
        self._offset = torch.tensor(all_offsets, dtype=torch.int32, device=device)
        return True

    def update(self, force: bool = False, scales: torch.Tensor | None = None) -> bool:
        r"""Compiles internal scale table CDFs."""
        if not force and self._quantized_cdf.numel() > 0:
            return False
        return self.update_scale_table(scales=scales)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for attr in ("_quantized_cdf", "_cdf_length", "_offset", "scale_table"):
            key = prefix + attr
            if key in state_dict:
                setattr(self, attr, state_dict[key].clone())
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


class EntropyBottleneck(nn.Module):
    r"""Factorized non-parametric entropy model for hyper-latents :math:`z`.

    Models marginal distribution :math:`p(z) = \prod_c p_{z_c}(z_c)` where each channel
    is parameterized by a monotonic multi-layer cumulative density network:
    :math:`c(x) = \operatorname{sigmoid}(f(x))`.

    Provides offline CDF compilation (`update()`) finding channel-specific support bounds
    and compiling 16-bit integer cumulative distributions for rANS coding.
    """

    def __init__(
        self,
        channels: int = 128,
        init_scale: float = 10.0,
        filters: tuple[int, ...] = (3, 3, 3),
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.filters = tuple(filters)

        # Monotonic network parameters per channel
        # Weights are initialized positive and constrained with LowerBound(w, 0.0)
        layer_sizes = [1, *list(self.filters), 1]
        self.num_layers = len(layer_sizes) - 1

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for i in range(self.num_layers):
            in_f = layer_sizes[i]
            out_f = layer_sizes[i + 1]
            # Initialize weights to positive values
            init_w = torch.full((self.channels, out_f, in_f), 0.5 / float(in_f))
            init_b = torch.zeros((self.channels, out_f, 1))

            self.weights.append(nn.Parameter(init_w))
            self.biases.append(nn.Parameter(init_b))

        self.lower_bound = LowerBound(0.0)

        # Pre-allocated CDF buffers
        self.register_buffer("_quantized_cdf", torch.empty((0, 0), dtype=torch.int32))
        self.register_buffer("_cdf_length", torch.empty(0, dtype=torch.int32))
        self.register_buffer("_offset", torch.empty(0, dtype=torch.int32))

    def _logits_cumulative(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Evaluates cumulative distribution :math:`c(x) \in (0, 1)` in strict FP32."""
        device_type = inputs.device.type if inputs.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # Input shape: (B, C, H, W) or (C, N)
            orig_shape = inputs.shape
            if inputs.dim() == 4:
                c = orig_shape[1]
            elif inputs.dim() == 2:
                c = orig_shape[0]
            else:
                c = orig_shape[1] if inputs.dim() > 1 else orig_shape[0]

            if c != self.channels:
                raise ValueError(f"Channel mismatch: expected {self.channels}, got {c}")

            # Permute to (C, 1, -1) for batched matrix multiplication
            if inputs.dim() == 4:
                batch_size, _, height, width = orig_shape
                x = inputs.float().permute(1, 0, 2, 3).reshape(self.channels, 1, batch_size * height * width)
            elif inputs.dim() == 2:
                x = inputs.float().unsqueeze(1)
            else:
                x = inputs.float().reshape(self.channels, 1, -1)

            # Forward through monotonic layers
            for i in range(self.num_layers):
                w = self.lower_bound(self.weights[i])
                bias = self.biases[i]
                # x is (C, in_f, N), w is (C, out_f, in_f) -> (C, out_f, N)
                x = torch.baddbmm(bias, w, x)
                # Monotonic activation g(v) = v + tanh(v) for hidden layers
                if i < self.num_layers - 1:
                    x = x + torch.tanh(x)

            # Final sigmoid mapping to (0, 1)
            cdf = torch.sigmoid(x)

            # Restore original shape
            if inputs.dim() == 4:
                cdf = cdf.reshape(self.channels, batch_size, height, width).permute(1, 0, 2, 3)
            elif inputs.dim() == 2:
                cdf = cdf.squeeze(1)
            else:
                cdf = cdf.reshape(orig_shape)

        return cdf

    def forward(
        self,
        x: torch.Tensor,
        training: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Evaluates quantized hyper-latents and factorized likelihoods in strict FP32.

        Args:
            x: Continuous hyper-latents z of shape (B, C, H, W).
            training: Mode override (True for noise relaxation, False for rounding).

        Returns:
            Tuple of (quantized_z, likelihoods).
        """
        device_type = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            x_f = x.float()
            is_training = self.training if training is None else bool(training)

            if is_training:
                noise = torch.empty_like(x_f).uniform_(-0.5, 0.5)
                x_hat = x_f + noise
            else:
                x_hat = torch.round(x_f)

            # Evaluate likelihood c(x + 0.5) - c(x - 0.5)
            upper = self._logits_cumulative(x_hat + 0.5)
            lower = self._logits_cumulative(x_hat - 0.5)
            likelihoods = torch.clamp(upper - lower, min=1e-9, max=1.0)

        return x_hat, likelihoods

    def quantize(self, x: torch.Tensor, mode: str = "noise") -> torch.Tensor:
        r"""Quantizes hyper-latents using noise or rounding."""
        x_f = x.float()
        if mode == "noise":
            noise = torch.empty_like(x_f).uniform_(-0.5, 0.5)
            return x_f + noise
        elif mode == "round":
            return torch.round(x_f)
        raise ValueError(f"Unknown quantization mode: {mode}")

    def dequantize(self, x: torch.Tensor) -> torch.Tensor:
        r"""Dequantizes hyper-latents."""
        return x.float()

    def loss(self) -> torch.Tensor:
        r"""Auxiliary loss for non-negative parameter constraints."""
        total_loss = torch.tensor(0.0, device=self.weights[0].device)
        return total_loss

    @torch.no_grad()
    def update(self, force: bool = False) -> bool:
        r"""Compiles 16-bit integer cumulative distribution tables for all channels.

        Determines integer support bounds [z_min, z_max] for each channel c
        by scanning integer coordinates until tail mass < 1/65536.

        Args:
            force: Recompiles even if already populated.

        Returns:
            True upon successful compilation.
        """
        if not force and self._quantized_cdf.numel() > 0:
            return False

        device = self.weights[0].device
        total_freq = 65536

        # Scan integer samples across [-256, 256] in FP32
        samples = torch.arange(-256, 257, dtype=torch.float32, device=device)
        # Shape: (channels, num_samples)
        samples_expanded = samples.unsqueeze(0).expand(self.channels, -1)
        cdf_vals = self._logits_cumulative(samples_expanded)

        all_cdfs: list[torch.Tensor] = []
        all_offsets: list[int] = []
        all_lengths: list[int] = []

        threshold = 1.0 / float(total_freq)

        for c in range(self.channels):
            channel_cdf = cdf_vals[c].cpu()
            # Find bounds where CDF is within [threshold, 1 - threshold]
            valid = (channel_cdf >= threshold) & (channel_cdf <= 1.0 - threshold)
            indices = torch.where(valid)[0]

            if indices.numel() == 0:
                # Fallback to narrow default support
                min_idx = 256 - 5
                max_idx = 256 + 5
            else:
                min_idx = max(0, indices[0].item() - 1)
                max_idx = min(samples.numel() - 1, indices[-1].item() + 1)

            z_min = int(samples[min_idx].item())
            z_max = int(samples[max_idx].item())
            if z_max <= z_min:
                z_max = z_min + 1

            syms = torch.arange(z_min, z_max + 1, dtype=torch.float32, device=device)
            # Evaluate upper and lower cumulative points for channel c
            # Need tensor of shape (channels, N) with row c populated
            full_syms_upper = torch.zeros((self.channels, syms.numel()), device=device)
            full_syms_lower = torch.zeros((self.channels, syms.numel()), device=device)
            full_syms_upper[c] = syms + 0.5
            full_syms_lower[c] = syms - 0.5

            up = self._logits_cumulative(full_syms_upper)[c].cpu()
            lo = self._logits_cumulative(full_syms_lower)[c].cpu()
            probs = torch.clamp(up - lo, min=1e-12)

            freqs = _normalize_frequencies(probs, total_freq=total_freq)
            cdf = torch.zeros(freqs.numel() + 1, dtype=torch.int32)
            cdf[1:] = torch.cumsum(freqs, dim=0)

            all_cdfs.append(cdf)
            all_offsets.append(z_min)
            all_lengths.append(cdf.numel())

        max_len = max(all_lengths)
        quantized_cdf = torch.zeros((self.channels, max_len), dtype=torch.int32)
        for i, cdf in enumerate(all_cdfs):
            quantized_cdf[i, : cdf.numel()] = cdf
            if cdf.numel() < max_len:
                quantized_cdf[i, cdf.numel() :] = total_freq

        self._quantized_cdf = quantized_cdf.to(device)
        self._cdf_length = torch.tensor(all_lengths, dtype=torch.int32, device=device)
        self._offset = torch.tensor(all_offsets, dtype=torch.int32, device=device)
        return True

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for attr in ("_quantized_cdf", "_cdf_length", "_offset"):
            key = prefix + attr
            if key in state_dict:
                setattr(self, attr, state_dict[key].clone())
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
