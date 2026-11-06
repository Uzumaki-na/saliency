"""Model B (bmshj2018 saliency-weighted implicit allocation baseline).

This module implements:
- ModelB: Variational image compression model subclassing ModelA (bmshj2018).
  Shares identical architecture and parameter shapes (N=128, M=192), serving
  as the implicit rate allocation baseline trained under the saliency-weighted
  distortion objective (S' = 0.2 + 0.8 S).
- remap_compressai_state_dict: Utility translating CompressAI Sequential layer
  naming (g_a.0 -> g_a.conv1, etc.) to ModelA/ModelB naming conventions.
- load_pretrained_bmshj2018: Checkpoint loader supporting raw state dicts,
  wrapped dictionaries, checkpoint files on disk, and CompressAI remapping.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from src.codec.model_a import ModelA

logger = logging.getLogger(__name__)

# Translation rules mapping CompressAI Sequential layer indexing to ModelA/ModelB naming
COMPRESSAI_RENAME_RULES: tuple[tuple[str, str], ...] = (
    ("g_a.0.", "g_a.conv1."),
    ("g_a.1.", "g_a.gdn1."),
    ("g_a.2.", "g_a.conv2."),
    ("g_a.3.", "g_a.gdn2."),
    ("g_a.4.", "g_a.conv3."),
    ("g_a.5.", "g_a.gdn3."),
    ("g_a.6.", "g_a.conv4."),
    ("g_s.0.", "g_s.deconv1."),
    ("g_s.1.", "g_s.igdn1."),
    ("g_s.2.", "g_s.deconv2."),
    ("g_s.3.", "g_s.igdn2."),
    ("g_s.4.", "g_s.deconv3."),
    ("g_s.5.", "g_s.igdn3."),
    ("g_s.6.", "g_s.deconv4."),
    ("h_a.0.", "h_a.conv1."),
    ("h_a.2.", "h_a.conv2."),
    ("h_a.4.", "h_a.conv3."),
    ("h_s.0.", "h_s.deconv1."),
    ("h_s.2.", "h_s.deconv2."),
    ("h_s.4.", "h_s.conv3."),
    ("entropy_bottleneck._matrices.", "entropy_bottleneck.weights."),
    ("entropy_bottleneck.matrices.", "entropy_bottleneck.weights."),
    ("entropy_bottleneck._biases.", "entropy_bottleneck.biases."),
)

_KNOWN_MODEL_PREFIXES: tuple[str, ...] = (
    "g_a.conv1.",
    "g_a.conv2.",
    "g_a.conv3.",
    "g_a.conv4.",
    "g_a.gdn1.",
    "g_a.gdn2.",
    "g_a.gdn3.",
    "g_s.deconv1.",
    "g_s.deconv2.",
    "g_s.deconv3.",
    "g_s.deconv4.",
    "g_s.igdn1.",
    "g_s.igdn2.",
    "g_s.igdn3.",
    "h_a.conv1.",
    "h_a.conv2.",
    "h_a.conv3.",
    "h_s.deconv1.",
    "h_s.deconv2.",
    "h_s.conv3.",
    "entropy_bottleneck.weights.",
    "entropy_bottleneck.biases.",
    "entropy_bottleneck._quantized_cdf",
    "entropy_bottleneck._cdf_length",
    "entropy_bottleneck._offset",
    "entropy_bottleneck.lower_bound.bound",
    "gaussian_conditional.scale_bound",
    "gaussian_conditional.scale_table",
    "gaussian_conditional._quantized_cdf",
    "gaussian_conditional._cdf_length",
    "gaussian_conditional._offset",
    "gaussian_conditional.lower_bound_scale.bound",
)

_DEFAULT_BUFFERS: dict[str, torch.Tensor] = {
    "entropy_bottleneck.lower_bound.bound": torch.tensor(0.0, dtype=torch.float32),
    "gaussian_conditional.scale_bound": torch.tensor(0.11, dtype=torch.float32),
    "gaussian_conditional.lower_bound_scale.bound": torch.tensor(0.11, dtype=torch.float32),
}


def _clean_key(key: str) -> str:
    """Strips distributed training (module.) and compilation (_orig_mod.) prefixes iteratively."""
    stripped = True
    while stripped:
        stripped = False
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                stripped = True
                break
    return key


def _is_known_model_key(key: str) -> bool:
    """Returns True if key starts with an expected ModelA/ModelB submodule prefix."""
    return any(key.startswith(prefix) for prefix in _KNOWN_MODEL_PREFIXES)


def _is_compressai_ignored_key(key: str) -> bool:
    """Returns True if key is a CompressAI-specific attribute unneeded in ModelA/ModelB."""
    return (
        key.startswith("entropy_bottleneck._factors.")
        or key.startswith("entropy_bottleneck.factors.")
        or key == "entropy_bottleneck.quantiles"
        or key.startswith("entropy_bottleneck.quantiles.")
        or key == "entropy_bottleneck._quantiles"
        or key.startswith("entropy_bottleneck._quantiles.")
        or key == "entropy_bottleneck.target"
        or key.startswith("entropy_bottleneck.target.")
        or "likelihood_lower_bound." in key
    )


def _unwrap_state_dict(checkpoint: Any) -> dict[str, Any]:
    r"""Unwraps nested state dict from common checkpoint dictionary wrappers.

    Handles wrappers such as {"state_dict": ...}, {"model": ...},
    {"model_state_dict": ...}, {"network": ...}, or {"net": ...}.
    """
    if not isinstance(checkpoint, (dict, Mapping)):
        raise TypeError(f"Expected dict-like checkpoint, got {type(checkpoint).__name__}")

    curr: Any = checkpoint
    while isinstance(curr, (dict, Mapping)):
        wrapper_found = False
        for wrapper_key in ("state_dict", "model", "model_state_dict", "network", "net"):
            if wrapper_key in curr and isinstance(curr[wrapper_key], (dict, Mapping)):
                curr = curr[wrapper_key]
                wrapper_found = True
                break
        if not wrapper_found:
            break

    return dict(curr)


def remap_compressai_state_dict(
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    r"""Remaps CompressAI bmshj2018 Sequential state dict keys to ModelA/ModelB naming.

    Translation mapping:
    - g_a.0 -> g_a.conv1, g_a.1 -> g_a.gdn1, g_a.2 -> g_a.conv2, g_a.3 -> g_a.gdn2,
      g_a.4 -> g_a.conv3, g_a.5 -> g_a.gdn3, g_a.6 -> g_a.conv4
    - g_s.0 -> g_s.deconv1, g_s.1 -> g_s.igdn1, g_s.2 -> g_s.deconv2, g_s.3 -> g_s.igdn2,
      g_s.4 -> g_s.deconv3, g_s.5 -> g_s.igdn3, g_s.6 -> g_s.deconv4
    - h_a.0 -> h_a.conv1, h_a.2 -> h_a.conv2, h_a.4 -> h_a.conv3
    - h_s.0 -> h_s.deconv1, h_s.2 -> h_s.deconv2, h_s.4 -> h_s.conv3
    - entropy_bottleneck._matrices.i -> entropy_bottleneck.weights.i
    - entropy_bottleneck._biases.i -> entropy_bottleneck.biases.i

    Preserves direct ModelA keys, entropy bottleneck buffers, and gaussian conditional buffers.
    Logs warnings for any unmapped or unexpected keys.

    Args:
        state_dict: Source state dictionary from CompressAI or direct ModelA checkpoint.

    Returns:
        dict[str, Any]: Translated state dictionary.
    """
    remapped: dict[str, Any] = {}
    unmapped_keys: list[str] = []

    for raw_k, v in state_dict.items():
        k = _clean_key(raw_k)

        # Ignore unneeded CompressAI-specific parameters and buffers
        if _is_compressai_ignored_key(k):
            continue

        new_k = k
        matched = False
        for src_prefix, dst_prefix in COMPRESSAI_RENAME_RULES:
            if k.startswith(src_prefix):
                new_k = dst_prefix + k[len(src_prefix) :]
                matched = True
                break

        if not matched and not _is_known_model_key(new_k):
            unmapped_keys.append(raw_k)

        remapped[new_k] = v

    # Populate default lower_bound buffers and scale bounds if missing from checkpoint
    for buf_key, default_val in _DEFAULT_BUFFERS.items():
        if buf_key not in remapped:
            remapped[buf_key] = default_val.clone()

    if unmapped_keys:
        logger.warning(
            "Found %d unmapped key(s) during CompressAI state dict translation: %s",
            len(unmapped_keys),
            unmapped_keys,
        )

    return remapped


def load_pretrained_bmshj2018(
    checkpoint: str | Path | Mapping[str, Any] | ModelA,
    target_or_checkpoint: str | Path | Mapping[str, Any] | ModelA | None = None,
    *,
    model: ModelA | None = None,
    strict: bool = True,
    device: str | torch.device | None = None,
    **model_kwargs: Any,
) -> ModelB | ModelA:
    r"""Loads pretrained bmshj2018 weights into a ModelB (or ModelA) instance.

    Features:
    - Supports loading from `.pt` or `.pth` file paths on disk, raising FileNotFoundError if missing.
    - Automatically unwraps nested checkpoint dictionaries (e.g. `{"state_dict": ...}`, `{"model": ...}`).
    - Automatically remaps CompressAI Sequential layer indexing to ModelA/ModelB naming.
    - Supports direct ModelA state dicts with 100% parameter match.
    - Accommodates flexible invocation styles:
        `load_pretrained_bmshj2018(path)` -> creates ModelB and loads
        `load_pretrained_bmshj2018(state_dict)` -> creates ModelB and loads
        `load_pretrained_bmshj2018(path, model=model)` -> loads into existing model
        `load_pretrained_bmshj2018(model, path)` -> loads into existing model

    Args:
        checkpoint: File path (str or Path), state dict mapping, or target model instance.
        target_or_checkpoint: Optional second positional argument for flexible signature support.
        model: Target model instance. If None and no model passed positionally, a new ModelB is instantiated.
        strict: Enforces strict key matching in load_state_dict (default: True).
        device: Device to map tensors to or move instantiated model to.
        **model_kwargs: Constructor keyword arguments passed to ModelB if instantiated.

    Returns:
        ModelB or ModelA instance with weights loaded.

    Raises:
        FileNotFoundError: If checkpoint file path does not exist.
        TypeError: If checkpoint is neither a valid path nor dict mapping.
        RuntimeError: If strict=True and keys do not match model parameters/buffers.
    """
    target_model: ModelA | None = None
    state_source: str | Path | Mapping[str, Any] | None = None

    if target_or_checkpoint is not None:
        if isinstance(checkpoint, ModelA) and not isinstance(target_or_checkpoint, ModelA):
            target_model = checkpoint
            state_source = target_or_checkpoint
        elif isinstance(target_or_checkpoint, ModelA) and not isinstance(checkpoint, ModelA):
            target_model = target_or_checkpoint
            state_source = checkpoint
        else:
            raise ValueError(
                f"Unexpected second positional argument '{type(target_or_checkpoint).__name__}'. "
                "When two positional arguments are passed, exactly one must be a ModelA instance."
            )
        if model is not None:
            raise ValueError("Target model was specified both positionally and via 'model' keyword argument.")
    else:
        if isinstance(checkpoint, ModelA):
            raise ValueError("Checkpoint file path or state dict must be provided.")
        state_source = checkpoint
        target_model = model

    if state_source is None:
        raise ValueError("Checkpoint file path or state dict must be provided.")

    if isinstance(state_source, (str, Path)):
        path = Path(state_source)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        map_loc = device if device is not None else "cpu"
        raw_checkpoint = torch.load(path, map_location=map_loc, weights_only=False)
    elif isinstance(state_source, (dict, Mapping)):
        raw_checkpoint = state_source
    else:
        raise TypeError(
            f"Expected checkpoint path (str, Path) or state dict mapping, got {type(state_source).__name__}"
        )

    state_dict = _unwrap_state_dict(raw_checkpoint)
    remapped_state_dict = remap_compressai_state_dict(state_dict)

    if target_model is None:
        target_model = ModelB(**model_kwargs)

    if device is not None:
        target_device = torch.device(device)
    else:
        try:
            target_device = next(target_model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")

    target_model.to(target_device)

    # Use super().load_state_dict on ModelB to avoid double-remapping and duplicate warning logs
    if isinstance(target_model, ModelB):
        super(ModelB, target_model).load_state_dict(remapped_state_dict, strict=strict)
    else:
        target_model.load_state_dict(remapped_state_dict, strict=strict)

    # Re-synchronize all parameters and buffers to target_device after loading
    target_model.to(target_device)
    return target_model


class ModelB(ModelA):
    r"""Model B baseline: loss-weighted implicit rate allocation.

    Shares identical network architecture and parameter dimensions with Model A (bmshj2018):
    - Analysis transform g_a (16x downsampling, 4 stages with GDN)
    - Synthesis transform g_s (16x upsampling, 4 stages with IGDN)
    - Hyper-analysis transform h_a (4x downsampling on latents y, total 64x)
    - Hyper-synthesis transform h_s (4x upsampling on hyper-latents z)
    - Factorized EntropyBottleneck for hyper-latents z
    - GaussianConditional for latents y conditioned on predicted scales sigma = exp(h_s(z_hat))

    Model B is designated for training under the saliency-weighted rate-distortion objective:
        loss = R_bpp + lambda * 255^2 * D_saliency
    where S' = 0.2 + 0.8 * S.

    Latent rate allocation is driven implicitly through the spatial weighting of the reconstruction
    loss rather than explicit latent quantization scaling (Delta).
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_channels: int = 128,
        m_channels: int = 192,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            n_channels=n_channels,
            m_channels=m_channels,
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        r"""Loads state dict with automatic unwrap, CompressAI key remapping, and device synchronization."""
        try:
            target_device = next(self.parameters()).device
        except StopIteration:
            target_device = None

        unwrapped = _unwrap_state_dict(state_dict)
        remapped = remap_compressai_state_dict(unwrapped)
        result = super().load_state_dict(remapped, strict=strict, assign=assign)

        if target_device is not None:
            self.to(target_device)

        return result

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path | Mapping[str, Any],
        strict: bool = True,
        device: str | torch.device | None = None,
        **kwargs: Any,
    ) -> ModelB:
        r"""Constructs a ModelB instance and loads weights from a checkpoint or state dict."""
        model = cls(**kwargs)
        return load_pretrained_bmshj2018(model=model, checkpoint=checkpoint, strict=strict, device=device)
