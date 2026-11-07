"""Codec package providing neural compression transforms and layers.

Exports:
- LowerBoundFunction, LowerBound, NonNegativeParametrizer: Bound operators.
- GDN, IGDN: Non-linear normalization layers.
- AnalysisTransform (g_a), SynthesisTransform (g_s): Main codec transforms.
- HyperAnalysisTransform (h_a), HyperSynthesisTransform (h_s): Hyperprior transforms.
- ModelOutput, ModelA: Baseline compression model and output container.
- ModelB: Saliency-weighted implicit allocation baseline model.
- load_pretrained_bmshj2018, remap_compressai_state_dict: Pretrained weight loading utilities.
- compute_log_delta, compute_effective_log_scale, quantize_ste_1_32, build_delta_table,
  lookup_delta_from_table, quantize_latents_adaptive, DELTA_TABLE_129: Saliency allocation engine.
"""

from src.codec.allocation import (
    DELTA_TABLE_129,
    STE1Over32Quantizer,
    build_delta_table,
    compute_effective_log_scale,
    compute_log_delta,
    lookup_delta_from_table,
    quantize_latents_adaptive,
    quantize_ste_1_32,
)
from src.codec.entropy_models import (
    EntropyBottleneck,
    GaussianConditional,
)
from src.codec.layers import (
    GDN,
    IGDN,
    LowerBound,
    LowerBoundFunction,
    NonNegativeParametrizer,
)
from src.codec.model_a import (
    AnalysisTransform,
    HyperAnalysisTransform,
    HyperSynthesisTransform,
    ModelA,
    ModelOutput,
    SynthesisTransform,
)
from src.codec.model_b import (
    ModelB,
    load_pretrained_bmshj2018,
    remap_compressai_state_dict,
)

__all__ = [
    "DELTA_TABLE_129",
    "GDN",
    "IGDN",
    "AnalysisTransform",
    "EntropyBottleneck",
    "GaussianConditional",
    "HyperAnalysisTransform",
    "HyperSynthesisTransform",
    "LowerBound",
    "LowerBoundFunction",
    "ModelA",
    "ModelB",
    "ModelOutput",
    "NonNegativeParametrizer",
    "STE1Over32Quantizer",
    "SynthesisTransform",
    "build_delta_table",
    "compute_effective_log_scale",
    "compute_log_delta",
    "load_pretrained_bmshj2018",
    "lookup_delta_from_table",
    "quantize_latents_adaptive",
    "quantize_ste_1_32",
    "remap_compressai_state_dict",
]
