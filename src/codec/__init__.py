"""Codec package providing neural compression transforms and layers.

Exports:
- LowerBoundFunction, LowerBound, NonNegativeParametrizer: Bound operators.
- GDN, IGDN: Non-linear normalization layers.
- AnalysisTransform (g_a), SynthesisTransform (g_s): Main codec transforms.
- HyperAnalysisTransform (h_a), HyperSynthesisTransform (h_s): Hyperprior transforms.
- ModelOutput, ModelA: Baseline compression model and output container.
- ModelB: Saliency-weighted implicit allocation baseline model.
- load_pretrained_bmshj2018, remap_compressai_state_dict: Pretrained weight loading utilities.
"""

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
    "SynthesisTransform",
    "load_pretrained_bmshj2018",
    "remap_compressai_state_dict",
]
