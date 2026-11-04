"""Codec package providing neural compression transforms and layers.

Exports:
- LowerBoundFunction, LowerBound, NonNegativeParametrizer: Bound operators.
- GDN, IGDN: Non-linear normalization layers.
- AnalysisTransform (g_a), SynthesisTransform (g_s): Main codec transforms.
- HyperAnalysisTransform (h_a), HyperSynthesisTransform (h_s): Hyperprior transforms.
- ModelOutput, ModelA: Baseline compression model and output container.
"""

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

__all__ = [
    "GDN",
    "IGDN",
    "AnalysisTransform",
    "HyperAnalysisTransform",
    "HyperSynthesisTransform",
    "LowerBound",
    "LowerBoundFunction",
    "ModelA",
    "ModelOutput",
    "NonNegativeParametrizer",
    "SynthesisTransform",
]
