"""Codec package providing neural compression transforms and layers.

Exports:
- LowerBoundFunction, LowerBound, NonNegativeParametrizer: Bound operators.
- GDN, IGDN: Non-linear normalization layers.
- AnalysisTransform (g_a), SynthesisTransform (g_s): Main codec transforms.
- HyperAnalysisTransform (h_a), HyperSynthesisTransform (h_s): Hyperprior transforms.
- ModelOutput, ModelA: Baseline compression model and output container.
- ModelB: Saliency-weighted implicit allocation baseline model.
- ModelC, ModelCOutput, DualHeadHyperSynthesisTransform: Explicit adaptive allocation model.
- ModelD: Mechanism-only architecture control variant.
- load_pretrained_bmshj2018, remap_compressai_state_dict: Pretrained weight loading utilities.
- compute_log_delta, compute_effective_log_scale, quantize_ste_1_32, build_delta_table,
  lookup_delta_from_table, quantize_latents_adaptive, DELTA_TABLE_129: Saliency allocation engine.
- compute_hyperprior_overhead: Hyperprior bitrate overhead Delta BPP_z calculation.
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
    MODEL_ID_A,
    AnalysisTransform,
    HyperAnalysisTransform,
    HyperSynthesisTransform,
    ModelA,
    ModelOutput,
    SynthesisTransform,
)
from src.codec.model_b import (
    MODEL_ID_B,
    ModelB,
    load_pretrained_bmshj2018,
    remap_compressai_state_dict,
)
from src.codec.model_c import (
    MODEL_ID_C,
    DualHeadHyperSynthesisTransform,
    ModelC,
    ModelCOutput,
)
from src.codec.model_c_transmitted import (
    MODEL_ID_C_TRANSMITTED,
    ModelCTransmitted,
    ModelCTransmittedOutput,
    compress_saliency_thumbnail,
    compute_sidecar_bpp,
    decompress_saliency_thumbnail,
)
from src.codec.model_d import (
    MODEL_ID_D,
    ModelD,
    compute_hyperprior_overhead,
)

__all__ = [
    "DELTA_TABLE_129",
    "GDN",
    "IGDN",
    "MODEL_ID_A",
    "MODEL_ID_B",
    "MODEL_ID_C",
    "MODEL_ID_C_TRANSMITTED",
    "MODEL_ID_D",
    "AnalysisTransform",
    "DualHeadHyperSynthesisTransform",
    "EntropyBottleneck",
    "GaussianConditional",
    "HyperAnalysisTransform",
    "HyperSynthesisTransform",
    "LowerBound",
    "LowerBoundFunction",
    "ModelA",
    "ModelB",
    "ModelC",
    "ModelCOutput",
    "ModelCTransmitted",
    "ModelCTransmittedOutput",
    "ModelD",
    "ModelOutput",
    "NonNegativeParametrizer",
    "STE1Over32Quantizer",
    "SynthesisTransform",
    "build_delta_table",
    "compress_saliency_thumbnail",
    "compute_effective_log_scale",
    "compute_hyperprior_overhead",
    "compute_log_delta",
    "compute_sidecar_bpp",
    "decompress_saliency_thumbnail",
    "load_pretrained_bmshj2018",
    "lookup_delta_from_table",
    "quantize_latents_adaptive",
    "quantize_ste_1_32",
    "remap_compressai_state_dict",
]
