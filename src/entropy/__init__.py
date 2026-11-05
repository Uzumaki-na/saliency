"""Entropy engine package for deterministic cross-device entropy coding.

Exports:
- DeterministicCPUEngine: Single-threaded strict FP32 CPU entropy parameter execution engine.
- configure_cpu_determinism: Configuration helper for bit-exact CPU inference.
- ConstrictionRAnsCoder: High-speed rANS range coder utilizing the constriction library.
- PurePythonRAnsCoder: Pure-Python 16-bit integer rANS coder fallback.
- RAnsEncoder: Unified integer rANS encoder supporting auto/constriction/python backends.
- RAnsDecoder: Unified integer rANS decoder supporting auto/constriction/python backends.
- encode_hyper_latents: Entropy-codes hyper-latent tensors q_z via EntropyBottleneck.
- decode_hyper_latents: Entropy-decodes hyper-latent tensors q_z via EntropyBottleneck.
- encode_latents: Entropy-codes main latent tensors q_y via GaussianConditional.
- decode_latents: Entropy-decodes main latent tensors q_y via GaussianConditional.
"""

from src.entropy.coder import (
    ConstrictionRAnsCoder,
    PurePythonRAnsCoder,
    RAnsDecoder,
    RAnsEncoder,
    decode_hyper_latents,
    decode_latents,
    encode_hyper_latents,
    encode_latents,
)
from src.entropy.cpu_engine import (
    DeterministicCPUEngine,
    configure_cpu_determinism,
)

__all__ = [
    "ConstrictionRAnsCoder",
    "DeterministicCPUEngine",
    "PurePythonRAnsCoder",
    "RAnsDecoder",
    "RAnsEncoder",
    "configure_cpu_determinism",
    "decode_hyper_latents",
    "decode_latents",
    "encode_hyper_latents",
    "encode_latents",
]
