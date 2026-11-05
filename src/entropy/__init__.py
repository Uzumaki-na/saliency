"""Entropy engine package for deterministic cross-device entropy coding.

Exports:
- DeterministicCPUEngine: Single-threaded strict FP32 CPU entropy parameter execution engine.
- configure_cpu_determinism: Configuration helper for bit-exact CPU inference.
"""

from src.entropy.cpu_engine import (
    DeterministicCPUEngine,
    configure_cpu_determinism,
)

__all__ = [
    "DeterministicCPUEngine",
    "configure_cpu_determinism",
]

