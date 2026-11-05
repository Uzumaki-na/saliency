"""Integer rANS (Asymmetric Numeral Systems) range coding core.

Provides lossless entropy serialization and deserialization for discrete latent symbols:
- Hyper-latents q_z via EntropyBottleneck CDF tables.
- Main latents q_y via GaussianConditional discrete scale index CDF tables.

Supports dual execution backends:
1. `constriction`: High-performance pre-compiled C/Rust rANS stack coder (default).
2. `python`: Fully standalone, pure-Python 16-bit integer rANS coder fallback with zero dependencies.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import torch

try:
    import constriction

    HAS_CONSTRICTION = True
except ImportError:
    constriction = None
    HAS_CONSTRICTION = False

if TYPE_CHECKING:
    from src.codec.entropy_models import EntropyBottleneck, GaussianConditional


class ConstrictionRAnsCoder:
    r"""High-speed integer rANS range coder utilizing the constriction library."""

    @staticmethod
    def _get_models(
        cdfs: np.ndarray,
        cdf_lengths: np.ndarray,
    ) -> list[constriction.stream.model.Categorical]:
        r"""Builds list of Categorical distribution models from CDF frequency arrays."""
        models: list[constriction.stream.model.Categorical] = []
        for k in range(len(cdf_lengths)):
            length = int(cdf_lengths[k])
            cdf_row = cdfs[k, :length]
            freqs = cdf_row[1:] - cdf_row[:-1]
            probs = freqs.astype(np.float32) / 65536.0
            models.append(constriction.stream.model.Categorical(probs, perfect=False))
        return models

    @classmethod
    def encode(
        cls,
        symbols: np.ndarray,
        indexes: np.ndarray,
        cdfs: np.ndarray,
        cdf_lengths: np.ndarray,
        offsets: np.ndarray,
    ) -> bytes:
        r"""Encodes an array of integer symbols into compressed bytes using constriction."""
        if not HAS_CONSTRICTION:
            raise RuntimeError("constriction library is not available.")

        models = cls._get_models(cdfs, cdf_lengths)
        sym_flat = symbols.astype(np.int32).reshape(-1)
        idx_flat = indexes.astype(np.int32).reshape(-1)

        if len(sym_flat) != len(idx_flat):
            raise ValueError(f"Symbols length ({len(sym_flat)}) does not match indexes length ({len(idx_flat)})")

        if len(sym_flat) == 0:
            return b""

        coder = constriction.stream.stack.AnsCoder()

        # rANS encodes in reverse (LIFO stack property)
        for s, k in zip(reversed(sym_flat), reversed(idx_flat)):
            off = int(offsets[k])
            sym_idx = int(s - off)
            num_syms = int(cdf_lengths[k] - 1)
            if sym_idx < 0 or sym_idx >= num_syms:
                raise ValueError(f"Symbol {s} out of support range [{off}, {off + num_syms - 1}] for distribution {k}")
            coder.encode_reverse(np.array([sym_idx], dtype=np.int32), models[k])

        compressed = coder.get_compressed()
        return compressed.tobytes()

    @classmethod
    def decode(
        cls,
        bitstream: bytes,
        indexes: np.ndarray,
        cdfs: np.ndarray,
        cdf_lengths: np.ndarray,
        offsets: np.ndarray,
    ) -> np.ndarray:
        r"""Decodes compressed bytes back into original integer symbols using constriction."""
        if not HAS_CONSTRICTION:
            raise RuntimeError("constriction library is not available.")

        models = cls._get_models(cdfs, cdf_lengths)
        idx_flat = indexes.astype(np.int32).reshape(-1)
        total_symbols = len(idx_flat)

        if total_symbols == 0:
            return np.empty(indexes.shape, dtype=np.int32)

        words = np.frombuffer(bitstream, dtype=np.uint32)
        decoder = constriction.stream.stack.AnsCoder(words)

        decoded = np.empty(total_symbols, dtype=np.int32)
        for i, k in enumerate(idx_flat):
            off = int(offsets[k])
            sym_idx = int(decoder.decode(models[k], 1)[0])
            decoded[i] = sym_idx + off

        return decoded.reshape(indexes.shape)


class PurePythonRAnsCoder:
    r"""Pure-Python 16-bit precision integer rANS (asymmetric numeral systems) coder.

    Guarantees cross-platform execution on any environment without external C/Rust dependencies.
    State precision: 16-bit frequency precision (M = 65536), 16-bit word normalization (L = 65536).
    """

    PRECISION: int = 16
    M: int = 1 << 16  # 65536
    L: int = 1 << 16  # 65536

    @classmethod
    def encode(
        cls,
        symbols: np.ndarray,
        indexes: np.ndarray,
        cdfs: np.ndarray,
        cdf_lengths: np.ndarray,
        offsets: np.ndarray,
    ) -> bytes:
        r"""Encodes an array of integer symbols into compressed bytes using pure Python."""
        sym_flat = symbols.astype(np.int32).reshape(-1)
        idx_flat = indexes.astype(np.int32).reshape(-1)

        if len(sym_flat) != len(idx_flat):
            raise ValueError(f"Symbols length ({len(sym_flat)}) does not match indexes length ({len(idx_flat)})")

        if len(sym_flat) == 0:
            return b""

        state = cls.L
        words: list[int] = []

        # rANS encodes in reverse order
        for s, k in zip(reversed(sym_flat), reversed(idx_flat)):
            off = int(offsets[k])
            sym_idx = int(s - off)
            num_syms = int(cdf_lengths[k] - 1)
            if sym_idx < 0 or sym_idx >= num_syms:
                raise ValueError(f"Symbol {s} out of support range [{off}, {off + num_syms - 1}] for distribution {k}")

            cdf_row = cdfs[k]
            start = int(cdf_row[sym_idx])
            freq = int(cdf_row[sym_idx + 1] - start)

            # Renormalization: emit 16-bit words while state would overflow
            # Condition: state >= (L >> PRECISION) * freq * M
            # Since L == M == 65536 and PRECISION == 16, (L >> 16) == 1, threshold is freq * 65536
            threshold = freq * cls.M
            while state >= threshold:
                words.append(state & 0xFFFF)
                state >>= 16

            # State transition
            state = (state // freq) * cls.M + start + (state % freq)

        # Flush final 64-bit state as four 16-bit words
        words.append(state & 0xFFFF)
        words.append((state >> 16) & 0xFFFF)
        words.append((state >> 32) & 0xFFFF)
        words.append((state >> 48) & 0xFFFF)

        return struct.pack(f"<{len(words)}H", *words)

    @classmethod
    def decode(
        cls,
        bitstream: bytes,
        indexes: np.ndarray,
        cdfs: np.ndarray,
        cdf_lengths: np.ndarray,
        offsets: np.ndarray,
    ) -> np.ndarray:
        r"""Decodes compressed bytes back into original integer symbols using pure Python."""
        idx_flat = indexes.astype(np.int32).reshape(-1)
        total_symbols = len(idx_flat)

        if total_symbols == 0:
            return np.empty(indexes.shape, dtype=np.int32)

        if len(bitstream) < 8:
            raise ValueError(
                f"Corrupted or truncated bitstream: length {len(bitstream)} bytes is smaller than 8-byte state header"
            )

        num_words = len(bitstream) // 2
        words = list(struct.unpack(f"<{num_words}H", bitstream))

        # Restore final state
        s3 = words.pop()
        s2 = words.pop()
        s1 = words.pop()
        s0 = words.pop()
        state = (s3 << 48) | (s2 << 32) | (s1 << 16) | s0

        decoded = np.empty(total_symbols, dtype=np.int32)

        for i, k in enumerate(idx_flat):
            off = int(offsets[k])
            length = int(cdf_lengths[k])
            cdf_row = cdfs[k, :length]

            # Invert cumulative frequency
            slot = state % cls.M
            # Find symbol where cdf[sym_idx] <= slot < cdf[sym_idx + 1]
            sym_idx = int(np.searchsorted(cdf_row, slot, side="right") - 1)
            start = int(cdf_row[sym_idx])
            freq = int(cdf_row[sym_idx + 1] - start)

            # Invert state transition
            state = freq * (state // cls.M) + (slot - start)

            # Renormalize: read 16-bit words from stream while state < L
            while state < cls.L and words:
                state = (state << 16) | words.pop()

            decoded[i] = sym_idx + off

        return decoded.reshape(indexes.shape)


class RAnsEncoder:
    r"""Unified interface for integer rANS encoding."""

    def __init__(self, backend: str = "auto") -> None:
        if backend == "auto":
            self.backend = "constriction" if HAS_CONSTRICTION else "python"
        elif backend in ("constriction", "python"):
            self.backend = backend
        else:
            raise ValueError(f"Unknown backend '{backend}'. Expected 'auto', 'constriction', or 'python'.")

    def encode_with_indexes(
        self,
        symbols: np.ndarray | torch.Tensor,
        indexes: np.ndarray | torch.Tensor,
        cdfs: np.ndarray | torch.Tensor,
        cdf_lengths: np.ndarray | torch.Tensor,
        offsets: np.ndarray | torch.Tensor,
    ) -> bytes:
        r"""Encodes symbol array with per-symbol distribution indices."""
        sym_np = symbols.cpu().numpy() if isinstance(symbols, torch.Tensor) else np.asarray(symbols)
        idx_np = indexes.cpu().numpy() if isinstance(indexes, torch.Tensor) else np.asarray(indexes)
        cdfs_np = cdfs.cpu().numpy() if isinstance(cdfs, torch.Tensor) else np.asarray(cdfs)
        lens_np = cdf_lengths.cpu().numpy() if isinstance(cdf_lengths, torch.Tensor) else np.asarray(cdf_lengths)
        offs_np = offsets.cpu().numpy() if isinstance(offsets, torch.Tensor) else np.asarray(offsets)

        if self.backend == "constriction":
            return ConstrictionRAnsCoder.encode(sym_np, idx_np, cdfs_np, lens_np, offs_np)
        return PurePythonRAnsCoder.encode(sym_np, idx_np, cdfs_np, lens_np, offs_np)


class RAnsDecoder:
    r"""Unified interface for integer rANS decoding."""

    def __init__(self, backend: str = "auto") -> None:
        if backend == "auto":
            self.backend = "constriction" if HAS_CONSTRICTION else "python"
        elif backend in ("constriction", "python"):
            self.backend = backend
        else:
            raise ValueError(f"Unknown backend '{backend}'. Expected 'auto', 'constriction', or 'python'.")

    def decode_with_indexes(
        self,
        bitstream: bytes,
        indexes: np.ndarray | torch.Tensor,
        cdfs: np.ndarray | torch.Tensor,
        cdf_lengths: np.ndarray | torch.Tensor,
        offsets: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        r"""Decodes compressed bitstream given distribution indices."""
        idx_np = indexes.cpu().numpy() if isinstance(indexes, torch.Tensor) else np.asarray(indexes)
        cdfs_np = cdfs.cpu().numpy() if isinstance(cdfs, torch.Tensor) else np.asarray(cdfs)
        lens_np = cdf_lengths.cpu().numpy() if isinstance(cdf_lengths, torch.Tensor) else np.asarray(cdf_lengths)
        offs_np = offsets.cpu().numpy() if isinstance(offsets, torch.Tensor) else np.asarray(offsets)

        if self.backend == "constriction":
            return ConstrictionRAnsCoder.decode(bitstream, idx_np, cdfs_np, lens_np, offs_np)
        return PurePythonRAnsCoder.decode(bitstream, idx_np, cdfs_np, lens_np, offs_np)


def encode_hyper_latents(
    q_z: torch.Tensor,
    entropy_bottleneck: EntropyBottleneck,
    backend: str = "auto",
) -> bytes:
    r"""Encodes quantized hyper-latents q_z into compressed bytes."""
    if entropy_bottleneck._quantized_cdf.numel() == 0:
        entropy_bottleneck.update()

    # Generate channel index tensor matching q_z
    if q_z.dim() == 4:
        c = q_z.shape[1]
        indices = (
            torch.arange(c, dtype=torch.int32, device="cpu").unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(q_z)
        )
    elif q_z.dim() == 3:
        c = q_z.shape[0]
        indices = torch.arange(c, dtype=torch.int32, device="cpu").unsqueeze(-1).unsqueeze(-1).expand_as(q_z)
    elif q_z.dim() == 2:
        c = q_z.shape[0]
        indices = torch.arange(c, dtype=torch.int32, device="cpu").unsqueeze(-1).expand_as(q_z)
    else:
        c = q_z.shape[0]
        indices = torch.arange(c, dtype=torch.int32, device="cpu").expand_as(q_z)

    encoder = RAnsEncoder(backend=backend)
    return encoder.encode_with_indexes(
        symbols=torch.round(q_z).to(torch.int32),
        indexes=indices,
        cdfs=entropy_bottleneck._quantized_cdf,
        cdf_lengths=entropy_bottleneck._cdf_length,
        offsets=entropy_bottleneck._offset,
    )


def decode_hyper_latents(
    bitstream: bytes,
    shape: tuple[int, ...] | torch.Size,
    entropy_bottleneck: EntropyBottleneck,
    backend: str = "auto",
) -> torch.Tensor:
    r"""Decodes compressed bytes back into hyper-latents tensor of given shape."""
    if entropy_bottleneck._quantized_cdf.numel() == 0:
        entropy_bottleneck.update()

    dummy = torch.empty(shape, dtype=torch.float32)

    if len(shape) == 4:
        c = shape[1]
        indices = (
            torch.arange(c, dtype=torch.int32, device="cpu").unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(dummy)
        )
    elif len(shape) == 3:
        c = shape[0]
        indices = torch.arange(c, dtype=torch.int32, device="cpu").unsqueeze(-1).unsqueeze(-1).expand_as(dummy)
    elif len(shape) == 2:
        c = shape[0]
        indices = torch.arange(c, dtype=torch.int32, device="cpu").unsqueeze(-1).expand_as(dummy)
    else:
        c = shape[0]
        indices = torch.arange(c, dtype=torch.int32, device="cpu").expand_as(dummy)

    decoder = RAnsDecoder(backend=backend)
    decoded_np = decoder.decode_with_indexes(
        bitstream=bitstream,
        indexes=indices,
        cdfs=entropy_bottleneck._quantized_cdf,
        cdf_lengths=entropy_bottleneck._cdf_length,
        offsets=entropy_bottleneck._offset,
    )
    return torch.from_numpy(decoded_np).float()


def encode_latents(
    q_y: torch.Tensor,
    scale_indexes: torch.Tensor,
    gaussian_conditional: GaussianConditional,
    backend: str = "auto",
) -> bytes:
    r"""Encodes quantized latents q_y using discrete scale table indices."""
    if gaussian_conditional._quantized_cdf.numel() == 0:
        gaussian_conditional.update()

    encoder = RAnsEncoder(backend=backend)
    return encoder.encode_with_indexes(
        symbols=torch.round(q_y).to(torch.int32),
        indexes=scale_indexes.to(torch.int32),
        cdfs=gaussian_conditional._quantized_cdf,
        cdf_lengths=gaussian_conditional._cdf_length,
        offsets=gaussian_conditional._offset,
    )


def decode_latents(
    bitstream: bytes,
    scale_indexes: torch.Tensor,
    gaussian_conditional: GaussianConditional,
    backend: str = "auto",
) -> torch.Tensor:
    r"""Decodes compressed bytes back into latents matching scale_indexes.shape."""
    if gaussian_conditional._quantized_cdf.numel() == 0:
        gaussian_conditional.update()

    decoder = RAnsDecoder(backend=backend)
    decoded_np = decoder.decode_with_indexes(
        bitstream=bitstream,
        indexes=scale_indexes.to(torch.int32),
        cdfs=gaussian_conditional._quantized_cdf,
        cdf_lengths=gaussian_conditional._cdf_length,
        offsets=gaussian_conditional._offset,
    )
    return torch.from_numpy(decoded_np).float().to(device=scale_indexes.device)
