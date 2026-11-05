"""Unit tests for integer rANS range coding core (Story 2.4).

Verifies:
1. Lossless symbol recovery across 100,000 random integer symbols (max |q - hat_q| == 0).
2. Dual backend roundtrip (constriction and pure Python).
3. 4D hyper-latent tensor encode/decode with EntropyBottleneck.
4. 4D main latent tensor encode/decode with GaussianConditional and scale indices.
5. Out-of-alphabet exception handling.
6. Empty tensor edge cases.
7. Compression efficiency on low-entropy distributions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.codec.entropy_models import EntropyBottleneck, GaussianConditional
from src.entropy.coder import (
    HAS_CONSTRICTION,
    ConstrictionRAnsCoder,
    PurePythonRAnsCoder,
    RAnsDecoder,
    RAnsEncoder,
    decode_hyper_latents,
    decode_latents,
    encode_hyper_latents,
    encode_latents,
)


@pytest.fixture
def compiled_entropy_bottleneck() -> EntropyBottleneck:
    r"""Fixture providing an updated EntropyBottleneck with compiled integer CDFs."""
    eb = EntropyBottleneck(channels=128)
    eb.update()
    return eb


@pytest.fixture
def compiled_gaussian_conditional() -> GaussianConditional:
    r"""Fixture providing an updated GaussianConditional with compiled integer CDFs."""
    gc = GaussianConditional()
    gc.update()
    return gc


class TestRAnsCoderDirect:
    r"""Direct tests for rANS coder backends."""

    def test_pure_python_coder_lossless_roundtrip(self) -> None:
        r"""Verifies PurePythonRAnsCoder recovers integer symbols with 100% precision."""
        rng = np.random.default_rng(42)
        k = 4
        cdfs = np.zeros((k, 16), dtype=np.int32)
        for i in range(k):
            raw_freqs = rng.integers(1, 100, size=8)
            freqs = np.round(raw_freqs / raw_freqs.sum() * 65536).astype(np.int32)
            freqs[freqs < 1] = 1
            freqs[-1] += 65536 - freqs.sum()
            cdfs[i, :9] = np.pad(np.cumsum(freqs), (1, 0))

        cdf_lengths = np.full(k, 9, dtype=np.int32)
        offsets = np.zeros(k, dtype=np.int32)

        # Generate 2,000 symbols
        n_syms = 2000
        indexes = rng.integers(0, k, size=n_syms, dtype=np.int32)
        symbols = rng.integers(0, 8, size=n_syms, dtype=np.int32)

        bitstream = PurePythonRAnsCoder.encode(symbols, indexes, cdfs, cdf_lengths, offsets)
        assert isinstance(bitstream, bytes)
        assert len(bitstream) > 0

        decoded = PurePythonRAnsCoder.decode(bitstream, indexes, cdfs, cdf_lengths, offsets)
        np.testing.assert_array_equal(symbols, decoded)

    @pytest.mark.skipif(not HAS_CONSTRICTION, reason="constriction library not installed")
    def test_constriction_coder_lossless_roundtrip(self) -> None:
        r"""Verifies ConstrictionRAnsCoder recovers integer symbols with 100% precision."""
        rng = np.random.default_rng(42)
        k = 4
        cdfs = np.zeros((k, 16), dtype=np.int32)
        for i in range(k):
            raw_freqs = rng.integers(1, 100, size=8)
            freqs = np.round(raw_freqs / raw_freqs.sum() * 65536).astype(np.int32)
            freqs[freqs < 1] = 1
            freqs[-1] += 65536 - freqs.sum()
            cdfs[i, :9] = np.pad(np.cumsum(freqs), (1, 0))

        cdf_lengths = np.full(k, 9, dtype=np.int32)
        offsets = np.zeros(k, dtype=np.int32)

        # Generate 2,000 symbols
        n_syms = 2000
        indexes = rng.integers(0, k, size=n_syms, dtype=np.int32)
        symbols = rng.integers(0, 8, size=n_syms, dtype=np.int32)

        bitstream = ConstrictionRAnsCoder.encode(symbols, indexes, cdfs, cdf_lengths, offsets)
        assert isinstance(bitstream, bytes)
        assert len(bitstream) > 0

        decoded = ConstrictionRAnsCoder.decode(bitstream, indexes, cdfs, cdf_lengths, offsets)
        np.testing.assert_array_equal(symbols, decoded)


class TestRAnsLargeScale:
    r"""Large-scale 100,000 symbol lossless benchmark."""

    @pytest.mark.parametrize("backend", ["python"] + (["constriction"] if HAS_CONSTRICTION else []))
    def test_100k_symbols_lossless_recovery(
        self,
        compiled_gaussian_conditional: GaussianConditional,
        backend: str,
    ) -> None:
        r"""Encodes and decodes 100,000 random integer symbols across 64 scale distributions."""
        gc = compiled_gaussian_conditional
        cdfs = gc._quantized_cdf.cpu().numpy()
        lengths = gc._cdf_length.cpu().numpy()
        offsets = gc._offset.cpu().numpy()
        num_distribs = cdfs.shape[0]

        rng = np.random.default_rng(12345)
        n_symbols = 100000

        # Select random distribution indices
        indexes = rng.integers(0, num_distribs, size=n_symbols, dtype=np.int32)

        # Sample valid symbols within support [offset, offset + length - 2]
        symbols = np.empty(n_symbols, dtype=np.int32)
        for i in range(n_symbols):
            idx = indexes[i]
            max_sym = lengths[idx] - 2  # num symbols = lengths - 1
            symbols[i] = rng.integers(0, max_sym) + offsets[idx]

        encoder = RAnsEncoder(backend=backend)
        decoder = RAnsDecoder(backend=backend)

        bitstream = encoder.encode_with_indexes(symbols, indexes, cdfs, lengths, offsets)
        assert isinstance(bitstream, bytes)
        assert len(bitstream) > 0

        decoded = decoder.decode_with_indexes(bitstream, indexes, cdfs, lengths, offsets)

        # Acceptance criterion: max |q - hat_q| == 0
        diff = np.abs(symbols - decoded)
        assert np.max(diff) == 0, f"Max difference {np.max(diff)} across {n_symbols} symbols"


class TestHyperLatentCoding:
    r"""Tests for hyper-latent encode/decode routines with EntropyBottleneck."""

    @pytest.mark.parametrize("backend", ["python"] + (["constriction"] if HAS_CONSTRICTION else []))
    def test_encode_decode_hyper_latents_4d(
        self,
        compiled_entropy_bottleneck: EntropyBottleneck,
        backend: str,
    ) -> None:
        r"""Encodes and decodes a 4D batch of hyper-latents (B, C, H, W)."""
        eb = compiled_entropy_bottleneck
        b, c, h, w = 2, 128, 8, 8

        # Quantized integer hyper-latents near 0
        rng = np.random.default_rng(999)
        # Offset bounds for each channel
        offsets = eb._offset.cpu().numpy()
        lengths = eb._cdf_length.cpu().numpy()

        q_z_np = np.zeros((b, c, h, w), dtype=np.int32)
        for ch in range(c):
            max_val = lengths[ch] - 2
            off = offsets[ch]
            # Center around 0, safely clamped to alphabet
            low = max(off, -5)
            high = min(off + max_val, 5)
            q_z_np[:, ch, :, :] = rng.integers(low, high + 1, size=(b, h, w))

        q_z = torch.from_numpy(q_z_np).float()

        bitstream = encode_hyper_latents(q_z, eb, backend=backend)
        assert isinstance(bitstream, bytes)
        assert len(bitstream) > 0

        decoded_q_z = decode_hyper_latents(bitstream, q_z.shape, eb, backend=backend)

        assert decoded_q_z.shape == q_z.shape
        diff = torch.abs(q_z - decoded_q_z)
        assert torch.max(diff).item() == 0.0

    @pytest.mark.parametrize("backend", ["python"] + (["constriction"] if HAS_CONSTRICTION else []))
    def test_encode_decode_hyper_latents_3d(
        self,
        compiled_entropy_bottleneck: EntropyBottleneck,
        backend: str,
    ) -> None:
        r"""Encodes and decodes a 3D batch of hyper-latents (C, H, W)."""
        eb = compiled_entropy_bottleneck
        c, h, w = 128, 4, 4

        rng = np.random.default_rng(1001)
        offsets = eb._offset.cpu().numpy()
        lengths = eb._cdf_length.cpu().numpy()

        q_z_np = np.zeros((c, h, w), dtype=np.int32)
        for ch in range(c):
            max_val = lengths[ch] - 2
            off = offsets[ch]
            low = max(off, -3)
            high = min(off + max_val, 3)
            q_z_np[ch, :, :] = rng.integers(low, high + 1, size=(h, w))

        q_z = torch.from_numpy(q_z_np).float()

        bitstream = encode_hyper_latents(q_z, eb, backend=backend)
        decoded = decode_hyper_latents(bitstream, q_z.shape, eb, backend=backend)

        assert decoded.shape == q_z.shape
        assert torch.max(torch.abs(q_z - decoded)).item() == 0.0


class TestMainLatentCoding:
    r"""Tests for main latent encode/decode routines with GaussianConditional."""

    @pytest.mark.parametrize("backend", ["python"] + (["constriction"] if HAS_CONSTRICTION else []))
    def test_encode_decode_main_latents_4d(
        self,
        compiled_gaussian_conditional: GaussianConditional,
        backend: str,
    ) -> None:
        r"""Encodes and decodes 4D main latents (B, C, H, W) with per-element scale indexes."""
        gc = compiled_gaussian_conditional
        b, c, h, w = 2, 192, 8, 8

        rng = np.random.default_rng(2026)
        scale_indexes = torch.randint(0, 64, size=(b, c, h, w), dtype=torch.int32)

        # Generate symbols within support of their corresponding scale index
        lengths = gc._cdf_length.cpu().numpy()
        offsets = gc._offset.cpu().numpy()

        q_y_np = np.zeros((b, c, h, w), dtype=np.int32)
        idx_np = scale_indexes.numpy()
        for i in range(b):
            for j in range(c):
                for row in range(h):
                    for col in range(w):
                        k = idx_np[i, j, row, col]
                        max_sym = lengths[k] - 2
                        off = offsets[k]
                        low = max(off, -4)
                        high = min(off + max_sym, 4)
                        q_y_np[i, j, row, col] = rng.integers(low, high + 1)

        q_y = torch.from_numpy(q_y_np).float()

        bitstream = encode_latents(q_y, scale_indexes, gc, backend=backend)
        assert isinstance(bitstream, bytes)
        assert len(bitstream) > 0

        decoded_q_y = decode_latents(bitstream, scale_indexes, gc, backend=backend)

        assert decoded_q_y.shape == q_y.shape
        assert torch.max(torch.abs(q_y - decoded_q_y)).item() == 0.0


class TestEdgeCasesAndErrors:
    r"""Tests for error handling and edge cases."""

    def test_out_of_alphabet_symbol_raises(
        self,
        compiled_entropy_bottleneck: EntropyBottleneck,
    ) -> None:
        r"""Asserts that symbols outside the alphabet support raise ValueError."""
        eb = compiled_entropy_bottleneck
        c = 128
        # Create an out-of-range symbol
        q_z = torch.zeros(1, c, 1, 1, dtype=torch.float32)
        # Put an extreme outlier symbol in channel 0
        q_z[0, 0, 0, 0] = 99999.0

        with pytest.raises(ValueError, match="out of support range"):
            encode_hyper_latents(q_z, eb, backend="python")

        if HAS_CONSTRICTION:
            with pytest.raises(ValueError, match="out of support range"):
                encode_hyper_latents(q_z, eb, backend="constriction")

    def test_empty_tensor_handling(
        self,
        compiled_entropy_bottleneck: EntropyBottleneck,
    ) -> None:
        r"""Asserts that empty latent tensors encode and decode gracefully."""
        eb = compiled_entropy_bottleneck
        empty_q_z = torch.empty(1, 128, 0, 0, dtype=torch.float32)

        bitstream = encode_hyper_latents(empty_q_z, eb, backend="python")
        decoded = decode_hyper_latents(bitstream, empty_q_z.shape, eb, backend="python")
        assert decoded.shape == empty_q_z.shape

        if HAS_CONSTRICTION:
            bitstream = encode_hyper_latents(empty_q_z, eb, backend="constriction")
            decoded = decode_hyper_latents(bitstream, empty_q_z.shape, eb, backend="constriction")
            assert decoded.shape == empty_q_z.shape

    def test_mismatched_length_raises(self) -> None:
        r"""Asserts that unequal lengths of symbols and indexes raise ValueError."""
        symbols = np.array([1, 2, 3], dtype=np.int32)
        indexes = np.array([0, 0], dtype=np.int32)
        cdfs = np.zeros((1, 16), dtype=np.int32)
        cdfs[0, :3] = [0, 32768, 65536]
        lens = np.array([3], dtype=np.int32)
        offs = np.array([0], dtype=np.int32)

        with pytest.raises(ValueError, match="does not match indexes length"):
            PurePythonRAnsCoder.encode(symbols, indexes, cdfs, lens, offs)

        if HAS_CONSTRICTION:
            with pytest.raises(ValueError, match="does not match indexes length"):
                ConstrictionRAnsCoder.encode(symbols, indexes, cdfs, lens, offs)

    def test_truncated_bitstream_raises(self) -> None:
        r"""Asserts that decoding corrupted or truncated bitstreams raises ValueError."""
        indexes = np.array([0], dtype=np.int32)
        cdfs = np.zeros((1, 16), dtype=np.int32)
        cdfs[0, :3] = [0, 32768, 65536]
        lens = np.array([3], dtype=np.int32)
        offs = np.array([0], dtype=np.int32)

        with pytest.raises(ValueError, match="smaller than 8-byte state header"):
            PurePythonRAnsCoder.decode(b"\x01\x02", indexes, cdfs, lens, offs)

    def test_invalid_backend_raises(self) -> None:
        r"""Asserts that invalid backend names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            RAnsEncoder(backend="nonexistent")

        with pytest.raises(ValueError, match="Unknown backend"):
            RAnsDecoder(backend="nonexistent")

    def test_entropy_compression_efficiency(
        self,
        compiled_gaussian_conditional: GaussianConditional,
    ) -> None:
        r"""Verifies that low-entropy symbols compress significantly smaller than raw bytes."""
        gc = compiled_gaussian_conditional
        n_symbols = 10000

        # All zero symbols with low scale (index 0)
        symbols = torch.zeros(n_symbols, dtype=torch.int32)
        indexes = torch.zeros(n_symbols, dtype=torch.int32)

        encoder = RAnsEncoder(backend="python")
        bitstream = encoder.encode_with_indexes(
            symbols=symbols,
            indexes=indexes,
            cdfs=gc._quantized_cdf,
            cdf_lengths=gc._cdf_length,
            offsets=gc._offset,
        )

        raw_bytes = n_symbols * 4  # 40,000 bytes raw int32
        compressed_bytes = len(bitstream)

        # Compressed bitstream should achieve high compression ratio (> 5x smaller)
        assert compressed_bytes < raw_bytes / 5, (
            f"Expected high compression for constant symbols, got {compressed_bytes} bytes vs {raw_bytes} raw"
        )

