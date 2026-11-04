"""Unit tests for centered symmetric reflection padding and valid latent mask computation.

Verifies:
  - Kodak benchmark dimensions (768x512 and 512x768): zero padding, all-true mask.
  - MIT1003 benchmark dimensions (500x375): exact pad_bounds (6, 6, 4, 5), (32, 24) latent mask.
  - Exact bit-identical recovery across 2D, 3D, and 4D tensors.
  - Device and dtype preservation (float32, float64, float16, bfloat16, int32, int64, uint8, bool).
  - Iterative reflection padding with replication fallback for tiny inputs (1x1, 10x10).
  - Paired image and saliency padding with identical margins and unpadding recovery.
  - Latent validity mask boundary precision using receptive field centers (16i + 8, 16j + 8).
  - Custom factor (32, 16) and custom stride (8, 32) support.
  - Parameter validation and clean error handling for negative/zero factor, stride, and dimensions.
"""

from __future__ import annotations

import pytest
import torch

from src.data.padding import (
    PadBounds,
    compute_padding_offsets,
    compute_valid_latent_mask,
    pad_image_and_saliency_centered,
    pad_symmetric_centered,
    unpad_centered,
)

# ==============================================================================
# 1. Benchmark Dimensions: Kodak & MIT1003
# ==============================================================================


def test_kodak_dimensions_divisible_by_64() -> None:
    """Verify Kodak benchmark dimensions (768x512 and 512x768) require zero padding."""
    for h, w in [(768, 512), (512, 768)]:
        x = torch.randn(3, h, w, dtype=torch.float32)
        x_pad, bounds = pad_symmetric_centered(x, factor=64)

        assert bounds == (0, 0, 0, 0)
        assert bounds.pad_top == 0
        assert bounds.pad_bottom == 0
        assert bounds.pad_left == 0
        assert bounds.pad_right == 0
        assert x_pad.shape == (3, h, w)

        # Bit-identical recovery
        x_recovered = unpad_centered(x_pad, bounds)
        assert torch.equal(x, x_recovered)
        assert (x - x_recovered).abs().max().item() == 0.0

        # Latent valid mask must be all True
        expected_hl = h // 16
        expected_wl = w // 16
        mask = compute_valid_latent_mask(x_pad, bounds, stride=16)
        assert mask.shape == (expected_hl, expected_wl)
        assert mask.dtype == torch.bool
        assert torch.all(mask)
        assert mask.sum().item() == expected_hl * expected_wl


def test_mit1003_dimensions_odd_and_padded() -> None:
    """Verify MIT1003 benchmark dimensions (500x375) produce exact pad_bounds (6, 6, 4, 5)."""
    h_orig, w_orig = 500, 375
    x = torch.randn(3, h_orig, w_orig, dtype=torch.float32)
    x_pad, bounds = pad_symmetric_centered(x, factor=64)

    # 500 -> 512: delta=12 -> top=6, bottom=6
    # 375 -> 384: delta=9  -> left=4, right=5
    assert bounds == (6, 6, 4, 5)
    assert x_pad.shape == (3, 512, 384)

    # Bit-identical unpadding recovery
    x_recovered = unpad_centered(x_pad, bounds)
    assert torch.equal(x, x_recovered)
    assert (x - x_recovered).abs().max().item() == 0.0

    # Latent valid mask computation
    mask = compute_valid_latent_mask(x_pad, bounds, stride=16)
    assert mask.shape == (32, 24)
    assert mask.dtype == torch.bool

    # Verify receptive field centers (16i + 8, 16j + 8)
    # Valid interval: [6, 506) for height, [4, 379) for width
    for i in range(32):
        ci = 16 * i + 8
        valid_i = 6 <= ci < 506
        for j in range(24):
            cj = 16 * j + 8
            valid_j = 4 <= cj < 379
            expected = valid_i and valid_j
            assert mask[i, j].item() == expected, f"Mismatch at cell ({i}, {j}): ci={ci}, cj={cj}"


def test_mit1003_transposed_dimensions() -> None:
    """Verify transposed MIT1003 dimension (375x500) produces pad_bounds (4, 5, 6, 6)."""
    h_orig, w_orig = 375, 500
    x = torch.randn(3, h_orig, w_orig, dtype=torch.float32)
    x_pad, bounds = pad_symmetric_centered(x, factor=64)

    assert bounds == (4, 5, 6, 6)
    assert x_pad.shape == (3, 384, 512)

    x_recovered = unpad_centered(x_pad, bounds)
    assert torch.equal(x, x_recovered)

    mask = compute_valid_latent_mask(x_pad, bounds, stride=16)
    assert mask.shape == (24, 32)


# ==============================================================================
# 2. Exact Bit-Identical Unpadding Recovery Across Shapes
# ==============================================================================


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (2, 2),
        (10, 10),
        (1, 101, 103),
        (3, 127, 255),
        (3, 200, 300),
        (2, 3, 47, 89),
        (4, 1, 64, 64),
        (1, 3, 128, 192),
    ],
)
def test_unpad_bit_identical_recovery(shape: tuple[int, ...]) -> None:
    """Verify unpad_centered recovers the exact original tensor bit-identically across arbitrary shapes."""
    x = torch.randn(shape, dtype=torch.float32)
    x_pad, bounds = pad_symmetric_centered(x, factor=64)

    # Check padded shape is multiple of 64
    assert x_pad.shape[-2] % 64 == 0
    assert x_pad.shape[-1] % 64 == 0

    x_rec = unpad_centered(x_pad, bounds)
    assert x_rec.shape == x.shape
    assert torch.equal(x, x_rec)
    assert (x - x_rec).abs().max().item() == 0.0


def test_tensor_dimensionality_support() -> None:
    """Verify 2D (H, W), 3D (C, H, W), and 4D (B, C, H, W) tensors are supported and shapes preserved."""
    # 2D
    x2d = torch.randn(100, 150)
    p2d, b2d = pad_symmetric_centered(x2d, factor=64)
    assert p2d.ndim == 2
    assert p2d.shape == (128, 192)
    r2d = unpad_centered(p2d, b2d)
    assert r2d.shape == (100, 150)
    assert torch.equal(x2d, r2d)

    # 3D
    x3d = torch.randn(3, 100, 150)
    p3d, b3d = pad_symmetric_centered(x3d, factor=64)
    assert p3d.ndim == 3
    assert p3d.shape == (3, 128, 192)
    r3d = unpad_centered(p3d, b3d)
    assert r3d.shape == (3, 100, 150)
    assert torch.equal(x3d, r3d)

    # 4D
    x4d = torch.randn(2, 3, 100, 150)
    p4d, b4d = pad_symmetric_centered(x4d, factor=64)
    assert p4d.ndim == 4
    assert p4d.shape == (2, 3, 128, 192)
    r4d = unpad_centered(p4d, b4d)
    assert r4d.shape == (2, 3, 100, 150)
    assert torch.equal(x4d, r4d)


# ==============================================================================
# 3. Dtype and Device Preservation & Non-Mutation
# ==============================================================================


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float64,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    ],
)
def test_dtype_preservation(dtype: torch.dtype) -> None:
    """Verify padding and unpadding operations preserve original tensor dtype."""
    if dtype == torch.bool:
        x = torch.randint(0, 2, (3, 40, 50), dtype=torch.bool)
    elif dtype in (torch.int32, torch.int64, torch.uint8):
        x = torch.randint(0, 100, (3, 40, 50), dtype=dtype)
    else:
        x = torch.randn(3, 40, 50, dtype=dtype)

    x_pad, bounds = pad_symmetric_centered(x, factor=64)
    assert x_pad.dtype == dtype

    x_rec = unpad_centered(x_pad, bounds)
    assert x_rec.dtype == dtype
    assert torch.equal(x, x_rec)


def test_device_preservation() -> None:
    """Verify padding and unpadding operations preserve tensor device."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))

    for dev in devices:
        x = torch.randn(3, 50, 70, device=dev)
        x_pad, bounds = pad_symmetric_centered(x, factor=64)
        assert x_pad.device == dev

        x_rec = unpad_centered(x_pad, bounds)
        assert x_rec.device == dev
        assert torch.equal(x, x_rec)

        mask = compute_valid_latent_mask(x_pad, bounds, stride=16)
        assert mask.device == dev


def test_input_tensor_not_mutated() -> None:
    """Verify input tensors are never altered or mutated in-place during padding."""
    x = torch.randn(3, 50, 50, dtype=torch.float32)
    x_copy = x.clone()

    x_pad, _ = pad_symmetric_centered(x, factor=64)
    assert torch.equal(x, x_copy)

    # Mutating padded tensor does not affect original
    x_pad[0, 0, 0] += 999.0
    assert torch.equal(x, x_copy)


# ==============================================================================
# 4. Tiny Inputs & Iterative Reflection Fallback
# ==============================================================================


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1),
        (3, 1, 1),
        (3, 2, 2),
        (3, 5, 5),
        (3, 10, 10),
        (1, 3, 1, 1),
    ],
)
def test_tiny_inputs_iterative_reflection(shape: tuple[int, ...]) -> None:
    """Verify tiny tensors smaller than factor pad cleanly to 64x64 without crashing."""
    x = torch.randn(shape, dtype=torch.float32)
    x_pad, bounds = pad_symmetric_centered(x, factor=64)

    assert x_pad.shape[-2] == 64
    assert x_pad.shape[-1] == 64

    x_rec = unpad_centered(x_pad, bounds)
    assert x_rec.shape == x.shape
    assert torch.equal(x, x_rec)


# ==============================================================================
# 5. Paired Image and Saliency Padding
# ==============================================================================


def test_pad_image_and_saliency_centered() -> None:
    """Verify paired image and saliency padding applies identical margins and unpads bit-identically."""
    h, w = 500, 375
    x = torch.randn(3, h, w, dtype=torch.float32)
    s = torch.rand(1, h, w, dtype=torch.float32)

    x_pad, s_pad, bounds = pad_image_and_saliency_centered(x, s, factor=64)

    assert bounds == (6, 6, 4, 5)
    assert x_pad.shape == (3, 512, 384)
    assert s_pad.shape == (1, 512, 384)

    x_rec = unpad_centered(x_pad, bounds)
    s_rec = unpad_centered(s_pad, bounds)

    assert torch.equal(x, x_rec)
    assert torch.equal(s, s_rec)


def test_pad_image_and_saliency_dimension_mismatch() -> None:
    """Verify pad_image_and_saliency_centered raises ValueError when spatial dimensions do not match."""
    x = torch.randn(3, 100, 100)
    s = torch.randn(1, 100, 101)

    with pytest.raises(ValueError, match="Spatial dimensions of x and S must match"):
        pad_image_and_saliency_centered(x, s)


# ==============================================================================
# 6. Latent Validity Mask Boundary Precision & Custom Parameters
# ==============================================================================


def test_latent_mask_boundary_precision_with_large_padding() -> None:
    """Verify boundary cells are strictly False when padding exceeds half-stride (8 pixels)."""
    # Create a scenario where pad_top = 20, pad_bottom = 20, pad_left = 20, pad_right = 20
    # Original image: 24 x 24. Padded image: 64 x 64.
    # Latent grid at stride 16: 4 x 4.
    # Centers at stride 16: [8, 24, 40, 56].
    # Valid interval: [20, 44).
    # i=0: center=8  -> False (8 < 20)
    # i=1: center=24 -> True (20 <= 24 < 44)
    # i=2: center=40 -> True (20 <= 40 < 44)
    # i=3: center=56 -> False (56 >= 44)
    bounds = PadBounds(pad_top=20, pad_bottom=20, pad_left=20, pad_right=20)
    # Passed as unpadded shape (24, 24)
    mask = compute_valid_latent_mask((24, 24), bounds, stride=16)

    assert mask.shape == (4, 4)
    expected_mask = torch.tensor(
        [
            [False, False, False, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected_mask)

    # Check consistency with explicit padded_shape and is_padded=True
    mask_padded = compute_valid_latent_mask(padded_shape=(64, 64), pad_bounds=bounds, stride=16)
    mask_flag = compute_valid_latent_mask((64, 64), bounds, is_padded=True, stride=16)
    assert torch.equal(mask, mask_padded)
    assert torch.equal(mask, mask_flag)


def test_latent_mask_calling_conventions() -> None:
    """Verify compute_valid_latent_mask accepts various calling signatures cleanly."""
    bounds = PadBounds(pad_top=6, pad_bottom=6, pad_left=4, pad_right=5)

    # 1. Padded tensor
    x_pad = torch.zeros(1, 3, 512, 384)
    m_pad_tensor = compute_valid_latent_mask(x_pad, bounds)
    # 2. Explicit padded_shape
    m_padded_shape = compute_valid_latent_mask(padded_shape=(512, 384), pad_bounds=bounds)
    # 3. Padded tuple with is_padded=True
    m_is_padded = compute_valid_latent_mask((512, 384), bounds, is_padded=True)
    # 4. Explicit H_pad, W_pad keywords
    m_h_pad = compute_valid_latent_mask(H_pad=512, W_pad=384, pad_bounds=bounds)

    # 5. Default tuple (treated deterministically as unpadded orig_shape)
    m_orig_tuple = compute_valid_latent_mask((500, 375), bounds)
    # 6. Default two integers (treated deterministically as unpadded orig_shape)
    m_orig_ints = compute_valid_latent_mask(500, 375, bounds)
    # 7. Explicit orig_shape
    m_orig_shape = compute_valid_latent_mask(orig_shape=(500, 375), pad_bounds=bounds)
    # 8. Explicit H_orig, W_orig keywords
    m_h_orig = compute_valid_latent_mask(H_orig=500, W_orig=375, pad_bounds=bounds)

    assert torch.equal(m_pad_tensor, m_padded_shape)
    assert torch.equal(m_pad_tensor, m_is_padded)
    assert torch.equal(m_pad_tensor, m_h_pad)
    assert torch.equal(m_pad_tensor, m_orig_tuple)
    assert torch.equal(m_pad_tensor, m_orig_ints)
    assert torch.equal(m_pad_tensor, m_orig_shape)
    assert torch.equal(m_pad_tensor, m_h_orig)


def test_latent_mask_orig_shape_divisible_by_stride_not_misidentified() -> None:
    """Verify that unpadded shapes divisible by stride (e.g. 160x160 with pad 16) are not misidentified as padded."""
    # H_orig = 160, W_orig = 160 (both 160 % 16 == 0)
    # pad_top=16, pad_bottom=16, pad_left=16, pad_right=16
    # Padded dimensions must be 192x192, latent grid must be 12x12
    bounds = PadBounds(pad_top=16, pad_bottom=16, pad_left=16, pad_right=16)

    m_orig = compute_valid_latent_mask((160, 160), bounds, stride=16)
    assert m_orig.shape == (12, 12)

    # Receptive field centers: 16*k + 8
    # Valid interval: [16, 176)
    # k=0: center=8 (< 16) -> False
    # k=1: center=24 (in [16, 176)) -> True
    # k=10: center=168 (in [16, 176)) -> True
    # k=11: center=184 (>= 176) -> False
    assert m_orig[0, 0].item() is False
    assert m_orig[1, 1].item() is True
    assert m_orig[10, 10].item() is True
    assert m_orig[11, 11].item() is False

    # Check consistency with padded tensor and explicit padded_shape
    x_pad = torch.zeros(1, 1, 192, 192)
    m_tensor = compute_valid_latent_mask(x_pad, bounds, stride=16)
    m_padded = compute_valid_latent_mask(padded_shape=(192, 192), pad_bounds=bounds, stride=16)
    m_flag = compute_valid_latent_mask((192, 192), bounds, is_padded=True, stride=16)
    assert torch.equal(m_orig, m_tensor)
    assert torch.equal(m_orig, m_padded)
    assert torch.equal(m_orig, m_flag)


def test_custom_factor_and_stride() -> None:
    """Verify padding with custom factor (32) and latent mask with custom stride (8)."""
    h, w = 100, 100
    x = torch.randn(1, h, w)
    x_pad, bounds = pad_symmetric_centered(x, factor=32)

    # 100 -> 128: delta=28 -> top=14, bottom=14, left=14, right=14
    assert bounds == (14, 14, 14, 14)
    assert x_pad.shape == (1, 128, 128)

    # Mask at stride 8: 128 // 8 = 16
    mask = compute_valid_latent_mask(x_pad, bounds, stride=8)
    assert mask.shape == (16, 16)

    # Centers: 8*k + 4
    # Valid interval: [14, 114)
    # k=0: 4 < 14 -> False
    # k=1: 12 < 14 -> False
    # k=2: 20 in [14, 114) -> True
    # k=13: 8*13 + 4 = 108 in [14, 114) -> True
    # k=14: 8*14 + 4 = 116 >= 114 -> False
    # k=15: 8*15 + 4 = 124 >= 114 -> False
    assert mask[0, 0].item() is False
    assert mask[1, 1].item() is False
    assert mask[2, 2].item() is True
    assert mask[13, 13].item() is True
    assert mask[14, 14].item() is False
    assert mask[15, 15].item() is False


# ==============================================================================
# 7. Parameter Validation & Error Handling
# ==============================================================================


def test_invalid_parameters_raise_value_error() -> None:
    """Verify invalid factors, strides, dimensions, and pad_bounds raise descriptive ValueErrors."""
    # Factor <= 0
    with pytest.raises(ValueError, match="factor must be a positive integer"):
        compute_padding_offsets(100, 100, factor=0)

    with pytest.raises(ValueError, match="factor must be a positive integer"):
        compute_padding_offsets(100, 100, factor=-16)

    with pytest.raises(ValueError, match="factor must be a positive integer"):
        pad_symmetric_centered(torch.randn(3, 100, 100), factor=0)

    # Factor <= 0 even when pad_bounds is passed
    with pytest.raises(ValueError, match="factor must be a positive integer"):
        pad_symmetric_centered(torch.randn(3, 100, 100), factor=0, pad_bounds=(0, 0, 0, 0))

    with pytest.raises(ValueError, match="factor must be a positive integer"):
        pad_symmetric_centered(torch.randn(3, 100, 100), factor=-32, pad_bounds=(0, 0, 0, 0))

    # Stride <= 0
    with pytest.raises(ValueError, match="stride must be a positive integer"):
        compute_valid_latent_mask((64, 64), (0, 0, 0, 0), stride=0)

    # 1D tensor input to compute_valid_latent_mask
    with pytest.raises(ValueError, match="Expected at least 2D tensor"):
        compute_valid_latent_mask(torch.randn(10), (0, 0, 0, 0))

    # Non-positive dimensions
    with pytest.raises(ValueError, match="Spatial dimensions must be positive"):
        compute_padding_offsets(0, 100)

    with pytest.raises(ValueError, match="Spatial dimensions must be positive"):
        compute_padding_offsets(-10, 100)

    # Invalid tensor dimensions (1D or 5D)
    with pytest.raises(ValueError, match="Expected 2D, 3D, or 4D tensor"):
        pad_symmetric_centered(torch.randn(10))

    with pytest.raises(ValueError, match="Expected 2D, 3D, or 4D tensor"):
        pad_symmetric_centered(torch.randn(1, 2, 3, 4, 5))

    # Invalid pad_bounds length
    with pytest.raises(ValueError, match="pad_bounds must have 4 elements"):
        unpad_centered(torch.randn(64, 64), (0, 0, 0))  # type: ignore

    # Negative pad_bounds entries
    with pytest.raises(ValueError, match="pad_bounds entries must be non-negative"):
        unpad_centered(torch.randn(64, 64), (0, -1, 0, 0))

    # Padding offsets exceed tensor dimensions
    with pytest.raises(ValueError, match="exceed tensor spatial dimensions"):
        unpad_centered(torch.randn(64, 64), (40, 40, 0, 0))

    # Non-tensor input
    with pytest.raises(TypeError, match=r"must be a torch\.Tensor"):
        pad_symmetric_centered([1, 2, 3])  # type: ignore

    with pytest.raises(TypeError, match=r"must be a torch\.Tensor"):
        unpad_centered([1, 2, 3], (0, 0, 0, 0))  # type: ignore
