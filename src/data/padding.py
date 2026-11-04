"""Centered symmetric reflection padding and valid latent mask computation for neural compression.

Provides centered reflection padding to align variable-dimension images to multiples of 64
for 4-stage convolutional neural codecs with stride-16 backbones and hyperpriors.
Guarantees bit-identical unpadding crop recovery, handles tiny inputs via iterative reflection
with replication fallback, and generates exact latent grid validity masks (M_valid)
based on receptive field centers in padded image coordinate space.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F


class PadBounds(NamedTuple):
    """Bounding margins for centered spatial padding: (pad_top, pad_bottom, pad_left, pad_right)."""

    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int


def compute_padding_offsets(
    height_or_shape: int | tuple[int, ...] | Sequence[int] | torch.Size,
    width: int | None = None,
    factor: int = 64,
) -> PadBounds:
    """Compute centered padding offsets so spatial dimensions align to multiples of factor.

    For height H and width W:
        H_pad = ceil(H / factor) * factor
        delta_H = H_pad - H
        pad_top = delta_H // 2
        pad_bottom = delta_H - pad_top

        W_pad = ceil(W / factor) * factor
        delta_W = W_pad - W
        pad_left = delta_W // 2
        pad_right = delta_W - pad_left

    Args:
        height_or_shape: Spatial height (int) or shape tuple containing (..., H, W).
        width: Spatial width (int), required if height_or_shape is an integer.
        factor: Multiplier to align spatial dimensions to (default 64). Must be positive.

    Returns:
        PadBounds: Tuple of (pad_top, pad_bottom, pad_left, pad_right).

    Raises:
        TypeError: If height_or_shape has an unsupported type.
        ValueError: If factor <= 0, or height/width <= 0, or shape has fewer than 2 dimensions.
    """
    if factor <= 0:
        raise ValueError(f"factor must be a positive integer, got factor={factor}")

    if isinstance(height_or_shape, int):
        if width is None:
            raise ValueError("width must be provided when height_or_shape is an integer")
        h = int(height_or_shape)
        w = int(width)
    elif isinstance(height_or_shape, (tuple, list, torch.Size)):
        if len(height_or_shape) < 2:
            raise ValueError(f"Shape must have at least 2 dimensions, got {height_or_shape}")
        h = int(height_or_shape[-2])
        w = int(height_or_shape[-1])
    else:
        raise TypeError(f"Unsupported type for height_or_shape: {type(height_or_shape)}")

    if h <= 0 or w <= 0:
        raise ValueError(f"Spatial dimensions must be positive, got height={h}, width={w}")

    h_pad = ((h + factor - 1) // factor) * factor
    w_pad = ((w + factor - 1) // factor) * factor

    delta_h = h_pad - h
    delta_w = w_pad - w

    pad_top = delta_h // 2
    pad_bottom = delta_h - pad_top

    pad_left = delta_w // 2
    pad_right = delta_w - pad_left

    return PadBounds(pad_top=pad_top, pad_bottom=pad_bottom, pad_left=pad_left, pad_right=pad_right)


def _pad_symmetric_iterative(
    t: torch.Tensor,
    pad_left: int,
    pad_right: int,
    pad_top: int,
    pad_bottom: int,
) -> torch.Tensor:
    """Apply centered symmetric reflection padding iteratively, falling back to replication if dimension is 1.

    Guarantees device and dtype preservation without crashing when padding width
    exceeds spatial dimensions.
    """
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        return t.clone()

    orig_dtype = t.dtype
    work_t = t
    if orig_dtype == torch.bool:
        work_t = work_t.to(torch.uint8)

    orig_ndim = work_t.ndim
    if orig_ndim == 2:
        work_t = work_t.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        work_t = work_t.unsqueeze(0)
    elif orig_ndim != 4:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {orig_ndim}D tensor of shape {tuple(t.shape)}")

    # Pad width iteratively on tensor's device
    rem_l, rem_r = pad_left, pad_right
    while rem_l > 0 or rem_r > 0:
        w = work_t.shape[-1]
        if w == 1:
            work_t = F.pad(work_t, (rem_l, rem_r, 0, 0), mode="replicate")
            break
        pl = min(rem_l, w - 1)
        pr = min(rem_r, w - 1)
        work_t = F.pad(work_t, (pl, pr, 0, 0), mode="reflect")
        rem_l -= pl
        rem_r -= pr

    # Pad height iteratively on tensor's device
    rem_t, rem_b = pad_top, pad_bottom
    while rem_t > 0 or rem_b > 0:
        h = work_t.shape[-2]
        if h == 1:
            work_t = F.pad(work_t, (0, 0, rem_t, rem_b), mode="replicate")
            break
        pt = min(rem_t, h - 1)
        pb = min(rem_b, h - 1)
        work_t = F.pad(work_t, (0, 0, pt, pb), mode="reflect")
        rem_t -= pt
        rem_b -= pb

    if orig_ndim == 2:
        work_t = work_t.squeeze(0).squeeze(0)
    elif orig_ndim == 3:
        work_t = work_t.squeeze(0)

    if orig_dtype == torch.bool:
        work_t = work_t.to(torch.bool)

    return work_t


def pad_symmetric_centered(
    x: torch.Tensor,
    factor: int = 64,
    pad_bounds: tuple[int, int, int, int] | PadBounds | None = None,
) -> tuple[torch.Tensor, PadBounds]:
    """Pad tensor with centered symmetric reflection so spatial dimensions are multiples of factor.

    Supports 2D (H, W), 3D (C, H, W), and 4D (B, C, H, W) tensors, preserving original
    device and dtype.

    Args:
        x: Input tensor of shape (H, W), (C, H, W), or (B, C, H, W).
        factor: Spatial dimension multiplier (default 64). Must be positive.
        pad_bounds: Optional pre-computed PadBounds or 4-tuple (pad_top, pad_bottom, pad_left, pad_right).
            If omitted, computed automatically from x's spatial shape and factor.

    Returns:
        tuple[torch.Tensor, PadBounds]: A tuple (x_padded, pad_bounds).

    Raises:
        TypeError: If x is not a torch.Tensor.
        ValueError: If factor <= 0, x has invalid dimensions, or pad_bounds entries are negative.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")

    if factor <= 0:
        raise ValueError(f"factor must be a positive integer, got factor={factor}")

    if x.ndim < 2 or x.ndim > 4:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got shape {tuple(x.shape)}")

    h, w = x.shape[-2], x.shape[-1]
    if h <= 0 or w <= 0:
        raise ValueError(f"Tensor spatial dimensions must be positive, got shape {tuple(x.shape)}")

    if pad_bounds is None:
        bounds = compute_padding_offsets(h, w, factor=factor)
    else:
        if len(pad_bounds) != 4:
            raise ValueError(f"pad_bounds must have 4 elements (top, bottom, left, right), got {pad_bounds}")
        pt, pb, pl, pr = pad_bounds
        if pt < 0 or pb < 0 or pl < 0 or pr < 0:
            raise ValueError(f"pad_bounds entries must be non-negative, got {pad_bounds}")
        bounds = PadBounds(pad_top=int(pt), pad_bottom=int(pb), pad_left=int(pl), pad_right=int(pr))

    x_pad = _pad_symmetric_iterative(
        x,
        pad_left=bounds.pad_left,
        pad_right=bounds.pad_right,
        pad_top=bounds.pad_top,
        pad_bottom=bounds.pad_bottom,
    )
    return x_pad, bounds


def unpad_centered(
    t: torch.Tensor,
    pad_bounds: tuple[int, int, int, int] | PadBounds | Sequence[int],
) -> torch.Tensor:
    """Crop padded tensor back to original unpadded dimensions using pad_bounds.

    Performs exact bit-identical slice recovery:
        t[..., pad_top : H_pad - pad_bottom, pad_left : W_pad - pad_right]

    Args:
        t: Padded tensor of shape (H_pad, W_pad), (C, H_pad, W_pad), or (B, C, H_pad, W_pad).
        pad_bounds: Tuple or PadBounds of (pad_top, pad_bottom, pad_left, pad_right).

    Returns:
        torch.Tensor: Cropped tensor matching original dimensions.

    Raises:
        TypeError: If t is not a torch.Tensor.
        ValueError: If t has invalid ndim, pad_bounds has invalid length/values, or
            padding offsets exceed tensor dimensions.
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"t must be a torch.Tensor, got {type(t)}")

    if t.ndim < 2 or t.ndim > 4:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got shape {tuple(t.shape)}")

    if len(pad_bounds) != 4:
        raise ValueError(f"pad_bounds must have 4 elements (top, bottom, left, right), got {pad_bounds}")

    pt, pb, pl, pr = int(pad_bounds[0]), int(pad_bounds[1]), int(pad_bounds[2]), int(pad_bounds[3])
    if pt < 0 or pb < 0 or pl < 0 or pr < 0:
        raise ValueError(f"pad_bounds entries must be non-negative, got {pad_bounds}")

    h_pad, w_pad = t.shape[-2], t.shape[-1]
    if h_pad < pt + pb or w_pad < pl + pr:
        raise ValueError(
            f"Padding offsets (top={pt}, bottom={pb}, left={pl}, right={pr}) exceed "
            f"tensor spatial dimensions ({h_pad}, {w_pad})"
        )

    h_end = h_pad - pb
    w_end = w_pad - pr

    return t[..., pt:h_end, pl:w_end]


def pad_image_and_saliency_centered(
    x: torch.Tensor,
    S: torch.Tensor,
    factor: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, PadBounds]:
    """Pad both RGB image and visual saliency map with identical centered symmetric reflection margins.

    Args:
        x: RGB image tensor of shape (H, W), (3, H, W), or (B, 3, H, W).
        S: Saliency map tensor of shape (H, W), (1, H, W), or (B, 1, H, W).
        factor: Multiplier to align spatial dimensions to (default 64). Must be positive.

    Returns:
        tuple[torch.Tensor, torch.Tensor, PadBounds]: (x_padded, S_padded, pad_bounds).

    Raises:
        TypeError: If x or S is not a torch.Tensor.
        ValueError: If spatial dimensions of x and S do not match, or factor <= 0.
    """
    if not isinstance(x, torch.Tensor) or not isinstance(S, torch.Tensor):
        raise TypeError("Both x and S must be torch.Tensor instances")

    if x.ndim < 2 or x.ndim > 4 or S.ndim < 2 or S.ndim > 4:
        raise ValueError(
            f"Expected 2D, 3D, or 4D tensors, got x shape {tuple(x.shape)}, S shape {tuple(S.shape)}"
        )

    if x.shape[-2:] != S.shape[-2:]:
        raise ValueError(
            f"Spatial dimensions of x and S must match, got x {tuple(x.shape[-2:])} vs S {tuple(S.shape[-2:])}"
        )

    bounds = compute_padding_offsets(x.shape[-2], x.shape[-1], factor=factor)
    x_pad, _ = pad_symmetric_centered(x, factor=factor, pad_bounds=bounds)
    S_pad, _ = pad_symmetric_centered(S, factor=factor, pad_bounds=bounds)
    return x_pad, S_pad, bounds


def compute_valid_latent_mask(
    input_or_h: int | tuple[int, ...] | Sequence[int] | torch.Size | torch.Tensor | None = None,
    width_or_bounds: int | tuple[int, int, int, int] | PadBounds | Sequence[int] | None = None,
    pad_bounds: tuple[int, int, int, int] | PadBounds | Sequence[int] | None = None,
    *,
    stride: int = 16,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bool,
    is_padded: bool = False,
    padded_shape: tuple[int, ...] | Sequence[int] | torch.Size | None = None,
    orig_shape: tuple[int, ...] | Sequence[int] | torch.Size | None = None,
    H_orig: int | None = None,
    W_orig: int | None = None,
    H_pad: int | None = None,
    W_pad: int | None = None,
) -> torch.Tensor:
    """Compute the valid latent validity mask (M_valid) on the downsampled latent grid.

    A latent grid cell (i, j) at stride s is valid (M_valid[i, j] = 1) if and only if
    its receptive field center (s*i + s//2, s*j + s//2) falls within the unpadded image:
        pad_top <= s*i + s//2 < pad_top + H_orig
        and
        pad_left <= s*j + s//2 < pad_left + W_orig

    Args:
        input_or_h: Padded tensor, shape tuple (..., H, W), or H dimension (int).
            If a torch.Tensor is provided, it is treated as the padded tensor.
            If a tuple or integer is provided, it is deterministically treated as unpadded
            orig_shape unless is_padded=True or padded_shape is passed.
        width_or_bounds: W dimension (int) if input_or_h is int, or pad_bounds if passed as 2nd arg.
        pad_bounds: 4-tuple of (pad_top, pad_bottom, pad_left, pad_right).
        stride: Analysis downsampling stride (default 16). Must be positive.
        device: PyTorch device on which to construct mask (inferred from tensor if input_or_h is Tensor).
        dtype: Return tensor dtype (default torch.bool).
        is_padded: Boolean flag indicating whether tuple or integer inputs represent padded dimensions
            (default False, treating tuple/int inputs deterministically as unpadded orig_shape).
        padded_shape: Explicit padded shape (..., H_pad, W_pad).
        orig_shape: Explicit unpadded original shape (..., H_orig, W_orig).
        H_orig: Explicit unpadded height override.
        W_orig: Explicit unpadded width override.
        H_pad: Explicit padded height override.
        W_pad: Explicit padded width override.

    Returns:
        torch.Tensor: Boolean tensor of shape (H_latent, W_latent) where H_latent = H_pad // stride
            and W_latent = W_pad // stride.

    Raises:
        ValueError: If stride <= 0, pad_bounds are invalid, dimensions are non-positive,
            tensor has fewer than 2 dimensions, or padded dimensions are not divisible by stride.
    """
    if stride <= 0:
        raise ValueError(f"stride must be a positive integer, got stride={stride}")

    # Determine pad_bounds
    actual_bounds: tuple[int, int, int, int] | PadBounds | None = None
    if pad_bounds is not None:
        actual_bounds = pad_bounds
    elif (
        width_or_bounds is not None
        and isinstance(width_or_bounds, (tuple, list, PadBounds))
        and len(width_or_bounds) == 4
    ):
        actual_bounds = width_or_bounds

    if actual_bounds is None:
        raise ValueError("pad_bounds must be provided as a 4-tuple of (top, bottom, left, right)")

    if len(actual_bounds) != 4:
        raise ValueError(f"pad_bounds must have 4 elements, got {actual_bounds}")

    pt, pb, pl, pr = int(actual_bounds[0]), int(actual_bounds[1]), int(actual_bounds[2]), int(actual_bounds[3])
    if pt < 0 or pb < 0 or pl < 0 or pr < 0:
        raise ValueError(f"pad_bounds elements must be non-negative, got {actual_bounds}")

    # Determine device and target spatial dimensions
    inferred_device = device
    is_input_tensor = False
    h_val: int | None = None
    w_val: int | None = None

    if isinstance(input_or_h, torch.Tensor):
        if input_or_h.ndim < 2:
            raise ValueError(
                f"Expected at least 2D tensor, got {input_or_h.ndim}D tensor of shape {tuple(input_or_h.shape)}"
            )
        if inferred_device is None:
            inferred_device = input_or_h.device
        h_val = int(input_or_h.shape[-2])
        w_val = int(input_or_h.shape[-1])
        is_input_tensor = True
    elif isinstance(input_or_h, (tuple, list, torch.Size)):
        if len(input_or_h) < 2:
            raise ValueError(f"Shape must have at least 2 dimensions, got {input_or_h}")
        h_val = int(input_or_h[-2])
        w_val = int(input_or_h[-1])
    elif isinstance(input_or_h, int):
        h_val = input_or_h
        if isinstance(width_or_bounds, int):
            w_val = width_or_bounds

    # Explicit shape overrides
    if padded_shape is not None:
        if len(padded_shape) < 2:
            raise ValueError(f"padded_shape must have at least 2 dimensions, got {padded_shape}")
        actual_h_pad = int(padded_shape[-2])
        actual_w_pad = int(padded_shape[-1])
        actual_h_orig = actual_h_pad - pt - pb
        actual_w_orig = actual_w_pad - pl - pr
    elif orig_shape is not None:
        if len(orig_shape) < 2:
            raise ValueError(f"orig_shape must have at least 2 dimensions, got {orig_shape}")
        actual_h_orig = int(orig_shape[-2])
        actual_w_orig = int(orig_shape[-1])
        actual_h_pad = actual_h_orig + pt + pb
        actual_w_pad = actual_w_orig + pl + pr
    elif H_pad is not None and W_pad is not None:
        actual_h_pad = int(H_pad)
        actual_w_pad = int(W_pad)
        actual_h_orig = actual_h_pad - pt - pb
        actual_w_orig = actual_w_pad - pl - pr
    elif H_orig is not None and W_orig is not None:
        actual_h_orig = int(H_orig)
        actual_w_orig = int(W_orig)
        actual_h_pad = actual_h_orig + pt + pb
        actual_w_pad = actual_w_orig + pl + pr
    elif h_val is not None and w_val is not None:
        # torch.Tensor is treated as padded tensor; or if is_padded=True
        if is_input_tensor or is_padded:
            actual_h_pad = h_val
            actual_w_pad = w_val
            actual_h_orig = actual_h_pad - pt - pb
            actual_w_orig = actual_w_pad - pl - pr
        else:
            # Default for tuple/int inputs: deterministically treat as unpadded orig_shape
            actual_h_orig = h_val
            actual_w_orig = w_val
            actual_h_pad = actual_h_orig + pt + pb
            actual_w_pad = actual_w_orig + pl + pr
    else:
        raise ValueError("Could not determine spatial dimensions for valid latent mask computation")

    if actual_h_orig <= 0 or actual_w_orig <= 0:
        raise ValueError(
            f"Original spatial dimensions must be positive, got H_orig={actual_h_orig}, W_orig={actual_w_orig}"
        )
    if actual_h_pad <= 0 or actual_w_pad <= 0:
        raise ValueError(
            f"Padded spatial dimensions must be positive, got H_pad={actual_h_pad}, W_pad={actual_w_pad}"
        )

    if actual_h_pad % stride != 0 or actual_w_pad % stride != 0:
        raise ValueError(
            f"Padded dimensions ({actual_h_pad}, {actual_w_pad}) must be divisible by stride {stride}"
        )

    h_latent = actual_h_pad // stride
    w_latent = actual_w_pad // stride

    i_coords = torch.arange(h_latent, device=inferred_device)
    j_coords = torch.arange(w_latent, device=inferred_device)

    # Receptive field centers: s * idx + s // 2
    ci = stride * i_coords + (stride // 2)
    cj = stride * j_coords + (stride // 2)

    # Valid condition:
    # pad_top <= ci < pad_top + H_orig AND pad_left <= cj < pad_left + W_orig
    valid_i = (ci >= pt) & (ci < pt + actual_h_orig)
    valid_j = (cj >= pl) & (cj < pl + actual_w_orig)

    mask = valid_i[:, None] & valid_j[None, :]
    if mask.dtype != dtype:
        mask = mask.to(dtype=dtype)

    return mask
