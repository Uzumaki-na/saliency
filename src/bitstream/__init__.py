"""Bitstream and container serialization package for .sanc compressed files.

Exports:
- SancContainer: Binary container format storing dimensions, padding margins, and bitstream payloads.
- encode_image: End-to-end image compression pipeline producing a SancContainer.
- decode_image: End-to-end decompression pipeline recovering an unpadded image tensor.
- load_image_tensor: Helper to load image files or arrays into normalized tensors.
"""

from src.bitstream.container import (
    FLAG_HAS_SIDECAR,
    SancContainer,
    create_model,
    decode_image,
    encode_image,
    load_image_tensor,
)

__all__ = [
    "FLAG_HAS_SIDECAR",
    "SancContainer",
    "create_model",
    "decode_image",
    "encode_image",
    "load_image_tensor",
]
