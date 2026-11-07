"""Standalone CLI script to decompress a .sanc container bitstream into an image file.

Usage:
    python scripts/decode.py -i input.sanc -o output.png [--checkpoint model.pt] [--device cpu]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path when executed directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.bitstream.container import SancContainer, create_model, decode_image  # noqa: E402
from src.codec.model_a import ModelA  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompress a .sanc binary container file into a reconstructed image.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input .sanc bitstream file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to output image file (PNG, JPEG, etc.).")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pt).")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="Model architecture override (A, B, C, D, C-Transmitted; defaults to model_id from container).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device for executing synthesis transforms (default: cpu).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "constriction", "python"],
        help="Entropy coding backend (default: auto).",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str | None, model_type_or_id: str | int = 0) -> ModelA:
    model = create_model(model_type_or_id)
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.is_file():
            sys.exit(f"Error: Checkpoint file not found: {path}")
        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)

    model.update()
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        sys.exit(f"Error: Input .sanc file not found: {input_path}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    container = SancContainer.load(input_path)
    model_type = args.model if args.model is not None else container.model_id
    model = load_model(args.checkpoint, model_type_or_id=model_type)

    t0 = time.perf_counter()
    x_hat = decode_image(
        container=container,
        model=model,
        device=device,
        backend=args.backend,
    )
    elapsed = time.perf_counter() - t0

    # Convert (1, 3, H, W) tensor in [0, 1] to uint8 PIL Image
    arr = (x_hat.squeeze(0).permute(1, 2, 0).float().cpu().numpy() * 255.0).round().astype(np.uint8)
    img = Image.fromarray(arr)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

    print(f"Decompressed {input_path.name} -> {output_path.name}")
    print(f"Recovered Dimensions: {container.orig_width}x{container.orig_height}")
    print(f"Bitstream size:       {container.total_bytes} bytes ({container.bpp:.4f} bpp)")
    print(f"Time:                 {elapsed:.3f}s")


if __name__ == "__main__":
    main()
