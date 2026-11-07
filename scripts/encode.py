"""Standalone CLI script to compress an image file into a .sanc container bitstream.

Usage:
    python scripts/encode.py -i input.png -o output.sanc [--checkpoint model.pt] [--device cuda]
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

import torch  # noqa: E402

from src.bitstream.container import SancContainer, create_model, encode_image, load_image_tensor  # noqa: E402
from src.codec.model_a import ModelA  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress an image file into a .sanc binary container using learned image compression."
    )
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input image file (PNG, JPEG, etc.).")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to destination .sanc bitstream file.")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Path to trained model checkpoint (.pt).")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="A",
        help="Model architecture (A, B, C, D, C-Transmitted, default: A).",
    )
    parser.add_argument(
        "-s",
        "--saliency",
        type=str,
        default=None,
        help="Optional path to visual saliency map for Model C-Transmitted.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for executing analysis transforms (default: cuda if available).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "constriction", "python"],
        help="Entropy coding backend (default: auto).",
    )
    return parser.parse_args()


def load_model(checkpoint_path: str | None, model_type: str = "A") -> ModelA:
    model = create_model(model_type)
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
        sys.exit(f"Error: Input image file not found: {input_path}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    model = load_model(args.checkpoint, model_type=args.model)

    if args.saliency is not None and args.model.strip().upper() not in ("C-TRANSMITTED", "CTRANSMITTED", "4"):
        print(
            f"Warning: Saliency map provided but model architecture '{args.model}' "
            "does not transmit or consume external saliency sidecars. The saliency map will be ignored."
        )

    t0 = time.perf_counter()
    image_tensor = load_image_tensor(input_path)
    container: SancContainer = encode_image(
        image=image_tensor,
        model=model,
        device=device,
        backend=args.backend,
        saliency_map=args.saliency,
    )
    elapsed = time.perf_counter() - t0

    bytes_written = container.save(output_path)
    h_pad = container.orig_height + container.pad_bounds.pad_top + container.pad_bounds.pad_bottom
    w_pad = container.orig_width + container.pad_bounds.pad_left + container.pad_bounds.pad_right

    print(f"Compressed {input_path.name} -> {output_path.name}")
    print(f"Dimensions: {container.orig_width}x{container.orig_height} (Padded: {w_pad}x{h_pad})")
    print(f"Bitstream:  {bytes_written} bytes ({container.bpp:.4f} bpp)")
    print(f"Hyper bits: {len(container.z_bitstream) * 8} bits | Main bits: {len(container.y_bitstream) * 8} bits")
    print(f"Time:       {elapsed:.3f}s")


if __name__ == "__main__":
    main()
