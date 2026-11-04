"""FastSal offline visual saliency extraction and percentile caching module.

Provides:
  - FastSal: Lightweight saliency model using MobileNetV2 backbone and multiscale decoder.
  - normalize_saliency_percentile: 1st-99th percentile dynamic range clipping and uint8 quantization.
  - extract_and_cache_saliency: Batch offline extraction pipeline with subsample and device selection support.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models

from src.data.manifests import (
    get_manifests_for_dataset,
    resolve_target_dir,
)

logger = logging.getLogger(__name__)


def get_default_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_saliency_percentile(
    saliency_map: np.ndarray | torch.Tensor,
    p_low: float = 1.0,
    p_high: float = 99.0,
    eps: float = 1e-7,
) -> np.ndarray:
    """Normalize raw continuous saliency activations to uint8 [0, 255] via percentile clipping.

    Applies the mathematical formulation:
        S_clip = clip(S_raw, p_1, p_99)
        S_norm = (S_clip - p_1) / (p_99 - p_1 + 10^-7)
        S_uint8 = round(255 * S_norm)

    Handles extreme outliers safely by discarding the outer 1% tail spikes,
    and safely normalizes constant/zero-variance flat maps without divide-by-zero.

    Args:
        saliency_map: Continuous saliency activations as NumPy array or PyTorch Tensor.
        p_low: Lower percentile for dynamic range clipping (default: 1.0).
        p_high: Upper percentile for dynamic range clipping (default: 99.0).
        eps: Small epsilon constant to prevent divide-by-zero (default: 1e-7).

    Returns:
        NumPy uint8 array with values in [0, 255] preserving spatial shape.
    """
    if isinstance(saliency_map, torch.Tensor):
        arr = saliency_map.detach().cpu().to(torch.float32).numpy()
    else:
        arr = np.asarray(saliency_map, dtype=np.float32)

    # Sanitize non-finite values if any
    if not np.all(np.isfinite(arr)):
        finfo = np.finfo(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=float(finfo.max), neginf=float(finfo.min))

    p1 = float(np.percentile(arr, p_low))
    p99 = float(np.percentile(arr, p_high))

    diff = p99 - p1
    if diff > eps:
        clipped = np.clip(arr, p1, p99).astype(np.float64)
        norm = (clipped - p1) / diff
    else:
        norm = np.zeros_like(arr, dtype=np.float64)

    norm = np.clip(norm, 0.0, 1.0)
    uint8_map = np.round(255.0 * norm).astype(np.uint8)
    return uint8_map


class MultiscaleDecoder(nn.Module):
    """Multiscale Feature Fusion Decoder for FastSal.

    Fuses intermediate MobileNetV2 feature representations from multiple stages:
      - Stage 1 (stride 4, 24 channels)
      - Stage 2 (stride 8, 32 channels)
      - Stage 3 (stride 16, 96 channels)
      - Stage 4 (stride 32, 1280 channels)

    Constructs a top-down feature pyramid with lateral projections and upsampling,
    terminating in a refinement block and spatial bilinear interpolation.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (24, 32, 96, 1280),
        decoder_dim: int = 64,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = in_channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(c4, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c3, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c2, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, decoder_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        f1, f2, f3, f4 = features

        p4 = self.conv4(f4)
        p3 = self.conv3(f3) + F.interpolate(
            p4, size=f3.shape[2:], mode="bilinear", align_corners=False
        )
        p2 = self.conv2(f2) + F.interpolate(
            p3, size=f2.shape[2:], mode="bilinear", align_corners=False
        )
        p1 = self.conv1(f1) + F.interpolate(
            p2, size=f1.shape[2:], mode="bilinear", align_corners=False
        )

        out = self.refine(p1)
        saliency = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return saliency


class FastSal(nn.Module):
    """FastSal: Lightweight visual saliency extractor using MobileNetV2 backbone.

    Executes in torch.no_grad() inference mode, outputting high-resolution
    continuous saliency maps matching the input spatial dimensions (H, W).
    """

    def __init__(
        self,
        pretrained: bool = False,
        weights: Any = None,
        decoder_channels: int = 64,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()

        # Initialize MobileNetV2 backbone
        backbone_weights = None
        if weights is not None:
            backbone_weights = weights
        elif pretrained:
            try:
                from torchvision.models import MobileNet_V2_Weights
                backbone_weights = MobileNet_V2_Weights.DEFAULT
            except (ImportError, AttributeError, RuntimeError, OSError):
                backbone_weights = None

        try:
            mobilenet = models.mobilenet_v2(weights=backbone_weights)
        except (RuntimeError, OSError, ValueError):
            mobilenet = models.mobilenet_v2(weights=None)

        # Slice backbone into 4 multiscale feature stages
        self.stage1 = mobilenet.features[:4]    # stride 4, channels 24
        self.stage2 = mobilenet.features[4:7]   # stride 8, channels 32
        self.stage3 = mobilenet.features[7:14]  # stride 16, channels 96
        self.stage4 = mobilenet.features[14:19] # stride 32, channels 1280

        self.decoder = MultiscaleDecoder(
            in_channels=(24, 32, 96, 1280),
            decoder_dim=decoder_channels,
        )

        target_device = torch.device(device) if device is not None else get_default_device()
        self.to(target_device)
        self.eval()

    @property
    def device(self) -> torch.device:
        """Active device for inference derived from model parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _prepare_input(
        self,
        x: torch.Tensor | np.ndarray | Image.Image,
    ) -> tuple[torch.Tensor, tuple[int, int], bool]:
        """Convert any supported image input into a (B, 3, H, W) float32 tensor."""
        if isinstance(x, Image.Image):
            rgb_img = x.convert("RGB")
            w, h = rgb_img.size
            arr = np.array(rgb_img, dtype=np.float32) / 255.0  # (H, W, 3)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            return tensor.to(self.device), (h, w), True

        if isinstance(x, np.ndarray):
            if x.ndim == 3 and x.shape[2] == 3:  # (H, W, 3)
                h, w, _ = x.shape
                arr = x.astype(np.float32)
                if x.dtype == np.uint8 or arr.max() > 1.0:
                    arr = arr / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
                return tensor.to(self.device), (h, w), True
            elif x.ndim == 3 and x.shape[0] == 3:  # (3, H, W)
                _, h, w = x.shape
                arr = x.astype(np.float32)
                if x.dtype == np.uint8 or arr.max() > 1.0:
                    arr = arr / 255.0
                tensor = torch.from_numpy(arr).unsqueeze(0)
                return tensor.to(self.device), (h, w), True
            elif x.ndim == 4:
                if x.shape[3] == 3:  # NHWC: (B, H, W, 3)
                    _, h, w, _ = x.shape
                    arr = x.astype(np.float32)
                    if x.dtype == np.uint8 or arr.max() > 1.0:
                        arr = arr / 255.0
                    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)
                    return tensor.to(self.device), (h, w), False
                elif x.shape[1] == 3:  # NCHW: (B, 3, H, W)
                    _, _, h, w = x.shape
                    arr = x.astype(np.float32)
                    if x.dtype == np.uint8 or arr.max() > 1.0:
                        arr = arr / 255.0
                    tensor = torch.from_numpy(arr)
                    return tensor.to(self.device), (h, w), False
                else:
                    raise ValueError(f"Unsupported NumPy 4D array shape: {x.shape}")
            else:
                raise ValueError(f"Unsupported NumPy array shape for FastSal: {x.shape}")

        if isinstance(x, torch.Tensor):
            if x.ndim == 3 and x.shape[2] == 3:  # (H, W, 3)
                h, w, _ = x.shape
                t = x.to(torch.float32)
                if x.dtype == torch.uint8 or t.max() > 1.0:
                    t = t / 255.0
                tensor = t.permute(2, 0, 1).unsqueeze(0)
                return tensor.to(self.device), (h, w), True
            elif x.ndim == 3 and x.shape[0] == 3:  # (3, H, W)
                _, h, w = x.shape
                t = x.to(torch.float32)
                if x.dtype == torch.uint8 or t.max() > 1.0:
                    t = t / 255.0
                tensor = t.unsqueeze(0)
                return tensor.to(self.device), (h, w), True
            elif x.ndim == 4:
                if x.shape[3] == 3:  # NHWC: (B, H, W, 3)
                    _, h, w, _ = x.shape
                    t = x.to(torch.float32)
                    if x.dtype == torch.uint8 or t.max() > 1.0:
                        t = t / 255.0
                    tensor = t.permute(0, 3, 1, 2)
                    return tensor.to(self.device), (h, w), False
                elif x.shape[1] == 3:  # NCHW: (B, 3, H, W)
                    _, _, h, w = x.shape
                    t = x.to(torch.float32)
                    if x.dtype == torch.uint8 or t.max() > 1.0:
                        t = t / 255.0
                    return t.to(self.device), (h, w), False
                else:
                    raise ValueError(f"Unsupported PyTorch 4D tensor shape: {x.shape}")
            else:
                raise ValueError(f"Unsupported PyTorch tensor shape for FastSal: {x.shape}")

        raise TypeError(f"Unsupported image type: {type(x)}")

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor | np.ndarray | Image.Image,
    ) -> torch.Tensor:
        """Extract continuous spatial saliency map.

        Args:
            x: Input RGB image as PIL Image, NumPy array, or PyTorch Tensor.
               Supported shapes include (H, W, 3), (3, H, W), (B, 3, H, W), or (B, H, W, 3).

        Returns:
            Continuous saliency map. For unbatched input (e.g. H, W, 3), returns shape (H, W).
            For batched input (B, 3, H, W), returns shape (B, 1, H, W).
        """
        tensor, (h, w), is_unbatched = self._prepare_input(x)

        f1 = self.stage1(tensor)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        saliency = self.decoder([f1, f2, f3, f4], target_size=(h, w))

        if is_unbatched:
            return saliency.squeeze(0).squeeze(0)  # (H, W)
        return saliency

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor | np.ndarray | Image.Image,
        p_low: float = 1.0,
        p_high: float = 99.0,
    ) -> np.ndarray:
        """Extract and normalize saliency to uint8 [0, 255] NumPy array.

        For batched inputs, normalizes each batch item independently.
        """
        raw = self.forward(x)
        if isinstance(raw, torch.Tensor):
            raw_arr = raw.detach().cpu().to(torch.float32).numpy()
        else:
            raw_arr = np.asarray(raw, dtype=np.float32)

        if raw_arr.ndim == 4:
            # (B, 1, H, W) -> normalize each batch item independently
            return np.stack([
                normalize_saliency_percentile(raw_arr[i, 0], p_low=p_low, p_high=p_high)
                for i in range(raw_arr.shape[0])
            ])
        elif raw_arr.ndim == 3 and raw_arr.shape[0] > 1:
            # Batched (B, H, W)
            return np.stack([
                normalize_saliency_percentile(raw_arr[i], p_low=p_low, p_high=p_high)
                for i in range(raw_arr.shape[0])
            ])
        return normalize_saliency_percentile(raw_arr, p_low=p_low, p_high=p_high)


def extract_and_cache_saliency(
    dataset_name: str = "all",
    base_dir: Path | str | None = None,
    sample: int | None = None,
    device: str | torch.device | None = None,
    model: FastSal | None = None,
    overwrite: bool = True,
    quiet: bool = False,
) -> dict[str, int]:
    """Extract and cache offline visual saliency maps for specified dataset splits.

    Processes full RGB source images through FastSal, applies 1st-99th percentile
    dynamic range normalization, and caches 8-bit single-channel ('L') PNG images
    under data/saliency_cache/ matching the source image filenames and resolutions.

    Args:
        dataset_name: Target dataset name or alias ('kodak', 'div2k', 'mit1003', 'all').
        base_dir: Optional base directory override.
        sample: Optional subsample count limiting the number of images per manifest.
        device: Hardware device to execute FastSal on ('cuda', 'cpu', or None for auto).
        model: Optional pre-instantiated FastSal model.
        overwrite: If True, recomputes and overwrites existing cached maps.
        quiet: If True, suppress stdout logging.

    Returns:
        Dictionary mapping manifest names to count of successfully cached saliency maps.

    Raises:
        KeyError: If dataset_name is not registered.
        ValueError: If sample is less than or equal to 0.
        RuntimeError: If source images are missing or cached count doesn't match sample.
    """
    if sample is not None and sample <= 0:
        raise ValueError(f"Sample count must be positive, got {sample}")

    manifests = get_manifests_for_dataset(dataset_name)

    # Determine execution device
    if device is not None:
        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            if not quiet:
                print("[DEVICE WARNING] CUDA requested but unavailable. Falling back to CPU.", file=sys.stderr)
            target_device = torch.device("cpu")
    else:
        target_device = get_default_device()

    if not quiet:
        device_name = torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else "CPU"
        print(f"[DEVICE] FastSal initialized on: {target_device} ({device_name})")

    if model is None:
        model = FastSal(device=target_device)
    else:
        model = model.to(target_device)

    model.eval()

    results: dict[str, int] = {}

    for manifest in manifests:
        raw_dir = resolve_target_dir(manifest.target_dir, base_dir)
        cache_dir = resolve_target_dir(manifest.cache_dir, base_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Determine target filenames
        if manifest.filenames:
            all_files = list(manifest.filenames)
        elif raw_dir.exists():
            all_files = sorted(
                p.name for p in raw_dir.iterdir()
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
            )
        else:
            all_files = []

        if sample is not None:
            target_files = all_files[:sample]
        else:
            target_files = all_files

        if not quiet:
            print(
                f"\n[SALIENCY] Extracting for '{manifest.name}': "
                f"{len(target_files)} target images -> {cache_dir}"
            )

        cached_count = 0
        missing_count = 0

        for fname in target_files:
            raw_path = raw_dir / fname
            if not raw_path.exists():
                if missing_count < 3 and not quiet:
                    print(f"  [MISSING RAW] {raw_path} not found, skipping.", file=sys.stderr)
                missing_count += 1
                continue

            # Determine cache filename: save non-PNG as canonical .png (stem + '.png')
            cache_fname = Path(fname).stem + ".png" if not fname.lower().endswith(".png") else fname
            cache_path = cache_dir / cache_fname

            if not overwrite and cache_path.exists():
                if not quiet:
                    print(f"  [EXISTS] Cached map for {fname} already present.")
                cached_count += 1
                continue

            try:
                with Image.open(raw_path) as img:
                    rgb_img = img.convert("RGB")
                    orig_w, orig_h = rgb_img.size

                    # Run FastSal inference
                    raw_saliency = model(rgb_img)

                    # 1st-99th percentile normalization
                    uint8_map = normalize_saliency_percentile(raw_saliency)

                    # Validate output spatial shape matches original image dimensions
                    if uint8_map.shape != (orig_h, orig_w):
                        raise RuntimeError(
                            f"Dimension mismatch for {fname}: image is ({orig_h}, {orig_w}), "
                            f"saliency map is {uint8_map.shape}"
                        )

                    # Save single-channel 8-bit PNG (mode 'L')
                    saliency_img = Image.fromarray(uint8_map, mode="L")
                    saliency_img.save(cache_path, format="PNG")

                    cached_count += 1
                    if not quiet:
                        print(
                            f"  [CACHED] {fname} -> {cache_path.name} "
                            f"({orig_w}x{orig_h}, uint8 [min={uint8_map.min()}, max={uint8_map.max()}])"
                        )

            except (RuntimeError, OSError, ValueError) as e:
                print(f"  [ERROR] Failed to extract saliency for {fname}: {e}", file=sys.stderr)

        # Validate subsample count matches requested sample count
        if sample is not None and cached_count < sample:
            raise RuntimeError(
                f"Requested sample={sample} for '{manifest.name}', but only {cached_count} "
                f"images were cached ({missing_count} missing source files)."
            )

        results[manifest.name] = cached_count
        if not quiet:
            print(
                f"[SALIENCY COMPLETE] '{manifest.name}': {cached_count}/{len(target_files)} cached "
                f"({missing_count} source images missing)."
            )

    return results
