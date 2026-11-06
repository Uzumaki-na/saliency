"""Synchronized PyTorch Dataset and Deterministic Transforms for Saliency-Aware Compression.

Pairs raw RGB images with cached 8-bit visual saliency maps, guaranteeing bit-identical
spatial augmentations (random cropping and horizontal flipping) with shared random state.
Provides symmetric reflection padding for sub-crop inputs and deterministic full-resolution
loading for evaluation and testing splits.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.manifests import (
    DATASET_REGISTRY,
    RAW_DATA_DIR,
    SALIENCY_CACHE_DIR,
    DatasetManifest,
    resolve_target_dir,
)


def _normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    """Normalize image or saliency tensor to [0.0, 1.0] float32 on the tensor's current device."""
    t_float = t.float()
    if t.dtype == torch.uint8 or (t_float.numel() > 0 and t_float.max() > 1.0):
        t_float = t_float / 255.0
    return t_float


def _prepare_image_tensor(img: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert input RGB image into torch.float32 tensor of shape (3, H, W) scaled to [0.0, 1.0]."""
    if isinstance(img, Image.Image):
        img_rgb = img.convert("RGB")
        return _normalize_tensor(TF.to_tensor(img_rgb))

    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
            t = torch.from_numpy(arr)
            return _normalize_tensor(t)

        t = torch.from_numpy(arr.transpose(2, 0, 1))
        return _normalize_tensor(t)

    if isinstance(img, torch.Tensor):
        t = img
        if t.ndim == 2:
            t = t.unsqueeze(0).repeat(3, 1, 1)
        elif t.ndim == 3 and t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        elif t.ndim == 3 and t.shape[2] == 3 and t.shape[0] != 3:
            t = t.permute(2, 0, 1)

        return _normalize_tensor(t)

    raise TypeError(f"Unsupported image input type: {type(img)}")


def _prepare_saliency_tensor(saliency: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert input saliency map into torch.float32 tensor of shape (1, H, W) scaled to [0.0, 1.0]."""
    if isinstance(saliency, Image.Image):
        sal_l = saliency.convert("L")
        return _normalize_tensor(TF.to_tensor(sal_l))

    if isinstance(saliency, np.ndarray):
        arr = saliency
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.transpose(2, 0, 1)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            pass
        elif arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, 0:1].transpose(2, 0, 1)
        elif arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr[0:1, :, :]

        t = torch.from_numpy(arr)
        return _normalize_tensor(t)

    if isinstance(saliency, torch.Tensor):
        t = saliency
        if t.ndim == 2:
            t = t.unsqueeze(0)
        elif t.ndim == 3 and t.shape[2] == 1 and t.shape[0] != 1:
            t = t.permute(2, 0, 1)
        elif t.ndim == 3 and t.shape[0] == 3:
            t = t[0:1, :, :]

        return _normalize_tensor(t)

    raise TypeError(f"Unsupported saliency map input type: {type(saliency)}")


def _pad_symmetric_reflect(
    t: torch.Tensor,
    pad_left: int,
    pad_right: int,
    pad_top: int,
    pad_bottom: int,
) -> torch.Tensor:
    """Pad tensor with symmetric reflection, handling arbitrary padding amounts iteratively on device."""
    # Pad width iteratively on tensor's device
    rem_l, rem_r = pad_left, pad_right
    while rem_l > 0 or rem_r > 0:
        w = t.shape[-1]
        if w == 1:
            t = F.pad(t, (rem_l, rem_r, 0, 0), mode="replicate")
            break
        pl = min(rem_l, w - 1)
        pr = min(rem_r, w - 1)
        t = F.pad(t, (pl, pr, 0, 0), mode="reflect")
        rem_l -= pl
        rem_r -= pr

    # Pad height iteratively on tensor's device
    rem_t, rem_b = pad_top, pad_bottom
    while rem_t > 0 or rem_b > 0:
        h = t.shape[-2]
        if h == 1:
            t = F.pad(t, (0, 0, rem_t, rem_b), mode="replicate")
            break
        pt = min(rem_t, h - 1)
        pb = min(rem_b, h - 1)
        t = F.pad(t, (0, 0, pt, pb), mode="reflect")
        rem_t -= pt
        rem_b -= pb

    return t


class SynchronizedTransforms:
    """Synchronized spatial transformations for paired RGB images and saliency maps.

    Ensures that geometric augmentations (random crop offset (top, left) and horizontal
    flip decision) are bit-identical between the RGB image x and the saliency map S.
    Non-geometric transformations (e.g., color jitter, normalization) are never applied
    to the saliency map.

    In training mode (`split='train'`), applies symmetric reflection padding if source
    dimensions are smaller than crop_size (default 256x256), extracts a synchronized random
    crop, and applies a synchronized horizontal flip with probability flip_prob (default 0.5).

    In evaluation / test mode (`split='test'` or `split='val'`), returns uncropped, unaugmented
    full-resolution tensors matching source image dimensions.
    """

    def __init__(
        self,
        crop_size: int | tuple[int, int] = 256,
        flip_prob: float = 0.5,
        split: str = "train",
        is_train: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if isinstance(crop_size, int):
            if crop_size <= 0:
                raise ValueError(f"crop_size must be positive, got {crop_size}")
            self.crop_size: tuple[int, int] = (crop_size, crop_size)
        elif isinstance(crop_size, (tuple, list)) and len(crop_size) == 2:
            if crop_size[0] <= 0 or crop_size[1] <= 0:
                raise ValueError(f"crop_size dimensions must be positive, got {crop_size}")
            self.crop_size = (int(crop_size[0]), int(crop_size[1]))
        else:
            raise ValueError(f"Invalid crop_size format: {crop_size}")

        if not (0.0 <= flip_prob <= 1.0):
            raise ValueError(f"flip_prob must be in range [0.0, 1.0], got {flip_prob}")
        self.flip_prob = float(flip_prob)

        self.split = str(split).lower().strip()
        if is_train is not None:
            self.is_train = bool(is_train)
        else:
            self.is_train = self.split == "train"

        self.generator = generator

    def __call__(
        self,
        img: Image.Image | np.ndarray | torch.Tensor,
        saliency: Image.Image | np.ndarray | torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply synchronized transforms to paired image and saliency map.

        Args:
            img: Input RGB image (PIL Image, ndarray, or Tensor).
            saliency: Input visual saliency map (PIL Image, ndarray, or Tensor).
            generator: Optional PyTorch generator override for RNG reproducibility.

        Returns:
            Tuple (x, S) where:
                x: torch.float32 Tensor of shape (3, H_out, W_out) in [0.0, 1.0].
                S: torch.float32 Tensor of shape (1, H_out, W_out) in [0.0, 1.0].

        Raises:
            ValueError: If source spatial dimensions (H, W) between img and saliency do not match.
        """
        rng = generator if generator is not None else self.generator

        x = _prepare_image_tensor(img)
        S = _prepare_saliency_tensor(saliency)

        target_device = x.device
        S = S.to(target_device)

        h_x, w_x = x.shape[-2], x.shape[-1]
        h_s, w_s = S.shape[-2], S.shape[-1]

        if (h_x, w_x) != (h_s, w_s):
            raise ValueError(
                f"Dimension mismatch between image ({h_x}, {w_x}) and saliency map ({h_s}, {w_s}). "
                "Inputs must share identical spatial dimensions."
            )

        # For evaluation/test splits, return full-resolution uncropped tensors
        if not self.is_train:
            return x.contiguous(), S.contiguous()

        crop_h, crop_w = self.crop_size

        # Apply symmetric reflection padding if smaller than crop size
        if h_x < crop_h or w_x < crop_w:
            pad_h = max(0, crop_h - h_x)
            pad_w = max(0, crop_w - w_x)

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            x = _pad_symmetric_reflect(x, pad_left, pad_right, pad_top, pad_bottom)
            S = _pad_symmetric_reflect(S, pad_left, pad_right, pad_top, pad_bottom)

            h_x, w_x = x.shape[-2], x.shape[-1]

        # Validate padded output shape
        if h_x < crop_h or w_x < crop_w:
            raise RuntimeError(
                f"Padded spatial dimensions ({h_x}, {w_x}) remain smaller than crop size ({crop_h}, {crop_w})."
            )

        # Compute shared crop coordinates
        if h_x > crop_h:
            top = int(torch.randint(0, h_x - crop_h + 1, (1,), generator=rng).item())
        else:
            top = 0

        if w_x > crop_w:
            left = int(torch.randint(0, w_x - crop_w + 1, (1,), generator=rng).item())
        else:
            left = 0

        # Bit-identical crop
        x = x[:, top : top + crop_h, left : left + crop_w]
        S = S[:, top : top + crop_h, left : left + crop_w]

        # Shared horizontal flip decision
        if self.flip_prob > 0.0:
            flip = float(torch.rand(1, generator=rng).item()) < self.flip_prob
            if flip:
                x = torch.flip(x, dims=[-1])
                S = torch.flip(S, dims=[-1])

        return x.contiguous(), S.contiguous()


class SaliencyImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset pairing raw RGB images with pre-computed cached visual saliency maps.

    Loads raw images from image_dir and corresponding cached 8-bit saliency maps from
    saliency_dir, validates original dimension parity, and applies synchronized
    transformations (random crop and horizontal flip) using shared RNG state.
    """

    def __init__(
        self,
        dataset_name: str | None = None,
        image_dir: str | Path | None = None,
        saliency_dir: str | Path | None = None,
        split: str = "train",
        crop_size: int | tuple[int, int] = 256,
        flip_prob: float = 0.5,
        transform: Callable[[Any, Any], tuple[torch.Tensor, torch.Tensor]] | None = None,
        base_dir: str | Path | None = None,
        sample: int | None = None,
        filenames: Sequence[str] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        self.split = str(split).lower().strip()
        self.crop_size = crop_size
        self.flip_prob = flip_prob
        self.base_dir = Path(base_dir) if base_dir is not None else None

        # Resolve dataset manifest if applicable
        manifest: DatasetManifest | None = None
        if dataset_name is not None or image_dir is None:
            resolved_name = (
                dataset_name if dataset_name is not None else self._default_dataset_name_for_split(self.split)
            )
            manifest = self._resolve_manifest(resolved_name, self.split)

        # Resolve image and saliency directories
        if image_dir is not None:
            self.image_dir = resolve_target_dir(Path(image_dir), self.base_dir)
        elif manifest is not None:
            self.image_dir = resolve_target_dir(manifest.target_dir, self.base_dir)
        else:
            raise ValueError("Unable to determine image directory: provide image_dir or dataset_name.")

        if saliency_dir is not None:
            self.saliency_dir = resolve_target_dir(Path(saliency_dir), self.base_dir)
        elif manifest is not None:
            self.saliency_dir = resolve_target_dir(manifest.cache_dir, self.base_dir)
        else:
            # Match image_dir against known manifests or reconstruct relative to RAW_DATA_DIR
            matched_cache_dir: Path | None = None
            resolved_raw_dir = resolve_target_dir(RAW_DATA_DIR, self.base_dir)

            for manifests in DATASET_REGISTRY.values():
                for m in manifests:
                    m_target = resolve_target_dir(m.target_dir, self.base_dir)
                    if self.image_dir.resolve() == m_target.resolve():
                        matched_cache_dir = resolve_target_dir(m.cache_dir, self.base_dir)
                        break
                if matched_cache_dir is not None:
                    break

            if matched_cache_dir is not None:
                self.saliency_dir = matched_cache_dir
            else:
                try:
                    rel_to_raw = self.image_dir.resolve().relative_to(resolved_raw_dir.resolve())
                    self.saliency_dir = resolve_target_dir(SALIENCY_CACHE_DIR, self.base_dir) / rel_to_raw
                except ValueError:
                    try:
                        rel_path = self.image_dir.relative_to(RAW_DATA_DIR)
                        self.saliency_dir = resolve_target_dir(SALIENCY_CACHE_DIR, self.base_dir) / rel_path
                    except ValueError:
                        self.saliency_dir = resolve_target_dir(SALIENCY_CACHE_DIR, self.base_dir) / self.image_dir.name

        # Resolve filenames list
        self.filenames: list[str] = self._resolve_filenames(filenames, manifest, sample)

        # Set up synchronized transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = SynchronizedTransforms(
                crop_size=self.crop_size,
                flip_prob=self.flip_prob,
                split=self.split,
                generator=generator,
            )

    @staticmethod
    def _default_dataset_name_for_split(split: str) -> str:
        """Derive standard benchmark dataset for split when not specified."""
        norm_split = split.lower().strip()
        if norm_split == "train":
            return "div2k"
        if norm_split in ("val", "valid", "validation"):
            return "div2k"
        if norm_split == "test":
            return "kodak"
        return "div2k"

    @staticmethod
    def _resolve_manifest(dataset_name: str, split: str) -> DatasetManifest | None:
        """Match dataset name and split to registered DatasetManifest."""
        norm_name = dataset_name.lower().strip()
        norm_split = split.lower().strip()
        if norm_split in ("validation", "valid"):
            norm_split = "val"

        # Check registry by dataset family name (e.g. 'div2k', 'kodak', 'mit1003')
        if norm_name in DATASET_REGISTRY:
            manifests = DATASET_REGISTRY[norm_name]
            if len(manifests) == 1:
                return manifests[0]

            for m in manifests:
                if norm_split in m.name:
                    return m
            return manifests[0]

        # Check directly by manifest name (e.g. 'div2k_train', 'div2k_val')
        for manifests in DATASET_REGISTRY.values():
            for m in manifests:
                if m.name.lower() == norm_name:
                    return m

        return None

    def _resolve_filenames(
        self,
        filenames: Sequence[str] | None,
        manifest: DatasetManifest | None,
        sample: int | None,
    ) -> list[str]:
        """Resolve list of filenames to be served by the dataset."""
        if filenames is not None:
            files = list(filenames)
        elif self.image_dir.exists():
            valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            disk_files = {p.name for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts}
            if manifest is not None and manifest.filenames:
                matched = [f for f in manifest.filenames if f in disk_files]
                if matched:
                    files = matched
                else:
                    files = list(manifest.filenames)
            else:
                files = sorted(disk_files)
        elif manifest is not None and manifest.filenames:
            files = list(manifest.filenames)
        else:
            files = []

        if sample is not None:
            if sample <= 0:
                raise ValueError(f"Sample count must be positive, got {sample}")
            files = files[:sample]

        return files

    @property
    def image_paths(self) -> list[Path]:
        """Return full Path objects for all images in the dataset."""
        return [self.image_dir / fname for fname in self.filenames]

    def get_pair_paths(self, idx: int) -> tuple[Path, Path]:
        """Return (raw_image_path, cached_saliency_path) for an item index."""
        if idx < 0:
            idx += len(self.filenames)
        if idx < 0 or idx >= len(self.filenames):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.filenames)} items.")

        fname = self.filenames[idx]
        img_path = self.image_dir / fname

        stem_png = Path(fname).stem + ".png"
        sal_path = self.saliency_dir / stem_png
        if not sal_path.exists():
            exact_path = self.saliency_dir / fname
            if exact_path.exists():
                sal_path = exact_path

        return img_path, sal_path

    def __len__(self) -> int:
        """Total number of image-saliency pairs in the dataset."""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load paired raw image and cached saliency map, validate dimensions, and transform.

        Returns:
            Tuple (x, S) where:
                x: torch.float32 Tensor of shape (3, H_crop, W_crop) in [0.0, 1.0].
                S: torch.float32 Tensor of shape (1, H_crop, W_crop) in [0.0, 1.0].

        Raises:
            IndexError: If index is out of bounds.
            FileNotFoundError: If raw image or cached saliency map does not exist on disk.
            ValueError: If original dimensions of image and saliency map do not match.
        """
        if idx < 0:
            idx += len(self.filenames)
        if idx < 0 or idx >= len(self.filenames):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.filenames)} items.")

        fname = self.filenames[idx]
        img_path = self.image_dir / fname
        if not img_path.exists():
            raise FileNotFoundError(f"Raw image file not found at '{img_path}'.")

        # Locate cached saliency map (canonical .png format)
        stem_png = Path(fname).stem + ".png"
        sal_path = self.saliency_dir / stem_png
        if not sal_path.exists():
            exact_path = self.saliency_dir / fname
            if exact_path.exists():
                sal_path = exact_path
            else:
                raise FileNotFoundError(
                    f"Cached saliency map not found at '{sal_path}' for raw image '{fname}'. "
                    "All saliency maps must be precomputed offline. "
                    "Please run 'python scripts/prepare_data.py --extract-saliency' to generate cached saliency maps."
                )

        try:
            with Image.open(img_path) as img_pil:
                img = img_pil.convert("RGB")
                w_rgb, h_rgb = img.size
        except Exception as e:
            raise OSError(f"Failed to open raw image at '{img_path}': {e}") from e

        try:
            with Image.open(sal_path) as sal_pil:
                sal = sal_pil.convert("L")
                w_sal, h_sal = sal.size
        except Exception as e:
            raise OSError(f"Failed to open cached saliency map at '{sal_path}': {e}") from e

        if (h_rgb, w_rgb) != (h_sal, w_sal):
            raise ValueError(
                f"Dimension mismatch: RGB image '{fname}' has dimensions ({h_rgb}, {w_rgb}) "
                f"but cached saliency map '{sal_path.name}' has dimensions ({h_sal}, {w_sal}). "
                "Saliency maps must have identical spatial dimensions to raw images."
            )

        x, S = self.transform(img, sal)
        return x, S


def create_saliency_dataloader(
    dataset: SaliencyImageDataset | Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int | None = None,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    generator: torch.Generator | None = None,
    **kwargs: Any,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a standard PyTorch DataLoader configured for SaliencyImageDataset.

    When split != 'train' (or is_train=False), defaults to batch_size=1 and shuffle=False
    to handle mixed-resolution or mixed-aspect-ratio evaluation datasets without collation crashes.

    Args:
        dataset: SaliencyImageDataset or compatible PyTorch Dataset instance.
        batch_size: Batch size override (defaults to 4 for training, 1 for evaluation).
        shuffle: Whether to shuffle data (defaults to True for training, False for evaluation).
        num_workers: Subprocess worker count for data loading (default: 0).
        pin_memory: Whether to copy tensors into CUDA pinned memory before returning (default: False).
        generator: Optional PyTorch RNG generator for batch sampling determinism.
        **kwargs: Additional keyword arguments forwarded to torch.utils.data.DataLoader.

    Returns:
        Configured PyTorch DataLoader instance.
    """
    is_train = True
    if hasattr(dataset, "split"):
        split_val = str(getattr(dataset, "split", "train")).lower()
        is_train = split_val == "train"
    elif hasattr(dataset, "is_train"):
        is_train = bool(getattr(dataset, "is_train", True))

    if batch_size is None:
        batch_size = 4 if is_train else 1
    if shuffle is None:
        shuffle = bool(is_train)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        **kwargs,
    )
