"""Comprehensive unit test suite for synchronized PyTorch dataset and deterministic transforms.

Validates:
  - Bit-identical crop coordinates and horizontal flip synchronization between image and saliency.
  - Coordinate tracking of synthetic landmarks across random crops and flips.
  - Symmetric reflection padding for sub-crop inputs (both dimensions and single dimensions).
  - Deterministic uncropped full-resolution loading for evaluation/test splits.
  - PyTorch DataLoader batching and collate compatibility.
  - Dtype and numerical range validation (float32, [0.0, 1.0]).
  - Informative FileNotFoundError when cached saliency map is missing (referencing prepare_data.py).
  - ValueError reporting (H_rgb, W_rgb) vs (H_S, W_S) on dimension mismatch.
  - Multiple input interfaces: PIL Image, NumPy array, PyTorch Tensor.
  - Manifest and split auto-resolution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.dataset import (
    SaliencyImageDataset,
    SynchronizedTransforms,
    create_saliency_dataloader,
)

# ==============================================================================
# 1. Landmark Alignment & Parity Across Random Crops and Flips
# ==============================================================================


def test_synchronized_crop_and_flip_landmark_alignment() -> None:
    """Verify that synthetic landmark coordinates in image x exactly match saliency S across crops and flips."""
    height, width = 400, 400
    crop_size = 256

    # Create synthetic base images with background 0.0
    img_arr = np.zeros((height, width, 3), dtype=np.float32)
    sal_arr = np.zeros((height, width), dtype=np.float32)

    # Embed a prominent high-intensity marker at coordinate (y=160, x=140)
    marker_y, marker_x = 160, 140
    img_arr[marker_y, marker_x] = [1.0, 1.0, 1.0]
    sal_arr[marker_y, marker_x] = 1.0

    transform = SynchronizedTransforms(
        crop_size=crop_size,
        flip_prob=0.5,
        split="train",
    )

    detected_count = 0
    # Evaluate across 40 seeds
    for seed in range(40):
        gen = torch.Generator().manual_seed(seed + 1000)
        x, S = transform(img_arr, sal_arr, generator=gen)

        assert x.shape == (3, crop_size, crop_size)
        assert S.shape == (1, crop_size, crop_size)

        # Check for marker in x (channel 0) and in S
        x_marker_coords = (x[0] > 0.5).nonzero(as_tuple=False)
        s_marker_coords = (S[0] > 0.5).nonzero(as_tuple=False)

        # The landmark must either be present in both x and S, or cropped out of both
        assert len(x_marker_coords) == len(s_marker_coords), (
            f"Seed {seed}: Inconsistent landmark presence! x count={len(x_marker_coords)}, S count={len(s_marker_coords)}"
        )

        if len(x_marker_coords) > 0:
            detected_count += 1
            # Coordinates must be bit-identical between x and S
            assert torch.equal(x_marker_coords, s_marker_coords), (
                f"Seed {seed}: Spatial landmark coordinates diverged! "
                f"x coords: {x_marker_coords.tolist()}, S coords: {s_marker_coords.tolist()}"
            )

    # Ensure landmark fell inside the crop window in at least some iterations
    assert detected_count > 0, "Landmark was never captured within crop across test iterations."


def test_crop_coordinate_parity_bit_identical() -> None:
    """Verify bit-identical pixel alignment across linear gradient patterns."""
    h, w = 384, 384
    crop_size = 200

    # Pattern where pixel value uniquely identifies (y, x) coordinates
    y_coords = np.arange(h, dtype=np.float32)[:, None]
    x_coords = np.arange(w, dtype=np.float32)[None, :]
    pattern = (y_coords * 1000.0 + x_coords) / 500000.0

    # Create 3-channel image where channel 0 is identical to saliency map
    img = np.stack([pattern, pattern * 0.5, pattern * 0.25], axis=0)
    sal = pattern[None, :, :]

    # 1. Test without horizontal flip (flip_prob=0.0)
    trans_noflip = SynchronizedTransforms(crop_size=crop_size, flip_prob=0.0, split="train")
    for seed in range(10):
        gen = torch.Generator().manual_seed(seed + 42)
        x, S = trans_noflip(img, sal, generator=gen)
        # Channel 0 of x must be bit-identical to S
        assert torch.equal(x[0:1], S), f"Pixel values diverged between x[0] and S without flip (seed {seed})"

    # 2. Test with guaranteed horizontal flip (flip_prob=1.0)
    trans_flip = SynchronizedTransforms(crop_size=crop_size, flip_prob=1.0, split="train")
    for seed in range(10):
        gen = torch.Generator().manual_seed(seed + 99)
        x, S = trans_flip(img, sal, generator=gen)
        assert torch.equal(x[0:1], S), f"Pixel values diverged between x[0] and S with flip (seed {seed})"


# ==============================================================================
# 2. Padding on Images Smaller than Crop Size
# ==============================================================================


def test_padding_both_dimensions_smaller() -> None:
    """Verify symmetric reflection padding when both dimensions are smaller than crop size."""
    h_orig, w_orig = 100, 150
    crop_size = 256

    img = np.random.uniform(0.1, 0.9, size=(h_orig, w_orig, 3)).astype(np.float32)
    sal = np.random.uniform(0.1, 0.9, size=(h_orig, w_orig)).astype(np.float32)

    transform = SynchronizedTransforms(crop_size=crop_size, split="train")
    x, S = transform(img, sal)

    assert x.shape == (3, crop_size, crop_size)
    assert S.shape == (1, crop_size, crop_size)
    assert x.dtype == torch.float32
    assert S.dtype == torch.float32

    # Verify symmetric padding margins:
    pad_h = 256 - h_orig  # 156
    pad_w = 256 - w_orig  # 106
    pad_top = pad_h // 2  # 78
    pad_left = pad_w // 2  # 53

    # In a run without flip, the center region must match the original content
    gen = torch.Generator().manual_seed(42)
    trans_noflip = SynchronizedTransforms(crop_size=crop_size, flip_prob=0.0, split="train")
    x_noflip, S_noflip = trans_noflip(img, sal, generator=gen)

    unpadded_x = x_noflip[:, pad_top : pad_top + h_orig, pad_left : pad_left + w_orig]
    expected_x = torch.from_numpy(img.transpose(2, 0, 1))
    assert torch.allclose(unpadded_x, expected_x, atol=1e-6)

    unpadded_s = S_noflip[0, pad_top : pad_top + h_orig, pad_left : pad_left + w_orig]
    expected_s = torch.from_numpy(sal)
    assert torch.allclose(unpadded_s, expected_s, atol=1e-6)


def test_padding_single_dimension_smaller() -> None:
    """Verify padding when only height or only width is smaller than crop size."""
    crop_size = 256

    # 1. Height smaller than crop_size (180 < 256), width larger (300 >= 256)
    img_h_small = np.zeros((180, 300, 3), dtype=np.float32)
    sal_h_small = np.zeros((180, 300), dtype=np.float32)

    transform = SynchronizedTransforms(crop_size=crop_size, split="train")
    x, S = transform(img_h_small, sal_h_small)
    assert x.shape == (3, 256, 256)
    assert S.shape == (1, 256, 256)

    # 2. Width smaller than crop_size (180 < 256), height larger (300 >= 256)
    img_w_small = np.zeros((300, 180, 3), dtype=np.float32)
    sal_w_small = np.zeros((300, 180), dtype=np.float32)

    x2, S2 = transform(img_w_small, sal_w_small)
    assert x2.shape == (3, 256, 256)
    assert S2.shape == (1, 256, 256)


def test_tiny_image_padding_64x64() -> None:
    """Verify iterative reflection padding handles tiny inputs (64x64) and preserves device."""
    crop_size = 256
    img = torch.rand(3, 64, 64)
    sal = torch.rand(1, 64, 64)

    transform = SynchronizedTransforms(crop_size=crop_size, split="train")
    x, S = transform(img, sal)

    assert x.shape == (3, crop_size, crop_size)
    assert S.shape == (1, crop_size, crop_size)
    assert x.dtype == torch.float32
    assert S.dtype == torch.float32
    assert x.device == img.device
    assert S.device == sal.device
    assert x.min().item() >= 0.0 and x.max().item() <= 1.0
    assert S.min().item() >= 0.0 and S.max().item() <= 1.0


# ==============================================================================
# 3. Deterministic Uncropped Loading for Evaluation Splits
# ==============================================================================


@pytest.mark.parametrize("split", ["test", "val", "validation"])
def test_eval_split_deterministic_uncropped(split: str) -> None:
    """Verify evaluation and validation splits preserve full resolution without cropping or flipping."""
    h_orig, w_orig = 512, 768  # Kodak PhotoCD dimensions

    img = np.random.uniform(0.0, 1.0, size=(h_orig, w_orig, 3)).astype(np.float32)
    sal = np.random.uniform(0.0, 1.0, size=(h_orig, w_orig)).astype(np.float32)

    transform = SynchronizedTransforms(crop_size=256, split=split)
    x, S = transform(img, sal)

    # Full resolution must be maintained exactly
    assert x.shape == (3, h_orig, w_orig)
    assert S.shape == (1, h_orig, w_orig)

    # Content must match without spatial alteration
    expected_x = torch.from_numpy(img.transpose(2, 0, 1))
    expected_s = torch.from_numpy(sal[None, :, :])
    assert torch.equal(x, expected_x)
    assert torch.equal(S, expected_s)


# ==============================================================================
# 4. Dataset Iteration, Dtypes, and Normalization
# ==============================================================================


def test_saliency_image_dataset_iteration(tmp_path: Path) -> None:
    """Verify SaliencyImageDataset loads paired items, converts dtypes, and normalizes uint8 saliency."""
    raw_dir = tmp_path / "data" / "raw" / "custom"
    cache_dir = tmp_path / "data" / "saliency_cache" / "custom"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 synthetic RGB images and 8-bit single-channel saliency maps
    num_samples = 3
    for i in range(num_samples):
        img_arr = np.random.randint(0, 256, size=(300, 350, 3), dtype=np.uint8)
        sal_arr = np.random.randint(0, 256, size=(300, 350), dtype=np.uint8)

        Image.fromarray(img_arr, mode="RGB").save(raw_dir / f"img_{i:04d}.png")
        Image.fromarray(sal_arr, mode="L").save(cache_dir / f"img_{i:04d}.png")

    dataset = SaliencyImageDataset(
        image_dir=raw_dir,
        saliency_dir=cache_dir,
        split="train",
        crop_size=256,
    )

    assert len(dataset) == num_samples
    assert len(dataset.image_paths) == num_samples

    for idx in range(len(dataset)):
        x, S = dataset[idx]

        # Standard shapes
        assert x.shape == (3, 256, 256)
        assert S.shape == (1, 256, 256)

        # Standard dtypes
        assert x.dtype == torch.float32
        assert S.dtype == torch.float32

        # Standard value bounds [0.0, 1.0]
        assert x.min().item() >= 0.0 and x.max().item() <= 1.0
        assert S.min().item() >= 0.0 and S.max().item() <= 1.0


def test_negative_indexing(tmp_path: Path) -> None:
    """Verify standard negative indexing (e.g. idx = -1) in SaliencyImageDataset."""
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    num_samples = 3
    for i in range(num_samples):
        Image.new("RGB", (300, 300), color=(i * 40, i * 40, i * 40)).save(raw_dir / f"img_{i}.png")
        Image.new("L", (300, 300), color=i * 50).save(cache_dir / f"img_{i}.png")

    dataset = SaliencyImageDataset(
        image_dir=raw_dir,
        saliency_dir=cache_dir,
        split="test",
    )

    # -1 should access the last element (index 2)
    x_neg1, s_neg1 = dataset[-1]
    x_pos2, s_pos2 = dataset[2]
    assert torch.equal(x_neg1, x_pos2)
    assert torch.equal(s_neg1, s_pos2)
    assert dataset.get_pair_paths(-1) == dataset.get_pair_paths(2)

    # -3 should access the first element (index 0)
    x_neg3, s_neg3 = dataset[-3]
    x_pos0, s_pos0 = dataset[0]
    assert torch.equal(x_neg3, x_pos0)
    assert torch.equal(s_neg3, s_pos0)

    # Out of range negative index should raise IndexError
    with pytest.raises(IndexError):
        _ = dataset[-4]

    with pytest.raises(IndexError):
        _ = dataset.get_pair_paths(-4)


def test_float_tensor_normalization_scaling() -> None:
    """Verify that float tensors in [0, 255] (max > 1.0) are divided by 255.0."""
    t_unnorm = torch.tensor([[[255.0, 127.5], [0.0, 51.0]]])
    img_unnorm = t_unnorm.repeat(3, 1, 1)

    trans = SynchronizedTransforms(split="val")
    x, S = trans(img_unnorm, t_unnorm)

    assert x.max().item() <= 1.0
    assert S.max().item() <= 1.0
    expected = torch.tensor([[[1.0, 0.5], [0.0, 0.2]]])
    assert torch.allclose(S, expected, atol=1e-5)


# ==============================================================================
# 5. PyTorch DataLoader Batching and Collation
# ==============================================================================


def test_dataloader_batch_collation(tmp_path: Path) -> None:
    """Verify that PyTorch DataLoader cleanly collates batches of (x, S) into (B, 3, H, W) and (B, 1, H, W)."""
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 4
    num_samples = 8

    for i in range(num_samples):
        img = Image.new("RGB", (320, 320), color=(i * 20, 100, 150))
        sal = Image.new("L", (320, 320), color=i * 30)
        img.save(raw_dir / f"sample_{i:02d}.png")
        sal.save(cache_dir / f"sample_{i:02d}.png")

    dataset = SaliencyImageDataset(
        image_dir=raw_dir,
        saliency_dir=cache_dir,
        split="train",
        crop_size=256,
    )

    loader = create_saliency_dataloader(dataset, batch_size=batch_size, shuffle=True)

    batches_tested = 0
    for x_batch, S_batch in loader:
        assert isinstance(x_batch, torch.Tensor)
        assert isinstance(S_batch, torch.Tensor)

        assert x_batch.shape == (batch_size, 3, 256, 256)
        assert S_batch.shape == (batch_size, 1, 256, 256)
        assert x_batch.dtype == torch.float32
        assert S_batch.dtype == torch.float32

        batches_tested += 1

    assert batches_tested == num_samples // batch_size


def test_create_saliency_dataloader_eval_defaults(tmp_path: Path) -> None:
    """Verify evaluation splits default to batch_size=1 and shuffle=False to support mixed resolutions."""
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create two images with different aspect ratios
    Image.new("RGB", (300, 400)).save(raw_dir / "img1.png")
    Image.new("L", (300, 400)).save(cache_dir / "img1.png")
    Image.new("RGB", (500, 350)).save(raw_dir / "img2.png")
    Image.new("L", (500, 350)).save(cache_dir / "img2.png")

    dataset = SaliencyImageDataset(
        image_dir=raw_dir,
        saliency_dir=cache_dir,
        split="val",
    )

    loader = create_saliency_dataloader(dataset)
    assert loader.batch_size == 1
    # Check sampler is not RandomSampler
    assert not isinstance(loader.sampler, torch.utils.data.RandomSampler)

    # Iterating must succeed without dimension collation mismatch
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0][0].shape == (1, 3, 400, 300)
    assert batches[1][0].shape == (1, 3, 350, 500)


# ==============================================================================
# 6. Error Handling: Missing Saliency Map & Dimension Mismatch
# ==============================================================================


def test_missing_saliency_map_raises_informative_error(tmp_path: Path) -> None:
    """Missing cached saliency map must raise FileNotFoundError pointing to prepare_data.py --extract-saliency."""
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Save raw image without corresponding saliency map in cache_dir
    Image.new("RGB", (300, 300)).save(raw_dir / "unprocessed.png")

    dataset = SaliencyImageDataset(
        image_dir=raw_dir,
        saliency_dir=cache_dir,
        split="train",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        _ = dataset[0]

    err_msg = str(exc_info.value)
    assert "prepare_data.py --extract-saliency" in err_msg
    assert "unprocessed.png" in err_msg


def test_dimension_mismatch_raises_value_error(tmp_path: Path) -> None:
    """Mismatch between raw image dimensions and cached saliency map dimensions must raise ValueError."""
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Raw image is 300x400 (H=300, W=400)
    Image.new("RGB", (400, 300)).save(raw_dir / "mismatched.png")
    # Saliency map is 250x400 (H=250, W=400)
    Image.new("L", (400, 250)).save(cache_dir / "mismatched.png")

    dataset = SaliencyImageDataset(
        image_dir=raw_dir,
        saliency_dir=cache_dir,
        split="train",
    )

    with pytest.raises(ValueError) as exc_info:
        _ = dataset[0]

    err_msg = str(exc_info.value)
    # Must report (H_rgb, W_rgb) vs (H_S, W_S)
    assert "(300, 400)" in err_msg
    assert "(250, 400)" in err_msg


def test_missing_raw_image_raises_error(tmp_path: Path) -> None:
    """Non-existent raw image must raise FileNotFoundError."""
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = SaliencyImageDataset(
        image_dir=raw_dir,
        saliency_dir=cache_dir,
        filenames=["non_existent_image.png"],
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        _ = dataset[0]

    assert "Raw image file not found" in str(exc_info.value)


# ==============================================================================
# 7. Input Interface Flexibility (PIL, NumPy, PyTorch Tensors)
# ==============================================================================


def test_synchronized_transforms_input_types() -> None:
    """SynchronizedTransforms must accept PIL Image, NumPy array, and PyTorch Tensor."""
    h, w = 300, 300
    crop_size = 256
    transform = SynchronizedTransforms(crop_size=crop_size, split="train")

    # 1. PIL Images
    pil_img = Image.new("RGB", (w, h), color=(100, 150, 200))
    pil_sal = Image.new("L", (w, h), color=128)
    x1, S1 = transform(pil_img, pil_sal)
    assert x1.shape == (3, crop_size, crop_size) and x1.dtype == torch.float32
    assert S1.shape == (1, crop_size, crop_size) and S1.dtype == torch.float32

    # 2. NumPy arrays (uint8)
    np_img = np.full((h, w, 3), 128, dtype=np.uint8)
    np_sal = np.full((h, w), 255, dtype=np.uint8)
    x2, S2 = transform(np_img, np_sal)
    assert x2.shape == (3, crop_size, crop_size) and x2.dtype == torch.float32
    assert S2.shape == (1, crop_size, crop_size) and S2.dtype == torch.float32

    # 3. PyTorch Tensors (float32 in [0, 1])
    torch_img = torch.rand(3, h, w, dtype=torch.float32)
    torch_sal = torch.rand(1, h, w, dtype=torch.float32)
    x3, S3 = transform(torch_img, torch_sal)
    assert x3.shape == (3, crop_size, crop_size) and x3.dtype == torch.float32
    assert S3.shape == (1, crop_size, crop_size) and S3.dtype == torch.float32


# ==============================================================================
# 8. Manifest Auto-Resolution, Aliases, and Path Reconstruction
# ==============================================================================


def test_manifest_resolution_kodak_and_div2k_validation() -> None:
    """Verify manifest resolution for dataset_name='kodak' and dataset_name='div2k' with split aliases."""
    # Kodak dataset
    ds_kodak = SaliencyImageDataset(dataset_name="kodak")
    assert "kodak" in ds_kodak.image_dir.as_posix()
    assert "kodak" in ds_kodak.saliency_dir.as_posix()

    # DIV2K with split='validation' alias
    ds_div2k_val = SaliencyImageDataset(dataset_name="div2k", split="validation")
    assert "val" in ds_div2k_val.image_dir.as_posix()
    assert "val" in ds_div2k_val.saliency_dir.as_posix()

    # DIV2K with split='valid' alias
    ds_div2k_valid = SaliencyImageDataset(dataset_name="div2k", split="valid")
    assert "val" in ds_div2k_valid.image_dir.as_posix()
    assert "val" in ds_div2k_valid.saliency_dir.as_posix()


def test_image_dir_reconstruct_saliency_dir_without_dataset_name() -> None:
    """Verify saliency_dir reconstruction when image_dir is given without dataset_name."""
    # Matches known DIV2K train manifest target_dir
    ds_train = SaliencyImageDataset(image_dir=Path("data/raw/div2k/train"))
    assert ds_train.saliency_dir.as_posix().endswith("data/saliency_cache/div2k/train")

    # Reconstructs relative path for nested raw directories without truncating parent paths
    ds_custom = SaliencyImageDataset(image_dir=Path("data/raw/custom/sub/train"))
    assert ds_custom.saliency_dir.as_posix().endswith("data/saliency_cache/custom/sub/train")


def test_manifest_and_sample_resolution(tmp_path: Path) -> None:
    """Verify dataset manifest lookup and sample subsampling."""
    # Test sample subsampling with custom filenames
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for i in range(10):
        Image.new("RGB", (256, 256)).save(raw_dir / f"test_{i:02d}.png")
        Image.new("L", (256, 256)).save(cache_dir / f"test_{i:02d}.png")

    dataset = SaliencyImageDataset(
        image_dir=raw_dir,
        saliency_dir=cache_dir,
        sample=4,
    )
    assert len(dataset) == 4

    # Test invalid sample value
    with pytest.raises(ValueError):
        SaliencyImageDataset(image_dir=raw_dir, saliency_dir=cache_dir, sample=0)

    # Test invalid crop size
    with pytest.raises(ValueError):
        SynchronizedTransforms(crop_size=0)

    # Test invalid flip probability
    with pytest.raises(ValueError):
        SynchronizedTransforms(flip_prob=1.5)

    # Test out of range index
    with pytest.raises(IndexError):
        _ = dataset[10]


def test_reproducibility_with_generator() -> None:
    """Independent transform runs with the same generator seed must produce bit-identical tensors."""
    img = np.random.uniform(0.0, 1.0, size=(400, 400, 3)).astype(np.float32)
    sal = np.random.uniform(0.0, 1.0, size=(400, 400)).astype(np.float32)

    gen1 = torch.Generator().manual_seed(9876)
    gen2 = torch.Generator().manual_seed(9876)

    trans1 = SynchronizedTransforms(crop_size=256, flip_prob=0.5, generator=gen1)
    trans2 = SynchronizedTransforms(crop_size=256, flip_prob=0.5, generator=gen2)

    x1, S1 = trans1(img, sal)
    x2, S2 = trans2(img, sal)

    assert torch.equal(x1, x2)
    assert torch.equal(S1, S2)
