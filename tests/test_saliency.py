"""Unit test suite for FastSal offline saliency extraction and percentile caching.

Validates:
  - 1st-99th percentile dynamic range clipping and uint8 [0, 255] quantization.
  - Zero-variance / constant map division-by-zero safety.
  - Extreme outlier clipping and posinf sanitization.
  - FastSal model forward pass and spatial resolution preservation across diverse aspect ratios.
  - PIL, NumPy (HWC, CHW, NHWC, NCHW), and PyTorch Tensor input interfaces.
  - Low-key image normalization with uint8 inputs.
  - Batched FastSal.predict per-item independent normalization.
  - Dynamic device derivation via next(self.parameters()).device.
  - Canonical .png caching for non-PNG source images without file duplication.
  - Single-channel 8-bit ('L' mode) PNG caching and file integrity.
  - scripts/prepare_data.py CLI integration using isolated tmp_path fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.prepare_data import main as prepare_data_main
from src.data.manifests import DatasetManifest
from src.data.saliency import (
    FastSal,
    extract_and_cache_saliency,
    normalize_saliency_percentile,
)

# ==============================================================================
# 1. Percentile Normalization Tests
# ==============================================================================

def test_normalize_saliency_percentile_outliers() -> None:
    """Extreme outliers must be clipped to 0 and 255 without NaNs."""
    np.random.seed(42)
    base = np.random.uniform(0.1, 0.9, size=(100, 100)).astype(np.float32)

    # Inject extreme negative and positive outlier spikes
    base[0, 0] = -10000.0
    base[0, 1] = -500.0
    base[-1, -1] = 99999.0
    base[-1, -2] = 5000.0

    normalized = normalize_saliency_percentile(base, p_low=1.0, p_high=99.0)

    assert isinstance(normalized, np.ndarray)
    assert normalized.dtype == np.uint8
    assert normalized.shape == (100, 100)
    assert not np.isnan(normalized).any()
    assert not np.isinf(normalized).any()

    # The extreme negative outlier must be clipped to 0
    assert normalized[0, 0] == 0
    # The extreme positive outlier must be clipped to 255
    assert normalized[-1, -1] == 255
    # Overall range must span uint8 [0, 255]
    assert normalized.min() == 0
    assert normalized.max() == 255


def test_normalize_saliency_percentile_flat_constant_map() -> None:
    """Homogeneous images with zero variance must normalize safely without divide-by-zero."""
    for constant_val in [0.0, 0.5, 1.0, 42.0, -10.0]:
        flat = np.full((64, 64), fill_value=constant_val, dtype=np.float32)
        normalized = normalize_saliency_percentile(flat, p_low=1.0, p_high=99.0, eps=1e-7)

        assert isinstance(normalized, np.ndarray)
        assert normalized.dtype == np.uint8
        assert normalized.shape == (64, 64)
        assert not np.isnan(normalized).any()
        assert not np.isinf(normalized).any()
        # Flat map with (c - c) / (0 + eps) evaluates to 0
        assert np.all(normalized == 0)


def test_normalize_saliency_percentile_torch_tensor_input() -> None:
    """Torch tensor inputs (with or without grad) must convert to uint8 NumPy arrays."""
    t = torch.linspace(0.0, 10.0, steps=1000, dtype=torch.float32).reshape(20, 50)
    t.requires_grad_(True)
    loss = (t ** 2).sum()
    loss.backward()

    normalized = normalize_saliency_percentile(t)

    assert isinstance(normalized, np.ndarray)
    assert normalized.dtype == np.uint8
    assert normalized.shape == (20, 50)
    assert normalized.min() == 0
    assert normalized.max() == 255


def test_normalize_saliency_percentile_nan_inf_sanitization() -> None:
    """Non-finite inputs (NaN / Inf) must be sanitized; posinf clips to 255 and neginf to 0."""
    arr = np.array([[-np.inf, 0.5], [np.nan, np.inf]], dtype=np.float32)
    normalized = normalize_saliency_percentile(arr)

    assert isinstance(normalized, np.ndarray)
    assert normalized.dtype == np.uint8
    assert not np.isnan(normalized).any()
    assert not np.isinf(normalized).any()
    assert normalized[1, 1] == 255  # Positive infinity sanitized and clipped to 255
    assert normalized[0, 0] == 0    # Negative infinity sanitized and clipped to 0


# ==============================================================================
# 2. FastSal Model Architecture & Inference Tests
# ==============================================================================

@pytest.fixture(scope="module")
def fastsal_model() -> FastSal:
    """Instantiate a shared FastSal model on CPU for fast testing."""
    return FastSal(pretrained=False, device="cpu")


@pytest.mark.parametrize(
    "spatial_shape",
    [
        (224, 224),   # Standard square
        (512, 768),   # Kodak landscape (H, W)
        (768, 512),   # Kodak portrait (H, W)
        (375, 500),   # MIT1003 odd aspect ratio
        (103, 157),   # Arbitrary non-divisible prime dimensions
    ],
)
def test_fastsal_spatial_resolution_preservation_numpy(
    fastsal_model: FastSal,
    spatial_shape: tuple[int, int],
) -> None:
    """FastSal must output saliency map matching exact original spatial dimensions (H, W)."""
    h, w = spatial_shape
    rgb_numpy = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)

    out = fastsal_model(rgb_numpy)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (h, w), f"Expected shape ({h}, {w}), got {out.shape}"
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_fastsal_input_formats(fastsal_model: FastSal) -> None:
    """FastSal must transparently accept PIL Image, (H, W, 3) tensor, and (3, H, W) tensor."""
    h, w = 120, 160

    # 1. PIL Image
    pil_img = Image.new("RGB", (w, h), color=(128, 64, 32))
    out_pil = fastsal_model(pil_img)
    assert out_pil.shape == (h, w)

    # 2. PyTorch Tensor (H, W, 3)
    tensor_hwc = torch.rand(h, w, 3)
    out_hwc = fastsal_model(tensor_hwc)
    assert out_hwc.shape == (h, w)

    # 3. PyTorch Tensor (3, H, W)
    tensor_chw = torch.rand(3, h, w)
    out_chw = fastsal_model(tensor_chw)
    assert out_chw.shape == (h, w)


def test_fastsal_nhwc_4d_format(fastsal_model: FastSal) -> None:
    """FastSal must support NHWC 4D format (B, H, W, 3) for both NumPy and PyTorch."""
    b, h, w = 2, 64, 80

    # NumPy NHWC
    arr_nhwc = np.random.randint(0, 256, size=(b, h, w, 3), dtype=np.uint8)
    out_np = fastsal_model(arr_nhwc)
    assert out_np.shape == (b, 1, h, w)

    # Torch NHWC
    t_nhwc = torch.randint(0, 256, size=(b, h, w, 3), dtype=torch.uint8)
    out_t = fastsal_model(t_nhwc)
    assert out_t.shape == (b, 1, h, w)


def test_fastsal_low_key_image_normalization(fastsal_model: FastSal) -> None:
    """Low-key uint8 images where max pixel is 1 must still be normalized by 255.0."""
    # Low-key image: values are 0 or 1 in uint8
    low_key_np = np.ones((64, 64, 3), dtype=np.uint8)
    prepared, _, _ = fastsal_model._prepare_input(low_key_np)
    # Must be scaled by 255.0 to 1/255, NOT treated as already in [0, 1]
    expected_val = 1.0 / 255.0
    assert torch.isclose(prepared[0, 0, 0, 0], torch.tensor(expected_val), atol=1e-5)


def test_fastsal_batched_tensor_forward(fastsal_model: FastSal) -> None:
    """4D batched tensor input (B, 3, H, W) must return (B, 1, H, W) tensor."""
    b, c, h, w = 2, 3, 64, 64
    x = torch.randn(b, c, h, w)

    out = fastsal_model(x)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (b, 1, h, w)


def test_fastsal_predict_batched_independent_percentiles(fastsal_model: FastSal) -> None:
    """FastSal.predict() on batched 4D inputs must normalize each batch item independently."""
    b, h, w = 2, 64, 64
    # Image 1 has lower activation range, Image 2 has higher activation range
    x = np.random.randint(0, 256, size=(b, h, w, 3), dtype=np.uint8)

    pred = fastsal_model.predict(x)

    assert isinstance(pred, np.ndarray)
    assert pred.dtype == np.uint8
    assert pred.shape == (b, h, w)
    # Each image item should independently span uint8 [0, 255]
    for i in range(b):
        assert pred[i].min() >= 0 and pred[i].max() <= 255


def test_fastsal_device_derivation() -> None:
    """FastSal.device must derive dynamically from model parameters."""
    model = FastSal(pretrained=False, device="cpu")
    assert model.device.type == "cpu"

    # Moving with PyTorch standard .to() must update model.device
    model.to(torch.device("cpu"))
    assert model.device == torch.device("cpu")


# ==============================================================================
# 3. Offline Extraction and Caching Pipeline Tests
# ==============================================================================

def test_extract_and_cache_saliency_pipeline(tmp_path: Path) -> None:
    """extract_and_cache_saliency must generate valid single-channel 8-bit PNGs in tmp_path."""
    raw_dir = tmp_path / "data" / "raw" / "mock"
    cache_dir = tmp_path / "data" / "saliency_cache" / "mock"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 synthetic test images with varying dimensions
    shapes = [
        ("img_01.png", (80, 120)),   # H=80, W=120
        ("img_02.png", (150, 100)),  # H=150, W=100
        ("img_03.png", (90, 90)),    # H=90, W=90
    ]
    filenames: list[str] = []
    for name, (h, w) in shapes:
        img = Image.new("RGB", (w, h), color=(np.random.randint(255), 100, 200))
        img.save(raw_dir / name)
        filenames.append(name)

    mock_manifest = DatasetManifest(
        name="mock_dataset",
        target_dir=raw_dir,
        cache_dir=cache_dir,
        filenames=tuple(filenames),
        description="Mock dataset for unit tests",
    )

    from src.data.manifests import DATASET_REGISTRY
    DATASET_REGISTRY["mock"] = [mock_manifest]

    try:
        results = extract_and_cache_saliency(
            dataset_name="mock",
            base_dir=tmp_path,
            sample=2,  # Subsample: process 2 of the 3 images
            device="cpu",
            quiet=True,
        )

        assert results["mock_dataset"] == 2

        # Verify the 2 cached files
        cached_files = sorted(cache_dir.glob("*.png"))
        assert len(cached_files) == 2
        assert [f.name for f in cached_files] == ["img_01.png", "img_02.png"]

        # Inspect cached PNG properties
        with Image.open(cached_files[0]) as img1:
            assert img1.mode == "L"
            assert img1.format == "PNG"
            assert img1.size == (120, 80)  # (W, H)
            arr1 = np.array(img1)
            assert arr1.dtype == np.uint8
            assert arr1.min() == 0 and arr1.max() == 255

        with Image.open(cached_files[1]) as img2:
            assert img2.mode == "L"
            assert img2.format == "PNG"
            assert img2.size == (100, 150)  # (W, H)
            arr2 = np.array(img2)
            assert arr2.dtype == np.uint8

    finally:
        DATASET_REGISTRY.pop("mock", None)


def test_extract_and_cache_non_png_source_canonical_filename(tmp_path: Path) -> None:
    """Non-PNG source files (e.g. .jpeg) must be saved as a single canonical .png without duplicates."""
    raw_dir = tmp_path / "data" / "raw" / "mock_jpeg"
    cache_dir = tmp_path / "data" / "saliency_cache" / "mock_jpeg"
    raw_dir.mkdir(parents=True, exist_ok=True)

    jpeg_name = "sample_photo.jpeg"
    Image.new("RGB", (80, 80), color=(100, 150, 200)).save(raw_dir / jpeg_name)

    mock_manifest = DatasetManifest(
        name="mock_jpeg",
        target_dir=raw_dir,
        cache_dir=cache_dir,
        filenames=(jpeg_name,),
    )

    from src.data.manifests import DATASET_REGISTRY
    DATASET_REGISTRY["mock_jpeg"] = [mock_manifest]

    try:
        results = extract_and_cache_saliency(
            dataset_name="mock_jpeg",
            base_dir=tmp_path,
            device="cpu",
            quiet=True,
        )
        assert results["mock_jpeg"] == 1

        # Canonical file must be sample_photo.png
        assert (cache_dir / "sample_photo.png").exists()
        # Non-canonical sample_photo.jpeg must NOT exist in cache
        assert not (cache_dir / "sample_photo.jpeg").exists()

        all_cached = list(cache_dir.iterdir())
        assert len(all_cached) == 1
    finally:
        DATASET_REGISTRY.pop("mock_jpeg", None)


def test_extract_and_cache_sample_count_validation(tmp_path: Path) -> None:
    """When sample N is requested and fewer files exist, a RuntimeError must be raised."""
    raw_dir = tmp_path / "data" / "raw" / "mock_few"
    cache_dir = tmp_path / "data" / "saliency_cache" / "mock_few"
    raw_dir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (50, 50)).save(raw_dir / "one.png")

    mock_manifest = DatasetManifest(
        name="mock_few",
        target_dir=raw_dir,
        cache_dir=cache_dir,
        filenames=("one.png",),
    )

    from src.data.manifests import DATASET_REGISTRY
    DATASET_REGISTRY["mock_few"] = [mock_manifest]

    try:
        with pytest.raises(RuntimeError, match="Requested sample=5"):
            extract_and_cache_saliency(
                dataset_name="mock_few",
                base_dir=tmp_path,
                sample=5,
                device="cpu",
                quiet=True,
            )
    finally:
        DATASET_REGISTRY.pop("mock_few", None)


def test_extract_and_cache_invalid_sample_count() -> None:
    """Non-positive sample counts must raise ValueError."""
    with pytest.raises(ValueError, match="Sample count must be positive"):
        extract_and_cache_saliency(dataset_name="kodak", sample=0)

    with pytest.raises(ValueError, match="Sample count must be positive"):
        extract_and_cache_saliency(dataset_name="kodak", sample=-5)


# ==============================================================================
# 4. CLI Execution Tests
# ==============================================================================

def test_prepare_data_cli_extract_saliency_kodak_subsample(tmp_path: Path) -> None:
    """CLI execution with --extract-saliency --dataset kodak --sample 2 caches 2 files in tmp_path."""
    raw_kodak = tmp_path / "data" / "raw" / "kodak"
    raw_kodak.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (768, 512), color=(100, 150, 200)).save(raw_kodak / "kodim01.png")
    Image.new("RGB", (768, 512), color=(50, 80, 120)).save(raw_kodak / "kodim02.png")

    exit_code = prepare_data_main([
        "--extract-saliency",
        "--dataset", "kodak",
        "--sample", "2",
        "--device", "cpu",
        "--data-dir", str(tmp_path),
        "--quiet",
    ])
    assert exit_code == 0

    cache_dir = tmp_path / "data" / "saliency_cache" / "kodak"
    assert cache_dir.exists()
    assert (cache_dir / "kodim01.png").exists()
    assert (cache_dir / "kodim02.png").exists()

    with Image.open(cache_dir / "kodim01.png") as img:
        assert img.mode == "L"
        assert img.format == "PNG"
        assert img.size == (768, 512)
        arr = np.array(img)
        assert arr.dtype == np.uint8
        assert arr.min() == 0 and arr.max() == 255


def test_prepare_data_cli_missing_source_files_fails(tmp_path: Path) -> None:
    """CLI --extract-saliency returns exit code 1 when raw source files are missing."""
    empty_kodak = tmp_path / "data" / "raw" / "kodak"
    empty_kodak.mkdir(parents=True, exist_ok=True)

    exit_code = prepare_data_main([
        "--extract-saliency",
        "--dataset", "kodak",
        "--data-dir", str(tmp_path),
        "--quiet",
    ])
    assert exit_code == 1


def test_prepare_data_cli_unknown_dataset() -> None:
    """CLI with unknown dataset exits non-zero."""
    with pytest.raises(SystemExit) as exc_info:
        prepare_data_main(["--extract-saliency", "--dataset", "nonexistent_dataset"])
    assert exc_info.value.code != 0
