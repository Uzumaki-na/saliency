"""Unit tests for dataset layout scaffolding, manifests, MD5 verifier, and CLI commands."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.prepare_data import (
    download_archive_dataset,
    download_file_with_retry,
    download_kodak,
    main,
    setup_directories,
)
from src.data.manifests import (
    ALL_REQUIRED_DIRS,
    DIV2K_TRAIN_MANIFEST,
    DIV2K_VAL_ARCHIVE_MD5,
    DIV2K_VAL_MANIFEST,
    KODAK_CHECKSUMS,
    KODAK_FILENAMES,
    KODAK_MANIFEST,
    MIT1003_TEST_MANIFEST,
    DatasetManifest,
    get_manifests_for_dataset,
    resolve_target_dir,
)
from src.data.verifier import (
    CHUNK_SIZE,
    DatasetVerificationReport,
    VerificationResult,
    compute_file_md5,
    format_failure_details,
    format_verification_summary,
    verify_dataset,
)

# =====================================================================
# 1. Directory Scaffolding and Rebasing Tests
# =====================================================================


def test_setup_directories_creates_all_paths(tmp_path: Path):
    """Test that setup_directories creates all required raw and cache directories under base_dir."""
    created = setup_directories(base_dir=tmp_path, quiet=True)

    assert len(created) == len(ALL_REQUIRED_DIRS)
    for path in created:
        assert path.exists()
        assert path.is_dir()

    # Leading 'data/' is preserved under base_dir
    assert (tmp_path / "data" / "raw" / "kodak").is_dir()
    assert (tmp_path / "data" / "raw" / "div2k" / "train").is_dir()
    assert (tmp_path / "data" / "raw" / "div2k" / "val").is_dir()
    assert (tmp_path / "data" / "raw" / "mit1003" / "test").is_dir()
    assert (tmp_path / "data" / "saliency_cache" / "kodak").is_dir()
    assert (tmp_path / "data" / "saliency_cache" / "div2k" / "train").is_dir()
    assert (tmp_path / "data" / "saliency_cache" / "div2k" / "val").is_dir()
    assert (tmp_path / "data" / "saliency_cache" / "mit1003" / "test").is_dir()


def test_setup_directories_is_idempotent(tmp_path: Path):
    """Calling setup_directories repeatedly on existing directories succeeds cleanly."""
    setup_directories(base_dir=tmp_path, quiet=True)
    second_run = setup_directories(base_dir=tmp_path, quiet=True)
    assert len(second_run) == len(ALL_REQUIRED_DIRS)


def test_resolve_target_dir_rebasing(tmp_path: Path):
    """Test directory rebasing so leading 'data/' is preserved or resolved without duplication."""
    rel = Path("data/raw/kodak")

    # 1. Without base_dir: returns unchanged
    assert resolve_target_dir(rel, None) == rel

    # 2. With root base_dir: preserves 'data/'
    resolved_root = resolve_target_dir(rel, tmp_path)
    assert resolved_root == tmp_path / "data" / "raw" / "kodak"

    # 3. With base_dir ending in 'data': avoids data/data duplication
    data_base = tmp_path / "data"
    resolved_data = resolve_target_dir(rel, data_base)
    assert resolved_data == tmp_path / "data" / "raw" / "kodak"


# =====================================================================
# 2. Manifest Definition Tests
# =====================================================================


def test_manifest_filenames_and_counts():
    """Verify official filename lists and expected counts."""
    assert KODAK_MANIFEST.expected_count == 24
    assert len(KODAK_FILENAMES) == 24
    assert KODAK_FILENAMES[0] == "kodim01.png"
    assert KODAK_FILENAMES[-1] == "kodim24.png"

    assert DIV2K_TRAIN_MANIFEST.expected_count == 800
    assert DIV2K_TRAIN_MANIFEST.filenames[0] == "0001.png"
    assert DIV2K_TRAIN_MANIFEST.filenames[-1] == "0800.png"

    assert DIV2K_VAL_MANIFEST.expected_count == 100
    assert DIV2K_VAL_MANIFEST.filenames[0] == "0801.png"
    assert DIV2K_VAL_MANIFEST.filenames[-1] == "0900.png"

    assert MIT1003_TEST_MANIFEST.expected_count == 500
    assert len(MIT1003_TEST_MANIFEST.filenames) == 500


def test_kodak_official_checksums():
    """Verify official 24 Kodak PhotoCD MD5 checksums are populated in manifest."""
    assert len(KODAK_MANIFEST.checksums) == 24
    assert KODAK_MANIFEST.checksums == KODAK_CHECKSUMS
    assert KODAK_CHECKSUMS["kodim01.png"] == "797a5ed9e768bac2143197febe74e9f0"
    assert KODAK_CHECKSUMS["kodim02.png"] == "40bfc5a525f4659c132e72b7a733e45e"
    assert KODAK_CHECKSUMS["kodim24.png"] == "45a4912aaa0458940d5554ade1db5141"


def test_manifest_urls_and_archive_md5s():
    """Verify that download endpoints and archive MD5s are defined for datasets."""
    assert len(KODAK_MANIFEST.file_urls) == 24
    for urls in KODAK_MANIFEST.file_urls.values():
        assert len(urls) >= 2  # Primary + fallback mirror
        assert any("r0k.us" in u for u in urls)
        assert any("github" in u for u in urls)

    assert len(DIV2K_TRAIN_MANIFEST.archive_urls) >= 1
    assert DIV2K_TRAIN_MANIFEST.archive_md5 == "bd710d0d82d92178082697a6fa347cd3"

    assert len(DIV2K_VAL_MANIFEST.archive_urls) >= 1
    assert DIV2K_VAL_MANIFEST.archive_md5 == DIV2K_VAL_ARCHIVE_MD5
    assert DIV2K_VAL_ARCHIVE_MD5 == "9fcdda83005a5e599c4ab75282198b88"

    assert len(MIT1003_TEST_MANIFEST.archive_urls) >= 1


def test_get_manifests_for_dataset():
    """Test registry lookups with aliases."""
    assert get_manifests_for_dataset("kodak") == [KODAK_MANIFEST]
    assert get_manifests_for_dataset("div2k_train") == [DIV2K_TRAIN_MANIFEST]
    assert get_manifests_for_dataset("div2k_val") == [DIV2K_VAL_MANIFEST]
    assert get_manifests_for_dataset("div2k") == [DIV2K_TRAIN_MANIFEST, DIV2K_VAL_MANIFEST]
    assert get_manifests_for_dataset("mit1003") == [MIT1003_TEST_MANIFEST]
    assert len(get_manifests_for_dataset("all")) == 4

    with pytest.raises(KeyError, match="Unknown dataset 'invalid_dataset'"):
        get_manifests_for_dataset("invalid_dataset")


# =====================================================================
# 3. Streaming MD5 Computation Tests
# =====================================================================


def test_compute_file_md5_small_file(tmp_path: Path):
    """Test compute_file_md5 on a small payload."""
    test_file = tmp_path / "test.bin"
    payload = b"Hello, Saliency Compression!"
    test_file.write_bytes(payload)

    expected = hashlib.md5(payload).hexdigest()
    assert compute_file_md5(test_file) == expected


def test_compute_file_md5_large_file(tmp_path: Path):
    """Test compute_file_md5 across multiple 64KB streaming chunks."""
    test_file = tmp_path / "large.bin"
    payload = b"A" * (CHUNK_SIZE * 3 + 1234)
    test_file.write_bytes(payload)

    expected = hashlib.md5(payload).hexdigest()
    assert compute_file_md5(test_file) == expected


def test_compute_file_md5_non_existent(tmp_path: Path):
    """compute_file_md5 raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        compute_file_md5(tmp_path / "missing.bin")


# =====================================================================
# 4. Verifier Logic Tests (Mock File Trees)
# =====================================================================


def test_verify_dataset_all_valid(tmp_path: Path):
    """Verify returns is_valid == True when all files exist and match checksums."""
    target_dir = tmp_path / "data" / "raw" / "kodak"
    target_dir.mkdir(parents=True)

    filenames = ("kodim01.png", "kodim02.png")
    checksums = {}
    for fname in filenames:
        content = f"content-for-{fname}".encode()
        (target_dir / fname).write_bytes(content)
        checksums[fname] = hashlib.md5(content).hexdigest()

    manifest = DatasetManifest(
        name="test_kodak",
        target_dir=Path("data/raw/kodak"),
        cache_dir=Path("data/saliency_cache/kodak"),
        filenames=filenames,
        checksums=checksums,
    )

    report = verify_dataset(manifest, base_dir=tmp_path)
    assert report.is_valid is True
    assert report.valid_count == 2
    assert report.missing_count == 0
    assert report.corrupted_count == 0


def test_verify_dataset_detects_missing_files(tmp_path: Path):
    """Verify flags missing files and provides filenames."""
    target_dir = tmp_path / "data" / "raw" / "kodak"
    target_dir.mkdir(parents=True)

    (target_dir / "kodim01.png").write_bytes(b"image 1 bytes")

    manifest = DatasetManifest(
        name="test_kodak",
        target_dir=Path("data/raw/kodak"),
        cache_dir=Path("data/saliency_cache/kodak"),
        filenames=("kodim01.png", "kodim02.png"),
    )

    report = verify_dataset(manifest, base_dir=tmp_path)
    assert report.is_valid is False
    assert report.valid_count == 1
    assert report.missing_count == 1
    assert report.missing_files == ["kodim02.png"]


def test_verify_dataset_detects_corrupted_hash(tmp_path: Path):
    """Verify flags hash mismatch as corrupted."""
    target_dir = tmp_path / "data" / "raw" / "kodak"
    target_dir.mkdir(parents=True)

    (target_dir / "kodim01.png").write_bytes(b"corrupted bytes")

    manifest = DatasetManifest(
        name="test_kodak",
        target_dir=Path("data/raw/kodak"),
        cache_dir=Path("data/saliency_cache/kodak"),
        filenames=("kodim01.png",),
        checksums={"kodim01.png": "expected_md5_that_will_not_match"},
    )

    report = verify_dataset(manifest, base_dir=tmp_path)
    assert report.is_valid is False
    assert report.valid_count == 0
    assert report.corrupted_count == 1
    assert "MD5 hash mismatch" in report.corrupted_files[0].error_message


def test_verify_dataset_detects_empty_file(tmp_path: Path):
    """Verify flags 0-byte file as corrupted."""
    target_dir = tmp_path / "data" / "raw" / "kodak"
    target_dir.mkdir(parents=True)

    (target_dir / "kodim01.png").write_bytes(b"")

    manifest = DatasetManifest(
        name="test_kodak",
        target_dir=Path("data/raw/kodak"),
        cache_dir=Path("data/saliency_cache/kodak"),
        filenames=("kodim01.png",),
    )

    report = verify_dataset(manifest, base_dir=tmp_path)
    assert report.is_valid is False
    assert report.corrupted_count == 1
    assert "empty (0 bytes)" in report.corrupted_files[0].error_message


def test_format_summary_and_failure_details():
    """Test string formatting of report summary table and diagnostic logs."""
    report = DatasetVerificationReport(
        dataset_name="kodak",
        target_dir=Path("data/raw/kodak"),
        expected_count=2,
        results=[
            VerificationResult("kodim01.png", "kodak", Path("data/raw/kodak/kodim01.png"), "VALID"),
            VerificationResult(
                "kodim02.png",
                "kodak",
                Path("data/raw/kodak/kodim02.png"),
                "CORRUPTED",
                error_message="Hash mismatch",
            ),
        ],
    )
    summary = format_verification_summary([report])
    assert "kodak" in summary
    assert "FAILED" in summary

    details = format_failure_details([report])
    assert "Hash mismatch" in details
    assert "kodim02.png" in details


# =====================================================================
# 5. CLI Execution & Acquisition Tests
# =====================================================================


def test_cli_setup_dirs(tmp_path: Path):
    """CLI --setup-dirs creates directories under base_dir and exits 0."""
    code = main(["--setup-dirs", "--data-dir", str(tmp_path)])
    assert code == 0
    assert (tmp_path / "data" / "raw" / "kodak").is_dir()
    assert (tmp_path / "data" / "saliency_cache" / "kodak").is_dir()


def test_cli_extract_saliency_stub(capsys: pytest.CaptureFixture):
    """CLI --extract-saliency informs user and exits 0."""
    code = main(["--extract-saliency"])
    assert code == 0
    captured = capsys.readouterr()
    assert "FastSal offline pre-extraction" in captured.out
    assert "Story 1.2" in captured.out


def test_cli_verify_data_success_mock(tmp_path: Path):
    """CLI --verify-data returns 0 when mock dataset matches manifest checksums."""
    target_dir = tmp_path / "data" / "raw" / "kodak"
    target_dir.mkdir(parents=True)
    mock_checksums = {}
    for fname in KODAK_FILENAMES:
        content = f"content-{fname}".encode()
        (target_dir / fname).write_bytes(content)
        mock_checksums[fname] = hashlib.md5(content).hexdigest()

    with patch.dict(KODAK_MANIFEST.checksums, mock_checksums, clear=True):
        code = main(["--verify-data", "--dataset", "kodak", "--data-dir", str(tmp_path)])
        assert code == 0


def test_cli_verify_data_missing_fails(tmp_path: Path, capsys: pytest.CaptureFixture):
    """CLI --verify-data returns 1 and diagnostics on stderr when files are missing."""
    target_dir = tmp_path / "data" / "raw" / "kodak"
    target_dir.mkdir(parents=True)
    (target_dir / "kodim01.png").write_bytes(b"valid")

    code = main(["--verify-data", "--dataset", "kodak", "--data-dir", str(tmp_path)])
    assert code == 1

    captured = capsys.readouterr()
    assert "FAILED" in captured.out
    assert "Missing files" in captured.err
    assert "kodim02.png" in captured.err
    assert "--download-only" in captured.err


def test_cli_download_retry_offline_handling(tmp_path: Path):
    """CLI download fails gracefully when offline without unhandled tracebacks."""
    dest = tmp_path / "kodim01.png"
    ok = download_file_with_retry(
        urls=["http://127.0.0.1:9/unreachable.png"],
        destination=dest,
        retries=2,
        backoff_factor=0.01,
        timeout=0.5,
    )
    assert ok is False
    assert not dest.exists()


def test_download_kodak_redownloads_corrupted_file(tmp_path: Path):
    """Existing file with hash mismatch is identified as corrupted and re-downloaded."""
    target_dir = tmp_path / "data" / "raw" / "kodak"
    target_dir.mkdir(parents=True)

    test_file = target_dir / "kodim01.png"
    test_file.write_bytes(b"corrupted previous content")

    valid_content = b"valid kodim01 content"
    valid_md5 = hashlib.md5(valid_content).hexdigest()

    def mock_download(urls, dest, **kwargs):
        dest.write_bytes(valid_content)
        return True

    test_manifest = DatasetManifest(
        name="kodak",
        target_dir=Path("data/raw/kodak"),
        cache_dir=Path("data/saliency_cache/kodak"),
        filenames=("kodim01.png",),
        checksums={"kodim01.png": valid_md5},
        file_urls={"kodim01.png": ("http://dummy.url/kodim01.png",)},
    )

    with patch("scripts.prepare_data.download_file_with_retry", side_effect=mock_download):
        ok = download_kodak(test_manifest, base_dir=tmp_path)
        assert ok is True

    assert test_file.read_bytes() == valid_content
    assert compute_file_md5(test_file) == valid_md5


def test_download_archive_dataset_skips_existing_valid(tmp_path: Path):
    """If archive already exists with matching MD5, download is skipped and files are extracted."""
    target_dir = tmp_path / "data" / "raw" / "div2k" / "train"
    archive_dest = target_dir.parent / "DIV2K_train_HR.zip"
    archive_dest.parent.mkdir(parents=True, exist_ok=True)

    # Create dummy zip with 0001.png
    with zipfile.ZipFile(archive_dest, "w") as zf:
        zf.writestr("0001.png", b"dummy 0001 image bytes")

    archive_md5 = compute_file_md5(archive_dest)

    manifest = DatasetManifest(
        name="div2k_train",
        target_dir=Path("data/raw/div2k/train"),
        cache_dir=Path("data/saliency_cache/div2k/train"),
        filenames=("0001.png",),
        archive_urls=("http://dummy/DIV2K_train_HR.zip",),
        archive_md5=archive_md5,
    )

    download_mock = patch("scripts.prepare_data.download_file_with_retry")
    with download_mock as mock_dl:
        ok = download_archive_dataset(manifest, base_dir=tmp_path)
        assert ok is True
        mock_dl.assert_not_called()  # Download was skipped!

    assert (target_dir / "0001.png").exists()
    assert (target_dir / "0001.png").read_bytes() == b"dummy 0001 image bytes"


def test_download_archive_dataset_flattens_nested_directory(tmp_path: Path):
    """Zip archive with nested root directories (e.g. DIV2K_train_HR/0001.png) extracts flattened."""
    target_dir = tmp_path / "data" / "raw" / "div2k" / "train"
    archive_dest = target_dir.parent / "DIV2K_train_HR.zip"
    archive_dest.parent.mkdir(parents=True, exist_ok=True)

    # Create zip with nested folder hierarchy
    with zipfile.ZipFile(archive_dest, "w") as zf:
        zf.writestr("DIV2K_train_HR/0001.png", b"image 1 content")
        zf.writestr("DIV2K_train_HR/0002.png", b"image 2 content")

    archive_md5 = compute_file_md5(archive_dest)

    manifest = DatasetManifest(
        name="div2k_train",
        target_dir=Path("data/raw/div2k/train"),
        cache_dir=Path("data/saliency_cache/div2k/train"),
        filenames=("0001.png", "0002.png"),
        archive_urls=("http://dummy/DIV2K_train_HR.zip",),
        archive_md5=archive_md5,
    )

    ok = download_archive_dataset(manifest, base_dir=tmp_path)
    assert ok is True

    # Check files are flattened directly into target_dir
    assert (target_dir / "0001.png").is_file()
    assert (target_dir / "0002.png").is_file()
    assert (target_dir / "0001.png").read_bytes() == b"image 1 content"
    assert not (target_dir / "DIV2K_train_HR").exists()  # No nested subfolder
