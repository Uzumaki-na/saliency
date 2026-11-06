#!/usr/bin/env python3
"""Dataset acquisition, scaffolding, integrity verification, and saliency extraction script.

Supports:
  --setup-dirs: Scaffold the standard data directory hierarchy.
  --verify-data: Verify raw image integrity and reference checksums.
  --download-only: Ingest public benchmark datasets (Kodak, DIV2K, MIT1003).
  --extract-saliency: Extract visual saliency maps offline and cache to disk.
  --sample N: Subsample count for fast verification and incremental extraction.
  --device: Execution device for FastSal ('cuda', 'cpu', etc.).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

# Ensure project root is in sys.path when executed directly
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.manifests import (  # noqa: E402
    ALL_REQUIRED_DIRS,
    DATASET_REGISTRY,
    DatasetManifest,
    get_manifests_for_dataset,
    resolve_target_dir,
)
from src.data.saliency import extract_and_cache_saliency  # noqa: E402
from src.data.verifier import (  # noqa: E402
    compute_file_md5,
    format_verification_summary,
    log_verification_failure_to_stderr,
    verify_datasets,
)


def setup_directories(base_dir: Path | None = None, quiet: bool = False) -> list[Path]:
    """Scaffold all required raw and saliency cache directories under base_dir.

    Args:
        base_dir: Base directory override (default: current working directory).
        quiet: If True, suppress stdout messages.

    Returns:
        List of created or existing directories.
    """
    created_dirs: list[Path] = []
    for rel_path in ALL_REQUIRED_DIRS:
        target_path = resolve_target_dir(rel_path, base_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        created_dirs.append(target_path)
        if not quiet:
            print(f"[SETUP] Ready directory: {target_path}")

    return created_dirs


def download_file_with_retry(
    urls: Sequence[str],
    destination: Path,
    retries: int = 3,
    backoff_factor: float = 1.0,
    timeout: float = 20.0,
) -> bool:
    """Download a file with fallback URLs and exponential backoff retry.

    Args:
        urls: Sequence of URL mirrors in priority order.
        destination: Local file path where the payload will be saved.
        retries: Number of retry attempts per URL.
        backoff_factor: Multiplier for retry sleep delay.
        timeout: Socket timeout in seconds.

    Returns:
        True if download succeeded, False if all URLs and retries were exhausted.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = destination.with_suffix(destination.suffix + ".tmp")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }

    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    if response.status not in (200, 206):
                        raise urllib.error.HTTPError(
                            url, response.status, f"HTTP status {response.status}", response.headers, None
                        )
                    with open(temp_dest, "wb") as out_file:
                        while chunk := response.read(64 * 1024):
                            out_file.write(chunk)

                # Atomically rename temporary file
                if temp_dest.exists():
                    temp_dest.replace(destination)
                return True
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                if temp_dest.exists():
                    temp_dest.unlink(missing_ok=True)
                if attempt < retries:
                    sleep_time = backoff_factor * (2 ** (attempt - 1))
                    time.sleep(sleep_time)
                else:
                    print(
                        f"[DOWNLOAD WARNING] Failed URL {url} (attempt {attempt}/{retries}): {e}",
                        file=sys.stderr,
                    )
    return False


def download_kodak(manifest: DatasetManifest, base_dir: Path | None = None) -> bool:
    """Download the 24 Kodak PhotoCD images into target directory."""
    target_dir = resolve_target_dir(manifest.target_dir, base_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DOWNLOAD] Acquiring Kodak PhotoCD Suite (24 images) -> {target_dir}")

    success_count = 0
    failures: list[str] = []

    for fname in manifest.filenames:
        dest_file = target_dir / fname
        expected_md5 = manifest.checksums.get(fname)

        if dest_file.exists() and dest_file.stat().st_size > 0:
            if expected_md5 is not None:
                try:
                    actual_md5 = compute_file_md5(dest_file)
                    if actual_md5.lower() == expected_md5.lower():
                        print(f"  [EXISTS & VALID] {fname}")
                        success_count += 1
                        continue
                    else:
                        print(
                            f"  [CORRUPTED] {fname} checksum mismatch (expected {expected_md5}, "
                            f"got {actual_md5}), re-downloading...",
                            file=sys.stderr,
                        )
                except OSError:
                    pass
            else:
                print(f"  [EXISTS] {fname}")
                success_count += 1
                continue

        urls = manifest.file_urls.get(fname, ())
        print(f"  [FETCHING] {fname}...")
        ok = download_file_with_retry(urls, dest_file)
        if ok:
            if expected_md5 is not None:
                actual_md5 = compute_file_md5(dest_file)
                if actual_md5.lower() != expected_md5.lower():
                    print(
                        f"  [FAIL] Downloaded {fname} MD5 mismatch: expected {expected_md5}, got {actual_md5}",
                        file=sys.stderr,
                    )
                    failures.append(fname)
                    continue
            print(f"  [OK] Saved {fname} ({dest_file.stat().st_size} bytes)")
            success_count += 1
        else:
            failures.append(fname)
            print(f"  [FAIL] Could not acquire {fname}", file=sys.stderr)

    if failures:
        print(
            f"\n[ERROR] Failed to download {len(failures)} Kodak images: {', '.join(failures)}.\n"
            f"If offline, please ensure network access or manually place PNG files into: {target_dir}",
            file=sys.stderr,
        )
        return False

    print(f"[DOWNLOAD COMPLETE] Successfully acquired all {success_count}/24 Kodak images.")
    return True


def download_archive_dataset(manifest: DatasetManifest, base_dir: Path | None = None) -> bool:
    """Download and extract a zip archive dataset (DIV2K or MIT1003)."""
    target_dir = resolve_target_dir(manifest.target_dir, base_dir)

    if not manifest.archive_urls:
        print(f"[DOWNLOAD ERROR] No archive URLs registered for dataset '{manifest.name}'.", file=sys.stderr)
        return False

    archive_name = Path(manifest.archive_urls[0]).name
    archive_dest = target_dir.parent / archive_name
    archive_dest.parent.mkdir(parents=True, exist_ok=True)

    need_download = True
    if archive_dest.exists() and archive_dest.stat().st_size > 0:
        if manifest.archive_md5:
            try:
                existing_md5 = compute_file_md5(archive_dest)
                if existing_md5.lower() == manifest.archive_md5.lower():
                    print(f"[EXISTS & VALID] Archive {archive_dest} matches expected MD5. Skipping download.")
                    need_download = False
                else:
                    print(
                        f"[CORRUPTED ARCHIVE] Existing {archive_dest} hash mismatch "
                        f"(expected {manifest.archive_md5}, got {existing_md5}), re-downloading...",
                        file=sys.stderr,
                    )
            except OSError:
                pass
        else:
            print(f"[EXISTS] Archive {archive_dest} already exists. Skipping download.")
            need_download = False

    if need_download:
        print(f"[DOWNLOAD] Ingesting archive for '{manifest.name}' from {manifest.archive_urls[0]}...")
        ok = download_file_with_retry(manifest.archive_urls, archive_dest)
        if not ok:
            print(
                f"[ERROR] Could not download archive {archive_name}.\n"
                f"Please check your internet connection or place {archive_name} into {archive_dest.parent}",
                file=sys.stderr,
            )
            return False

        if manifest.archive_md5:
            print(f"[VERIFY ARCHIVE] Computing MD5 for {archive_dest}...")
            computed_md5 = compute_file_md5(archive_dest)
            if computed_md5.lower() != manifest.archive_md5.lower():
                print(
                    f"[ERROR] Corrupted archive {archive_dest}: expected MD5 {manifest.archive_md5}, "
                    f"got {computed_md5}",
                    file=sys.stderr,
                )
                return False
            print(f"[OK] Archive MD5 verified: {computed_md5}")

    print(f"[EXTRACT] Unpacking {archive_dest} -> {target_dir}...")
    target_dir.mkdir(parents=True, exist_ok=True)
    target_filenames_set = set(manifest.filenames) if manifest.filenames else None
    try:
        with zipfile.ZipFile(archive_dest, "r") as zip_ref:
            for member in zip_ref.infolist():
                if member.is_dir():
                    continue
                fname = Path(member.filename).name
                if fname.startswith(".") or member.filename.startswith("__MACOSX"):
                    continue
                if manifest.name == "mit1003_test" and target_filenames_set and fname not in target_filenames_set:
                    continue
                dest_file = target_dir / fname
                with zip_ref.open(member) as source, open(dest_file, "wb") as target:
                    shutil.copyfileobj(source, target)
        print(f"[EXTRACT COMPLETE] Unpacked and flattened into {target_dir}")
        return True
    except (zipfile.BadZipFile, OSError) as e:
        print(f"[ERROR] Failed to extract archive {archive_dest}: {e}", file=sys.stderr)
        return False


def run_download(manifests: Sequence[DatasetManifest], base_dir: Path | None = None) -> bool:
    """Execute download routines for all specified manifests."""
    overall_success = True
    for manifest in manifests:
        if manifest.name == "kodak":
            ok = download_kodak(manifest, base_dir=base_dir)
        else:
            ok = download_archive_dataset(manifest, base_dir=base_dir)
        if not ok:
            overall_success = False
    return overall_success


def run_verification(manifests: Sequence[DatasetManifest], base_dir: Path | None = None) -> int:
    """Verify integrity for given manifests and return CLI exit code (0 on success, 1 on failure)."""
    reports = verify_datasets(manifests, base_dir=base_dir)
    print("\n" + format_verification_summary(reports) + "\n")

    all_valid = all(rep.is_valid for rep in reports)
    if not all_valid:
        log_verification_failure_to_stderr(reports)
        print(
            "\n[ACTION REQUIRED] Some dataset files are missing or corrupted.\n"
            "To acquire missing files, run:\n"
            "    python scripts/prepare_data.py --download-only --dataset <name>\n"
            "or manually place files according to the layout in data/raw/.\n",
            file=sys.stderr,
        )
        return 1

    print("[SUCCESS] 100% integrity verified across checked datasets.")
    return 0


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dataset pipeline automation: directory scaffolding, acquisition, and verification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--setup-dirs",
        action="store_true",
        help="Scaffold data/raw and data/saliency_cache directory hierarchies.",
    )
    parser.add_argument(
        "--verify-data",
        action="store_true",
        help="Verify file existence, non-emptiness, and MD5 checksums for datasets.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download raw datasets without extracting saliency.",
    )
    parser.add_argument(
        "--extract-saliency",
        action="store_true",
        help="Extract visual saliency maps offline and cache to disk.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Subsample count limiting images processed per dataset split (for testing/fast verification).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target hardware device for FastSal execution ('cuda', 'cpu', etc.). Defaults to auto-selection.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=list(DATASET_REGISTRY.keys()),
        help="Target dataset to verify or download.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Base directory path override for data storage.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational stdout output.",
    )
    return parser.parse_args(args)


def main(argv: Sequence[str] | None = None) -> int:
    """Main execution entrypoint."""
    args = parse_args(argv)

    # Determine target manifests
    try:
        manifests = get_manifests_for_dataset(args.dataset)
    except KeyError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    # 1. Scaffolding action
    if args.setup_dirs:
        setup_directories(base_dir=args.data_dir, quiet=args.quiet)
        if not (args.verify_data or args.download_only or args.extract_saliency):
            return 0

    # 2. Download action (runs before extraction if both are requested)
    if args.download_only:
        setup_directories(base_dir=args.data_dir, quiet=True)
        download_ok = run_download(manifests, base_dir=args.data_dir)
        if not download_ok:
            return 1
        verify_code = run_verification(manifests, base_dir=args.data_dir)
        if verify_code != 0:
            return verify_code
        if not (args.verify_data or args.extract_saliency):
            return 0

    # 3. Saliency extraction action
    if args.extract_saliency:
        setup_directories(base_dir=args.data_dir, quiet=True)

        # Validate that raw image directories exist and are non-empty
        missing_source_manifests: list[str] = []
        for manifest in manifests:
            raw_dir = resolve_target_dir(manifest.target_dir, args.data_dir)
            if not raw_dir.exists() or not any(raw_dir.iterdir()):
                missing_source_manifests.append(manifest.name)

        if missing_source_manifests:
            print(
                f"[ERROR] Raw source files missing for dataset(s): {', '.join(missing_source_manifests)}.\n"
                f"Please download raw data first using --download-only.",
                file=sys.stderr,
            )
            return 1

        if not args.quiet:
            print("[INFO] Saliency extraction engine (FastSal offline pre-extraction & percentile caching) active.")
        try:
            results = extract_and_cache_saliency(
                dataset_name=args.dataset,
                base_dir=args.data_dir,
                sample=args.sample,
                device=args.device,
                quiet=args.quiet,
            )
            total_cached = sum(results.values())
            if total_cached == 0:
                print("[ERROR] No saliency maps were cached. Source files missing or empty.", file=sys.stderr)
                return 1

            if not args.quiet:
                print(f"\n[SUCCESS] Offline saliency extraction complete: {total_cached} total maps cached.")
        except (RuntimeError, OSError, ValueError, KeyError) as e:
            print(f"[ERROR] Saliency extraction failed: {e}", file=sys.stderr)
            return 1

        if not args.verify_data:
            return 0

    # 4. Verification action
    if args.verify_data:
        return run_verification(manifests, base_dir=args.data_dir)

    # If no functional action specified, display help and return 0
    if not (args.setup_dirs or args.extract_saliency or args.verify_data or args.download_only):
        parse_args(["--help"])
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
