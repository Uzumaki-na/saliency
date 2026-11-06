"""Dataset integrity verifier supporting streaming MD5 verification and reporting.

Validates that raw datasets match expected file lists and reference checksums,
detecting missing files, empty files, or corrupted byte payloads.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from src.data.manifests import DatasetManifest, resolve_target_dir

CHUNK_SIZE = 64 * 1024  # 64 KB streaming chunks


def compute_file_md5(file_path: Path | str, chunk_size: int = CHUNK_SIZE) -> str:
    """Compute the MD5 hash of a file using streaming chunked reads.

    Args:
        file_path: Path to the target file.
        chunk_size: Size of chunks in bytes (default: 64 KB).

    Returns:
        Hexadecimal MD5 hash digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be accessed.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    md5_hasher = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5_hasher.update(chunk)
    return md5_hasher.hexdigest()


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying an individual file."""

    filename: str
    dataset_name: str
    file_path: Path
    status: str  # "VALID", "MISSING", "CORRUPTED"
    expected_md5: str | None = None
    actual_md5: str | None = None
    file_size_bytes: int | None = None
    error_message: str | None = None


@dataclass
class DatasetVerificationReport:
    """Consolidated integrity verification report for a dataset split."""

    dataset_name: str
    target_dir: Path
    expected_count: int
    results: list[VerificationResult]

    @property
    def valid_count(self) -> int:
        """Total count of verified valid files."""
        return sum(1 for r in self.results if r.status == "VALID")

    @property
    def missing_count(self) -> int:
        """Total count of missing files."""
        return sum(1 for r in self.results if r.status == "MISSING")

    @property
    def corrupted_count(self) -> int:
        """Total count of corrupted or hash-mismatched files."""
        return sum(1 for r in self.results if r.status == "CORRUPTED")

    @property
    def is_valid(self) -> bool:
        """Return True if all expected files are present, valid, and uncorrupted."""
        return self.missing_count == 0 and self.corrupted_count == 0 and self.valid_count == self.expected_count

    @property
    def missing_files(self) -> list[str]:
        """List of filenames detected as missing."""
        return [r.filename for r in self.results if r.status == "MISSING"]

    @property
    def corrupted_files(self) -> list[VerificationResult]:
        """List of verification results for files detected as corrupted."""
        return [r for r in self.results if r.status == "CORRUPTED"]


def _resolve_target_dir(manifest_target_dir: Path, base_dir: Path | None) -> Path:
    """Resolve the dataset target directory against an optional custom base directory."""
    return resolve_target_dir(manifest_target_dir, base_dir)


def verify_dataset(manifest: DatasetManifest, base_dir: Path | None = None) -> DatasetVerificationReport:
    """Verify integrity of all files defined in a dataset manifest.

    Args:
        manifest: The DatasetManifest specification to evaluate.
        base_dir: Optional base directory override for tests or custom paths.

    Returns:
        DatasetVerificationReport with per-file and aggregated validation metrics.
    """
    target_dir = _resolve_target_dir(manifest.target_dir, base_dir)
    results: list[VerificationResult] = []

    for filename in manifest.filenames:
        file_path = target_dir / filename
        expected_md5 = manifest.checksums.get(filename)

        if not file_path.exists():
            results.append(
                VerificationResult(
                    filename=filename,
                    dataset_name=manifest.name,
                    file_path=file_path,
                    status="MISSING",
                    expected_md5=expected_md5,
                    error_message=f"File does not exist: {file_path}",
                )
            )
            continue

        try:
            stat_info = file_path.stat()
            file_size = stat_info.st_size
        except OSError as e:
            results.append(
                VerificationResult(
                    filename=filename,
                    dataset_name=manifest.name,
                    file_path=file_path,
                    status="CORRUPTED",
                    expected_md5=expected_md5,
                    error_message=f"Cannot stat file: {e}",
                )
            )
            continue

        if file_size == 0:
            results.append(
                VerificationResult(
                    filename=filename,
                    dataset_name=manifest.name,
                    file_path=file_path,
                    status="CORRUPTED",
                    expected_md5=expected_md5,
                    file_size_bytes=0,
                    error_message="File is empty (0 bytes)",
                )
            )
            continue

        try:
            actual_md5 = compute_file_md5(file_path)
        except (OSError, PermissionError) as e:
            results.append(
                VerificationResult(
                    filename=filename,
                    dataset_name=manifest.name,
                    file_path=file_path,
                    status="CORRUPTED",
                    expected_md5=expected_md5,
                    file_size_bytes=file_size,
                    error_message=f"Error reading file for MD5 computation: {e}",
                )
            )
            continue

        # Check hash match if an expected checksum is defined
        if expected_md5 is not None and actual_md5.lower() != expected_md5.lower():
            results.append(
                VerificationResult(
                    filename=filename,
                    dataset_name=manifest.name,
                    file_path=file_path,
                    status="CORRUPTED",
                    expected_md5=expected_md5,
                    actual_md5=actual_md5,
                    file_size_bytes=file_size,
                    error_message=(f"MD5 hash mismatch: expected {expected_md5}, got {actual_md5}"),
                )
            )
        else:
            results.append(
                VerificationResult(
                    filename=filename,
                    dataset_name=manifest.name,
                    file_path=file_path,
                    status="VALID",
                    expected_md5=expected_md5,
                    actual_md5=actual_md5,
                    file_size_bytes=file_size,
                )
            )

    return DatasetVerificationReport(
        dataset_name=manifest.name,
        target_dir=target_dir,
        expected_count=manifest.expected_count,
        results=results,
    )


def verify_datasets(
    manifests: Sequence[DatasetManifest], base_dir: Path | None = None
) -> list[DatasetVerificationReport]:
    """Verify a collection of dataset manifests."""
    return [verify_dataset(m, base_dir=base_dir) for m in manifests]


def format_verification_summary(reports: Sequence[DatasetVerificationReport]) -> str:
    """Format verification reports into an ASCII summary table."""
    headers = ["Dataset", "Expected", "Valid", "Missing", "Corrupted", "Status"]
    col_widths = [16, 10, 8, 9, 11, 10]

    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    divider = "-+-".join("-" * w for w in col_widths)

    rows = [header_line, divider]
    for rep in reports:
        status_str = "OK" if rep.is_valid else "FAILED"
        row_values = [
            rep.dataset_name[: col_widths[0]].ljust(col_widths[0]),
            str(rep.expected_count).rjust(col_widths[1]),
            str(rep.valid_count).rjust(col_widths[2]),
            str(rep.missing_count).rjust(col_widths[3]),
            str(rep.corrupted_count).rjust(col_widths[4]),
            status_str.ljust(col_widths[5]),
        ]
        rows.append(" | ".join(row_values))

    return "\n".join(rows)


def format_failure_details(reports: Sequence[DatasetVerificationReport]) -> str:
    """Format detailed error diagnostic for any missing or corrupted files."""
    lines: list[str] = []
    for rep in reports:
        if rep.is_valid:
            continue

        lines.append(f"\n--- Diagnostic: Dataset '{rep.dataset_name}' ({rep.target_dir}) ---")
        if rep.missing_count > 0:
            lines.append(f"  Missing files ({rep.missing_count}/{rep.expected_count}):")
            # Show up to 10 missing files explicitly
            for fname in rep.missing_files[:10]:
                lines.append(f"    - {fname}")
            if rep.missing_count > 10:
                lines.append(f"    ... and {rep.missing_count - 10} more missing files.")

        if rep.corrupted_count > 0:
            lines.append(f"  Corrupted files ({rep.corrupted_count}/{rep.expected_count}):")
            for c in rep.corrupted_files[:10]:
                lines.append(f"    - {c.filename}: {c.error_message}")
            if rep.corrupted_count > 10:
                lines.append(f"    ... and {rep.corrupted_count - 10} more corrupted files.")

    return "\n".join(lines)


def log_verification_failure_to_stderr(reports: Sequence[DatasetVerificationReport]) -> None:
    """Print formatted diagnostic failure details to sys.stderr."""
    details = format_failure_details(reports)
    if details:
        print(details, file=sys.stderr)
