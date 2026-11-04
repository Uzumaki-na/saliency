"""Dataset manifests and path definitions for raw image sets and saliency caches.

Defines directory hierarchy, filenames, download endpoints, and integrity checksums
for Kodak PhotoCD (24 images), DIV2K (800 train, 100 val), and MIT1003 (500 test split).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Base Directory Paths
RAW_DATA_DIR: Path = Path("data/raw")
SALIENCY_CACHE_DIR: Path = Path("data/saliency_cache")

# Raw Dataset Target Directories
KODAK_RAW_DIR: Path = RAW_DATA_DIR / "kodak"
DIV2K_TRAIN_RAW_DIR: Path = RAW_DATA_DIR / "div2k" / "train"
DIV2K_VAL_RAW_DIR: Path = RAW_DATA_DIR / "div2k" / "val"
MIT1003_TEST_RAW_DIR: Path = RAW_DATA_DIR / "mit1003" / "test"

# Saliency Cache Directories
KODAK_CACHE_DIR: Path = SALIENCY_CACHE_DIR / "kodak"
DIV2K_TRAIN_CACHE_DIR: Path = SALIENCY_CACHE_DIR / "div2k" / "train"
DIV2K_VAL_CACHE_DIR: Path = SALIENCY_CACHE_DIR / "div2k" / "val"
MIT1003_TEST_CACHE_DIR: Path = SALIENCY_CACHE_DIR / "mit1003" / "test"

# Directory Aggregations for Scaffolding
ALL_RAW_DIRS: tuple[Path, ...] = (
    KODAK_RAW_DIR,
    DIV2K_TRAIN_RAW_DIR,
    DIV2K_VAL_RAW_DIR,
    MIT1003_TEST_RAW_DIR,
)

ALL_CACHE_DIRS: tuple[Path, ...] = (
    KODAK_CACHE_DIR,
    DIV2K_TRAIN_CACHE_DIR,
    DIV2K_VAL_CACHE_DIR,
    MIT1003_TEST_CACHE_DIR,
)

ALL_REQUIRED_DIRS: tuple[Path, ...] = ALL_RAW_DIRS + ALL_CACHE_DIRS

# Official Filenames
KODAK_FILENAMES: tuple[str, ...] = tuple(f"kodim{i:02d}.png" for i in range(1, 25))
DIV2K_TRAIN_FILENAMES: tuple[str, ...] = tuple(f"{i:04d}.png" for i in range(1, 801))
DIV2K_VAL_FILENAMES: tuple[str, ...] = tuple(f"{i:04d}.png" for i in range(801, 901))
def _load_mit1003_test_filenames() -> tuple[str, ...]:
    manifest_path = Path(__file__).resolve().parent.parent.parent / "data" / "manifests" / "mit1003_test_split.json"
    if manifest_path.exists():
        try:
            import json
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                filenames = data.get("filenames", [])
                if filenames and len(filenames) == 500:
                    return tuple(filenames)
        except (OSError, json.JSONDecodeError, ValueError):
            return tuple(f"mit1003_test_{i:04d}.jpg" for i in range(1, 501))
    return tuple(f"mit1003_test_{i:04d}.jpg" for i in range(1, 501))


MIT1003_TEST_FILENAMES: tuple[str, ...] = _load_mit1003_test_filenames()

# Remote URL Templates
KODAK_PRIMARY_BASE_URL: str = "http://r0k.us/graphics/kodak/kodak"
KODAK_FALLBACK_BASE_URL: str = "https://raw.githubusercontent.com/lemire/kodakimagecollection/master"

DIV2K_TRAIN_ARCHIVE_URL: str = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_TRAIN_ARCHIVE_FALLBACK: str = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_VAL_ARCHIVE_URL: str = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
DIV2K_VAL_ARCHIVE_FALLBACK: str = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"

DIV2K_TRAIN_ARCHIVE_MD5: str = "bd710d0d82d92178082697a6fa347cd3"
DIV2K_VAL_ARCHIVE_MD5: str = "9fcdda83005a5e599c4ab75282198b88"

MIT1003_STIMULI_ARCHIVE_URL: str = "http://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip"
MIT1003_FIXATIONS_ARCHIVE_URL: str = "http://people.csail.mit.edu/tjudd/WherePeopleLook/ALLFIXATIONMAPS.zip"

# Official 24 Kodak PhotoCD MD5 Checksums
KODAK_CHECKSUMS: dict[str, str] = {
    "kodim01.png": "797a5ed9e768bac2143197febe74e9f0",
    "kodim02.png": "40bfc5a525f4659c132e72b7a733e45e",
    "kodim03.png": "5e9f48bd613eacfb6252c77ae7306a73",
    "kodim04.png": "f05d34fa98bd7fed4ecbe3597b71918e",
    "kodim05.png": "b79782844dae1e15aa9419c502a84f0a",
    "kodim06.png": "4be8c416ad84226df5ba3d289625770a",
    "kodim07.png": "4d97e7987524aff3bc495776179842be",
    "kodim08.png": "98454f0a6c3a35e70e69c20642c3bd1b",
    "kodim09.png": "07d3dba47087b5758a53010434cce622",
    "kodim10.png": "929943052e22f71b29bff2fbe197daff",
    "kodim11.png": "c04773608d99a39546de3d028a75bd35",
    "kodim12.png": "c6c8dea947e5431173a7347bdb2c2262",
    "kodim13.png": "9ddecfeef6f7915f5e5ce0d09f0c6c61",
    "kodim14.png": "a5b6b880ba060ad02805bbab60c399d0",
    "kodim15.png": "a858d33c424eaac1322cf3cab6d3d568",
    "kodim16.png": "a2308ff210e30c1969f7445135607f20",
    "kodim17.png": "52dac72df0b33bacc643ad6d42759964",
    "kodim18.png": "d1c85cf7358cd0bf4dd26c884f2e80b8",
    "kodim19.png": "d60e46f8c6fd8f8ae33a39e6eef22af3",
    "kodim20.png": "7d814638029aa1606caf78c7085414f4",
    "kodim21.png": "e7608675ced1dbc0b8d3a1878d8be675",
    "kodim22.png": "ceb46494f96589f8926a84bb5b66056e",
    "kodim23.png": "e1328ba2c55d5a72eb77e283c9630b90",
    "kodim24.png": "45a4912aaa0458940d5554ade1db5141",
}


def resolve_target_dir(target_dir: Path | str, base_dir: Path | str | None = None) -> Path:
    """Resolve target directory against an optional base directory, preserving 'data/' consistently."""
    path = Path(target_dir)
    if base_dir is None:
        return path
    base = Path(base_dir)
    if path.parts and path.parts[0] == "data":
        if base.name == "data":
            # Avoid data/data duplication if base_dir is already the data directory
            return base / Path(*path.parts[1:])
        return base / path
    return base / path


@dataclass(frozen=True)
class DatasetManifest:
    """Specification and integrity contract for a dataset split."""

    name: str
    target_dir: Path
    cache_dir: Path
    filenames: tuple[str, ...]
    checksums: dict[str, str] = field(default_factory=dict)
    archive_urls: tuple[str, ...] = ()
    archive_md5: str | None = None
    file_urls: dict[str, tuple[str, ...]] = field(default_factory=dict)
    description: str = ""

    @property
    def expected_count(self) -> int:
        """Total number of expected files in this dataset split."""
        return len(self.filenames)


# Kodak Manifest: 24 TrueColor Lossless PNG images
KODAK_FILE_URLS: dict[str, tuple[str, ...]] = {
    fname: (
        f"{KODAK_PRIMARY_BASE_URL}/{fname}",
        f"{KODAK_FALLBACK_BASE_URL}/{fname}",
    )
    for fname in KODAK_FILENAMES
}

KODAK_MANIFEST = DatasetManifest(
    name="kodak",
    target_dir=KODAK_RAW_DIR,
    cache_dir=KODAK_CACHE_DIR,
    filenames=KODAK_FILENAMES,
    checksums=KODAK_CHECKSUMS,
    archive_urls=(),
    file_urls=KODAK_FILE_URLS,
    description="Kodak PhotoCD Suite (24 uncompressed PNG images, 768x512 / 512x768)",
)

# DIV2K Training Split: 800 HR images (0001.png - 0800.png)
DIV2K_TRAIN_MANIFEST = DatasetManifest(
    name="div2k_train",
    target_dir=DIV2K_TRAIN_RAW_DIR,
    cache_dir=DIV2K_TRAIN_CACHE_DIR,
    filenames=DIV2K_TRAIN_FILENAMES,
    checksums={},
    archive_urls=(DIV2K_TRAIN_ARCHIVE_URL, DIV2K_TRAIN_ARCHIVE_FALLBACK),
    archive_md5=DIV2K_TRAIN_ARCHIVE_MD5,
    description="DIV2K Training Set (800 high-resolution PNG images, 2K resolution)",
)

# DIV2K Validation Split: 100 HR images (0801.png - 0900.png)
DIV2K_VAL_MANIFEST = DatasetManifest(
    name="div2k_val",
    target_dir=DIV2K_VAL_RAW_DIR,
    cache_dir=DIV2K_VAL_CACHE_DIR,
    filenames=DIV2K_VAL_FILENAMES,
    checksums={},
    archive_urls=(DIV2K_VAL_ARCHIVE_URL, DIV2K_VAL_ARCHIVE_FALLBACK),
    archive_md5=DIV2K_VAL_ARCHIVE_MD5,
    description="DIV2K Validation Set (100 high-resolution PNG images, 2K resolution)",
)

# MIT1003 Test Split: 500 images
MIT1003_TEST_MANIFEST = DatasetManifest(
    name="mit1003_test",
    target_dir=MIT1003_TEST_RAW_DIR,
    cache_dir=MIT1003_TEST_CACHE_DIR,
    filenames=MIT1003_TEST_FILENAMES,
    checksums={},
    archive_urls=(MIT1003_STIMULI_ARCHIVE_URL,),
    description="MIT1003 Human Attention Benchmark (500-image test split)",
)

ALL_MANIFESTS: tuple[DatasetManifest, ...] = (
    KODAK_MANIFEST,
    DIV2K_TRAIN_MANIFEST,
    DIV2K_VAL_MANIFEST,
    MIT1003_TEST_MANIFEST,
)

# Lookup registry supporting shorthand aliases
DATASET_REGISTRY: dict[str, list[DatasetManifest]] = {
    "kodak": [KODAK_MANIFEST],
    "div2k_train": [DIV2K_TRAIN_MANIFEST],
    "div2k_val": [DIV2K_VAL_MANIFEST],
    "div2k": [DIV2K_TRAIN_MANIFEST, DIV2K_VAL_MANIFEST],
    "mit1003_test": [MIT1003_TEST_MANIFEST],
    "mit1003": [MIT1003_TEST_MANIFEST],
    "all": list(ALL_MANIFESTS),
}


def get_manifests_for_dataset(name: str) -> list[DatasetManifest]:
    """Retrieve the list of manifests corresponding to a dataset key or alias.

    Args:
        name: Name of the dataset ('kodak', 'div2k', 'div2k_train', 'div2k_val', 'mit1003', 'all')

    Returns:
        List of matching DatasetManifest instances.

    Raises:
        KeyError: If dataset name is not registered.
    """
    key = name.strip().lower()
    if key not in DATASET_REGISTRY:
        valid_keys = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {valid_keys}")
    return DATASET_REGISTRY[key]
