"""Automated unit tests for DVC repository initialization, dvc.yaml specification, and reproduction."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.data.manifests import ALL_REQUIRED_DIRS, resolve_target_dir
from src.data.verifier import compute_file_md5

_REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_PREPARE_DATA_DEPS = [
    "data/manifests/mit1003_test_split.json",
    "scripts/prepare_data.py",
    "src/data/manifests.py",
    "src/data/verifier.py",
    "src/data/saliency.py",
]


def _setup_isolated_dvc_repo(target_dir: Path) -> None:
    """Scaffold a minimal isolated DVC configuration inside a temporary directory."""
    dvc_dir = target_dir / ".dvc"
    dvc_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (target_dir / ".dvc" / "tmp" / "cache").as_posix()
    (dvc_dir / "config").write_text(
        f"[core]\n    no_scm = true\n    site_cache_dir = {cache_dir}\n",
        encoding="utf-8",
    )
    (dvc_dir / ".gitignore").write_text("", encoding="utf-8")
    (target_dir / ".dvcignore").write_text("", encoding="utf-8")


# =====================================================================
# 1. DVC Initialization and Directory Structure Tests
# =====================================================================


def test_dvc_internal_directory_structure():
    """Verify that .dvc directory and standard internal files exist."""
    dvc_dir = _REPO_ROOT / ".dvc"
    assert dvc_dir.exists(), "DVC repository directory .dvc/ must exist"
    assert dvc_dir.is_dir(), ".dvc must be a directory"

    dvc_config = dvc_dir / "config"
    assert dvc_config.exists(), ".dvc/config file must exist"
    assert dvc_config.is_file(), ".dvc/config must be a regular file"

    dvc_gitignore = dvc_dir / ".gitignore"
    assert dvc_gitignore.exists(), ".dvc/.gitignore must exist"


def test_dvcignore_configuration():
    """Verify that .dvcignore exists, is non-empty, and contains core ignore rules."""
    dvcignore = _REPO_ROOT / ".dvcignore"
    assert dvcignore.exists(), ".dvcignore must exist in repository root"
    assert dvcignore.is_file(), ".dvcignore must be a file"

    content = dvcignore.read_text(encoding="utf-8")
    assert len(content.strip()) > 0, ".dvcignore must not be empty"

    # Verify standard ignoring rules
    assert ".venv" in content
    assert "__pycache__" in content
    assert "/data/raw/" in content
    assert "/data/saliency_cache/" in content


def test_gitignore_protects_dvc_cache_and_raw_data():
    """Verify that .gitignore excludes .dvc/cache and heavy raw image folders."""
    gitignore = _REPO_ROOT / ".gitignore"
    assert gitignore.exists(), ".gitignore must exist in repository root"
    content = gitignore.read_text(encoding="utf-8")

    assert ".dvc/cache" in content
    assert "/data/raw/" in content or "data/raw/" in content
    assert "/data/saliency_cache/" in content or "data/saliency_cache/" in content


# =====================================================================
# 2. dvc.yaml Pipeline Specification Tests
# =====================================================================


def test_dvc_yaml_exists_and_is_valid():
    """Verify that dvc.yaml exists and parses cleanly as valid YAML."""
    dvc_yaml_path = _REPO_ROOT / "dvc.yaml"
    assert dvc_yaml_path.exists(), "dvc.yaml must exist in repository root"

    with open(dvc_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), "dvc.yaml root must be a mapping"
    assert "stages" in data, "dvc.yaml must define a 'stages' block"


def test_prepare_data_stage_definition():
    """Verify that the prepare_data stage has the correct command and dependency list."""
    dvc_yaml_path = _REPO_ROOT / "dvc.yaml"
    with open(dvc_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    stages = data.get("stages", {})
    assert "prepare_data" in stages, "prepare_data stage must be defined in dvc.yaml"

    stage = stages["prepare_data"]
    assert stage.get("cmd") == "python scripts/prepare_data.py --setup-dirs"

    deps = stage.get("deps", [])
    assert sorted(deps) == sorted(EXPECTED_PREPARE_DATA_DEPS), (
        f"prepare_data deps mismatch. Expected: {EXPECTED_PREPARE_DATA_DEPS}, got: {deps}"
    )


def test_prepare_data_dependencies_exist_on_disk():
    """Verify that all files declared as dependencies in dvc.yaml exist on disk."""
    dvc_yaml_path = _REPO_ROOT / "dvc.yaml"
    with open(dvc_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    deps = data["stages"]["prepare_data"]["deps"]
    for rel_dep in deps:
        # Cross-platform relative path check (no backslashes or leading slashes in dvc.yaml)
        assert "\\" not in rel_dep, f"Dependency path '{rel_dep}' must use forward slashes"
        assert not rel_dep.startswith("/"), f"Dependency path '{rel_dep}' must be relative"

        dep_path = _REPO_ROOT / rel_dep
        assert dep_path.exists(), f"Declared dependency does not exist on disk: {dep_path}"
        assert dep_path.is_file(), f"Declared dependency is not a file: {dep_path}"
        assert dep_path.stat().st_size > 0, f"Declared dependency is empty: {dep_path}"


def test_dvc_python_api_stage_schema():
    """Verify that DVC Python API parses the pipeline stage and dependencies identically."""
    from dvc.repo import Repo

    with Repo(str(_REPO_ROOT)) as repo:
        stage_names = [s.name for s in repo.index.stages]
        assert "prepare_data" in stage_names

        stage = next(s for s in repo.index.stages if s.name == "prepare_data")
        assert stage.cmd == "python scripts/prepare_data.py --setup-dirs"
        actual_deps = [str(d.def_path).replace("\\", "/") for d in stage.deps]
        assert sorted(actual_deps) == sorted(EXPECTED_PREPARE_DATA_DEPS)


# =====================================================================
# 3. dvc.lock Hash Verification Tests
# =====================================================================


def test_dvc_lock_exists_and_matches_dependencies():
    """Verify that dvc.lock is generated and contains accurate hashes for all dependencies."""
    dvc_lock_path = _REPO_ROOT / "dvc.lock"
    assert dvc_lock_path.exists(), "dvc.lock must exist after pipeline reproduction"

    with open(dvc_lock_path, "r", encoding="utf-8") as f:
        lock_data = yaml.safe_load(f)

    assert "stages" in lock_data, "dvc.lock must contain 'stages'"
    assert "prepare_data" in lock_data["stages"], "prepare_data stage must be locked"

    stage_lock = lock_data["stages"]["prepare_data"]
    assert stage_lock.get("cmd") == "python scripts/prepare_data.py --setup-dirs"

    lock_deps = stage_lock.get("deps", [])
    locked_paths = [d["path"] for d in lock_deps]
    assert sorted(locked_paths) == sorted(EXPECTED_PREPARE_DATA_DEPS)

    for dep_entry in lock_deps:
        rel_path = dep_entry["path"]
        locked_md5 = dep_entry["md5"]
        locked_size = dep_entry["size"]

        file_path = _REPO_ROOT / rel_path
        assert file_path.exists(), f"Locked dependency file {file_path} must exist"
        assert file_path.stat().st_size == locked_size, f"Size mismatch for {rel_path}"

        actual_md5 = compute_file_md5(file_path)
        assert actual_md5.lower() == locked_md5.lower(), (
            f"MD5 mismatch for {rel_path}: lock={locked_md5}, disk={actual_md5}"
        )


# =====================================================================
# 4. Pipeline Reproduction and CLI Execution Tests
# =====================================================================


EXPECTED_PIPELINE_STAGES = [
    "prepare_data",
    "train_sweep",
    "benchmark",
    "evaluate",
]


def test_all_dvc_stages_defined_in_yaml():
    """Verify that all core pipeline stages are defined in dvc.yaml."""
    dvc_yaml_path = _REPO_ROOT / "dvc.yaml"
    with open(dvc_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    stages = data.get("stages", {})
    for stage_name in EXPECTED_PIPELINE_STAGES:
        assert stage_name in stages, f"Stage '{stage_name}' must be defined in dvc.yaml"
        assert "cmd" in stages[stage_name], f"Stage '{stage_name}' must define a command"
        assert "deps" in stages[stage_name], f"Stage '{stage_name}' must declare dependencies"


def test_all_declared_pipeline_dependencies_exist_on_disk():
    """Verify that every dependency across all dvc.yaml stages exists on disk."""
    dvc_yaml_path = _REPO_ROOT / "dvc.yaml"
    with open(dvc_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    stages = data.get("stages", {})
    for stage_name, stage_cfg in stages.items():
        deps = stage_cfg.get("deps", [])
        for rel_dep in deps:
            assert "\\" not in rel_dep, f"Stage '{stage_name}': Dep '{rel_dep}' must use forward slashes"
            assert not rel_dep.startswith("/"), f"Stage '{stage_name}': Dep '{rel_dep}' must be relative"
            dep_path = _REPO_ROOT / rel_dep
            assert dep_path.exists(), f"Stage '{stage_name}': Declared dep missing: {dep_path}"
            assert dep_path.is_file(), f"Stage '{stage_name}': Declared dep is not a file: {dep_path}"
            assert dep_path.stat().st_size > 0, f"Stage '{stage_name}': Declared dep is empty: {dep_path}"


def test_all_stages_locked_in_dvc_lock():
    """Verify that dvc.lock contains locked dependencies and commands for all pipeline stages."""
    dvc_lock_path = _REPO_ROOT / "dvc.lock"
    assert dvc_lock_path.exists(), "dvc.lock must exist"

    with open(dvc_lock_path, "r", encoding="utf-8") as f:
        lock_data = yaml.safe_load(f)

    locked_stages = lock_data.get("stages", {})
    for stage_name in EXPECTED_PIPELINE_STAGES:
        assert stage_name in locked_stages, f"Stage '{stage_name}' must be locked in dvc.lock"
        s_lock = locked_stages[stage_name]
        assert "cmd" in s_lock
        assert "deps" in s_lock
        for dep in s_lock["deps"]:
            dep_file = _REPO_ROOT / dep["path"]
            assert dep_file.exists(), f"Locked dependency file {dep_file} missing"
            assert compute_file_md5(dep_file).lower() == dep["md5"].lower()


def test_dvc_status_is_clean():
    """Verify dvc status reports entire pipeline is clean and up to date."""
    cmd = [sys.executable, "-m", "dvc", "status"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"dvc status failed with stderr: {result.stderr}"
    assert "Data and pipelines are up to date" in result.stdout or "up to date" in result.stdout.lower()


def test_dvc_repro_when_unchanged_skips():
    """Verify dvc repro skips execution when dependencies have not changed."""
    cmd = [sys.executable, "-m", "dvc", "repro", "prepare_data"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"dvc repro failed with stderr: {result.stderr}"
    assert "didn't change, skipping" in result.stdout or "up to date" in result.stdout.lower()


def test_prepare_data_stage_command_creates_all_directories(tmp_path: Path):
    """Verify the stage command creates all required directories when executed."""
    cmd = [sys.executable, "scripts/prepare_data.py", "--setup-dirs", "--data-dir", str(tmp_path)]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Command failed: {result.stderr}"

    for rel_dir in ALL_REQUIRED_DIRS:
        target = resolve_target_dir(rel_dir, tmp_path)
        assert target.exists(), f"Directory {target} was not created"
        assert target.is_dir(), f"Path {target} is not a directory"


def test_dvc_repro_detects_hash_change_and_reexecutes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify modifying a dependency causes DVC to detect the change and re-execute updating dvc.lock."""
    from dvc.repo import Repo

    _setup_isolated_dvc_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    dep_file = tmp_path / "dep.txt"
    dep_file.write_text("initial dependency payload", encoding="utf-8")

    script_file = tmp_path / "step.py"
    script_file.write_text("print('executed step')\n", encoding="utf-8")

    dvc_yaml_data = {
        "stages": {
            "demo_stage": {
                "cmd": f'"{sys.executable}" step.py',
                "deps": ["dep.txt", "step.py"],
            }
        }
    }
    with open(tmp_path / "dvc.yaml", "w", encoding="utf-8") as f:
        yaml.dump(dvc_yaml_data, f)

    # Initial reproduction generates dvc.lock
    with Repo(".") as repo:
        ran_stages = repo.reproduce("demo_stage")
        assert len(ran_stages) == 1
        assert ran_stages[0].name == "demo_stage"

    lock_file = tmp_path / "dvc.lock"
    assert lock_file.exists()
    with open(lock_file, "r", encoding="utf-8") as f:
        lock1 = yaml.safe_load(f)

    dep_entry1 = next(d for d in lock1["stages"]["demo_stage"]["deps"] if d["path"] == "dep.txt")
    initial_md5 = dep_entry1["md5"]

    # Re-run when unchanged skips execution
    with Repo(".") as repo:
        skipped = repo.reproduce("demo_stage")
        assert len(skipped) == 0

    # Modify dependency content
    dep_file.write_text("modified dependency payload - different hash", encoding="utf-8")

    # DVC detects change in status and re-executes stage updating lock
    with Repo(".") as repo:
        status = repo.status("demo_stage")
        assert "demo_stage" in status
        stage_status = status["demo_stage"]
        assert any("changed deps" in item for item in stage_status)

        re_executed = repo.reproduce("demo_stage")
        assert len(re_executed) == 1
        assert re_executed[0].name == "demo_stage"

    with open(lock_file, "r", encoding="utf-8") as f:
        lock2 = yaml.safe_load(f)

    dep_entry2 = next(d for d in lock2["stages"]["demo_stage"]["deps"] if d["path"] == "dep.txt")
    updated_md5 = dep_entry2["md5"]
    assert initial_md5 != updated_md5


def test_dvc_repro_missing_dependency_raises_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that reproduction raises a validation error when a required dependency is missing."""
    from dvc.exceptions import DvcException
    from dvc.repo import Repo

    _setup_isolated_dvc_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    dvc_yaml_data = {
        "stages": {
            "invalid_stage": {
                "cmd": f'"{sys.executable}" -c "print(1)"',
                "deps": ["missing_dependency_file.json"],
            }
        }
    }
    with open(tmp_path / "dvc.yaml", "w", encoding="utf-8") as f:
        yaml.dump(dvc_yaml_data, f)

    # Reproduction via DVC Repo API raises DvcException
    with Repo(".") as repo, pytest.raises(DvcException):
        repo.reproduce("invalid_stage")

    # Reproduction via CLI returns non-zero exit code
    cmd = [sys.executable, "-m", "dvc", "repro", "invalid_stage"]
    result = subprocess.run(
        cmd,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
