"""Tests for the reproducibility provenance manifest (adaptivehb.provenance)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adaptivehb import provenance as P
from adaptivehb.dataset import generate_synthetic_dataset
from adaptivehb.pipeline import HbPipeline


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

def test_collect_environment_shape() -> None:
    env = P.collect_environment()
    assert {"python_version", "platform", "packages"} <= set(env)
    assert isinstance(env["packages"], dict)
    # PyYAML is a hard runtime dependency, so it must be reported.
    assert "PyYAML" in env["packages"]


def test_collect_environment_omits_absent_packages() -> None:
    env = P.collect_environment(packages=("definitely-not-installed-xyz",))
    assert env["packages"] == {}


# --------------------------------------------------------------------------- #
# Git revision (synthetic .git, no subprocess)
# --------------------------------------------------------------------------- #

def test_git_revision_none_without_repo(tmp_path: Path) -> None:
    assert P.git_revision(tmp_path) is None


def test_git_revision_reads_branch_head(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    sha = "a" * 40
    (git_dir / "refs" / "heads" / "main").write_text(sha + "\n", encoding="utf-8")
    rev = P.git_revision(tmp_path)
    assert rev is not None
    assert rev["branch"] == "main"
    assert rev["commit"] == sha
    assert rev["short"] == sha[:7]
    assert rev["detached"] is False


def test_git_revision_detached_head(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    sha = "b" * 40
    (git_dir / "HEAD").write_text(sha + "\n", encoding="utf-8")
    rev = P.git_revision(tmp_path)
    assert rev is not None
    assert rev["detached"] is True
    assert rev["commit"] == sha
    assert rev["branch"] is None


def test_git_revision_from_packed_refs(tmp_path: Path) -> None:
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    sha = "c" * 40
    (git_dir / "packed-refs").write_text(
        f"# pack-refs with: peeled fully-peeled sorted\n{sha} refs/heads/main\n",
        encoding="utf-8",
    )
    rev = P.git_revision(tmp_path)
    assert rev is not None and rev["commit"] == sha


# --------------------------------------------------------------------------- #
# Fingerprints
# --------------------------------------------------------------------------- #

def test_config_fingerprint_is_deterministic(framework_config) -> None:
    a = P.config_fingerprint(framework_config)
    b = P.config_fingerprint(framework_config)
    assert a == b
    assert len(a["sha256"]) == 64
    assert a["digest"] == a["sha256"][:12]


def test_dataset_fingerprint_unavailable_without_root(framework_config) -> None:
    # A pipeline with no dataset root yields an "unavailable" fingerprint.
    from adaptivehb.dataset.manager import DatasetManager

    manager = DatasetManager(framework_config, base_dir=".", dataset_root=None)
    fp = P.dataset_fingerprint(manager)
    assert fp["available"] is False


def test_dataset_fingerprint_counts_and_hash(framework_config, tmp_path: Path) -> None:
    from adaptivehb.dataset.manager import DatasetManager

    root = tmp_path / "ds"
    generate_synthetic_dataset(root, num_patients=5, seed=1)
    manager = DatasetManager(framework_config, base_dir=tmp_path, dataset_root=root)
    manager.initialize()
    manager.split()
    fp = P.dataset_fingerprint(manager)
    assert fp["available"] is True
    assert fp["num_patients"] == 5
    assert fp["num_samples"] > 0
    assert sum(fp["split_sizes"].values()) == fp["num_samples"]
    assert len(fp["roster_sha256"]) == 64
    # Same dataset -> identical roster hash (deterministic).
    assert P.dataset_fingerprint(manager)["roster_sha256"] == fp["roster_sha256"]


# --------------------------------------------------------------------------- #
# Manifest + experiment integration
# --------------------------------------------------------------------------- #

def test_build_manifest_shape(framework_config, tmp_path: Path) -> None:
    root = tmp_path / "ds"
    generate_synthetic_dataset(root, num_patients=6, seed=3)
    pipeline = HbPipeline(framework_config, base_dir=tmp_path, dataset_root=root).initialize()
    manifest = P.build_manifest(pipeline, extra={"experiment_id": "abc"})
    assert manifest["experiment_id"] == "abc"
    assert manifest["seed"] == framework_config.project.seed
    assert manifest["framework_version"]
    assert "environment" in manifest and "config" in manifest and "dataset" in manifest
    # Fully JSON-serializable.
    json.dumps(manifest)


def test_experiment_archives_provenance(framework_config, tmp_path: Path) -> None:
    root = tmp_path / "ds"
    generate_synthetic_dataset(root, num_patients=8, seed=5)
    pipeline = HbPipeline(framework_config, base_dir=tmp_path, dataset_root=root)
    result = pipeline.experiment("provtest", epochs=2)
    # Result carries the manifest.
    assert result.provenance
    assert "config" in result.provenance and "dataset" in result.provenance
    # And it is archived on disk in the experiment's configuration directory.
    archived = list(Path(result.root).glob("configuration/provenance.json"))
    assert len(archived) == 1
    on_disk = json.loads(archived[0].read_text())
    assert on_disk["experiment_id"] == result.experiment_id
