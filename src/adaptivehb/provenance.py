"""Reproducibility provenance for experiments (Decision 032).

Reproducibility is a core project pillar and an explicit requirement of the
target journals: a published experiment must be accompanied by enough provenance
to reconstruct it. This module assembles a single, self-describing manifest for
an experiment run and writes it alongside the archived metrics/figures, capturing:

* the framework version and the RNG seed that drives every deterministic split
  and resampler,
* the software environment (Python, platform, and the versions of the relevant
  scientific packages actually installed — torch/numpy/… when present),
* the current git revision, read directly from the ``.git`` directory without
  shelling out (so it works in restricted/sandboxed environments), and
* content fingerprints of the configuration and of the dataset, so a reviewer can
  detect whether either changed between runs.

Everything here is standard-library only and degrades gracefully: missing
packages, a non-git checkout, or an unset dataset root each yield ``None``/absent
fields rather than raising, so building a manifest never breaks an experiment.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from adaptivehb.core.utils import utcnow_iso, write_json
from adaptivehb.version import __version__

if TYPE_CHECKING:  # pragma: no cover
    from adaptivehb.config.loader import FrameworkConfig
    from adaptivehb.dataset.manager import DatasetManager
    from adaptivehb.pipeline import HbPipeline

# Scientific/runtime packages whose versions are worth pinning in a manifest.
# Absent packages are simply omitted (this framework is deliberately torch-optional).
_TRACKED_PACKAGES: tuple[str, ...] = (
    "adaptivehb",
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "opencv-python",
    "Pillow",
    "torch",
    "torchvision",
    "torchmetrics",
    "albumentations",
    "PyYAML",
    "openpyxl",
)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #

def collect_environment(packages: tuple[str, ...] = _TRACKED_PACKAGES) -> dict[str, Any]:
    """Capture the software environment for the current process.

    Args:
        packages: Distribution names whose installed versions to record. Names
            that are not installed are omitted.

    Returns:
        A mapping with interpreter/platform details and a ``packages`` mapping of
        distribution name to installed version (only those present).
    """
    installed: dict[str, str] = {}
    for name in packages:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "packages": installed,
    }


# --------------------------------------------------------------------------- #
# Git revision (read without a subprocess)
# --------------------------------------------------------------------------- #

def git_revision(start: str | Path = ".") -> dict[str, Any] | None:
    """Return the current git revision by reading ``.git`` directly.

    Walks up from ``start`` to locate a ``.git`` directory (or a ``.git`` file
    for worktrees/submodules), then resolves ``HEAD``. No ``git`` executable is
    invoked, so this works in sandboxes where subprocesses/networking are blocked.

    Returns:
        ``{"commit": <full sha or None>, "short": <7-char sha or None>,
        "branch": <name or None>, "detached": <bool>}``, or ``None`` when no git
        metadata is found or it cannot be parsed.
    """
    git_dir = _find_git_dir(Path(start).resolve())
    if git_dir is None:
        return None
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None

    if head.startswith("ref:"):
        ref = head[4:].strip()
        branch = ref.rsplit("/", 1)[-1]
        commit = _read_ref(git_dir, ref)
        return {
            "commit": commit,
            "short": commit[:7] if commit else None,
            "branch": branch,
            "detached": False,
        }
    # Detached HEAD: the file holds the commit hash directly.
    return {"commit": head, "short": head[:7], "branch": None, "detached": True}


def _find_git_dir(start: Path) -> Path | None:
    """Locate the ``.git`` directory at or above ``start`` (handles gitdir files)."""
    for directory in (start, *start.parents):
        candidate = directory / ".git"
        if candidate.is_dir():
            return candidate
        if candidate.is_file():
            try:
                content = candidate.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            if content.startswith("gitdir:"):
                pointer = Path(content[len("gitdir:"):].strip())
                resolved = pointer if pointer.is_absolute() else (directory / pointer)
                return resolved if resolved.is_dir() else None
    return None


def _read_ref(git_dir: Path, ref: str) -> str | None:
    """Resolve a symbolic ref to a commit hash (loose ref or packed-refs)."""
    loose = git_dir / ref
    if loose.is_file():
        try:
            return loose.read_text(encoding="utf-8").strip()
        except OSError:
            return None
    packed = git_dir / "packed-refs"
    if packed.is_file():
        try:
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith(("#", "^")):
                    sha, _, name = line.partition(" ")
                    if name.strip() == ref:
                        return sha.strip()
        except OSError:
            return None
    return None


# --------------------------------------------------------------------------- #
# Fingerprints
# --------------------------------------------------------------------------- #

def _sha256_of(obj: Any) -> str:
    """Stable SHA-256 hex digest of a JSON-canonicalizable object."""
    payload = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def config_fingerprint(config: FrameworkConfig) -> dict[str, str]:
    """Content fingerprint of the full framework configuration.

    Returns a mapping with the canonical ``sha256`` and a short ``digest`` (first
    12 hex chars) over the typed project/logging sections plus every raw extras
    section, so any config change is detectable across runs.
    """
    snapshot = {
        "project": _to_plain(config.project),
        "logging": _to_plain(config.logging),
        "extras": config.extras,
    }
    digest = _sha256_of(snapshot)
    return {"sha256": digest, "digest": digest[:12]}


def dataset_fingerprint(dataset: DatasetManager) -> dict[str, Any]:
    """Structural fingerprint of the dataset feeding an experiment.

    Captures counts (patients, samples), per-tissue and per-split sizes, and a
    SHA-256 over the sorted ``(patient_id, tissue, filename, hb)`` tuples — image
    *content* is not hashed (that belongs to the data pipeline), but any change to
    the sample roster or labels is detectable. Returns a minimal mapping when no
    dataset root/samples are available.
    """
    try:
        samples = dataset.samples()
    except Exception:  # noqa: BLE001 - provenance must never break an experiment
        return {"available": False}
    if not samples:
        return {"available": False}

    patients = sorted({s.patient_id for s in samples})
    tissue_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for s in samples:
        tissue_counts[s.tissue] = tissue_counts.get(s.tissue, 0) + 1
        split_name = s.split or "unassigned"
        split_counts[split_name] = split_counts.get(split_name, 0) + 1

    roster = sorted(
        (s.patient_id, s.tissue, Path(s.image_path).name, s.hb)
        for s in samples
    )
    return {
        "available": True,
        "root": str(dataset.root) if dataset.root is not None else None,
        "num_patients": len(patients),
        "num_samples": len(samples),
        "tissue_counts": dict(sorted(tissue_counts.items())),
        "split_sizes": dict(sorted(split_counts.items())),
        "roster_sha256": _sha256_of(roster),
    }


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def build_manifest(pipeline: HbPipeline, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the full provenance manifest for a pipeline's next experiment.

    Args:
        pipeline: The initialized :class:`~adaptivehb.pipeline.HbPipeline`.
        extra: Optional additional fields to merge into the manifest (e.g. the
            experiment id once it is known).

    Returns:
        A JSON-serializable manifest mapping.
    """
    config = pipeline.config
    dataset = pipeline.manager.dataset
    manifest: dict[str, Any] = {
        "generated_at": utcnow_iso(),
        "framework_version": __version__,
        "project": config.project.name,
        "seed": config.project.seed,
        "environment": collect_environment(),
        "git": git_revision(dataset.root or "."),
        "config": config_fingerprint(config),
        "dataset": dataset_fingerprint(dataset),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    """Write a manifest to ``path`` as JSON and return the path."""
    return write_json(path, manifest)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _to_plain(obj: Any) -> Any:
    """Best-effort conversion of a (possibly dataclass) config object to plain data."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


__all__ = [
    "collect_environment",
    "git_revision",
    "config_fingerprint",
    "dataset_fingerprint",
    "build_manifest",
    "write_manifest",
]
