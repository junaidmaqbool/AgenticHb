"""Shared, dependency-light utilities for the AdaptiveHb framework.

These helpers underpin the infrastructure managers: timestamps, directory
creation, atomic JSON persistence, and reproducible seeding. NumPy and PyTorch
are seeded only if they are installed, so this module imports cleanly without
the ML stack.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def timestamp_slug() -> str:
    """Return a filesystem-safe UTC timestamp usable in directory names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed and return it.

    Args:
        path: Directory path.

    Returns:
        The directory as a :class:`~pathlib.Path`.
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> Any:
    """Read and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        The decoded JSON content.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any, *, atomic: bool = True) -> Path:
    """Serialize ``data`` to JSON, creating parent directories as needed.

    Args:
        path: Destination path.
        data: JSON-serializable content (non-serializable values fall back to
            ``str`` via ``default=str``).
        atomic: When true, write to a temporary file and atomically replace the
            destination to avoid partially written files.

    Returns:
        The destination path.
    """
    destination = Path(path)
    ensure_dir(destination.parent)
    if atomic:
        tmp = destination.with_name(destination.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=str, sort_keys=False)
        os.replace(tmp, destination)
    else:
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=str, sort_keys=False)
    return destination


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    NumPy and PyTorch are seeded only when importable, so this function is safe
    to call in a torch-free environment.

    Args:
        seed: Random seed.
        deterministic: When true and PyTorch is present, request deterministic
            cuDNN behaviour.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy optional at Phase 2
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch optional at Phase 2
        pass


__all__ = [
    "utcnow_iso",
    "timestamp_slug",
    "ensure_dir",
    "read_json",
    "write_json",
    "set_global_seed",
]
