"""CheckpointManager — recoverable training checkpoints.

Training must never be lost: every important stage saves a recoverable
checkpoint containing model/optimizer/scheduler state plus metadata (epoch,
metrics, seed, config, history) (PROJECT_DESIGN_SPECIFICATION Ch.11). Each model
keeps a ``latest`` and a ``best`` checkpoint.

Persistence is intentionally torch-free: the arbitrary state payload is pickled
(``pickle`` also round-trips torch tensors when torch is present) and a
human-readable JSON metadata sidecar is written alongside it. The serializer can
be swapped for ``torch.save`` in a later phase without changing the API.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.core.interfaces import BaseManager
from adaptivehb.core.utils import ensure_dir, read_json, utcnow_iso, write_json
from adaptivehb.exceptions import CheckpointError

_LATEST = "latest"
_BEST = "best"


class CheckpointManager(BaseManager):
    """Saves and restores per-model ``latest`` and ``best`` checkpoints."""

    def __init__(self, config: FrameworkConfig, base_dir: str | Path = ".") -> None:
        """Initialize the checkpoint manager.

        Args:
            config: Validated framework configuration.
            base_dir: Root directory; checkpoints live under
                ``base_dir / project.paths.checkpoints``.
        """
        super().__init__(config, base_dir)
        self._root = self._base_dir / config.project.paths.checkpoints

    def _on_initialize(self) -> None:
        ensure_dir(self._root)

    # -- saving ------------------------------------------------------------

    def save(
        self,
        name: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        *,
        is_best: bool = False,
    ) -> Path:
        """Save the ``latest`` checkpoint for a model (and ``best`` if flagged).

        Args:
            name: Model/checkpoint name (a subdirectory is created for it).
            payload: Arbitrary state (weights, optimizer, scheduler, epoch, ...).
            metadata: Human-readable metadata; a timestamp is added automatically.
            is_best: Also write this checkpoint as the model's ``best``.

        Returns:
            Path to the written ``latest`` payload file.
        """
        meta = dict(metadata or {})
        meta.setdefault("name", name)
        meta["saved_at"] = utcnow_iso()

        latest_path = self._write(name, _LATEST, payload, meta)
        if is_best:
            self._write(name, _BEST, payload, {**meta, "is_best": True})
            self._log.info("Saved checkpoint %s (latest + best).", name)
        else:
            self._log.info("Saved checkpoint %s (latest).", name)
        return latest_path

    # -- loading -----------------------------------------------------------

    def load_latest(self, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load the ``latest`` checkpoint. Returns ``(payload, metadata)``."""
        return self._load(name, _LATEST)

    def load_best(self, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load the ``best`` checkpoint. Returns ``(payload, metadata)``."""
        return self._load(name, _BEST)

    # -- queries -----------------------------------------------------------

    def exists(self, name: str, tag: str = _LATEST) -> bool:
        """Return whether a checkpoint exists for ``name`` with the given tag."""
        return self._payload_path(name, tag).is_file()

    def list_checkpoints(self) -> list[str]:
        """Return the names of all models that have at least one checkpoint."""
        if not self._root.is_dir():
            return []
        return sorted(
            entry.name for entry in self._root.iterdir() if entry.is_dir()
        )

    # -- internals ---------------------------------------------------------

    def _payload_path(self, name: str, tag: str) -> Path:
        return self._root / name / f"{tag}.pkl"

    def _meta_path(self, name: str, tag: str) -> Path:
        return self._root / name / f"{tag}.json"

    def _write(
        self, name: str, tag: str, payload: dict[str, Any], meta: dict[str, Any]
    ) -> Path:
        ensure_dir(self._root / name)
        payload_path = self._payload_path(name, tag)
        tmp = payload_path.with_name(payload_path.name + ".tmp")
        with tmp.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, payload_path)
        write_json(self._meta_path(name, tag), meta)
        return payload_path

    def _load(self, name: str, tag: str) -> tuple[dict[str, Any], dict[str, Any]]:
        payload_path = self._payload_path(name, tag)
        if not payload_path.is_file():
            raise CheckpointError(f"No '{tag}' checkpoint for {name!r} at {payload_path}.")
        with payload_path.open("rb") as handle:
            payload = pickle.load(handle)
        meta_path = self._meta_path(name, tag)
        metadata = read_json(meta_path) if meta_path.is_file() else {}
        return payload, metadata


__all__ = ["CheckpointManager"]
