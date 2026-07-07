"""ExperimentManager — immutable, reproducible experiment directories.

Every experiment receives a unique ID and a self-contained directory holding its
configuration snapshot, logs, checkpoints, metrics, figures, predictions,
reports, and summary (PROJECT_DESIGN_SPECIFICATION Ch.13, PIPELINE_SPEC Ch.22).
Experiments are never overwritten, keeping every run permanently reproducible.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.core.interfaces import BaseManager
from adaptivehb.core.utils import ensure_dir, timestamp_slug, utcnow_iso, write_json
from adaptivehb.exceptions import PipelineError
from adaptivehb.version import __version__

_SUBDIRS: tuple[str, ...] = (
    "configuration",
    "logs",
    "checkpoints",
    "registry",
    "metrics",
    "figures",
    "predictions",
    "tensorboard",
    "excel",
    "csv",
    "reports",
    "pipeline_state",
    "summary",
)


@dataclass(frozen=True)
class Experiment:
    """Handle to a created experiment directory."""

    experiment_id: str
    name: str
    root: Path
    created_at: str

    def path_for(self, subdir: str) -> Path:
        """Return the path to a standard experiment subdirectory."""
        return self.root / subdir


class ExperimentManager(BaseManager):
    """Creates and tracks reproducible experiment directories."""

    def __init__(self, config: FrameworkConfig, base_dir: str | Path = ".") -> None:
        """Initialize the experiment manager.

        Args:
            config: Validated framework configuration.
            base_dir: Root directory; experiments live under
                ``base_dir / project.paths.experiments``.
        """
        super().__init__(config, base_dir)
        self._root = self._base_dir / config.project.paths.experiments

    def _on_initialize(self) -> None:
        ensure_dir(self._root)

    def create(self, name: str, config_snapshot: dict[str, Any] | None = None) -> Experiment:
        """Create a new, unique experiment directory.

        Args:
            name: Human-readable experiment name.
            config_snapshot: Configuration to persist for reproducibility.

        Returns:
            An :class:`Experiment` handle.

        Raises:
            PipelineError: If the generated experiment directory already exists.
        """
        # A short random suffix guarantees uniqueness even for runs created
        # within the same second, so experiments are never overwritten.
        experiment_id = f"{name}_{timestamp_slug()}_{uuid4().hex[:6]}"
        root = self._root / experiment_id
        if root.exists():  # pragma: no cover - astronomically unlikely collision
            raise PipelineError(f"Experiment directory already exists: {root}")
        for subdir in _SUBDIRS:
            ensure_dir(root / subdir)

        created_at = utcnow_iso()
        experiment = Experiment(experiment_id, name, root, created_at)

        write_json(
            experiment.path_for("configuration") / "meta.json",
            {
                "experiment_id": experiment_id,
                "name": name,
                "created_at": created_at,
                "framework_version": __version__,
                "environment": self._environment(),
            },
        )
        if config_snapshot is not None:
            write_json(
                experiment.path_for("configuration") / "config.json", config_snapshot
            )
        self._log.info("Created experiment %s at %s.", experiment_id, root)
        return experiment

    def save_summary(self, experiment: Experiment, summary: dict[str, Any]) -> Path:
        """Persist an experiment summary and return its path."""
        return write_json(experiment.path_for("summary") / "summary.json", summary)

    def list_experiments(self) -> list[str]:
        """Return the IDs of all existing experiments, sorted."""
        if not self._root.is_dir():
            return []
        return sorted(entry.name for entry in self._root.iterdir() if entry.is_dir())

    @staticmethod
    def _environment() -> dict[str, str]:
        """Capture minimal environment information for reproducibility."""
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        }


__all__ = ["ExperimentManager", "Experiment"]
