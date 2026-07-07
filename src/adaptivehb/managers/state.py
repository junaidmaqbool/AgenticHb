"""StateManager — persistent pipeline progress for crash recovery.

The framework should always know exactly where it stopped (PROJECT_DESIGN_
SPECIFICATION Ch.12). A single ``pipeline_state.json`` file records the current
phase, module, epoch, completed work, and latest checkpoint, and is rewritten
after every important operation so execution can resume after interruption.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.core.interfaces import BaseManager
from adaptivehb.core.utils import read_json, utcnow_iso, write_json
from adaptivehb.exceptions import StateError


@dataclass
class PipelineState:
    """Serializable snapshot of pipeline progress."""

    experiment_id: str | None = None
    mode: str | None = None
    current_phase: str | None = None
    current_module: str | None = None
    current_epoch: int = 0
    completed_modules: list[str] = field(default_factory=list)
    pending_modules: list[str] = field(default_factory=list)
    last_checkpoint: str | None = None
    status: str = "idle"
    elapsed_seconds: float = 0.0
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineState:
        """Reconstruct a state from a stored dictionary, ignoring extra keys."""
        known = {key: data[key] for key in cls().to_dict() if key in data}
        return cls(**known)


class StateManager(BaseManager):
    """Reads, updates, and persists the pipeline state file."""

    def __init__(
        self,
        config: FrameworkConfig,
        base_dir: str | Path = ".",
        filename: str = "pipeline_state.json",
    ) -> None:
        """Initialize the state manager.

        Args:
            config: Validated framework configuration.
            base_dir: Root directory for the state file.
            filename: State file name.
        """
        super().__init__(config, base_dir)
        self._state_path = self._base_dir / filename
        self._state = PipelineState()

    @property
    def state(self) -> PipelineState:
        """Return the in-memory pipeline state."""
        return self._state

    @property
    def state_path(self) -> Path:
        """Return the path to the state file."""
        return self._state_path

    def exists(self) -> bool:
        """Return whether a persisted state file exists."""
        return self._state_path.is_file()

    def _on_initialize(self) -> None:
        if self.exists():
            self._state = PipelineState.from_dict(read_json(self._state_path))
            self._log.info("Pipeline state recovered (status=%s).", self._state.status)
        else:
            self.save()
            self._log.info("Pipeline state created at %s.", self._state_path)

    def update(self, **fields: Any) -> PipelineState:
        """Update state fields and persist.

        Args:
            **fields: Attributes of :class:`PipelineState` to overwrite.

        Returns:
            The updated state.

        Raises:
            StateError: If an unknown field is supplied.
        """
        valid = self._state.to_dict().keys()
        for key, value in fields.items():
            if key not in valid:
                raise StateError(f"Unknown pipeline-state field: {key!r}")
            setattr(self._state, key, value)
        self.save()
        return self._state

    def mark_completed(self, module: str) -> None:
        """Record a module as completed and remove it from pending."""
        if module not in self._state.completed_modules:
            self._state.completed_modules.append(module)
        if module in self._state.pending_modules:
            self._state.pending_modules.remove(module)
        self.save()
        self._log.info("Module completed: %s.", module)

    def is_completed(self, module: str) -> bool:
        """Return whether a module has already completed (skip on resume)."""
        return module in self._state.completed_modules

    def snapshot(self) -> dict[str, Any]:
        """Return a dictionary copy of the current state."""
        return self._state.to_dict()

    def reset(self) -> None:
        """Reset to a fresh state and persist (used when starting anew)."""
        self._state = PipelineState()
        self.save()
        self._log.info("Pipeline state reset.")

    def save(self) -> None:
        """Persist the current state, refreshing the update timestamp."""
        self._state.updated_at = utcnow_iso()
        write_json(self._state_path, self._state.to_dict())


__all__ = ["StateManager", "PipelineState"]
