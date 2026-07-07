"""Common segmentation model interface.

Every segmentation model exposes the same interface so they are interchangeable
and can be trained by the generic TrainingManager and selected by the adaptive
framework (PROJECT_DESIGN_SPECIFICATION Ch.20, IMPLEMENTATION_ROADMAP Phase 5).

The interface implements the :class:`~adaptivehb.managers.training.Trainable`
contract (``train_epoch``/``validate``/``state_dict``/``load_state_dict``) plus
segmentation-specific ``build`` and ``predict``. It is torch-free at this level;
concrete backends (real networks) live in ``torch_models`` behind guarded
imports, while a torch-free :class:`ReferenceSegmentationModel` keeps the whole
framework runnable and testable without the ML stack.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from adaptivehb.core.utils import ensure_dir


class SegmentationModel(ABC):
    """Abstract, interchangeable segmentation model."""

    def __init__(
        self,
        name: str,
        num_classes: int = 1,
        in_channels: int = 3,
        config: Any = None,
    ) -> None:
        """Initialize common attributes.

        Args:
            name: Architecture/instance name (e.g. ``"unet"``).
            num_classes: Number of segmentation output classes.
            in_channels: Number of input image channels.
            config: Optional typed segmentation configuration.
        """
        self.name = name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self._config = config
        self._built = False

    @property
    def is_built(self) -> bool:
        """Whether :meth:`build` has completed."""
        return self._built

    # -- interface ---------------------------------------------------------

    @abstractmethod
    def build(self) -> None:
        """Construct the underlying model (idempotent)."""

    @abstractmethod
    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch and return metrics."""

    @abstractmethod
    def validate(self, epoch: int) -> dict[str, float]:
        """Run validation for an epoch and return metrics."""

    @abstractmethod
    def predict(self, image: Any) -> Any:
        """Produce a segmentation mask for a single image or batch."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of model state."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore model state from a snapshot."""

    # -- persistence (concrete; torch backends may override) ---------------

    def save(self, path: str | Path) -> Path:
        """Persist the model state via pickle. Returns the written path."""
        destination = Path(path)
        ensure_dir(destination.parent)
        with destination.open("wb") as handle:
            pickle.dump(self.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return destination

    def load(self, path: str | Path) -> None:
        """Restore the model state from a pickle file."""
        with Path(path).open("rb") as handle:
            self.load_state_dict(pickle.load(handle))

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError(f"Segmentation model {self.name!r} is not built; call build().")


__all__ = ["SegmentationModel"]
