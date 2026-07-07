"""Dummy trainables for framework dry-runs and testing.

These implement the :class:`~adaptivehb.managers.training.Trainable` protocol
with a deterministic, monotonically-improving loss, so the data-driven pipeline
modes (training/evaluation/inference) can be exercised end-to-end before real
segmentation and prediction models exist (Phases 5-6). They contain no learning
and no heavy dependencies.
"""

from __future__ import annotations

from typing import Any


class DummyTrainable:
    """A no-op trainable with a deterministic decreasing validation loss."""

    def __init__(self, name: str, base_loss: float = 1.0) -> None:
        """Initialize the dummy trainable.

        Args:
            name: Identifier (typically the model/checkpoint name).
            base_loss: Loss scale; ``val_loss`` at epoch ``e`` is
                ``base_loss / e`` so lower epochs improve monotonically.
        """
        self.name = name
        self._base_loss = base_loss
        self.epochs_seen: list[int] = []

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Record the epoch and return a decreasing training loss."""
        self.epochs_seen.append(epoch)
        return {"train_loss": self._base_loss / float(epoch)}

    def validate(self, epoch: int) -> dict[str, float]:
        """Return a decreasing validation loss for the epoch."""
        return {"val_loss": self._base_loss / float(epoch)}

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of the dummy state."""
        return {"name": self.name, "seen": list(self.epochs_seen)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the dummy state (used to verify resume)."""
        self.epochs_seen = list(state.get("seen", []))


def make_dummy_factory() -> Any:
    """Return a factory mapping a training plan to a :class:`DummyTrainable`."""

    def _factory(plan: Any) -> DummyTrainable:
        return DummyTrainable(getattr(plan, "name", "dummy"))

    return _factory


__all__ = ["DummyTrainable", "make_dummy_factory"]
