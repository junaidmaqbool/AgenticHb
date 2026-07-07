"""Torch-free reference prediction model.

A deterministic, dependency-free implementation of :class:`PredictionModel` used
to exercise the prediction interface, factory, manager, and pipeline integration
without the ML stack. It performs no learning: the loss/MAE improve monotonically
with the epoch and ``predict`` returns a fixed hemoglobin estimate. Real training
uses the torch backbones during the experiment phase.
"""

from __future__ import annotations

from typing import Any

from adaptivehb.prediction.base import PredictionModel
from adaptivehb.prediction.registry import register_prediction

# A clinically plausible mid-range hemoglobin value (g/dL) used as the constant
# reference estimate. Real models learn per-tissue estimates.
_DEFAULT_ESTIMATE = 13.0


class ReferencePredictionModel(PredictionModel):
    """A no-op, deterministic regressor for framework verification."""

    def __init__(
        self,
        name: str = "reference",
        tissue: str | None = None,
        in_channels: int = 3,
        config: Any = None,
        estimate: float = _DEFAULT_ESTIMATE,
    ) -> None:
        """Initialize the reference regressor."""
        super().__init__(name, tissue=tissue, in_channels=in_channels, config=config)
        self._estimate = float(estimate)
        self._epochs_seen: list[int] = []

    def build(self) -> None:
        """Mark the model as built (no parameters to construct)."""
        self._built = True

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Return a decreasing loss/MAE for the epoch."""
        self._require_built()
        self._epochs_seen.append(epoch)
        return {"train_loss": 1.0 / float(epoch), "train_mae": _falling(epoch)}

    def validate(self, epoch: int) -> dict[str, float]:
        """Return a decreasing validation loss/MAE for the epoch."""
        self._require_built()
        return {"val_loss": 1.0 / float(epoch), "val_mae": _falling(epoch)}

    def predict(self, image: Any, metadata: dict[str, Any] | None = None) -> float:
        """Return the constant hemoglobin estimate."""
        self._require_built()
        return self._estimate

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot."""
        return {"name": self.name, "tissue": self.tissue, "epochs_seen": list(self._epochs_seen)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the snapshot (used to verify resume)."""
        self.name = state.get("name", self.name)
        self.tissue = state.get("tissue", self.tissue)
        self._epochs_seen = list(state.get("epochs_seen", []))
        self._built = True


def _falling(epoch: int) -> float:
    """A bounded, monotonically decreasing pseudo-MAE (g/dL)."""
    return max(0.1, 2.0 / float(epoch))


@register_prediction("reference")
def _build_reference(name: str = "reference", **kwargs: Any) -> ReferencePredictionModel:
    return ReferencePredictionModel(name=name, **kwargs)


__all__ = ["ReferencePredictionModel"]
