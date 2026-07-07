"""Torch-free reference segmentation model.

A deterministic, dependency-free implementation of :class:`SegmentationModel`
used to exercise the segmentation interface, factory, manager, and pipeline
integration without the ML stack. It performs no learning: metrics improve
monotonically with the epoch and ``predict`` returns a trivial constant mask.
Real training uses the torch backends during the experiment phase.
"""

from __future__ import annotations

from typing import Any

from adaptivehb.segmentation.base import SegmentationModel
from adaptivehb.segmentation.registry import register_segmentation


class ReferenceSegmentationModel(SegmentationModel):
    """A no-op, deterministic segmentation model for framework verification."""

    def __init__(
        self,
        name: str = "reference",
        num_classes: int = 1,
        in_channels: int = 3,
        config: Any = None,
    ) -> None:
        """Initialize the reference model."""
        super().__init__(name, num_classes=num_classes, in_channels=in_channels, config=config)
        self._epochs_seen: list[int] = []

    def build(self) -> None:
        """Mark the model as built (no parameters to construct)."""
        self._built = True

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Return a decreasing loss and increasing Dice for the epoch."""
        self._require_built()
        self._epochs_seen.append(epoch)
        return {"train_loss": 1.0 / float(epoch), "train_dice": _rising(epoch)}

    def validate(self, epoch: int) -> dict[str, float]:
        """Return a decreasing validation loss and increasing Dice."""
        self._require_built()
        return {"val_loss": 1.0 / float(epoch), "val_dice": _rising(epoch)}

    def predict(self, image: Any) -> dict[str, Any]:
        """Return a trivial, deterministic mask descriptor for ``image``."""
        self._require_built()
        return {"model": self.name, "num_classes": self.num_classes, "mask": 0}

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot (epochs seen)."""
        return {"name": self.name, "epochs_seen": list(self._epochs_seen)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the snapshot (used to verify resume)."""
        self._epochs_seen = list(state.get("epochs_seen", []))
        self._built = True


def _rising(epoch: int) -> float:
    """A bounded, monotonically increasing pseudo-Dice score."""
    return min(0.99, epoch / (epoch + 1.0))


@register_segmentation("reference")
def _build_reference(name: str = "reference", **kwargs: Any) -> ReferenceSegmentationModel:
    return ReferenceSegmentationModel(name=name, **kwargs)


__all__ = ["ReferenceSegmentationModel"]
