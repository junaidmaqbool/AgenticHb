"""Typed configuration for the segmentation subsystem (segmentation.yaml)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EarlyStoppingSpec:
    """Early-stopping configuration."""

    enabled: bool = True
    patience: int = 15

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EarlyStoppingSpec:
        """Build from a raw mapping."""
        return cls(
            enabled=bool(data.get("enabled", True)),
            patience=int(data.get("patience", 15)),
        )


@dataclass(frozen=True)
class SegmentationConfig:
    """Typed view of the ``segmentation`` configuration section."""

    available_models: tuple[str, ...] = ("unet", "segformer", "deeplabv3plus")
    default_model: str = "unet"
    loss: str = "dice_bce"
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    batch_size: int = 8
    epochs: int = 100
    scheduler: str = "cosine"
    early_stopping: EarlyStoppingSpec = field(default_factory=EarlyStoppingSpec)
    threshold: float = 0.5
    checkpoint_dir: str = "checkpoints/segmentation"

    @classmethod
    def from_section(cls, section: Mapping[str, Any]) -> SegmentationConfig:
        """Build a :class:`SegmentationConfig` from the parsed section.

        Accepts either the full ``{"segmentation": {...}}`` mapping or the inner
        mapping directly.
        """
        if "segmentation" in section and isinstance(section["segmentation"], Mapping):
            section = section["segmentation"]
        training = dict(section.get("training", {}))
        inference = dict(section.get("inference", {}))
        return cls(
            available_models=tuple(
                str(m) for m in section.get("available_models", ["unet", "segformer", "deeplabv3plus"])
            ),
            default_model=str(section.get("default_model", "unet")),
            loss=str(training.get("loss", "dice_bce")),
            optimizer=str(training.get("optimizer", "adam")),
            learning_rate=float(training.get("learning_rate", 1e-4)),
            batch_size=int(training.get("batch_size", 8)),
            epochs=int(training.get("epochs", 100)),
            scheduler=str(training.get("scheduler", "cosine")),
            early_stopping=EarlyStoppingSpec.from_dict(training.get("early_stopping", {})),
            threshold=float(inference.get("threshold", 0.5)),
            checkpoint_dir=str(section.get("checkpoint_dir", "checkpoints/segmentation")),
        )


__all__ = ["SegmentationConfig", "EarlyStoppingSpec"]
