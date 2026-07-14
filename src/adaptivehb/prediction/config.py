"""Typed configuration for the prediction subsystem (prediction.yaml)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EarlyStoppingSpec:
    """Early-stopping configuration."""

    enabled: bool = True
    patience: int = 20

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EarlyStoppingSpec:
        """Build from a raw mapping."""
        return cls(
            enabled=bool(data.get("enabled", True)),
            patience=int(data.get("patience", 20)),
        )


@dataclass(frozen=True)
class PredictionConfig:
    """Typed view of the ``prediction`` configuration section."""

    available_models: tuple[str, ...] = ("efficientnet", "resnet", "densenet", "vit", "convnext")
    default_model: str = "efficientnet"
    tissue_models: dict[str, str] = field(default_factory=dict)
    loss: str = "mse"
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    batch_size: int = 16
    epochs: int = 120
    scheduler: str = "cosine"
    weight_decay: float = 0.01
    pretrained: bool = True
    early_stopping: EarlyStoppingSpec = field(default_factory=EarlyStoppingSpec)
    input_resolution: tuple[int, int] = (224, 224)
    normalize: bool = True
    metadata_fusion: bool = False
    metadata_features: tuple[str, ...] = ()

    @classmethod
    def from_section(cls, section: Mapping[str, Any]) -> PredictionConfig:
        """Build a :class:`PredictionConfig` from the parsed section.

        Accepts either the full ``{"prediction": {...}}`` mapping or the inner
        mapping directly.
        """
        if "prediction" in section and isinstance(section["prediction"], Mapping):
            section = section["prediction"]
        training = dict(section.get("training", {}))
        input_spec = dict(section.get("input", {}))
        fusion = dict(section.get("metadata_fusion", {}))
        resolution = input_spec.get("resolution", [224, 224])
        return cls(
            available_models=tuple(
                str(m) for m in section.get(
                    "available_models", ["efficientnet", "resnet", "densenet", "vit", "convnext"]
                )
            ),
            default_model=str(section.get("default_model", "efficientnet")),
            tissue_models={str(k): str(v) for k, v in dict(section.get("tissue_models", {})).items()},
            loss=str(training.get("loss", "mse")),
            optimizer=str(training.get("optimizer", "adamw")),
            learning_rate=float(training.get("learning_rate", 3e-4)),
            batch_size=int(training.get("batch_size", 16)),
            epochs=int(training.get("epochs", 120)),
            scheduler=str(training.get("scheduler", "cosine")),
            weight_decay=float(training.get("weight_decay", 0.01)),
            pretrained=bool(training.get("pretrained", True)),
            early_stopping=EarlyStoppingSpec.from_dict(training.get("early_stopping", {})),
            input_resolution=(int(resolution[0]), int(resolution[1])),
            normalize=bool(input_spec.get("normalize", True)),
            metadata_fusion=bool(fusion.get("enabled", False)),
            metadata_features=tuple(str(f) for f in fusion.get("features", [])),
        )

    def architecture_for_tissue(self, tissue: str) -> str:
        """Return the configured architecture for a tissue (or the default)."""
        return self.tissue_models.get(tissue, self.default_model)


__all__ = ["PredictionConfig", "EarlyStoppingSpec"]
