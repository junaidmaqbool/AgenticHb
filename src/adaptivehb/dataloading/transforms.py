"""Preprocessing / augmentation specification for the training-data bridge.

``TransformSpec`` is parsed from the ``dataset`` configuration (resolution,
normalization, augmentation) and is fully testable without any vision library.
``build_transform`` constructs an Albumentations pipeline when that optional
dependency is present and returns ``None`` otherwise, so the module imports and
the spec parses cleanly without the ML/vision stack (Decision 025). Augmentation
is applied only to the training split (DATASET_SPEC Ch.17).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TransformSpec:
    """Typed preprocessing / augmentation settings."""

    resolution: tuple[int, int] = (224, 224)
    normalize: bool = True
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    augment: bool = True
    horizontal_flip: float = 0.5
    rotation: int = 15
    brightness_contrast: float = 0.2

    @classmethod
    def from_section(cls, section: Mapping[str, Any]) -> TransformSpec:
        """Build a :class:`TransformSpec` from the parsed ``dataset`` mapping."""
        if "dataset" in section and isinstance(section["dataset"], Mapping):
            section = section["dataset"]
        image = dict(section.get("image", {}))
        preprocessing = dict(section.get("preprocessing", {}))
        augmentation = dict(section.get("augmentation", {}))
        resolution = image.get("resolution", [224, 224])
        return cls(
            resolution=(int(resolution[0]), int(resolution[1])),
            normalize=bool(preprocessing.get("normalize", True)),
            mean=tuple(float(v) for v in preprocessing.get("mean", [0.485, 0.456, 0.406])),
            std=tuple(float(v) for v in preprocessing.get("std", [0.229, 0.224, 0.225])),
            augment=bool(augmentation.get("enabled", True)),
            horizontal_flip=float(augmentation.get("horizontal_flip", 0.5)),
            rotation=int(augmentation.get("rotation", 15)),
            brightness_contrast=float(augmentation.get("brightness_contrast", 0.2)),
        )


def transform_available() -> bool:
    """Whether the Albumentations transform backend is installed."""
    return importlib.util.find_spec("albumentations") is not None


def build_transform(spec: TransformSpec, *, training: bool = True) -> Any | None:
    """Build an Albumentations transform pipeline (or ``None`` if unavailable).

    Args:
        spec: The preprocessing/augmentation specification.
        training: When true, include augmentation ops (train split only).

    Returns:
        An ``albumentations.Compose`` pipeline, or ``None`` when Albumentations
        is not installed (real training requires it).
    """
    if not transform_available():
        return None
    import albumentations as A  # pragma: no cover - requires albumentations

    ops: list[Any] = [A.Resize(spec.resolution[1], spec.resolution[0])]
    if training and spec.augment:
        if spec.horizontal_flip > 0:
            ops.append(A.HorizontalFlip(p=spec.horizontal_flip))
        if spec.rotation > 0:
            ops.append(A.Rotate(limit=spec.rotation))
        if spec.brightness_contrast > 0:
            ops.append(A.RandomBrightnessContrast(p=spec.brightness_contrast))
    if spec.normalize:
        ops.append(A.Normalize(mean=spec.mean, std=spec.std))
    return A.Compose(ops)


__all__ = ["TransformSpec", "build_transform", "transform_available"]
