"""Typed configuration for the dataset subsystem.

Parses the ``dataset`` section of ``configs/dataset.yaml`` into strongly typed
objects. Following the framework convention, dataset-specific values (paths,
tissues, image size, column names, split ratios) originate only from
configuration — never hardcoded in source.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from adaptivehb.exceptions import ConfigError


@dataclass(frozen=True)
class ImageSpec:
    """Image expectations for the dataset."""

    channels: int = 3
    resolution: tuple[int, int] = (224, 224)
    formats: tuple[str, ...] = ("jpg", "jpeg", "png", "tiff", "bmp")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ImageSpec:
        """Build an :class:`ImageSpec` from a raw mapping."""
        resolution = data.get("resolution", [224, 224])
        formats = data.get("formats", ["jpg", "jpeg", "png", "tiff", "bmp"])
        return cls(
            channels=int(data.get("channels", 3)),
            resolution=(int(resolution[0]), int(resolution[1])),
            formats=tuple(str(fmt).lower().lstrip(".") for fmt in formats),
        )


@dataclass(frozen=True)
class MetadataSpec:
    """Metadata column expectations."""

    patient_id_column: str = "Patient_ID"
    target_column: str = "Hemoglobin"
    mandatory_columns: tuple[str, ...] = ("Patient_ID", "Hemoglobin", "Age", "Gender")
    optional_columns: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetadataSpec:
        """Build a :class:`MetadataSpec` from a raw mapping."""
        return cls(
            patient_id_column=str(data.get("patient_id_column", "Patient_ID")),
            target_column=str(data.get("target_column", "Hemoglobin")),
            mandatory_columns=tuple(
                str(c) for c in data.get("mandatory_columns", ["Patient_ID", "Hemoglobin"])
            ),
            optional_columns=tuple(str(c) for c in data.get("optional_columns", [])),
        )


@dataclass(frozen=True)
class SplitSpec:
    """Dataset splitting configuration."""

    strategy: str = "patient_level"
    train: float = 0.80
    validation: float = 0.10
    test: float = 0.10
    seed: int = 42

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SplitSpec:
        """Build a :class:`SplitSpec`, validating that ratios sum to ~1.0."""
        spec = cls(
            strategy=str(data.get("strategy", "patient_level")),
            train=float(data.get("train", 0.80)),
            validation=float(data.get("validation", 0.10)),
            test=float(data.get("test", 0.10)),
            seed=int(data.get("seed", 42)),
        )
        total = spec.train + spec.validation + spec.test
        if abs(total - 1.0) > 1e-6:
            raise ConfigError(f"Split ratios must sum to 1.0 (got {total:.4f}).")
        return spec


@dataclass(frozen=True)
class DatasetConfig:
    """Typed view of the ``dataset`` configuration section."""

    name: str | None = None
    version: str | None = None
    root: str | None = None
    images_dir: str = "images"
    masks_dir: str = "masks"
    metadata_file: str = "metadata/patients.csv"
    splits_dir: str = "splits"
    # Optional SEPARATE dataset for training segmentation (images + masks). When
    # segmentation_root is set, segmentation trains on this source instead of the
    # main (Hb) root; the main root is used for prediction + evaluation.
    segmentation_root: str | None = None
    segmentation_images_dir: str = "images"
    segmentation_masks_dir: str = "masks"
    segmentation_metadata_file: str | None = None
    tissues: tuple[str, ...] = ("eye", "palm", "tongue", "nail")
    tissue_sides: dict[str, list[str]] = field(default_factory=dict)
    image: ImageSpec = field(default_factory=ImageSpec)
    metadata: MetadataSpec = field(default_factory=MetadataSpec)
    split: SplitSpec = field(default_factory=SplitSpec)

    @classmethod
    def from_section(cls, section: Mapping[str, Any]) -> DatasetConfig:
        """Build a :class:`DatasetConfig` from the parsed ``dataset`` mapping.

        Args:
            section: The mapping under the top-level ``dataset`` key.

        Returns:
            A typed :class:`DatasetConfig`.

        Raises:
            ConfigError: If the mapping is malformed.
        """
        if "dataset" in section and isinstance(section["dataset"], Mapping):
            section = section["dataset"]
        images_dir = str(section.get("images_dir", "images"))
        masks_dir = str(section.get("masks_dir", "masks"))
        seg_src = dict(section.get("segmentation_source", {}))
        return cls(
            name=_opt_str(section.get("name")),
            version=_opt_str(section.get("version")),
            root=_opt_str(section.get("root")),
            images_dir=images_dir,
            masks_dir=masks_dir,
            metadata_file=str(section.get("metadata_file", "metadata/patients.csv")),
            splits_dir=str(section.get("splits_dir", "splits")),
            segmentation_root=_opt_str(seg_src.get("root")),
            segmentation_images_dir=str(seg_src.get("images_dir", images_dir)),
            segmentation_masks_dir=str(seg_src.get("masks_dir", masks_dir)),
            segmentation_metadata_file=_opt_str(seg_src.get("metadata_file")),
            tissues=tuple(str(t) for t in section.get("tissues", ["eye", "palm", "tongue", "nail"])),
            tissue_sides={
                str(k): [str(s) for s in v]
                for k, v in dict(section.get("tissue_sides", {})).items()
            },
            image=ImageSpec.from_dict(section.get("image", {})),
            metadata=MetadataSpec.from_dict(section.get("metadata", {})),
            split=SplitSpec.from_dict(section.get("split", {})),
        )


def _opt_str(value: Any) -> str | None:
    """Return ``None`` for null-like values, otherwise the string form."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["DatasetConfig", "ImageSpec", "MetadataSpec", "SplitSpec"]
