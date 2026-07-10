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
    """Metadata column expectations.

    In addition to the mandatory/optional column contract, this spec names the
    columns used to *derive* BMI algorithmically (``height_column`` /
    ``weight_column`` -> ``bmi_column``) and the ``needed_columns`` a client
    (e.g. the training notebook) wants carried through the pipeline. All column
    names originate from configuration — never hardcoded in source.
    """

    patient_id_column: str = "Patient_ID"
    target_column: str = "Hemoglobin"
    mandatory_columns: tuple[str, ...] = ("Patient_ID", "Hemoglobin", "Age", "Gender")
    optional_columns: tuple[str, ...] = ()
    # Columns a client explicitly wants available downstream (patient id, target,
    # anthropometrics, …). Informational: used by clients and reporting; the
    # validator still enforces ``mandatory_columns`` separately.
    needed_columns: tuple[str, ...] = ()
    # Anthropometric columns used to compute BMI when it is absent/blank.
    height_column: str = "Height"
    weight_column: str = "Weight"
    bmi_column: str = "BMI"
    height_unit: str = "cm"  # cm | m — unit of the height column, for the BMI formula
    compute_bmi: bool = True  # derive BMI from height/weight when missing

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
            needed_columns=tuple(str(c) for c in data.get("needed_columns", [])),
            height_column=str(data.get("height_column", "Height")),
            weight_column=str(data.get("weight_column", "Weight")),
            bmi_column=str(data.get("bmi_column", "BMI")),
            height_unit=str(data.get("height_unit", "cm")).lower(),
            compute_bmi=bool(data.get("compute_bmi", True)),
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


# Sentinel sampling modes for pooling the sides of a tissue against one patient.
SAMPLING_EXTENDED = "extended"  # each image is an independent data point (agentic)
SAMPLING_SINGLE = "single"      # one representative image per (patient, tissue)
SAMPLING_MODES = (SAMPLING_EXTENDED, SAMPLING_SINGLE)


@dataclass(frozen=True)
class SideSource:
    """One image (and optional mask) directory for a single tissue *side*.

    A tissue may have several sides (e.g. ``left``/``right`` eyes), each living in
    its own directory and possibly its own dataset. ``side`` is ``None`` for a
    single-source tissue (e.g. tongue).
    """

    images: str
    masks: str | None = None
    side: str | None = None


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
    # Optional per-tissue image/mask directories, normalized to a list of
    # :class:`SideSource`. A tissue entry may be either a flat mapping
    # (``{"images": <dir>, "masks": <dir>}`` — a single unsided source) or a sided
    # mapping (``{"sides": {"left": {...}, "right": {...}}}``). When a tissue has an
    # entry here, those directories are scanned directly instead of the conventional
    # ``root/images_dir/<tissue>`` layout, so each side may live in a separate
    # dataset. A tissue with no usable images directory is simply skipped, letting
    # experiments mix datasets that each cover a different subset of tissues.
    tissue_sources: dict[str, tuple[SideSource, ...]] = field(default_factory=dict)
    # How to pool the sides of a tissue against one patient's label. ``extended``
    # (default) keeps every image as an independent data point; ``single`` keeps one
    # representative image per (patient, tissue). ``tissue_sampling_mode`` overrides
    # the global ``sampling_mode`` per tissue.
    sampling_mode: str = SAMPLING_EXTENDED
    tissue_sampling_mode: dict[str, str] = field(default_factory=dict)
    image: ImageSpec = field(default_factory=ImageSpec)
    metadata: MetadataSpec = field(default_factory=MetadataSpec)
    split: SplitSpec = field(default_factory=SplitSpec)

    def sampling_mode_for(self, tissue: str) -> str:
        """Return the sampling mode for ``tissue`` (per-tissue override or global)."""
        return self.tissue_sampling_mode.get(tissue, self.sampling_mode)

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
            tissue_sources={
                str(k): sources
                for k, v in dict(section.get("tissue_sources", {})).items()
                if isinstance(v, Mapping) and (sources := _parse_tissue_source(v))
            },
            sampling_mode=_parse_sampling_mode(section.get("sampling_mode", SAMPLING_EXTENDED)),
            tissue_sampling_mode={
                str(k): _parse_sampling_mode(v)
                for k, v in dict(section.get("tissue_sampling_mode", {})).items()
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


def _parse_sampling_mode(value: Any) -> str:
    """Validate and normalize a sampling-mode token."""
    mode = str(value).strip().lower()
    if mode not in SAMPLING_MODES:
        raise ConfigError(
            f"Unknown sampling_mode {value!r}; expected one of {list(SAMPLING_MODES)}."
        )
    return mode


def _parse_tissue_source(entry: Mapping[str, Any]) -> tuple[SideSource, ...]:
    """Normalize one ``tissue_sources`` entry into a tuple of :class:`SideSource`.

    Accepts either a flat ``{"images": ..., "masks": ...}`` mapping (a single
    unsided source) or a sided ``{"sides": {"left": {...}, "right": {...}}}``
    mapping. Sides or sources whose ``images`` is null/blank are dropped so a
    partially-populated dataset degrades gracefully.
    """
    sides = entry.get("sides")
    if isinstance(sides, Mapping):
        out: list[SideSource] = []
        for side_name, spec in sides.items():
            if not isinstance(spec, Mapping):
                continue
            source = _side_source(spec, side=str(side_name))
            if source is not None:
                out.append(source)
        return tuple(out)
    source = _side_source(entry, side=None)
    return (source,) if source is not None else ()


def _side_source(spec: Mapping[str, Any], *, side: str | None) -> SideSource | None:
    """Build a :class:`SideSource` from a mapping, or ``None`` if it has no images."""
    images = _opt_str(spec.get("images"))
    if images is None:
        return None
    return SideSource(images=images, masks=_opt_str(spec.get("masks")), side=side)


__all__ = [
    "DatasetConfig",
    "ImageSpec",
    "MetadataSpec",
    "SplitSpec",
    "SideSource",
    "SAMPLING_EXTENDED",
    "SAMPLING_SINGLE",
    "SAMPLING_MODES",
]
