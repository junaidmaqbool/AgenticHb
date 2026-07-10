"""DatasetManager — the single entry point for dataset access (DATASET_SPEC Ch.23).

No other module accesses dataset files directly. The manager resolves paths from
configuration, loads metadata, scans images/masks into standardized
:class:`~adaptivehb.dataset.schema.Sample` records, validates the dataset,
produces patient-level splits, and computes statistics.

Image decoding, preprocessing, and augmentation belong to the training data
pipeline (Phases 5-6, which pull in albumentations/opencv) and are intentionally
not part of this dependency-light core.
"""

from __future__ import annotations

from pathlib import Path

from adaptivehb.config.loader import FrameworkConfig
from adaptivehb.core.interfaces import BaseManager
from adaptivehb.core.utils import ensure_dir, write_json
from adaptivehb.dataset.config import SAMPLING_SINGLE, DatasetConfig
from adaptivehb.dataset.metadata import MetadataTable
from adaptivehb.dataset.schema import DatasetStatistics, Sample, ValidationReport
from adaptivehb.dataset.splitting import invert_split, patient_level_split
from adaptivehb.dataset.statistics import compute_statistics
from adaptivehb.dataset.validation import DatasetValidator
from adaptivehb.exceptions import DatasetError


class DatasetManager(BaseManager):
    """Loads, validates, splits, and summarizes a spec-conformant dataset."""

    def __init__(
        self,
        config: FrameworkConfig,
        base_dir: str | Path = ".",
        dataset_root: str | Path | None = None,
        *,
        images_dir: str | None = None,
        masks_dir: str | None = None,
        metadata_file: str | None = None,
        metadata_optional: bool = False,
    ) -> None:
        """Initialize the dataset manager.

        Args:
            config: Validated framework configuration.
            base_dir: Root directory for resolving relative dataset paths.
            dataset_root: Explicit dataset root; overrides the config value.
            images_dir: Override for the images subdirectory (defaults to config).
            masks_dir: Override for the masks subdirectory (defaults to config).
            metadata_file: Override for the metadata CSV path (defaults to config).
            metadata_optional: When true, a missing metadata CSV is tolerated (the
                dataset then has no labels; used for a mask-only segmentation source).
        """
        super().__init__(config, base_dir)
        self._ds_config = DatasetConfig.from_section(config.section("dataset"))
        root = dataset_root if dataset_root is not None else self._ds_config.root
        self._root: Path | None = self._resolve(root) if root else None
        self._images_dir = images_dir or self._ds_config.images_dir
        self._masks_dir = masks_dir or self._ds_config.masks_dir
        self._metadata_file = metadata_file if metadata_file is not None else self._ds_config.metadata_file
        self._metadata_optional = metadata_optional
        self._metadata: MetadataTable | None = None
        self._index: list[Sample] | None = None
        self._splits: dict[str, list[str]] | None = None
        self._pinned_split: dict[str, list[str]] | None = None

    # -- properties --------------------------------------------------------

    @property
    def dataset_config(self) -> DatasetConfig:
        """The typed dataset configuration."""
        return self._ds_config

    @property
    def root(self) -> Path | None:
        """The resolved dataset root, if set."""
        return self._root

    # -- loading -----------------------------------------------------------

    def load_metadata(self) -> MetadataTable:
        """Load (and cache) the metadata table."""
        if self._metadata is None:
            path = self._require_root() / self._metadata_file
            id_column = self._ds_config.metadata.patient_id_column
            if self._metadata_optional and not path.is_file():
                self._metadata = MetadataTable([], [], id_column)
                self._log.info("No metadata file (optional source); proceeding without labels.")
            else:
                self._metadata = MetadataTable.load(path, id_column)
                self._log.info("Loaded metadata: %d patient(s).", len(self._metadata.patient_ids))
            self._apply_derived_columns(self._metadata)
        return self._metadata

    def _apply_derived_columns(self, metadata: MetadataTable) -> None:
        """Compute configured derived columns (currently BMI) on the metadata."""
        meta_spec = self._ds_config.metadata
        if meta_spec.compute_bmi:
            filled = metadata.derive_bmi(
                height_column=meta_spec.height_column,
                weight_column=meta_spec.weight_column,
                bmi_column=meta_spec.bmi_column,
                height_unit=meta_spec.height_unit,
            )
            if filled:
                self._log.info("Derived %s for %d patient(s) from height/weight.",
                               meta_spec.bmi_column, filled)

    def build_index(self) -> list[Sample]:
        """Scan images and masks into standardized samples (cached)."""
        if self._index is not None:
            return self._index
        root = self._require_root()
        metadata = self.load_metadata()
        images_root = root / self._images_dir
        masks_root = root / self._masks_dir
        formats = self._ds_config.image.formats
        target = self._ds_config.metadata.target_column

        samples: list[Sample] = []
        for tissue in self._ds_config.tissues:
            tissue_samples: list[Sample] = []
            for images_dir, masks_dir, side in self._tissue_sources(
                tissue, images_root, masks_root
            ):
                if not images_dir.is_dir():
                    # A source with no usable images directory is simply not
                    # applicable (e.g. this dataset does not cover this tissue/side);
                    # skip it silently so experiments can mix datasets covering
                    # different tissue/side subsets.
                    self._log.debug(
                        "Tissue %r side %r has no images directory (%s); skipping.",
                        tissue, side, images_dir,
                    )
                    continue
                for image_path in sorted(images_dir.iterdir()):
                    if not image_path.is_file():
                        continue
                    if image_path.suffix.lower().lstrip(".") not in formats:
                        continue
                    stem = image_path.stem
                    patient_id = stem.split("_")[0]
                    row = metadata.get(patient_id) or {}
                    tissue_samples.append(
                        Sample(
                            patient_id=patient_id,
                            tissue=tissue,
                            image_path=str(image_path),
                            mask_path=self._find_mask(masks_dir, stem, formats),
                            hb=_to_float(row.get(target)),
                            metadata=row,
                            side=side or _side_from_stem(stem),
                        )
                    )
            samples.extend(self._apply_sampling_mode(tissue, tissue_samples))
        self._index = samples
        self._log.info("Built dataset index: %d sample(s).", len(samples))
        return samples

    def _apply_sampling_mode(self, tissue: str, tissue_samples: list[Sample]) -> list[Sample]:
        """Pool a tissue's samples according to its sampling mode.

        ``extended`` keeps every image as an independent data point; ``single``
        keeps one representative image per patient (deterministically the first by
        image path), which correlates each patient to a single image of the tissue.
        """
        mode = self._ds_config.sampling_mode_for(tissue)
        if mode == SAMPLING_SINGLE:
            chosen: dict[str, Sample] = {}
            for sample in sorted(tissue_samples, key=lambda s: s.image_path):
                chosen.setdefault(sample.patient_id, sample)
            return list(chosen.values())
        return tissue_samples

    def load(self) -> list[Sample]:
        """Load metadata and build the sample index."""
        self.load_metadata()
        return self.build_index()

    # -- operations --------------------------------------------------------

    def validate(self) -> ValidationReport:
        """Validate the dataset and return a report."""
        report = DatasetValidator(self._ds_config).validate(
            self.load_metadata(), self.build_index()
        )
        level = "valid" if report.is_valid else "INVALID"
        self._log.info(
            "Dataset validation: %s (%d error(s), %d warning(s)).",
            level, len(report.errors), len(report.warnings),
        )
        return report

    def split(self) -> dict[str, list[str]]:
        """Generate patient-level splits and tag the sample index.

        When an explicit split has been pinned via :meth:`apply_split` (e.g. by the
        cross-validation runner), that split is (re)applied instead of drawing a
        fresh random one, so callers that trigger ``split()`` internally do not
        clobber an externally-controlled fold assignment.
        """
        if self._pinned_split is not None:
            return self._retag(self._pinned_split)
        mapping = patient_level_split(self._split_patient_ids(), self._ds_config.split)
        return self._retag(mapping)

    def apply_split(self, mapping: dict[str, list[str]]) -> dict[str, list[str]]:
        """Pin an explicit patient-level split and tag the sample index.

        The pinned split survives subsequent internal ``split()`` calls until
        :meth:`clear_pinned_split` is called. Used by the cross-validation runner to
        drive per-fold train/validation/test assignments.
        """
        self._pinned_split = {name: list(ids) for name, ids in mapping.items()}
        return self._retag(self._pinned_split)

    def clear_pinned_split(self) -> None:
        """Remove any pinned split so future ``split()`` calls draw a fresh one."""
        self._pinned_split = None

    def _retag(self, mapping: dict[str, list[str]]) -> dict[str, list[str]]:
        """Tag the sample index with a split mapping and record it."""
        self._splits = mapping
        lookup = invert_split(mapping)
        self._index = [s.with_split(lookup.get(s.patient_id)) for s in self.build_index()]
        sizes = {name: len(ids) for name, ids in mapping.items()}
        self._log.info("Patient-level split: %s.", sizes)
        return self._splits

    def _split_patient_ids(self) -> list[str]:
        """Patient IDs for splitting: from metadata when present, else from images."""
        ids = self.load_metadata().patient_ids
        if ids:
            return ids
        seen: dict[str, None] = {}
        for sample in self.build_index():
            seen.setdefault(sample.patient_id, None)
        return list(seen)

    def samples(self, split: str | None = None) -> list[Sample]:
        """Return samples, optionally filtered by split name."""
        index = self.build_index()
        if split is None:
            return list(index)
        return [s for s in index if s.split == split]

    def statistics(self) -> DatasetStatistics:
        """Compute dataset statistics."""
        return compute_statistics(self._ds_config, self.load_metadata(), self.build_index())

    def summary(self) -> dict[str, object]:
        """Return a compact summary combining validation, splits, and statistics."""
        report = self.validate()
        splits = self._splits or {}
        return {
            "root": str(self._root),
            "valid": report.is_valid,
            "num_patients": report.num_patients,
            "num_images": report.num_images,
            "num_masks": report.num_masks,
            "split_sizes": {name: len(ids) for name, ids in splits.items()},
            "statistics": self.statistics().to_dict(),
        }

    def export_report(self, report: ValidationReport, filename: str = "validation_report.json") -> Path:
        """Write a validation report into the dataset ``statistics`` directory."""
        out_dir = ensure_dir(self._require_root() / "statistics")
        return write_json(out_dir / filename, report.to_dict())

    # -- internals ---------------------------------------------------------

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self._base_dir / candidate

    def _tissue_sources(
        self, tissue: str, images_root: Path, masks_root: Path
    ) -> list[tuple[Path, Path, str | None]]:
        """Resolve the image/mask directories (and side) for a single tissue.

        When ``dataset.tissue_sources`` defines explicit sources for the tissue —
        either a single unsided source or several sided ones — each is used
        directly, allowing every side to originate from a different dataset. Paths
        may be absolute or relative to ``base_dir``. Otherwise the conventional
        ``root/images_dir/<tissue>`` (and matching masks) layout is used (side
        inferred from filenames).

        Args:
            tissue: The tissue name being resolved.
            images_root: The default ``root/images_dir`` directory.
            masks_root: The default ``root/masks_dir`` directory.

        Returns:
            A list of ``(images_dir, masks_dir, side)`` resolved tuples.
        """
        sources = self._ds_config.tissue_sources.get(tissue, ())
        if not sources:
            return [(images_root / tissue, masks_root / tissue, None)]
        resolved: list[tuple[Path, Path, str | None]] = []
        for source in sources:
            image_dir = self._resolve(source.images)
            # Explicit masks dir if given; otherwise look for masks beside images.
            mask_dir = self._resolve(source.masks) if source.masks else image_dir
            resolved.append((image_dir, mask_dir, source.side))
        return resolved

    def _require_root(self) -> Path:
        if self._root is None:
            raise DatasetError(
                "No dataset root configured. Set dataset.root or pass dataset_root."
            )
        if not self._root.is_dir():
            raise DatasetError(f"Dataset root does not exist: {self._root}")
        return self._root

    @staticmethod
    def _find_mask(mask_dir: Path, stem: str, formats: tuple[str, ...]) -> str | None:
        if not mask_dir.is_dir():
            return None
        for ext in formats:
            candidate = mask_dir / f"{stem}_mask.{ext}"
            if candidate.is_file():
                return str(candidate)
        return None


def _to_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


# Recognized side tokens for inferring a sample's side from its filename stem
# (filenames follow ``<patient>_<tissue>_<side>``; see the synthetic generator).
_SIDE_TOKENS = frozenset({"left", "right", "center", "centre", "l", "r"})


def _side_from_stem(stem: str) -> str | None:
    """Infer a capture side from a ``patient_tissue_side`` filename stem."""
    parts = stem.split("_")
    if len(parts) >= 3:
        token = parts[-1].lower()
        if token in _SIDE_TOKENS:
            return token
    return None


__all__ = ["DatasetManager"]
