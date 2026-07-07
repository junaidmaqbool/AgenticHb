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
from adaptivehb.dataset.config import DatasetConfig
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
    ) -> None:
        """Initialize the dataset manager.

        Args:
            config: Validated framework configuration.
            base_dir: Root directory for resolving relative dataset paths.
            dataset_root: Explicit dataset root; overrides the config value.
        """
        super().__init__(config, base_dir)
        self._ds_config = DatasetConfig.from_section(config.section("dataset"))
        root = dataset_root if dataset_root is not None else self._ds_config.root
        self._root: Path | None = self._resolve(root) if root else None
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
            path = self._require_root() / self._ds_config.metadata_file
            self._metadata = MetadataTable.load(path, self._ds_config.metadata.patient_id_column)
            self._log.info("Loaded metadata: %d patient(s).", len(self._metadata.patient_ids))
        return self._metadata

    def build_index(self) -> list[Sample]:
        """Scan images and masks into standardized samples (cached)."""
        if self._index is not None:
            return self._index
        root = self._require_root()
        metadata = self.load_metadata()
        images_root = root / self._ds_config.images_dir
        masks_root = root / self._ds_config.masks_dir
        formats = self._ds_config.image.formats
        target = self._ds_config.metadata.target_column

        samples: list[Sample] = []
        for tissue in self._ds_config.tissues:
            tissue_dir = images_root / tissue
            if not tissue_dir.is_dir():
                continue
            for image_path in sorted(tissue_dir.iterdir()):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower().lstrip(".") not in formats:
                    continue
                stem = image_path.stem
                patient_id = stem.split("_")[0]
                row = metadata.get(patient_id) or {}
                samples.append(
                    Sample(
                        patient_id=patient_id,
                        tissue=tissue,
                        image_path=str(image_path),
                        mask_path=self._find_mask(masks_root / tissue, stem, formats),
                        hb=_to_float(row.get(target)),
                        metadata=row,
                    )
                )
        self._index = samples
        self._log.info("Built dataset index: %d sample(s).", len(samples))
        return samples

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
        metadata = self.load_metadata()
        mapping = patient_level_split(metadata.patient_ids, self._ds_config.split)
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


__all__ = ["DatasetManager"]
