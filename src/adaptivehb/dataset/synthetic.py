"""Synthetic dataset generator for tests, examples, and BUILD-mode checks.

Creates a spec-conformant dataset directory (metadata CSV, per-tissue image and
mask folders) populated with lightweight placeholder image files. This lets the
DatasetManager and pipeline be exercised end-to-end without a real dataset,
mirroring the true schema (paths, tissues, filename convention, columns).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from adaptivehb.core.utils import ensure_dir

# A tiny placeholder written in place of a real image. Existence/scan logic does
# not decode pixels; decoding belongs to the training data pipeline (Phase 5+).
_PLACEHOLDER = b"ADAPTIVEHB_SYNTHETIC_IMAGE"

_DEFAULT_TISSUE_SIDES: dict[str, list[str]] = {
    "eye": ["left", "right"],
    "palm": ["left", "right"],
    "tongue": ["center"],
    "nail": ["left", "right"],
}


def generate_synthetic_dataset(
    root: str | Path,
    *,
    num_patients: int = 12,
    tissues: list[str] | None = None,
    tissue_sides: dict[str, list[str]] | None = None,
    with_masks: bool = True,
    image_ext: str = "png",
    seed: int = 42,
) -> Path:
    """Generate a synthetic, spec-conformant dataset on disk.

    Args:
        root: Directory to create the dataset in.
        num_patients: Number of patients to generate.
        tissues: Tissue classes to include (defaults to eye/palm/tongue/nail).
        tissue_sides: Sides per tissue (defaults to left/right, tongue=center).
        with_masks: Whether to also create matching mask files.
        image_ext: Image file extension.
        seed: RNG seed for reproducible metadata values.

    Returns:
        The dataset root path.
    """
    root_path = ensure_dir(root)
    tissues = tissues or list(_DEFAULT_TISSUE_SIDES.keys())
    sides = tissue_sides or _DEFAULT_TISSUE_SIDES
    rng = random.Random(seed)

    patient_ids = [f"P{index:04d}" for index in range(1, num_patients + 1)]
    _write_metadata(root_path, patient_ids, rng)

    for tissue in tissues:
        images_dir = ensure_dir(root_path / "images" / tissue)
        masks_dir = ensure_dir(root_path / "masks" / tissue) if with_masks else None
        for pid in patient_ids:
            for side in sides.get(tissue, ["center"]):
                stem = f"{pid}_{tissue}_{side}"
                (images_dir / f"{stem}.{image_ext}").write_bytes(_PLACEHOLDER)
                if masks_dir is not None:
                    (masks_dir / f"{stem}_mask.{image_ext}").write_bytes(_PLACEHOLDER)

    return root_path


def _write_metadata(root: Path, patient_ids: list[str], rng: random.Random) -> None:
    """Write a patients.csv with mandatory and a few optional columns."""
    meta_dir = ensure_dir(root / "metadata")
    columns = ["Patient_ID", "Hemoglobin", "Age", "Gender", "Height", "Weight", "BMI"]
    with (meta_dir / "patients.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for pid in patient_ids:
            height = rng.uniform(1.5, 1.9)
            weight = rng.uniform(45, 95)
            bmi = weight / (height * height)
            writer.writerow(
                [
                    pid,
                    round(rng.uniform(8.0, 17.5), 1),
                    rng.randint(1, 90),
                    rng.choice(["M", "F"]),
                    round(height * 100, 1),
                    round(weight, 1),
                    round(bmi, 1),
                ]
            )


__all__ = ["generate_synthetic_dataset"]
