"""Synthetic dataset generator for tests, examples, and BUILD-mode checks.

Creates a spec-conformant dataset directory (metadata CSV, per-tissue image and
mask folders) populated with small, **decodable** PNG images. This lets the
DatasetManager and the full pipeline — including the real torch training path that
decodes images — be exercised end-to-end without a real dataset, mirroring the true
schema (paths, tissues, filename convention, columns).

Images are encoded with a tiny standard-library PNG writer (no Pillow/OpenCV/numpy
dependency), so generation works in any environment while still producing files
that OpenCV/Pillow can decode.
"""

from __future__ import annotations

import csv
import random
import struct
import zlib
from pathlib import Path

from adaptivehb.core.utils import ensure_dir

# Size (pixels) of the generated square images/masks. Kept small so generation is
# fast; the training transforms resize to the configured input resolution.
_IMAGE_SIZE = 32

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
        image_ext: Image file extension (content is always PNG-encoded).
        seed: RNG seed for reproducible metadata and pixel colors.

    Returns:
        The dataset root path.
    """
    root_path = ensure_dir(root)
    tissues = tissues or list(_DEFAULT_TISSUE_SIDES.keys())
    sides = tissue_sides or _DEFAULT_TISSUE_SIDES
    rng = random.Random(seed)

    patient_ids = [f"P{index:04d}" for index in range(1, num_patients + 1)]
    _write_metadata(root_path, patient_ids, rng)

    mask_bytes = _mask_png(_IMAGE_SIZE)
    for tissue in tissues:
        images_dir = ensure_dir(root_path / "images" / tissue)
        masks_dir = ensure_dir(root_path / "masks" / tissue) if with_masks else None
        for pid in patient_ids:
            for side in sides.get(tissue, ["center"]):
                stem = f"{pid}_{tissue}_{side}"
                (images_dir / f"{stem}.{image_ext}").write_bytes(_image_png(rng, _IMAGE_SIZE))
                if masks_dir is not None:
                    (masks_dir / f"{stem}_mask.{image_ext}").write_bytes(mask_bytes)

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


# --------------------------------------------------------------------------- #
# Minimal standard-library PNG encoding (no third-party dependency)
# --------------------------------------------------------------------------- #

def _png_bytes(width: int, height: int, rows: list[bytes]) -> bytes:
    """Encode 8-bit RGB pixel rows into a valid PNG byte string.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        rows: ``height`` rows, each ``width * 3`` bytes of RGB pixels.
    """
    def _chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit, colour type 2 (RGB)
    raw = bytearray()
    for row in rows:
        raw.append(0)  # filter type 0 (None) for each scanline
        raw.extend(row)
    idat = zlib.compress(bytes(raw), 9)
    return signature + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _image_png(rng: random.Random, size: int) -> bytes:
    """A small solid-colour RGB PNG (a different colour per call)."""
    pixel = bytes((rng.randint(60, 200), rng.randint(40, 160), rng.randint(50, 170)))
    row = pixel * size
    return _png_bytes(size, size, [row] * size)


def _mask_png(size: int) -> bytes:
    """A binary-style mask PNG: a white central square on a black background."""
    lo, hi = size // 4, size - size // 4
    white, black = bytes((255, 255, 255)), bytes((0, 0, 0))
    foreground_row = black * lo + white * (hi - lo) + black * (size - hi)
    background_row = black * size
    rows = [foreground_row if lo <= y < hi else background_row for y in range(size)]
    return _png_bytes(size, size, rows)


__all__ = ["generate_synthetic_dataset"]
