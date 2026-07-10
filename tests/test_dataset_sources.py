"""Tests for per-side tissue sources, sampling modes, BMI derivation, and
Excel metadata loading (the independent-path dataset wiring)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from adaptivehb.config import ConfigLoader
from adaptivehb.dataset.bmi import add_bmi, compute_bmi
from adaptivehb.dataset.config import (
    SAMPLING_EXTENDED,
    SAMPLING_SINGLE,
    DatasetConfig,
    SideSource,
)
from adaptivehb.dataset.manager import DatasetManager
from adaptivehb.dataset.metadata import MetadataTable
from adaptivehb.dataset.synthetic import _image_png, _mask_png
from adaptivehb.exceptions import ConfigError


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #

def test_flat_tissue_source_parses_to_single_unsided_source() -> None:
    section = {"dataset": {"tissue_sources": {"tongue": {"images": "/data/tongue"}}}}
    cfg = DatasetConfig.from_section(section)
    assert cfg.tissue_sources["tongue"] == (
        SideSource(images="/data/tongue", masks=None, side=None),
    )


def test_sided_tissue_source_parses_each_side() -> None:
    section = {
        "dataset": {
            "tissue_sources": {
                "eye": {
                    "sides": {
                        "left": {"images": "/l", "masks": "/lm"},
                        "right": {"images": "/r"},
                    }
                }
            }
        }
    }
    cfg = DatasetConfig.from_section(section)
    sources = cfg.tissue_sources["eye"]
    assert {s.side for s in sources} == {"left", "right"}
    left = next(s for s in sources if s.side == "left")
    assert left.images == "/l" and left.masks == "/lm"


def test_side_without_images_is_dropped() -> None:
    section = {
        "dataset": {
            "tissue_sources": {
                "eye": {"sides": {"left": {"images": "/l"}, "right": {"images": None}}}
            }
        }
    }
    cfg = DatasetConfig.from_section(section)
    assert {s.side for s in cfg.tissue_sources["eye"]} == {"left"}


def test_sampling_mode_defaults_and_override() -> None:
    section = {
        "dataset": {"sampling_mode": "single", "tissue_sampling_mode": {"eye": "extended"}}
    }
    cfg = DatasetConfig.from_section(section)
    assert cfg.sampling_mode == SAMPLING_SINGLE
    assert cfg.sampling_mode_for("eye") == SAMPLING_EXTENDED
    assert cfg.sampling_mode_for("tongue") == SAMPLING_SINGLE


def test_invalid_sampling_mode_raises() -> None:
    with pytest.raises(ConfigError):
        DatasetConfig.from_section({"dataset": {"sampling_mode": "nonsense"}})


# --------------------------------------------------------------------------- #
# BMI
# --------------------------------------------------------------------------- #

def test_compute_bmi_cm_and_m_agree() -> None:
    assert compute_bmi(60.0, 150.0, height_unit="cm") == pytest.approx(26.6667, abs=1e-3)
    assert compute_bmi(60.0, 1.5, height_unit="m") == pytest.approx(26.6667, abs=1e-3)


def test_compute_bmi_rejects_nonpositive() -> None:
    assert compute_bmi(0.0, 150.0) is None
    assert compute_bmi(60.0, 0.0) is None


def test_add_bmi_fills_only_blank_values() -> None:
    rows = [
        {"height": "150", "weight": "60", "BMI": ""},
        {"height": "160", "weight": "70", "BMI": "99.9"},  # recorded -> untouched
        {"height": "", "weight": "70", "BMI": ""},          # unusable -> skipped
    ]
    filled = add_bmi(rows, height_column="height", weight_column="weight", bmi_column="BMI")
    assert filled == 1
    assert rows[0]["BMI"] == "26.7"
    assert rows[1]["BMI"] == "99.9"
    assert rows[2]["BMI"] == ""


def test_metadata_table_derive_bmi_registers_column() -> None:
    table = MetadataTable(
        [{"pid": "P1", "height": "150", "weight": "60"}], ["pid", "height", "weight"], "pid"
    )
    filled = table.derive_bmi(height_column="height", weight_column="weight", bmi_column="BMI")
    assert filled == 1
    assert "BMI" in table.columns
    assert table.get("P1")["BMI"] == "26.7"


# --------------------------------------------------------------------------- #
# Excel metadata loading
# --------------------------------------------------------------------------- #

def test_excel_metadata_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["pid", "Hgb", "height", "weight"])
    ws.append(["P001", 12.3, 150, 60])
    ws.append(["P002", 9.9, 160, 70])
    path = tmp_path / "labels.xlsx"
    wb.save(path)

    table = MetadataTable.load(path, "pid")
    assert table.patient_ids == ["P001", "P002"]
    assert table.get("P001")["Hgb"] == "12.3"


# --------------------------------------------------------------------------- #
# End-to-end: per-side scan + sampling mode through the DatasetManager
# --------------------------------------------------------------------------- #

def _make_side_dataset(root: Path) -> tuple[Path, Path, Path, Path]:
    """Create left/right eye image dirs, a shared mask dir, and an xlsx labels file."""
    left, right, masks = root / "left", root / "right", root / "masks"
    for d in (left, right, masks):
        d.mkdir(parents=True)
    rng = random.Random(0)
    pids = [f"P{i:03d}" for i in range(1, 6)]
    for pid in pids:
        (left / f"{pid}_eye_left.png").write_bytes(_image_png(rng, 16))
        (right / f"{pid}_eye_right.png").write_bytes(_image_png(rng, 16))
        (masks / f"{pid}_eye_left_mask.png").write_bytes(_mask_png(16))
        (masks / f"{pid}_eye_right_mask.png").write_bytes(_mask_png(16))

    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["pid", "Hgb", "height", "weight"])
    for i, pid in enumerate(pids):
        ws.append([pid, round(9 + i * 0.5, 1), 150 + i, 60 + i])
    labels = root / "labels.xlsx"
    wb.save(labels)
    return left, right, masks, labels


def _config_with(section_overrides: dict) -> ConfigLoader:
    cfg = ConfigLoader("configs").load()
    ds = cfg.section("dataset")["dataset"]
    ds.update(section_overrides)
    return cfg


def _base_metadata() -> dict:
    return {
        "patient_id_column": "pid",
        "target_column": "Hgb",
        "mandatory_columns": ["pid", "Hgb"],
        "height_column": "height",
        "weight_column": "weight",
        "bmi_column": "BMI",
        "compute_bmi": True,
    }


def test_extended_mode_scans_all_sides(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    left, right, masks, labels = _make_side_dataset(tmp_path)
    cfg = _config_with(
        {
            "metadata_file": str(labels),
            "tissues": ["eye"],
            "sampling_mode": "extended",
            "tissue_sources": {
                "eye": {
                    "sides": {
                        "left": {"images": str(left), "masks": str(masks)},
                        "right": {"images": str(right), "masks": str(masks)},
                    }
                }
            },
            "metadata": _base_metadata(),
        }
    )
    manager = DatasetManager(cfg, base_dir=".", dataset_root=str(tmp_path))
    samples = manager.load()
    assert len(samples) == 10  # 5 patients x 2 sides
    assert {s.side for s in samples} == {"left", "right"}
    assert all(s.mask_path for s in samples)  # masks wired from the segmentation dir
    # BMI derived and correlated to each image's patient row.
    assert manager.load_metadata().get("P001")["BMI"] == "26.7"
    assert manager.validate().is_valid


def test_single_mode_keeps_one_image_per_patient(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    left, right, masks, labels = _make_side_dataset(tmp_path)
    cfg = _config_with(
        {
            "metadata_file": str(labels),
            "tissues": ["eye"],
            "sampling_mode": "single",
            "tissue_sources": {
                "eye": {
                    "sides": {
                        "left": {"images": str(left)},
                        "right": {"images": str(right)},
                    }
                }
            },
            "metadata": _base_metadata(),
        }
    )
    manager = DatasetManager(cfg, base_dir=".", dataset_root=str(tmp_path))
    samples = manager.load()
    assert len(samples) == 5
    assert len({s.patient_id for s in samples}) == 5
