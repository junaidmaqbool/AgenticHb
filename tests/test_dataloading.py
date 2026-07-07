"""Unit tests for the training-data bridge (batching, decoding, transforms)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.dataloading import (
    Batch,
    ImageDecoder,
    TransformSpec,
    batches_for_split,
    build_dataloader,
    build_transform,
    decode_available,
    iter_batches,
    tissue_batches,
    torch_available,
    transform_available,
)
from adaptivehb.dataset import DatasetManager, Sample, generate_synthetic_dataset
from adaptivehb.exceptions import DatasetError


def _samples() -> list[Sample]:
    return [
        Sample("P1", "eye", "/tmp/p1_eye.png", hb=12.0),
        Sample("P2", "eye", "/tmp/p2_eye.png", hb=13.0),
        Sample("P3", "palm", "/tmp/p3_palm.png", hb=None),  # unlabelled
        Sample("P4", "palm", "/tmp/p4_palm.png", hb=11.0),
    ]


# --------------------------------------------------------------------------- #
# Batching + labels (dependency-free)
# --------------------------------------------------------------------------- #

def test_iter_batches_sizes_and_labels() -> None:
    batches = list(iter_batches(_samples(), batch_size=2))
    # The unlabelled P3 is dropped, leaving 3 labelled samples -> [2, 1].
    assert [len(b) for b in batches] == [2, 1]
    assert batches[0].labels == [12.0, 13.0]
    assert isinstance(batches[0], Batch)


def test_iter_batches_can_keep_unlabelled() -> None:
    batches = list(iter_batches(_samples(), batch_size=10, require_label=False))
    assert len(batches[0]) == 4  # all samples retained for inference


def test_iter_batches_drop_last() -> None:
    batches = list(iter_batches(_samples(), batch_size=2, drop_last=True))
    assert [len(b) for b in batches] == [2]  # trailing partial batch dropped


def test_iter_batches_rejects_bad_size() -> None:
    with pytest.raises(ValueError):
        list(iter_batches(_samples(), batch_size=0))


def test_tissue_batches_filter() -> None:
    batches = list(tissue_batches(_samples(), "eye", 5))
    assert batches[0].tissues == ["eye", "eye"]
    assert batches[0].labels == [12.0, 13.0]


def test_batches_for_split(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    root = tmp_path / "ds"
    generate_synthetic_dataset(root, num_patients=8, seed=1)
    manager = DatasetManager(framework_config, base_dir=tmp_path, dataset_root=root)
    manager.initialize()
    manager.split()
    batches = list(batches_for_split(manager, "train", batch_size=4))
    assert batches
    assert all(b.split == "train" for b in batches)


# --------------------------------------------------------------------------- #
# Decoding (real backend when installed)
# --------------------------------------------------------------------------- #

def test_decode_missing_file_raises() -> None:
    with pytest.raises(DatasetError):
        ImageDecoder().decode("/does/not/exist.png")


def test_decode_available_is_bool() -> None:
    assert isinstance(decode_available(), bool)


def test_decode_real_image_when_backend_present(tmp_path: Path) -> None:
    if not decode_available():
        pytest.skip("no image-decoding backend installed")
    image_module = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "img.png"
    image_module.new("RGB", (8, 6), (120, 30, 200)).save(image_path)
    array = ImageDecoder().decode(image_path)
    assert array.shape == (6, 8, 3)


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

def test_transform_spec_parses(framework_config: FrameworkConfig) -> None:
    spec = TransformSpec.from_section(framework_config.section("dataset"))
    assert spec.resolution == (224, 224)
    assert spec.normalize is True
    assert len(spec.mean) == 3
    assert spec.augment is True


def test_build_transform_availability() -> None:
    spec = TransformSpec()
    transform = build_transform(spec, training=True)
    if transform_available():
        assert transform is not None  # Albumentations pipeline
    else:
        assert transform is None  # gracefully unavailable


# --------------------------------------------------------------------------- #
# Torch DataLoader adapter (guarded)
# --------------------------------------------------------------------------- #

def test_dataloader_requires_torch() -> None:
    if torch_available():
        loader = build_dataloader(_samples(), batch_size=2)
        assert loader is not None
    else:
        with pytest.raises(DatasetError):
            build_dataloader(_samples(), batch_size=2)
