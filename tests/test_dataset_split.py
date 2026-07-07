"""Unit tests for patient-level dataset splitting."""

from __future__ import annotations

import pytest

from adaptivehb.dataset import invert_split, patient_level_split
from adaptivehb.dataset.config import SplitSpec
from adaptivehb.exceptions import DatasetError

_IDS = [f"P{i:03d}" for i in range(100)]


def test_split_is_deterministic() -> None:
    spec = SplitSpec(seed=7)
    assert patient_level_split(_IDS, spec) == patient_level_split(_IDS, spec)


def test_no_patient_leakage() -> None:
    splits = patient_level_split(_IDS, SplitSpec(seed=1))
    all_assigned = splits["train"] + splits["validation"] + splits["test"]
    # Every patient appears exactly once across all splits.
    assert sorted(all_assigned) == sorted(_IDS)
    assert len(set(all_assigned)) == len(_IDS)


def test_ratios_are_respected() -> None:
    splits = patient_level_split(_IDS, SplitSpec(train=0.8, validation=0.1, test=0.1, seed=3))
    assert len(splits["train"]) == 80
    assert len(splits["validation"]) == 10
    assert len(splits["test"]) == 10


def test_different_seeds_change_assignment() -> None:
    a = patient_level_split(_IDS, SplitSpec(seed=1))
    b = patient_level_split(_IDS, SplitSpec(seed=2))
    assert a["train"] != b["train"]


def test_duplicate_ids_raise() -> None:
    with pytest.raises(DatasetError):
        patient_level_split(["P1", "P1", "P2"], SplitSpec())


def test_invert_split_maps_each_patient() -> None:
    splits = patient_level_split(_IDS, SplitSpec(seed=5))
    lookup = invert_split(splits)
    assert len(lookup) == len(_IDS)
    assert lookup[splits["test"][0]] == "test"
