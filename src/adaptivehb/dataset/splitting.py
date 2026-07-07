"""Patient-level dataset splitting (DATASET_SPEC Ch.13).

Splitting always occurs at the patient level: all images belonging to a patient
land in exactly one split, preventing information leakage. Splits are
deterministic for a given seed so experiments are reproducible.
"""

from __future__ import annotations

import random

from adaptivehb.dataset.config import SplitSpec
from adaptivehb.dataset.schema import TEST, TRAIN, VALIDATION
from adaptivehb.exceptions import DatasetError


def patient_level_split(patient_ids: list[str], spec: SplitSpec) -> dict[str, list[str]]:
    """Partition patient IDs into train/validation/test splits.

    Args:
        patient_ids: Unique patient identifiers.
        spec: Split configuration (ratios + seed).

    Returns:
        Mapping of split name to the list of patient IDs assigned to it. Every
        input patient appears in exactly one split.

    Raises:
        DatasetError: If ``patient_ids`` contains duplicates.
    """
    unique = list(dict.fromkeys(patient_ids))
    if len(unique) != len(patient_ids):
        raise DatasetError("Cannot split: duplicate patient IDs supplied.")

    ordered = sorted(unique)
    rng = random.Random(spec.seed)
    rng.shuffle(ordered)

    total = len(ordered)
    n_train = int(total * spec.train)
    n_val = int(total * spec.validation)
    # Remainder goes to test so every patient is assigned.
    train_ids = ordered[:n_train]
    val_ids = ordered[n_train : n_train + n_val]
    test_ids = ordered[n_train + n_val :]

    return {TRAIN: train_ids, VALIDATION: val_ids, TEST: test_ids}


def k_fold_split(
    patient_ids: list[str], k: int, *, seed: int = 0, val_fraction: float = 0.0
) -> list[dict[str, list[str]]]:
    """Partition patients into ``k`` cross-validation folds (DATASET_SPEC Ch.13).

    Splitting is at the patient level, so a patient never appears in more than one
    fold's test set. For each fold the held-out fold is the ``test`` set and the
    remaining patients form ``train`` (optionally carving a ``validation`` slice of
    ``val_fraction`` from the training pool). Folds are balanced (round-robin over a
    seeded shuffle) and deterministic for a given ``seed``.

    Args:
        patient_ids: Unique patient identifiers.
        k: Number of folds (``2 <= k <= len(patient_ids)``).
        seed: Seed for the deterministic shuffle.
        val_fraction: Fraction of each fold's training pool to reserve for
            validation (``0.0`` disables the validation split).

    Returns:
        A list of ``k`` split mappings, each with ``train``/``validation``/``test``
        patient-ID lists. Every patient appears in exactly one fold's test set.

    Raises:
        DatasetError: If ``patient_ids`` has duplicates, ``k < 2``, or ``k`` exceeds
            the number of unique patients.
    """
    unique = list(dict.fromkeys(patient_ids))
    if len(unique) != len(patient_ids):
        raise DatasetError("Cannot fold: duplicate patient IDs supplied.")
    if k < 2:
        raise DatasetError("k-fold cross-validation requires k >= 2.")
    if k > len(unique):
        raise DatasetError(f"k={k} exceeds the number of patients ({len(unique)}).")
    if not 0.0 <= val_fraction < 1.0:
        raise DatasetError("val_fraction must be in [0.0, 1.0).")

    ordered = sorted(unique)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    # Round-robin assignment keeps fold sizes balanced (differ by at most one).
    folds = [ordered[i::k] for i in range(k)]

    splits: list[dict[str, list[str]]] = []
    for i in range(k):
        test_ids = list(folds[i])
        train_pool = [pid for j, fold in enumerate(folds) if j != i for pid in fold]
        if val_fraction > 0.0:
            n_val = int(len(train_pool) * val_fraction)
            val_ids = train_pool[:n_val]
            train_ids = train_pool[n_val:]
        else:
            val_ids, train_ids = [], train_pool
        splits.append({TRAIN: train_ids, VALIDATION: val_ids, TEST: test_ids})
    return splits


def invert_split(splits: dict[str, list[str]]) -> dict[str, str]:
    """Build a patient-ID → split-name lookup from a split mapping."""
    lookup: dict[str, str] = {}
    for split_name, ids in splits.items():
        for pid in ids:
            lookup[pid] = split_name
    return lookup


__all__ = ["patient_level_split", "k_fold_split", "invert_split"]
