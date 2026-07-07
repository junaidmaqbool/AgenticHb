"""Batching and label extraction for the training-data bridge.

This layer turns standardized :class:`~adaptivehb.dataset.schema.Sample` records
into batches with extracted hemoglobin labels and mask references. It is
dependency-free (no torch/opencv), so batching and label logic are fully
testable without the ML stack; image decoding and tensor conversion are handled
separately by the (guarded) decoder and torch DataLoader adapter.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from adaptivehb.dataset.schema import Sample

if TYPE_CHECKING:  # pragma: no cover
    from adaptivehb.dataset.manager import DatasetManager


@dataclass
class Batch:
    """A mini-batch of samples with extracted labels.

    Attributes:
        patient_ids: Patient identifier per item.
        tissues: Tissue class per item.
        image_paths: Image path per item (decoded lazily downstream).
        mask_paths: Mask path per item (``None`` when absent).
        labels: Hemoglobin label per item.
        split: Split name the batch was drawn from, if uniform.
    """

    patient_ids: list[str] = field(default_factory=list)
    tissues: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    mask_paths: list[str | None] = field(default_factory=list)
    labels: list[float] = field(default_factory=list)
    split: str | None = None

    def __len__(self) -> int:
        """Number of items in the batch."""
        return len(self.patient_ids)


def _to_batch(samples: list[Sample], split: str | None) -> Batch:
    return Batch(
        patient_ids=[s.patient_id for s in samples],
        tissues=[s.tissue for s in samples],
        image_paths=[s.image_path for s in samples],
        mask_paths=[s.mask_path for s in samples],
        labels=[float(s.hb) for s in samples if s.hb is not None],
        split=split,
    )


def iter_batches(
    samples: Iterable[Sample],
    batch_size: int,
    *,
    require_label: bool = True,
    drop_last: bool = False,
    split: str | None = None,
) -> Iterator[Batch]:
    """Yield fixed-size batches of samples.

    Args:
        samples: Samples to batch.
        batch_size: Items per batch (must be positive).
        require_label: Skip samples without a hemoglobin label (training needs
            labels; inference may not).
        drop_last: Drop a trailing partial batch.
        split: Optional split name recorded on each batch.

    Yields:
        :class:`Batch` objects.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    items = [s for s in samples if s.hb is not None or not require_label]
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        if drop_last and len(chunk) < batch_size:
            break
        yield _to_batch(chunk, split)


def tissue_batches(
    samples: Iterable[Sample], tissue: str, batch_size: int, **kwargs: object
) -> Iterator[Batch]:
    """Yield batches restricted to a single tissue class."""
    selected = [s for s in samples if s.tissue == tissue]
    return iter_batches(selected, batch_size, **kwargs)  # type: ignore[arg-type]


def batches_for_split(
    manager: DatasetManager, split: str, batch_size: int, **kwargs: object
) -> Iterator[Batch]:
    """Yield batches for a dataset split via the DatasetManager."""
    return iter_batches(manager.samples(split), batch_size, split=split, **kwargs)  # type: ignore[arg-type]


__all__ = ["Batch", "iter_batches", "tissue_batches", "batches_for_split"]
