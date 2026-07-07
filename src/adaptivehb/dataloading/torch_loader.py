"""Torch DataLoader adapter for the training-data bridge (guarded).

Imports cleanly without PyTorch: :func:`torch_available` reports availability and
:func:`build_dataloader` raises a clear :class:`DatasetError` when torch is
absent. When torch is present, a ``Dataset`` decodes each sample's image (via the
guarded :class:`ImageDecoder`), applies the transform, and yields
``(image, target)`` tensors for prediction (target = Hb) or segmentation
(target = mask). Real training loops consume this adapter in the experiment phase.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from typing import Any

from adaptivehb.dataloading.decoding import ImageDecoder
from adaptivehb.dataset.schema import Sample
from adaptivehb.exceptions import DatasetError

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def torch_available() -> bool:
    """Whether PyTorch is importable in this environment."""
    return _TORCH_AVAILABLE


if _TORCH_AVAILABLE:  # pragma: no cover - requires torch
    import torch
    from torch.utils.data import Dataset

    class SampleDataset(Dataset):
        """A ``torch`` dataset decoding samples to ``(image, target)`` tensors."""

        def __init__(
            self,
            samples: Sequence[Sample],
            *,
            task: str = "prediction",
            decoder: ImageDecoder | None = None,
            transform: Any = None,
        ) -> None:
            self._samples = list(samples)
            self._task = task
            self._decoder = decoder or ImageDecoder()
            self._transform = transform

        def __len__(self) -> int:
            return len(self._samples)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            sample = self._samples[index]
            image = self._decoder.decode(sample.image_path)
            if self._transform is not None:
                image = self._transform(image=image)["image"]
            image_tensor = torch.as_tensor(image).float()
            if image_tensor.ndim == 3 and image_tensor.shape[-1] in (1, 3):
                image_tensor = image_tensor.permute(2, 0, 1)

            if self._task == "segmentation":
                if sample.mask_path:
                    mask = self._decoder.decode(sample.mask_path)
                    target = torch.as_tensor(mask).float()
                else:
                    target = torch.zeros(1)
                return image_tensor, target

            label = float(sample.hb) if sample.hb is not None else 0.0
            return image_tensor, torch.tensor(label).float()


def build_dataloader(
    samples: Sequence[Sample],
    *,
    batch_size: int,
    task: str = "prediction",
    shuffle: bool = False,
    decoder: ImageDecoder | None = None,
    transform: Any = None,
    num_workers: int = 0,
) -> Any:
    """Build a ``torch`` DataLoader over samples.

    Args:
        samples: Samples to load.
        batch_size: Items per batch.
        task: ``"prediction"`` (target = Hb) or ``"segmentation"`` (target = mask).
        shuffle: Shuffle each epoch (typically true for training).
        decoder: Optional image decoder (defaults to a fresh one).
        transform: Optional transform applied to each decoded image.
        num_workers: DataLoader worker processes.

    Returns:
        A ``torch.utils.data.DataLoader``.

    Raises:
        DatasetError: If PyTorch is not installed.
    """
    if not _TORCH_AVAILABLE:
        raise DatasetError("PyTorch is required to build a DataLoader (install the 'ml' extra).")
    from torch.utils.data import DataLoader  # pragma: no cover - requires torch

    dataset = SampleDataset(samples, task=task, decoder=decoder, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


__all__ = ["build_dataloader", "torch_available"]
