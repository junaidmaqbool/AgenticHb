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

            if self._task == "segmentation":
                mask = self._decoder.decode(sample.mask_path) if sample.mask_path else None
                # Transform image and mask together so a Resize/flip/rotate keeps
                # them spatially aligned (the mask must match the model's output).
                if self._transform is not None:
                    if mask is not None:
                        result = self._transform(image=image, mask=mask)
                        image, mask = result["image"], result["mask"]
                    else:
                        image = self._transform(image=image)["image"]
                return self._to_chw(image), self._mask_target(mask)

            if self._transform is not None:
                image = self._transform(image=image)["image"]
            label = float(sample.hb) if sample.hb is not None else 0.0
            return self._to_chw(image), torch.tensor(label).float()

        @staticmethod
        def _to_chw(image: Any) -> Any:
            """Return a float CHW image tensor from an HWC array (or passthrough)."""
            tensor = torch.as_tensor(image).float()
            if tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1)
            return tensor

        @staticmethod
        def _mask_target(mask: Any) -> Any:
            """Build a single-channel [1, H, W] binary float mask target.

            Collapses a decoded HWC mask to one channel, scales 0-255 to 0-1, and
            binarizes, so the target matches a ``[N, 1, H, W]`` segmentation logit.
            """
            if mask is None:
                return torch.zeros(1)
            tensor = torch.as_tensor(mask).float()
            if tensor.ndim == 3:  # H, W, C -> take a single channel
                tensor = tensor[..., 0]
            if float(tensor.max()) > 1.0:
                tensor = tensor / 255.0
            tensor = (tensor >= 0.5).float()
            return tensor.unsqueeze(0)  # [1, H, W]


def build_dataloader(
    samples: Sequence[Sample],
    *,
    batch_size: int,
    task: str = "prediction",
    shuffle: bool = False,
    decoder: ImageDecoder | None = None,
    transform: Any = None,
    num_workers: int = 0,
    sample_weights: Sequence[float] | None = None,
) -> Any:
    """Build a ``torch`` DataLoader over samples.

    Args:
        samples: Samples to load.
        batch_size: Items per batch.
        task: ``"prediction"`` (target = Hb) or ``"segmentation"`` (target = mask).
        shuffle: Shuffle each epoch (typically true for training). Ignored when
            ``sample_weights`` is given (a weighted sampler already randomizes).
        decoder: Optional image decoder (defaults to a fresh one).
        transform: Optional transform applied to each decoded image.
        num_workers: DataLoader worker processes.
        sample_weights: Optional per-sample draw weights. When provided, a
            :class:`~torch.utils.data.WeightedRandomSampler` is used so that
            under-represented Hb bins are oversampled each epoch (train split
            only). Must have one weight per sample.

    Returns:
        A ``torch.utils.data.DataLoader``.

    Raises:
        DatasetError: If PyTorch is not installed, or ``sample_weights`` length
            does not match ``samples``.
    """
    if not _TORCH_AVAILABLE:
        raise DatasetError("PyTorch is required to build a DataLoader (install the 'ml' extra).")
    from torch.utils.data import (  # pragma: no cover - requires torch
        DataLoader,
        WeightedRandomSampler,
    )

    dataset = SampleDataset(samples, task=task, decoder=decoder, transform=transform)

    sampler = None
    if sample_weights is not None:  # pragma: no cover - requires torch
        weights = list(sample_weights)
        if len(weights) != len(dataset):
            raise DatasetError(
                f"sample_weights length ({len(weights)}) must match samples ({len(dataset)})."
            )
        # Balanced oversampling with replacement: draw one epoch's worth of items
        # according to the weights so every Hb bin contributes roughly equally.
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False  # a sampler and shuffle are mutually exclusive

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
    )


__all__ = ["build_dataloader", "torch_available"]
