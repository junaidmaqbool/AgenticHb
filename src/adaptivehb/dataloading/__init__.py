"""Training-data bridge: Sample -> batches / tensors (Stage B).

A dependency-free batching and label-extraction layer, plus guarded image
decoding, preprocessing/augmentation, and a torch DataLoader adapter. The bridge
turns standardized samples into the batches and tensors that the real
segmentation/prediction training loops consume during the experiment phase. All
ML/vision dependencies are optional (Decision 025).
"""

from adaptivehb.dataloading.batch import (
    Batch,
    batches_for_split,
    iter_batches,
    tissue_batches,
)
from adaptivehb.dataloading.decoding import ImageDecoder, decode_available
from adaptivehb.dataloading.torch_loader import build_dataloader, torch_available
from adaptivehb.dataloading.transforms import (
    TransformSpec,
    build_transform,
    transform_available,
)

__all__ = [
    "Batch",
    "iter_batches",
    "tissue_batches",
    "batches_for_split",
    "ImageDecoder",
    "decode_available",
    "TransformSpec",
    "build_transform",
    "transform_available",
    "build_dataloader",
    "torch_available",
]
