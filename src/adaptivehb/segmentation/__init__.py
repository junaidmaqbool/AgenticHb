"""Segmentation subsystem.

Provides a common segmentation model interface, a name-keyed model factory,
concrete backends (torch-free reference model + guarded real UNet/DeepLabV3+/
SegFormer), and the SegmentationManager. Importing the package registers all
available builders.
"""

from adaptivehb.segmentation.base import SegmentationModel
from adaptivehb.segmentation.config import SegmentationConfig
from adaptivehb.segmentation.manager import SegmentationManager
from adaptivehb.segmentation.metrics import (
    SegmentationMetrics,
    dice_score,
    iou_score,
    pixel_accuracy,
    segmentation_metrics,
)
from adaptivehb.segmentation.reference import ReferenceSegmentationModel
from adaptivehb.segmentation.registry import (
    available_segmentation,
    build_segmentation,
    is_registered,
    register_segmentation,
)
from adaptivehb.segmentation.torch_models import torch_available

__all__ = [
    "SegmentationModel",
    "SegmentationConfig",
    "SegmentationManager",
    "SegmentationMetrics",
    "segmentation_metrics",
    "iou_score",
    "dice_score",
    "pixel_accuracy",
    "ReferenceSegmentationModel",
    "build_segmentation",
    "available_segmentation",
    "is_registered",
    "register_segmentation",
    "torch_available",
]
