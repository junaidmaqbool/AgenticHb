"""Prediction subsystem.

Provides a common hemoglobin-regression model interface, a name-keyed model
factory, concrete backends (torch-free reference regressor + guarded real
EfficientNet/ResNet/DenseNet/ViT/ConvNeXt), and the PredictionManager. Importing
the package registers all available builders.
"""

from adaptivehb.prediction.base import PredictionModel
from adaptivehb.prediction.config import PredictionConfig
from adaptivehb.prediction.manager import PredictionManager
from adaptivehb.prediction.reference import ReferencePredictionModel
from adaptivehb.prediction.registry import (
    available_prediction,
    build_prediction,
    is_registered,
    register_prediction,
)
from adaptivehb.prediction.torch_models import torch_available

__all__ = [
    "PredictionModel",
    "PredictionConfig",
    "PredictionManager",
    "ReferencePredictionModel",
    "build_prediction",
    "available_prediction",
    "is_registered",
    "register_prediction",
    "torch_available",
]
