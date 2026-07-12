"""Real (PyTorch) prediction backbones, imported behind a guard.

Imports cleanly without PyTorch: when torch is absent no builder registers and
the factory falls back to the reference regressor. When torch (and torchvision)
are available, standard classification backbones are adapted to single-output
regression heads and trained over an attached DataLoader (Decision 026).

Real training consumes data attached via :meth:`TorchPredictionModel.attach_data`
(built from the dataset by the pipeline). ``train_epoch``/``validate`` raise a
clear error if no data is attached, so models remain safe to construct and
inspect.
"""

from __future__ import annotations

from typing import Any

from adaptivehb.exceptions import ModelError
from adaptivehb.prediction.base import PredictionModel
from adaptivehb.prediction.registry import register_prediction
from adaptivehb import training_ops as ops

try:  # pragma: no cover - exercised only where torch is installed
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - default in the framework-only env
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def torch_available() -> bool:
    """Return whether PyTorch is importable in this environment."""
    return _TORCH_AVAILABLE


if _TORCH_AVAILABLE:  # pragma: no cover - requires torch

    class TorchPredictionModel(PredictionModel):
        """A torchvision backbone with a regression head, trained over a loader."""

        #: Learned backbones require a real decoded image tensor at inference.
        consumes_images = True

        def __init__(
            self,
            name: str,
            module_factory: Any,
            tissue: str | None = None,
            in_channels: int = 3,
            config: Any = None,
        ) -> None:
            super().__init__(name, tissue=tissue, in_channels=in_channels, config=config)
            self._module_factory = module_factory
            self._module: Any = None
            self._train_loader: Any = None
            self._val_loader: Any = None
            self._optimizer: Any = None
            self._scheduler: Any = None
            self._loss_fn: Any = None
            self._device = ops.resolve_device()

        def build(self) -> None:
            self._module = self._module_factory()
            self._built = True

        def attach_data(self, train_loader: Any, val_loader: Any) -> None:
            """Attach dataloaders used by ``train_epoch``/``validate``."""
            self._train_loader, self._val_loader = train_loader, val_loader

        def _ensure_setup(self) -> None:
            if self._optimizer is not None:
                return
            self._module.to(self._device)
            cfg = self._config
            lr = float(getattr(cfg, "learning_rate", 3e-4))
            wd = float(getattr(cfg, "weight_decay", 0.0))
            optimizer_name = str(getattr(cfg, "optimizer", "adamw"))
            scheduler_name = str(getattr(cfg, "scheduler", "cosine"))
            loss_name = str(getattr(cfg, "loss", "mse"))
            epochs = int(getattr(cfg, "epochs", 1))
            self._optimizer = ops.build_optimizer(optimizer_name, self._module.parameters(), lr, wd)
            self._scheduler = ops.build_scheduler(scheduler_name, self._optimizer, epochs)
            self._loss_fn = ops.build_regression_loss(loss_name)

        def train_epoch(self, epoch: int) -> dict[str, float]:
            self._require_built()
            if self._train_loader is None:
                raise ModelError("No training data attached; call attach_data() first.")
            self._ensure_setup()
            metrics = ops.run_regression_epoch(
                self._module, self._train_loader, self._loss_fn, self._optimizer, device=self._device
            )
            if self._scheduler is not None:
                self._scheduler.step()
            return {"train_loss": metrics["loss"], "train_mae": metrics["mae"]}

        def validate(self, epoch: int) -> dict[str, float]:
            self._require_built()
            if self._val_loader is None:
                raise ModelError("No validation data attached; call attach_data() first.")
            self._ensure_setup()
            metrics = ops.run_regression_epoch(
                self._module, self._val_loader, self._loss_fn, None, device=self._device
            )
            return {"val_loss": metrics["loss"], "val_mae": metrics["mae"]}

        def predict(self, image: Any, metadata: dict[str, Any] | None = None) -> float:
            """Return the Hb estimate (g/dL) for a single image.

            Accepts a CHW/HWC tensor or array (a batch dimension is added when
            missing). Raises a clear :class:`~adaptivehb.exceptions.ModelError`
            when no image is supplied, since a learned backbone cannot predict
            without one (unlike the torch-free reference model).
            """
            return self.predict_batch(self._as_batch(image))[0]

        def predict_batch(self, images: Any) -> list[float]:
            """Return per-sample Hb estimates for a batch of images.

            Args:
                images: A ``[N, C, H, W]`` tensor (or an array/tensor coercible to
                    one; a leading batch dimension is added when absent).

            Returns:
                One float estimate per sample, in input order.
            """
            self._require_built()
            batch = self._as_batch(images)
            self._module.eval()
            with torch.no_grad():
                output = self._module(batch.to(self._device))
                if isinstance(output, dict):
                    output = next(iter(output.values()))
                return [float(value) for value in output.reshape(batch.shape[0], -1)[:, 0]]

        def _as_batch(self, image: Any) -> "torch.Tensor":
            """Coerce an image (tensor/array) into a ``[N, C, H, W]`` batch tensor."""
            if image is None:
                raise ModelError(
                    f"Prediction model {self.name!r} requires a decoded image; got None. "
                    "Feed real images (e.g. via PredictionManager.predict_samples) "
                    "instead of relying on the torch-free reference behaviour."
                )
            tensor = image if isinstance(image, torch.Tensor) else torch.as_tensor(image)
            tensor = tensor.float()
            if tensor.ndim == 3:  # C,H,W -> add batch dimension
                tensor = tensor.unsqueeze(0)
            return tensor

        def state_dict(self) -> dict[str, Any]:
            self._require_built()
            return {"module": self._module.state_dict()}

        def load_state_dict(self, state: dict[str, Any]) -> None:
            if self._module is None:
                self.build()
            self._module.load_state_dict(state["module"])

    def _regression_head(in_features: int) -> "nn.Module":
        return nn.Linear(in_features, 1)

    def _efficientnet() -> "nn.Module":
        from torchvision.models import efficientnet_b0

        model = efficientnet_b0(weights=None)
        model.classifier[1] = _regression_head(model.classifier[1].in_features)
        return model

    def _resnet() -> "nn.Module":
        from torchvision.models import resnet50

        model = resnet50(weights=None)
        model.fc = _regression_head(model.fc.in_features)
        return model

    def _densenet() -> "nn.Module":
        from torchvision.models import densenet121

        model = densenet121(weights=None)
        model.classifier = _regression_head(model.classifier.in_features)
        return model

    def _vit() -> "nn.Module":
        from torchvision.models import vit_b_16

        model = vit_b_16(weights=None)
        model.heads.head = _regression_head(model.heads.head.in_features)
        return model

    def _convnext() -> "nn.Module":
        from torchvision.models import convnext_tiny

        model = convnext_tiny(weights=None)
        model.classifier[2] = _regression_head(model.classifier[2].in_features)
        return model

    _FACTORIES = {
        "efficientnet": _efficientnet,
        "resnet": _resnet,
        "densenet": _densenet,
        "vit": _vit,
        "convnext": _convnext,
    }

    def _register_all() -> None:
        for arch, factory in _FACTORIES.items():
            def _make_builder(fac: Any) -> Any:
                def _builder(name: str, **kwargs: Any) -> TorchPredictionModel:
                    return TorchPredictionModel(name, fac, **kwargs)

                return _builder

            register_prediction(arch)(_make_builder(factory))

    _register_all()


__all__ = ["torch_available"]
