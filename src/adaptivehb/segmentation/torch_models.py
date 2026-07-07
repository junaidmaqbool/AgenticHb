"""Real (PyTorch) segmentation backends, imported behind a guard.

Imports cleanly without PyTorch: in that case no builder is registered and the
factory falls back to the reference model. When torch is available, a
from-scratch ``UNet`` is registered, and DeepLabV3+ / SegFormer builders are
provided via optional libraries (``torchvision`` / ``segmentation_models_
pytorch``) with a clear error if those extras are missing.

Real training consumes data attached via :meth:`TorchSegmentationModel.attach_
data` (built from the dataset by the pipeline). ``train_epoch``/``validate`` run
over the attached dataloaders and raise if none is attached (Decision 026).
"""

from __future__ import annotations

from typing import Any

from adaptivehb.exceptions import SegmentationError
from adaptivehb.segmentation.base import SegmentationModel
from adaptivehb.segmentation.registry import register_segmentation
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

    class _DoubleConv(nn.Module):
        """(conv -> BN -> ReLU) x 2, the standard U-Net block."""

        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.block(x)

    class _UNet(nn.Module):
        """A compact, from-scratch U-Net."""

        def __init__(self, in_channels: int, num_classes: int, base: int = 64) -> None:
            super().__init__()
            self.d1 = _DoubleConv(in_channels, base)
            self.d2 = _DoubleConv(base, base * 2)
            self.d3 = _DoubleConv(base * 2, base * 4)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = _DoubleConv(base * 4, base * 8)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.u3 = _DoubleConv(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.u2 = _DoubleConv(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.u1 = _DoubleConv(base * 2, base)
            self.head = nn.Conv2d(base, num_classes, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            c1 = self.d1(x)
            c2 = self.d2(self.pool(c1))
            c3 = self.d3(self.pool(c2))
            b = self.bottleneck(self.pool(c3))
            x = self.u3(torch.cat([self.up3(b), c3], dim=1))
            x = self.u2(torch.cat([self.up2(x), c2], dim=1))
            x = self.u1(torch.cat([self.up1(x), c1], dim=1))
            return self.head(x)

    class TorchSegmentationModel(SegmentationModel):
        """Adapter wrapping an ``nn.Module`` as a :class:`SegmentationModel`."""

        def __init__(
            self,
            name: str,
            module_factory: Any,
            num_classes: int = 1,
            in_channels: int = 3,
            config: Any = None,
        ) -> None:
            super().__init__(name, num_classes=num_classes, in_channels=in_channels, config=config)
            self._module_factory = module_factory
            self._module: Any = None
            self._train_loader: Any = None
            self._val_loader: Any = None
            self._optimizer: Any = None
            self._scheduler: Any = None
            self._loss_fn: Any = None
            self._device = ops.resolve_device()

        def build(self) -> None:
            self._module = self._module_factory(self.in_channels, self.num_classes)
            self._built = True

        def attach_data(self, train_loader: Any, val_loader: Any) -> None:
            """Attach dataloaders used by ``train_epoch``/``validate``."""
            self._train_loader, self._val_loader = train_loader, val_loader

        def _ensure_setup(self) -> None:
            if self._optimizer is not None:
                return
            self._module.to(self._device)
            cfg = self._config
            lr = float(getattr(cfg, "learning_rate", 1e-4))
            wd = float(getattr(cfg, "weight_decay", 0.0))
            optimizer_name = str(getattr(cfg, "optimizer", "adam"))
            scheduler_name = str(getattr(cfg, "scheduler", "cosine"))
            loss_name = str(getattr(cfg, "loss", "dice_bce"))
            epochs = int(getattr(cfg, "epochs", 1))
            self._optimizer = ops.build_optimizer(optimizer_name, self._module.parameters(), lr, wd)
            self._scheduler = ops.build_scheduler(scheduler_name, self._optimizer, epochs)
            self._loss_fn = ops.build_segmentation_loss(loss_name)

        def train_epoch(self, epoch: int) -> dict[str, float]:
            self._require_built()
            if self._train_loader is None:
                raise SegmentationError("No training data attached; call attach_data() first.")
            self._ensure_setup()
            metrics = ops.run_segmentation_epoch(
                self._module, self._train_loader, self._loss_fn, self._optimizer, device=self._device
            )
            if self._scheduler is not None:
                self._scheduler.step()
            return {"train_loss": metrics["loss"], "train_dice": metrics["dice"]}

        def validate(self, epoch: int) -> dict[str, float]:
            self._require_built()
            if self._val_loader is None:
                raise SegmentationError("No validation data attached; call attach_data() first.")
            self._ensure_setup()
            metrics = ops.run_segmentation_epoch(
                self._module, self._val_loader, self._loss_fn, None, device=self._device
            )
            return {"val_loss": metrics["loss"], "val_dice": metrics["dice"]}

        def predict(self, image: Any) -> Any:
            self._require_built()
            self._module.eval()
            with torch.no_grad():
                output = self._module(image.to(self._device))
                return next(iter(output.values())) if isinstance(output, dict) else output

        def state_dict(self) -> dict[str, Any]:
            self._require_built()
            return {"module": self._module.state_dict()}

        def load_state_dict(self, state: dict[str, Any]) -> None:
            if self._module is None:
                self.build()
            self._module.load_state_dict(state["module"])

    @register_segmentation("unet")
    def _build_unet(name: str = "unet", **kwargs: Any) -> TorchSegmentationModel:
        return TorchSegmentationModel(name, _make_unet_module, **kwargs)

    def _make_unet_module(in_channels: int, num_classes: int) -> "nn.Module":
        return _UNet(in_channels, num_classes)

    @register_segmentation("deeplabv3plus")
    def _build_deeplab(name: str = "deeplabv3plus", **kwargs: Any) -> TorchSegmentationModel:
        return TorchSegmentationModel(name, _make_deeplab_module, **kwargs)

    def _make_deeplab_module(in_channels: int, num_classes: int) -> "nn.Module":
        try:
            from torchvision.models.segmentation import deeplabv3_resnet50
        except ImportError as exc:  # pragma: no cover
            raise SegmentationError("DeepLabV3+ requires torchvision (the 'ml' extra).") from exc
        return deeplabv3_resnet50(weights=None, num_classes=num_classes)

    @register_segmentation("segformer")
    def _build_segformer(name: str = "segformer", **kwargs: Any) -> TorchSegmentationModel:
        return TorchSegmentationModel(name, _make_segformer_module, **kwargs)

    def _make_segformer_module(in_channels: int, num_classes: int) -> "nn.Module":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:  # pragma: no cover
            raise SegmentationError(
                "SegFormer requires 'segmentation_models_pytorch'; install it for the experiment phase."
            ) from exc
        return smp.Segformer(in_channels=in_channels, classes=num_classes)


__all__ = ["torch_available"]
