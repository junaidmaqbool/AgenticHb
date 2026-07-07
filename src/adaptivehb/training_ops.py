"""Shared training operations for the real (torch) model loops.

Separates the parts that are testable without PyTorch (metric accumulation,
optimizer/loss name validation) from the torch-dependent builders and epoch
loops (guarded). Segmentation and prediction backbones share these loops so the
training logic lives in one place (Decision 026).
"""

from __future__ import annotations

from typing import Any

from adaptivehb.exceptions import ModelError

try:  # pragma: no cover - exercised only where torch is installed
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - default in the framework-only env
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

_OPTIMIZERS = frozenset({"adam", "adamw", "sgd"})
_SCHEDULERS = frozenset({"cosine", "step", "none"})
_REGRESSION_LOSSES = frozenset({"mse", "mae", "smoothl1", "huber"})
_SEGMENTATION_LOSSES = frozenset({"dice_bce", "bce", "dice"})


# -- torch-free helpers ------------------------------------------------------ #


class LossAccumulator:
    """Accumulates a sample-weighted running average of a scalar metric."""

    def __init__(self) -> None:
        self._sum = 0.0
        self._count = 0

    def add(self, value: float, count: int = 1) -> None:
        """Add ``value`` observed over ``count`` samples."""
        self._sum += float(value) * count
        self._count += count

    @property
    def average(self) -> float:
        """The sample-weighted mean (0.0 when empty)."""
        return self._sum / self._count if self._count else 0.0

    @property
    def count(self) -> int:
        """Total number of samples accumulated."""
        return self._count


def supported_optimizers() -> set[str]:
    """Return the supported optimizer names."""
    return set(_OPTIMIZERS)


def supported_losses(task: str) -> set[str]:
    """Return the supported loss names for ``task`` (prediction/segmentation)."""
    return set(_REGRESSION_LOSSES if task == "prediction" else _SEGMENTATION_LOSSES)


def validate_optimizer(name: str) -> str:
    """Validate an optimizer name, returning its canonical (lower) form."""
    key = str(name).lower()
    if key not in _OPTIMIZERS:
        raise ModelError(f"Unsupported optimizer {name!r}; choose from {sorted(_OPTIMIZERS)}.")
    return key


def validate_loss(name: str, task: str) -> str:
    """Validate a loss name for a task, returning its canonical (lower) form."""
    key = str(name).lower()
    if key not in supported_losses(task):
        raise ModelError(f"Unsupported {task} loss {name!r}; choose from {sorted(supported_losses(task))}.")
    return key


def torch_available() -> bool:
    """Whether PyTorch is importable in this environment."""
    return _TORCH_AVAILABLE


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ModelError("PyTorch is required for real training (install the 'ml' extra).")


def resolve_device() -> str:
    """Return ``'cuda'`` when available, else ``'cpu'`` (``'cpu'`` without torch)."""
    if not _TORCH_AVAILABLE:  # pragma: no cover - trivial
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"  # pragma: no cover - GPU-dependent


# -- guarded builders -------------------------------------------------------- #


def build_optimizer(name: str, params: Any, learning_rate: float, weight_decay: float = 0.0) -> Any:
    """Construct an optimizer (validates the name first; requires torch)."""
    key = validate_optimizer(name)
    _require_torch()
    if key == "adam":  # pragma: no cover - requires torch
        return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    if key == "adamw":  # pragma: no cover - requires torch
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)  # pragma: no cover


def build_scheduler(name: str, optimizer: Any, epochs: int) -> Any:
    """Construct an LR scheduler (or ``None`` for ``'none'``; requires torch)."""
    _require_torch()
    key = str(name or "none").lower()
    if key == "cosine":  # pragma: no cover - requires torch
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(epochs), 1))
    if key == "step":  # pragma: no cover - requires torch
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(int(epochs) // 3, 1), gamma=0.1)
    return None


def build_regression_loss(name: str) -> Any:
    """Construct a regression loss module (requires torch)."""
    key = validate_loss(name, "prediction")
    _require_torch()
    if key == "mse":  # pragma: no cover - requires torch
        return nn.MSELoss()
    if key == "mae":  # pragma: no cover - requires torch
        return nn.L1Loss()
    return nn.SmoothL1Loss()  # pragma: no cover - smoothl1/huber


def build_segmentation_loss(name: str) -> Any:
    """Construct a segmentation loss module (requires torch)."""
    key = validate_loss(name, "segmentation")
    _require_torch()
    if key == "bce":  # pragma: no cover - requires torch
        return nn.BCEWithLogitsLoss()
    if key == "dice":  # pragma: no cover - requires torch
        return _DiceLoss()
    return _DiceBCELoss()  # pragma: no cover - default dice_bce


if _TORCH_AVAILABLE:  # pragma: no cover - requires torch

    class _DiceLoss(nn.Module):
        """Soft Dice loss for binary/multi-class segmentation logits."""

        def forward(self, logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
            probs = torch.sigmoid(logits)
            targets = targets.float()
            dims = tuple(range(1, probs.ndim))
            intersection = (probs * targets).sum(dims)
            union = probs.sum(dims) + targets.sum(dims)
            dice = (2 * intersection + 1.0) / (union + 1.0)
            return 1.0 - dice.mean()

    class _DiceBCELoss(nn.Module):
        """Combined BCE + Dice loss."""

        def __init__(self) -> None:
            super().__init__()
            self._bce = nn.BCEWithLogitsLoss()
            self._dice = _DiceLoss()

        def forward(self, logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
            return self._bce(logits, targets.float()) + self._dice(logits, targets)

    def _unwrap(output: Any) -> "torch.Tensor":
        return next(iter(output.values())) if isinstance(output, dict) else output

    def run_regression_epoch(
        module: Any, loader: Any, loss_fn: Any, optimizer: Any = None, *, device: str = "cpu"
    ) -> dict[str, float]:
        """Run one regression epoch (train if ``optimizer`` given, else validate)."""
        training = optimizer is not None
        module.train() if training else module.eval()
        loss_acc, mae_acc = LossAccumulator(), LossAccumulator()
        grad_ctx = torch.enable_grad() if training else torch.no_grad()
        with grad_ctx:
            for images, targets in loader:
                images = images.to(device)
                targets = targets.to(device).float().flatten()
                preds = _unwrap(module(images)).flatten()
                loss = loss_fn(preds, targets)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                n = int(targets.numel())
                loss_acc.add(float(loss.item()), n)
                mae_acc.add(float((preds - targets).abs().mean().item()), n)
        return {"loss": loss_acc.average, "mae": mae_acc.average}

    def run_segmentation_epoch(
        module: Any, loader: Any, loss_fn: Any, optimizer: Any = None, *, device: str = "cpu"
    ) -> dict[str, float]:
        """Run one segmentation epoch (train if ``optimizer`` given, else validate)."""
        training = optimizer is not None
        module.train() if training else module.eval()
        loss_acc, dice_acc = LossAccumulator(), LossAccumulator()
        grad_ctx = torch.enable_grad() if training else torch.no_grad()
        with grad_ctx:
            for images, masks in loader:
                images = images.to(device)
                masks = masks.to(device).float()
                logits = _unwrap(module(images))
                loss = loss_fn(logits, masks)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                n = int(images.shape[0])
                loss_acc.add(float(loss.item()), n)
                probs = (torch.sigmoid(logits) > 0.5).float()
                inter = (probs * masks).sum()
                union = probs.sum() + masks.sum()
                dice_acc.add(float(((2 * inter + 1.0) / (union + 1.0)).item()), n)
        return {"loss": loss_acc.average, "dice": dice_acc.average}


__all__ = [
    "LossAccumulator",
    "supported_optimizers",
    "supported_losses",
    "validate_optimizer",
    "validate_loss",
    "torch_available",
    "resolve_device",
    "build_optimizer",
    "build_scheduler",
    "build_regression_loss",
    "build_segmentation_loss",
]
