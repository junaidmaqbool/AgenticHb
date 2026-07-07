"""Image decoding for the training-data bridge (guarded optional backends).

Decodes an image path to an array using OpenCV or Pillow if available. The
module imports cleanly without any vision library; :meth:`ImageDecoder.decode`
raises a clear :class:`DatasetError` when no backend is installed, so the
framework stays importable and testable without the ML/vision stack (Decision
025). Real decoding activates automatically once opencv or Pillow is present.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from adaptivehb.exceptions import DatasetError


def _detect_backend() -> str | None:
    """Return the available image backend name, or ``None``."""
    if importlib.util.find_spec("cv2") is not None:
        return "opencv"
    if importlib.util.find_spec("PIL") is not None:
        return "pillow"
    return None


def decode_available() -> bool:
    """Whether an image-decoding backend is installed."""
    return _detect_backend() is not None


class ImageDecoder:
    """Decodes image files to RGB arrays via OpenCV or Pillow."""

    def __init__(self, channels: int = 3) -> None:
        """Initialize the decoder.

        Args:
            channels: Expected number of image channels (3 = RGB).
        """
        self.channels = channels
        self._backend = _detect_backend()

    @property
    def backend(self) -> str | None:
        """The active backend name (``opencv``/``pillow``) or ``None``."""
        return self._backend

    @property
    def available(self) -> bool:
        """Whether decoding is possible in this environment."""
        return self._backend is not None

    def decode(self, path: str | Path) -> Any:
        """Decode an image file into an RGB array.

        Args:
            path: Path to the image file.

        Returns:
            An ``H x W x C`` array (numpy) in RGB order.

        Raises:
            DatasetError: If no backend is installed or the file is missing.
        """
        image_path = Path(path)
        if not image_path.is_file():
            raise DatasetError(f"Image file not found: {image_path}")
        if self._backend == "opencv":  # pragma: no cover - requires opencv
            import cv2

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise DatasetError(f"Failed to decode image: {image_path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self._backend == "pillow":  # pragma: no cover - requires Pillow
            import numpy as np
            from PIL import Image

            with Image.open(image_path) as handle:
                return np.asarray(handle.convert("RGB"))
        raise DatasetError(
            "No image-decoding backend installed (opencv or Pillow required for real training)."
        )


__all__ = ["ImageDecoder", "decode_available"]
