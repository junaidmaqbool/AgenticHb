"""AdaptiveHb — Adaptive Multi-Agent Framework for Non-Invasive Hemoglobin Estimation.

The public surface is intentionally small. Import subpackages explicitly, e.g.::

    from adaptivehb.config import ConfigLoader
    from adaptivehb.logging import setup_logging, get_logger

Heavy ML dependencies are not imported at package import time, so the
infrastructure layer remains usable without a GPU or PyTorch.
"""

from adaptivehb.version import __version__

__all__ = ["__version__"]
