"""Logging subsystem for the AdaptiveHb framework.

Provides :func:`setup_logging` to configure the framework's root logger from a
:class:`~adaptivehb.config.schemas.LoggingConfig`, and :func:`get_logger` to
obtain namespaced child loggers. All framework logging lives under the
``adaptivehb`` logger namespace.

Note:
    Within this module, ``import logging`` refers to the Python standard
    library (imports are absolute in Python 3); ``adaptivehb.logging`` is this
    package's own subpackage and does not shadow it.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from adaptivehb.config.schemas import LoggingConfig

_ROOT_LOGGER_NAME = "adaptivehb"


def setup_logging(config: LoggingConfig, *, root_dir: str | Path = ".") -> logging.Logger:
    """Configure the framework root logger.

    Console and/or rotating-file handlers are attached according to ``config``.
    The function is idempotent: existing handlers on the framework logger are
    cleared before new ones are attached, so repeated calls do not duplicate
    log output.

    Args:
        config: Logging configuration.
        root_dir: Base directory against which ``config.log_dir`` is resolved.

    Returns:
        The configured ``adaptivehb`` root logger.
    """
    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(config.level.upper())
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(fmt=config.format, datefmt=config.datefmt)

    if config.console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.file:
        log_dir = Path(root_dir) / config.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / config.filename
        file_handler: logging.Handler
        if config.rotation.enabled:
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=config.rotation.max_bytes,
                backupCount=config.rotation.backup_count,
                encoding="utf-8",
            )
        else:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.debug("Logging configured (level=%s).", config.level)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced framework logger.

    Args:
        name: Optional child name; the returned logger is
            ``adaptivehb.<name>``. When omitted, the framework root logger is
            returned.

    Returns:
        A logger under the ``adaptivehb`` namespace.
    """
    if name:
        return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
    return logging.getLogger(_ROOT_LOGGER_NAME)


__all__ = ["setup_logging", "get_logger"]
