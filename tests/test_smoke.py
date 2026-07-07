"""Phase 1 smoke tests.

These verify that the repository infrastructure is sound: the package imports,
every configuration file loads and validates, the typed project/logging schemas
are populated, the logging subsystem initializes, and the exception hierarchy is
well-formed. No machine-learning code is exercised.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import adaptivehb
from adaptivehb.config import ConfigLoader, FrameworkConfig, LoggingConfig, ProjectConfig
from adaptivehb.exceptions import (
    AdaptiveHbError,
    ConfigError,
    PipelineError,
    RegistryError,
)
from adaptivehb.logging import get_logger, setup_logging

EXPECTED_EXTRA_SECTIONS = {
    "dataset",
    "segmentation",
    "prediction",
    "agents",
    "evaluation",
    "deployment",
    "registry",
}


def test_package_exposes_version() -> None:
    assert isinstance(adaptivehb.__version__, str)
    assert adaptivehb.__version__.count(".") == 2


def test_all_configs_load(configs_dir: Path) -> None:
    config = ConfigLoader(configs_dir).load()
    assert isinstance(config, FrameworkConfig)
    assert isinstance(config.project, ProjectConfig)
    assert isinstance(config.logging, LoggingConfig)


def test_project_config_typed_values(configs_dir: Path) -> None:
    config = ConfigLoader(configs_dir).load()
    project = config.project
    assert project.name
    assert isinstance(project.seed, int)
    assert project.hardware.device in {"auto", "cpu", "cuda"}
    # Paths must be sourced from config, not hardcoded in code.
    assert project.paths.checkpoints == "checkpoints"


def test_logging_config_typed_values(configs_dir: Path) -> None:
    config = ConfigLoader(configs_dir).load()
    log = config.logging
    assert log.level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    assert log.rotation.backup_count >= 0


def test_extra_sections_present(configs_dir: Path) -> None:
    config = ConfigLoader(configs_dir).load()
    assert EXPECTED_EXTRA_SECTIONS.issubset(config.extras.keys())
    assert "prediction" in config.section("prediction")


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        ConfigLoader(tmp_path / "does_not_exist").load()


def test_missing_file_raises(tmp_path: Path) -> None:
    # Directory exists but contains no configuration files.
    with pytest.raises(ConfigError):
        ConfigLoader(tmp_path).load()


def test_setup_logging_is_idempotent(configs_dir: Path, tmp_path: Path) -> None:
    config = ConfigLoader(configs_dir).load()
    logger_first = setup_logging(config.logging, root_dir=tmp_path)
    handler_count = len(logger_first.handlers)
    logger_second = setup_logging(config.logging, root_dir=tmp_path)
    assert logger_first is logger_second
    # Repeated setup must not accumulate duplicate handlers.
    assert len(logger_second.handlers) == handler_count
    assert handler_count >= 1


def test_get_logger_namespacing() -> None:
    child = get_logger("dataset")
    assert child.name == "adaptivehb.dataset"
    assert get_logger().name == "adaptivehb"


def test_exception_hierarchy() -> None:
    for error in (ConfigError, PipelineError, RegistryError):
        assert issubclass(error, AdaptiveHbError)
    assert issubclass(AdaptiveHbError, Exception)


def test_logging_writes_file(configs_dir: Path, tmp_path: Path) -> None:
    config = ConfigLoader(configs_dir).load()
    logger = setup_logging(config.logging, root_dir=tmp_path)
    logger.info("smoke-test-message")
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
    log_file = tmp_path / config.logging.log_dir / config.logging.filename
    assert log_file.is_file()
