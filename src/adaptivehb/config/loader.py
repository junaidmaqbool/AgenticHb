"""High-level configuration loading and validation for AdaptiveHb.

``ConfigLoader`` reads the nine framework configuration files from a directory,
validates that each contains its required top-level keys, and returns a
:class:`FrameworkConfig`. The ``project`` and ``logging`` sections are returned
as strongly typed objects; the remaining sections are returned as validated raw
mappings until their consuming managers (and typed schemas) are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adaptivehb.config.base import load_yaml, require_keys
from adaptivehb.config.schemas import LoggingConfig, ProjectConfig
from adaptivehb.exceptions import ConfigError

# Configuration file stem -> required top-level keys.
_EXPECTED_FILES: dict[str, list[str]] = {
    "project": ["project", "hardware", "paths"],
    "logging": ["logging"],
    "dataset": ["dataset"],
    "segmentation": ["segmentation"],
    "prediction": ["prediction"],
    "agents": ["agents"],
    "evaluation": ["evaluation"],
    "deployment": ["deployment"],
    "registry": ["registry"],
}

# Sections that are exposed as typed objects rather than raw mappings.
_TYPED_SECTIONS = frozenset({"project", "logging"})


@dataclass(frozen=True)
class FrameworkConfig:
    """Aggregated, validated framework configuration.

    Attributes:
        project: Typed project configuration.
        logging: Typed logging configuration.
        extras: Validated but not-yet-typed sections keyed by file stem
            (``dataset``, ``segmentation``, ``prediction``, ``agents``,
            ``evaluation``, ``deployment``, ``registry``).
    """

    project: ProjectConfig
    logging: LoggingConfig
    extras: dict[str, dict[str, Any]] = field(default_factory=dict)

    def section(self, name: str) -> dict[str, Any]:
        """Return a raw configuration section by its file stem.

        Args:
            name: Configuration file stem (e.g. ``"dataset"``).

        Returns:
            The raw mapping for that section.

        Raises:
            ConfigError: If the requested section is unknown.
        """
        if name not in self.extras:
            raise ConfigError(f"Unknown configuration section: {name!r}")
        return self.extras[name]


class ConfigLoader:
    """Loads and validates the framework configuration directory."""

    def __init__(self, config_dir: str | Path) -> None:
        """Initialize the loader.

        Args:
            config_dir: Directory containing the nine ``*.yaml`` files.
        """
        self._config_dir = Path(config_dir)

    @property
    def config_dir(self) -> Path:
        """The configuration directory this loader reads from."""
        return self._config_dir

    def load(self) -> FrameworkConfig:
        """Load and validate every framework configuration file.

        Returns:
            A fully validated :class:`FrameworkConfig`.

        Raises:
            ConfigError: If the directory or any file is missing, malformed,
                or missing a required top-level key.
        """
        if not self._config_dir.is_dir():
            raise ConfigError(
                f"Configuration directory not found: {self._config_dir}"
            )
        raw: dict[str, dict[str, Any]] = {}
        for name, required in _EXPECTED_FILES.items():
            data = load_yaml(self._config_dir / f"{name}.yaml")
            require_keys(data, required, f"{name}.yaml")
            raw[name] = data
        project = ProjectConfig.from_dict(raw["project"])
        logging_cfg = LoggingConfig.from_dict(raw["logging"])
        extras = {
            name: data for name, data in raw.items() if name not in _TYPED_SECTIONS
        }
        return FrameworkConfig(project=project, logging=logging_cfg, extras=extras)


__all__ = ["ConfigLoader", "FrameworkConfig"]
