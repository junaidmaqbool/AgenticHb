"""Low-level configuration utilities: YAML loading and key validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from adaptivehb.exceptions import ConfigError


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed mapping (an empty mapping for an empty file).

    Raises:
        ConfigError: If the file is missing, unparseable, or does not contain
            a mapping at the top level.
    """
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:  # pragma: no cover - exercised via malformed files
        raise ConfigError(f"Failed to parse YAML file {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"Configuration file {path} must contain a mapping at the top level."
        )
    return data


def require_keys(mapping: Mapping[str, Any], keys: Iterable[str], context: str) -> None:
    """Validate that all required keys are present in a mapping.

    Args:
        mapping: Mapping to validate.
        keys: Keys that must be present.
        context: Human-readable context used in the error message.

    Raises:
        ConfigError: If any required key is missing.
    """
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"Missing required key(s) {missing} in {context}.")


__all__ = ["load_yaml", "require_keys"]
