"""Shared pytest fixtures for the AdaptiveHb test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def configs_dir() -> Path:
    """Absolute path to the repository configs directory."""
    return REPO_ROOT / "configs"
