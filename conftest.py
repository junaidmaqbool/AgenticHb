"""Root-level pytest fixtures for the AdaptiveHb test suite.

pytest loads the rootdir ``conftest.py`` in addition to ``tests/conftest.py``;
shared, package-dependent fixtures live here so they are available to every test
module regardless of collection path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.config import ConfigLoader, FrameworkConfig

_REPO_ROOT = Path(__file__).resolve().parent


@pytest.fixture()
def framework_config() -> FrameworkConfig:
    """A fully loaded, validated framework configuration."""
    return ConfigLoader(_REPO_ROOT / "configs").load()
