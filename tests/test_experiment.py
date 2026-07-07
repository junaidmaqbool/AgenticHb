"""Unit tests for ExperimentManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.managers import ExperimentManager

_SUBDIRS = ("configuration", "logs", "checkpoints", "metrics", "figures", "summary")


@pytest.fixture()
def experiments(framework_config: FrameworkConfig, tmp_path: Path) -> ExperimentManager:
    manager = ExperimentManager(framework_config, base_dir=tmp_path)
    manager.initialize()
    return manager


def test_create_builds_structure(experiments: ExperimentManager) -> None:
    snapshot = {"project": {"name": "AdaptiveHb"}}
    experiment = experiments.create("demo", config_snapshot=snapshot)

    assert experiment.experiment_id.startswith("demo_")
    assert experiment.root.is_dir()
    for subdir in _SUBDIRS:
        assert experiment.path_for(subdir).is_dir()

    meta_path = experiment.path_for("configuration") / "meta.json"
    config_path = experiment.path_for("configuration") / "config.json"
    assert meta_path.is_file()
    assert config_path.is_file()


def test_experiments_are_listed(experiments: ExperimentManager) -> None:
    experiments.create("run_a")
    experiments.create("run_b")
    listed = experiments.list_experiments()
    assert len(listed) == 2
    assert any(name.startswith("run_a_") for name in listed)


def test_save_summary(experiments: ExperimentManager) -> None:
    experiment = experiments.create("demo")
    path = experiments.save_summary(experiment, {"mae": 0.42, "status": "done"})
    assert path.is_file()


def test_create_is_unique(experiments: ExperimentManager) -> None:
    first = experiments.create("demo")
    second = experiments.create("demo")
    # Distinct IDs even for the same name (timestamp-based); no overwrite.
    assert first.experiment_id != second.experiment_id or first.root != second.root
