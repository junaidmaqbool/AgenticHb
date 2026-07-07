"""Unit tests for StateManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.exceptions import StateError
from adaptivehb.managers import StateManager


@pytest.fixture()
def state(framework_config: FrameworkConfig, tmp_path: Path) -> StateManager:
    manager = StateManager(framework_config, base_dir=tmp_path)
    manager.initialize()
    return manager


def test_state_file_created(state: StateManager) -> None:
    assert state.exists()
    assert state.state.status == "idle"


def test_update_fields(state: StateManager) -> None:
    state.update(current_phase="segmentation", current_epoch=3, status="running")
    assert state.state.current_phase == "segmentation"
    assert state.state.current_epoch == 3


def test_update_unknown_field_raises(state: StateManager) -> None:
    with pytest.raises(StateError):
        state.update(not_a_field=1)


def test_mark_completed_and_query(state: StateManager) -> None:
    state.update(pending_modules=["segmentation", "prediction"])
    state.mark_completed("segmentation")
    assert state.is_completed("segmentation")
    assert "segmentation" not in state.state.pending_modules
    assert "prediction" in state.state.pending_modules


def test_recovery_across_instances(
    framework_config: FrameworkConfig, tmp_path: Path
) -> None:
    first = StateManager(framework_config, base_dir=tmp_path)
    first.initialize()
    first.update(experiment_id="exp_001", current_phase="prediction")
    first.mark_completed("segmentation")

    second = StateManager(framework_config, base_dir=tmp_path)
    second.initialize()
    assert second.state.experiment_id == "exp_001"
    assert second.state.current_phase == "prediction"
    assert second.is_completed("segmentation")


def test_reset(state: StateManager) -> None:
    state.update(experiment_id="exp_001", current_epoch=10)
    state.reset()
    assert state.state.experiment_id is None
    assert state.state.current_epoch == 0
