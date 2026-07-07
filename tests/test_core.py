"""Unit tests for the core layer (types, utils, interfaces)."""

from __future__ import annotations

from pathlib import Path

from adaptivehb.config import FrameworkConfig
from adaptivehb.core import (
    BaseManager,
    ModelCategory,
    ModelRecord,
    ModelStatus,
    ensure_dir,
    read_json,
    set_global_seed,
    write_json,
)


def test_model_record_round_trip() -> None:
    record = ModelRecord(
        name="unet",
        category=ModelCategory.SEGMENTATION,
        task="segmentation",
        architecture="UNet",
        metrics={"dice": 0.91},
        status=ModelStatus.STABLE,
    )
    restored = ModelRecord.from_dict(record.to_dict())
    assert restored == record
    assert restored.category is ModelCategory.SEGMENTATION
    assert restored.status is ModelStatus.STABLE


def test_write_and_read_json_atomic(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "data.json"
    write_json(target, {"a": 1, "b": [1, 2, 3]})
    assert target.is_file()
    assert read_json(target) == {"a": 1, "b": [1, 2, 3]}


def test_ensure_dir_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "x" / "y"
    assert ensure_dir(path).is_dir()
    assert ensure_dir(path).is_dir()  # second call must not raise


def test_set_global_seed_runs_without_ml_stack() -> None:
    # Must not raise even when numpy/torch are absent.
    set_global_seed(123, deterministic=True)


def test_base_manager_lifecycle(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    class _Dummy(BaseManager):
        def __init__(self, config: FrameworkConfig, base_dir: Path) -> None:
            super().__init__(config, base_dir)
            self.init_calls = 0

        def _on_initialize(self) -> None:
            self.init_calls += 1

    manager = _Dummy(framework_config, tmp_path)
    assert manager.name == "_Dummy"
    assert not manager.initialized
    manager.initialize()
    manager.initialize()  # idempotent
    assert manager.initialized
    assert manager.init_calls == 1
    manager.shutdown()
    assert not manager.initialized
