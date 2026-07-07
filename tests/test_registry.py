"""Unit tests for RegistryManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.core import ModelCategory, ModelRecord, ModelStatus
from adaptivehb.exceptions import RegistryError
from adaptivehb.managers import RegistryManager


def _record(name: str, metrics: dict[str, float], status: ModelStatus) -> ModelRecord:
    return ModelRecord(
        name=name,
        category=ModelCategory.PREDICTION,
        task="hb_estimation",
        architecture=name,
        metrics=metrics,
        status=status,
    )


@pytest.fixture()
def registry(framework_config: FrameworkConfig, tmp_path: Path) -> RegistryManager:
    manager = RegistryManager(framework_config, base_dir=tmp_path)
    manager.initialize()
    return manager


def test_register_assigns_version_and_unique_id(registry: RegistryManager) -> None:
    record = registry.register(_record("vit", {"val_mae": 0.6}, ModelStatus.STABLE))
    assert record.version == 1
    assert record.unique_id == "HB_VIT_V001"


def test_versions_increment_without_overwrite(registry: RegistryManager) -> None:
    registry.register(_record("vit", {"val_mae": 0.6}, ModelStatus.STABLE))
    registry.register(_record("vit", {"val_mae": 0.5}, ModelStatus.STABLE))
    history = registry.history(ModelCategory.PREDICTION, "vit")
    assert [r.version for r in history] == [1, 2]
    assert history[-1].unique_id == "HB_VIT_V002"


def test_load_latest_and_best(registry: RegistryManager) -> None:
    registry.register(_record("vit", {"val_mae": 0.6}, ModelStatus.STABLE))
    registry.register(_record("vit", {"val_mae": 0.4}, ModelStatus.STABLE))
    registry.register(_record("vit", {"val_mae": 0.5}, ModelStatus.EXPERIMENTAL))

    assert registry.load_latest(ModelCategory.PREDICTION, "vit").version == 3
    best = registry.load_best(ModelCategory.PREDICTION, "vit", "val_mae", direction="min")
    assert best.version == 2  # 0.4 is lowest among deployable (stable) models


def test_load_best_respects_deployable_filter(registry: RegistryManager) -> None:
    registry.register(_record("vit", {"val_mae": 0.9}, ModelStatus.STABLE))
    registry.register(_record("vit", {"val_mae": 0.1}, ModelStatus.EXPERIMENTAL))
    # Experimental 0.1 is better but excluded by default.
    best = registry.load_best(ModelCategory.PREDICTION, "vit", "val_mae")
    assert best.status is ModelStatus.STABLE
    # When experimental is allowed, the lower value wins.
    best_any = registry.load_best(
        ModelCategory.PREDICTION, "vit", "val_mae", deployable_only=False
    )
    assert best_any.status is ModelStatus.EXPERIMENTAL


def test_update_status_and_find(registry: RegistryManager) -> None:
    record = registry.register(_record("vit", {"val_mae": 0.6}, ModelStatus.EXPERIMENTAL))
    registry.update(record.unique_id, status=ModelStatus.PRODUCTION)
    found = registry.find(ModelCategory.PREDICTION, status=ModelStatus.PRODUCTION)
    assert len(found) == 1
    assert found[0].unique_id == record.unique_id


def test_update_unknown_raises(registry: RegistryManager) -> None:
    with pytest.raises(RegistryError):
        registry.update("HB_MISSING_V001", status=ModelStatus.STABLE)


def test_report_counts(registry: RegistryManager) -> None:
    registry.register(_record("vit", {"val_mae": 0.6}, ModelStatus.STABLE))
    registry.register(_record("resnet", {"val_mae": 0.7}, ModelStatus.EXPERIMENTAL))
    report = registry.report()
    assert report["total"] == 2
    assert report["by_category"]["prediction"] == 2


def test_persistence_across_instances(
    framework_config: FrameworkConfig, tmp_path: Path
) -> None:
    first = RegistryManager(framework_config, base_dir=tmp_path)
    first.initialize()
    first.register(_record("vit", {"val_mae": 0.6}, ModelStatus.STABLE))

    second = RegistryManager(framework_config, base_dir=tmp_path)
    second.initialize()
    assert second.load_latest(ModelCategory.PREDICTION, "vit").unique_id == "HB_VIT_V001"


def test_requires_initialization(
    framework_config: FrameworkConfig, tmp_path: Path
) -> None:
    manager = RegistryManager(framework_config, base_dir=tmp_path)
    with pytest.raises(RegistryError):
        manager.list_models()
