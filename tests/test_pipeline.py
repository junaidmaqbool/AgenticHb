"""Unit tests for PipelineManager core dispatch and the HbPipeline facade.

Data-driven modes (training/evaluation/inference) are covered in
``test_pipeline_modes.py``; this module focuses on BUILD, dispatch guards, the
job-submission engine, and deployment deferral.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptivehb.config import FrameworkConfig
from adaptivehb.core.types import PipelineMode
from adaptivehb.exceptions import PipelineError
from adaptivehb.managers import Job, PipelineManager
from adaptivehb.pipeline import HbPipeline


@pytest.fixture()
def pipeline_manager(
    framework_config: FrameworkConfig, tmp_path: Path
) -> PipelineManager:
    manager = PipelineManager(framework_config, tmp_path)
    manager.initialize()
    return manager


def test_build_mode_runs_all_checks(pipeline_manager: PipelineManager) -> None:
    result = pipeline_manager.run(PipelineMode.BUILD)
    assert result["mode"] == "build"
    assert set(result["jobs"].values()) == {"completed"}
    assert pipeline_manager.state.state.status == "completed"


def test_run_before_initialize_raises(
    framework_config: FrameworkConfig, tmp_path: Path
) -> None:
    manager = PipelineManager(framework_config, tmp_path)
    with pytest.raises(PipelineError):
        manager.run(PipelineMode.BUILD)


def test_deployment_without_models_raises(pipeline_manager: PipelineManager) -> None:
    # Deployment is implemented (Phase 9) but requires trained prediction models.
    with pytest.raises(PipelineError):
        pipeline_manager.run(PipelineMode.DEPLOYMENT)


def test_unknown_mode_raises(pipeline_manager: PipelineManager) -> None:
    with pytest.raises(PipelineError):
        pipeline_manager.run("not_a_mode")


def test_submit_with_resume_skips_completed(pipeline_manager: PipelineManager) -> None:
    executed: list[str] = []
    pipeline_manager.state.update(completed_modules=["step_a"])
    jobs = [
        Job("step_a", lambda: executed.append("a")),
        Job("step_b", lambda: executed.append("b"), depends_on=["step_a"]),
    ]
    statuses = pipeline_manager.submit(jobs, resume=True)
    assert executed == ["b"]
    assert statuses["step_a"] == "skipped"
    assert statuses["step_b"] == "completed"


def test_facade_build(framework_config: FrameworkConfig, tmp_path: Path) -> None:
    pipeline = HbPipeline(framework_config, base_dir=tmp_path)
    result = pipeline.build()
    assert result["mode"] == "build"
    assert pipeline.config.project.name


def test_facade_from_config_dir(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = HbPipeline.from_config_dir(repo_root / "configs", base_dir=tmp_path)
    assert pipeline.build()["mode"] == "build"


def test_facade_deploy_without_models_raises(
    framework_config: FrameworkConfig, tmp_path: Path
) -> None:
    pipeline = HbPipeline(framework_config, base_dir=tmp_path)
    with pytest.raises(PipelineError):
        pipeline.deploy()
