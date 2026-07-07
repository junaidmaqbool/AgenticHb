"""Unit tests for the Job / JobQueue engine."""

from __future__ import annotations

import pytest

from adaptivehb.core.types import JobStatus
from adaptivehb.exceptions import PipelineError
from adaptivehb.managers import Job, JobQueue


def test_linear_execution_records_results() -> None:
    log: list[str] = []
    queue = JobQueue()
    queue.add(Job("a", lambda: log.append("a") or "ra"))
    queue.add(Job("b", lambda: log.append("b") or "rb", depends_on=["a"]))
    statuses = queue.run()
    assert log == ["a", "b"]
    assert statuses == {"a": "completed", "b": "completed"}
    assert queue.jobs["a"].result == "ra"


def test_dependencies_are_ordered_before_dependents() -> None:
    queue = JobQueue()
    # Insertion order deliberately reversed relative to dependency order.
    queue.add(Job("last", lambda: None, depends_on=["mid"]))
    queue.add(Job("mid", lambda: None, depends_on=["first"]))
    queue.add(Job("first", lambda: None))
    assert queue.resolve_order() == ["first", "mid", "last"]


def test_duplicate_job_id_raises() -> None:
    queue = JobQueue()
    queue.add(Job("a", lambda: None))
    with pytest.raises(PipelineError):
        queue.add(Job("a", lambda: None))


def test_missing_dependency_raises() -> None:
    queue = JobQueue()
    queue.add(Job("a", lambda: None, depends_on=["ghost"]))
    with pytest.raises(PipelineError):
        queue.resolve_order()


def test_cycle_is_detected() -> None:
    queue = JobQueue()
    queue.add(Job("a", lambda: None, depends_on=["b"]))
    queue.add(Job("b", lambda: None, depends_on=["a"]))
    with pytest.raises(PipelineError):
        queue.resolve_order()


def test_failure_propagates_and_marks_failed() -> None:
    def boom() -> None:
        raise ValueError("kaboom")

    queue = JobQueue()
    queue.add(Job("ok", lambda: None))
    queue.add(Job("bad", boom, depends_on=["ok"]))
    with pytest.raises(PipelineError):
        queue.run()
    assert queue.jobs["bad"].status is JobStatus.FAILED
    assert "kaboom" in (queue.jobs["bad"].error or "")


def test_resume_skips_completed_jobs() -> None:
    executed: list[str] = []
    completed = {"a"}
    marked: list[str] = []
    queue = JobQueue()
    queue.add(Job("a", lambda: executed.append("a")))
    queue.add(Job("b", lambda: executed.append("b"), depends_on=["a"]))
    statuses = queue.run(
        is_completed=lambda jid: jid in completed,
        on_complete=marked.append,
    )
    assert executed == ["b"]  # 'a' was skipped
    assert statuses["a"] == "skipped"
    assert statuses["b"] == "completed"
    assert marked == ["b"]
