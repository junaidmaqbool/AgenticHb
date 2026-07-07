"""Job queue with dependency resolution for pipeline execution.

Every pipeline operation is represented as a :class:`Job` (PIPELINE_SPEC Ch.18).
The :class:`JobQueue` executes jobs sequentially in dependency order, verifies
dependencies before running, supports resume (skipping already-completed jobs),
and stops on the first failure with an informative error (PIPELINE_SPEC Ch.19,
Ch.20).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from adaptivehb.core.types import JobStatus
from adaptivehb.exceptions import PipelineError

_DONE = frozenset({JobStatus.COMPLETED, JobStatus.SKIPPED})


@dataclass
class Job:
    """A single unit of pipeline work.

    Attributes:
        job_id: Unique identifier within a queue.
        run: Zero-argument callable performing the work; its return value is
            stored on ``result``.
        depends_on: IDs of jobs that must finish before this one runs.
        status: Current lifecycle status.
        result: Value returned by ``run`` on success.
        error: Error message when the job fails.
    """

    job_id: str
    run: Callable[[], Any]
    depends_on: list[str] = field(default_factory=list)
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: str | None = None


class JobQueue:
    """Executes jobs sequentially in dependency-resolved order."""

    def __init__(self, logger: Any = None) -> None:
        """Initialize an empty queue.

        Args:
            logger: Optional logger for job lifecycle messages.
        """
        self._jobs: dict[str, Job] = {}
        self._insertion: list[str] = []
        self._log = logger

    def add(self, job: Job) -> Job:
        """Add a job to the queue.

        Args:
            job: The job to add.

        Returns:
            The added job.

        Raises:
            PipelineError: If a job with the same ID already exists.
        """
        if job.job_id in self._jobs:
            raise PipelineError(f"Duplicate job id: {job.job_id!r}")
        self._jobs[job.job_id] = job
        self._insertion.append(job.job_id)
        return job

    @property
    def jobs(self) -> dict[str, Job]:
        """Return the mapping of job IDs to jobs."""
        return self._jobs

    def resolve_order(self) -> list[str]:
        """Return job IDs in a valid execution order.

        Uses Kahn's algorithm over the dependency graph, preserving insertion
        order among ready jobs for deterministic execution.

        Returns:
            Topologically ordered job IDs.

        Raises:
            PipelineError: If a dependency is unknown or a cycle exists.
        """
        indegree = {jid: 0 for jid in self._insertion}
        dependents: dict[str, list[str]] = {jid: [] for jid in self._insertion}
        for jid in self._insertion:
            for dep in self._jobs[jid].depends_on:
                if dep not in self._jobs:
                    raise PipelineError(
                        f"Job {jid!r} depends on unknown job {dep!r}."
                    )
                indegree[jid] += 1
                dependents[dep].append(jid)

        ready = [jid for jid in self._insertion if indegree[jid] == 0]
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for nxt in dependents[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)

        if len(order) != len(self._insertion):
            unresolved = [jid for jid in self._insertion if jid not in order]
            raise PipelineError(f"Cyclic job dependencies among: {unresolved}")
        return order

    def run(
        self,
        *,
        is_completed: Callable[[str], bool] | None = None,
        on_complete: Callable[[str], None] | None = None,
    ) -> dict[str, str]:
        """Execute all jobs in dependency order.

        Args:
            is_completed: Optional predicate; when it returns true for a job ID,
                the job is skipped (resume support).
            on_complete: Optional callback invoked with the job ID after a job
                completes successfully (e.g. to persist pipeline state).

        Returns:
            Mapping of job ID to final status value.

        Raises:
            PipelineError: If a dependency is unsatisfied or a job raises.
        """
        for jid in self.resolve_order():
            job = self._jobs[jid]

            if is_completed is not None and is_completed(jid):
                job.status = JobStatus.SKIPPED
                self._debug("Skipping completed job %s.", jid)
                continue

            unmet = [
                dep for dep in job.depends_on if self._jobs[dep].status not in _DONE
            ]
            if unmet:
                raise PipelineError(f"Job {jid!r} has unmet dependencies: {unmet}")

            try:
                job.status = JobStatus.RUNNING
                self._debug("Running job %s.", jid)
                job.result = job.run()
                job.status = JobStatus.COMPLETED
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                raise PipelineError(f"Job {jid!r} failed: {exc}") from exc

            if on_complete is not None:
                on_complete(jid)

        return {jid: self._jobs[jid].status.value for jid in self._insertion}

    def _debug(self, message: str, *args: Any) -> None:
        if self._log is not None:
            self._log.debug(message, *args)


__all__ = ["Job", "JobQueue"]
