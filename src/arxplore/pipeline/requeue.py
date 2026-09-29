from dataclasses import dataclass
from datetime import date
from typing import Any

from arxplore.db.pool import AnyPool
from arxplore.db.repositories import jobs
from arxplore.domain.jobs import DEFAULT_JOB_MODE, RequeuedJob


@dataclass(frozen=True, slots=True, kw_only=True)
class RequeueResult:
    mode: str
    since: date | None
    force: bool
    applied: bool
    jobs: tuple[RequeuedJob, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": "requeue_failed",
            "mode": self.mode,
            "since": self.since.isoformat() if self.since else None,
            "force": self.force,
            "applied": self.applied,
            "job_count": len(self.jobs),
            "jobs": [
                {"job_id": job.job_id, "target_date": job.target_date.isoformat(), "attempt_count": job.attempt_count}
                for job in self.jobs
            ],
        }


async def requeue_failed_jobs(
    pool: AnyPool,
    *,
    mode: str = DEFAULT_JOB_MODE,
    since: date | None = None,
    force: bool = False,
    apply: bool = False,
) -> RequeueResult:
    requeued = await jobs.requeue_failed(pool, mode=mode, since_date=since, dry_run=not apply, force=force)
    return RequeueResult(mode=mode, since=since, force=force, applied=apply, jobs=tuple(requeued))
