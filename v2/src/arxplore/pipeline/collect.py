from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from arxplore.db.pool import AnyPool, transaction
from arxplore.db.repositories import jobs, raw
from arxplore.domain.jobs import DEFAULT_JOB_MODE, DEFAULT_JOB_SOURCE, EnqueueResult

SAMPLE_ARXIV_ID_LIMIT = 5


class DailyPapersSource(Protocol):
    async def fetch_daily_papers(self, target_date: date) -> list[Any]: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class CollectResult:
    target_date: date
    fetched_count: int
    stored_record_id: str
    raw_revision: int
    raw_payload_changed: bool
    job: EnqueueResult | None
    sample_arxiv_ids: tuple[Any, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": "collect_papers",
            "status": "success",
            "target_date": self.target_date.isoformat(),
            "fetched_count": self.fetched_count,
            "stored_record_id": self.stored_record_id,
            "raw_revision": self.raw_revision,
            "raw_payload_changed": self.raw_payload_changed,
            "prepare_job_enqueued": bool(self.job and self.job.enqueued),
            "prepare_job_id": self.job.job_id if self.job else None,
            "prepare_job_status": self.job.status.value if self.job else None,
            "sample_arxiv_ids": list(self.sample_arxiv_ids),
        }


def sample_arxiv_ids(payload: Sequence[Any]) -> tuple[Any, ...]:
    ids = [
        item["paper"]["id"]
        for item in payload
        if isinstance(item, dict) and isinstance(item.get("paper"), dict) and item["paper"].get("id")
    ]
    return tuple(ids[:SAMPLE_ARXIV_ID_LIMIT])


async def collect_daily_papers(
    pool: AnyPool,
    source: DailyPapersSource,
    *,
    target_date: date,
    enqueue_prepare: bool = True,
    job_mode: str = DEFAULT_JOB_MODE,
    job_source: str = DEFAULT_JOB_SOURCE,
) -> CollectResult:
    payload = await source.fetch_daily_papers(target_date)
    queued: EnqueueResult | None = None
    async with transaction(pool) as connection:
        saved = await raw.save_daily_papers(connection, target_date=target_date, payload=payload)
        if enqueue_prepare:
            queued = await jobs.enqueue(
                connection,
                target_date=target_date,
                mode=job_mode,
                source=job_source,
                raw_revision=saved.revision,
            )
    return CollectResult(
        target_date=target_date,
        fetched_count=len(payload),
        stored_record_id=saved.record_id,
        raw_revision=saved.revision,
        raw_payload_changed=saved.changed,
        job=queued,
        sample_arxiv_ids=sample_arxiv_ids(payload),
    )
