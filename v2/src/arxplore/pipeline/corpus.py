import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from psycopg.rows import dict_row

from arxplore.db.pool import AnyPool, transaction
from arxplore.db.repositories import embeddings, fulltexts, jobs, papers, raw
from arxplore.domain.chunks import ChunkInput, StoredChunk
from arxplore.domain.jobs import DEFAULT_JOB_MODE, JobStatus, job_notification_payload
from arxplore.parsing.chunker import build_chunks, summarize_chunks
from arxplore.parsing.models import Chunk
from arxplore.parsing.roles import REFERENCES_ROLE

REPARSE_JOB_SOURCE = "reparse"
REOPENABLE_STATUSES = (JobStatus.PENDING.value, JobStatus.DONE.value, JobStatus.FAILED.value)

JOB_STATUS_BY_DATE_SQL = """
SELECT id, target_date, status
FROM prepare_jobs
WHERE mode = %s AND target_date = ANY(%s)
ORDER BY target_date
"""

REOPEN_JOBS_SQL = """
UPDATE prepare_jobs
SET
    status = 'pending',
    payload = CASE WHEN %(force)s THEN payload || '{"force": true}'::jsonb ELSE payload - 'force' END,
    attempt_count = 0,
    next_attempt_at = NULL,
    pending_refresh = FALSE,
    worker_id = NULL,
    error = NULL,
    claimed_at = NULL,
    heartbeat_at = NULL,
    finished_at = NULL,
    updated_at = NOW()
WHERE mode = %(mode)s
  AND target_date = ANY(%(dates)s)
  AND status = ANY(%(statuses)s)
RETURNING target_date
"""


def _is_reference(role: Any) -> bool:
    return bool(role == REFERENCES_ROLE)


def _stored_role(chunk: StoredChunk) -> Any:
    return chunk.metadata.get("content_role")


def _chunk_role(chunk: Chunk) -> str:
    return chunk["metadata"]["content_role"]


@dataclass(frozen=True, slots=True, kw_only=True)
class RechunkPaper:
    arxiv_id: str
    chunks_before: int
    chunks_after: int
    changed: bool
    embedded_before: int
    estimated_embedding_tokens: int


@dataclass(slots=True)
class RechunkResult:
    applied: bool
    papers: list[RechunkPaper] = field(default_factory=list)
    without_fulltext: list[str] = field(default_factory=list)

    @property
    def changed(self) -> list[RechunkPaper]:
        return [paper for paper in self.papers if paper.changed]

    def as_dict(self) -> dict[str, Any]:
        changed = self.changed
        return {
            "stage": "rechunk",
            "applied": self.applied,
            "paper_count": len(self.papers),
            "without_fulltext_count": len(self.without_fulltext),
            "changed_paper_count": len(changed),
            "chunks_before": sum(paper.chunks_before for paper in self.papers),
            "chunks_after": sum(paper.chunks_after for paper in self.papers),
            "embeddings_dropped": sum(paper.embedded_before for paper in changed),
            "estimated_embedding_tokens": sum(paper.estimated_embedding_tokens for paper in changed),
            "changed_arxiv_ids": [paper.arxiv_id for paper in changed],
        }


def chunks_differ(stored: Sequence[StoredChunk], rebuilt: Sequence[Chunk]) -> bool:
    before = sorted((chunk.chunk_index, chunk.chunk_text) for chunk in stored)
    after = sorted((chunk["chunk_index"], chunk["chunk_text"]) for chunk in rebuilt)
    return before != after


async def _embedded_count(pool: AnyPool, arxiv_id: str, stored: Sequence[StoredChunk]) -> int:
    embeddable = [chunk for chunk in stored if not _is_reference(_stored_role(chunk))]
    if not embeddable:
        return 0
    missing = await embeddings.list_chunks_missing_embeddings(pool, limit=len(embeddable), arxiv_id=arxiv_id)
    return len(embeddable) - len(missing)


async def rechunk_paper(pool: AnyPool, arxiv_id: str, *, apply: bool) -> RechunkPaper | None:
    fulltext = await fulltexts.get_fulltext(pool, arxiv_id)
    if fulltext is None or not fulltext.text:
        return None
    stored = await fulltexts.list_chunks(pool, arxiv_id)
    rebuilt = await asyncio.to_thread(build_chunks, fulltext.text, fulltext.sections)
    changed = chunks_differ(stored, rebuilt)
    paper = RechunkPaper(
        arxiv_id=arxiv_id,
        chunks_before=len(stored),
        chunks_after=len(rebuilt),
        changed=changed,
        embedded_before=await _embedded_count(pool, arxiv_id, stored),
        estimated_embedding_tokens=sum(
            chunk["token_count"] for chunk in rebuilt if not _is_reference(_chunk_role(chunk))
        ),
    )
    if apply and rebuilt:
        state = await fulltexts.get_fulltext_state(pool, arxiv_id)
        await fulltexts.save_chunks(pool, arxiv_id, [ChunkInput.from_mapping(chunk) for chunk in rebuilt])
        await fulltexts.save_fulltext(
            pool,
            arxiv_id,
            text=fulltext.text,
            sections=fulltext.sections,
            source=fulltext.source,
            quality_metrics={**fulltext.quality_metrics, **summarize_chunks(rebuilt)},
            artifacts=fulltext.artifacts,
            parser_metadata=fulltext.parser_metadata,
            content_hash=state.content_hash if state else None,
        )
    return paper


async def rechunk_corpus(
    pool: AnyPool, *, arxiv_ids: Sequence[str] | None = None, apply: bool = False
) -> RechunkResult:
    result = RechunkResult(applied=apply)
    for arxiv_id in list(arxiv_ids) if arxiv_ids else await papers.list_paper_ids(pool):
        paper = await rechunk_paper(pool, arxiv_id, apply=apply)
        if paper is None:
            result.without_fulltext.append(arxiv_id)
        else:
            result.papers.append(paper)
    return result


@dataclass(frozen=True, slots=True, kw_only=True)
class ReparseDate:
    target_date: date
    job_id: int | None
    status: str | None

    @property
    def action(self) -> str:
        if self.status is None:
            return "enqueue"
        if self.status in REOPENABLE_STATUSES:
            return "reopen"
        return "skip_processing"

    def as_dict(self) -> dict[str, Any]:
        return {
            "date": self.target_date.isoformat(),
            "job_id": self.job_id,
            "status": self.status,
            "action": self.action,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ReparseResult:
    mode: str
    force: bool
    applied: bool
    dates: tuple[ReparseDate, ...]

    def _count(self, action: str) -> int:
        return sum(1 for entry in self.dates if entry.action == action)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": "reparse",
            "mode": self.mode,
            "force": self.force,
            "applied": self.applied,
            "date_count": len(self.dates),
            "reopened": self._count("reopen"),
            "enqueued": self._count("enqueue"),
            "skipped_processing": self._count("skip_processing"),
            "dates": [entry.as_dict() for entry in self.dates],
        }


async def plan_reparse(
    pool: AnyPool, *, mode: str = DEFAULT_JOB_MODE, since: date | None = None, until: date | None = None
) -> list[ReparseDate]:
    dates = await raw.list_daily_papers_dates(pool, date_gte=since, date_lte=until)
    if not dates:
        return []
    async with pool.connection() as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(JOB_STATUS_BY_DATE_SQL, (mode, dates))
        rows = {row["target_date"]: row for row in await cursor.fetchall()}
    return [
        ReparseDate(
            target_date=day,
            job_id=int(rows[day]["id"]) if day in rows else None,
            status=str(rows[day]["status"]) if day in rows else None,
        )
        for day in dates
    ]


async def reparse_dates(
    pool: AnyPool,
    *,
    mode: str = DEFAULT_JOB_MODE,
    since: date | None = None,
    until: date | None = None,
    force: bool = True,
    apply: bool = False,
) -> ReparseResult:
    planned = await plan_reparse(pool, mode=mode, since=since, until=until)
    if apply and planned:
        reopen = [entry.target_date for entry in planned if entry.action == "reopen"]
        enqueue = [entry.target_date for entry in planned if entry.action == "enqueue"]
        async with transaction(pool) as connection, connection.cursor(row_factory=dict_row) as cursor:
            if reopen:
                await cursor.execute(
                    REOPEN_JOBS_SQL,
                    {"force": force, "mode": mode, "dates": reopen, "statuses": list(REOPENABLE_STATUSES)},
                )
                for row in await cursor.fetchall():
                    await cursor.execute(
                        jobs.NOTIFY_SQL, (jobs.JOB_CHANNEL, job_notification_payload(mode, row["target_date"]))
                    )
            for day in enqueue:
                await jobs.enqueue(
                    connection,
                    target_date=day,
                    mode=mode,
                    source=REPARSE_JOB_SOURCE,
                    payload={"force": True} if force else None,
                )
    return ReparseResult(mode=mode, force=force, applied=apply, dates=tuple(planned))
