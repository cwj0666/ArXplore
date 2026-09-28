import argparse
import asyncio
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import aclosing, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import IO, Any

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row

from arxplore.db.migrate import migrate
from arxplore.db.pool import Pool, create_pool, open_pool, transaction
from arxplore.db.repositories import ai_cache, embeddings, fulltexts, jobs, papers, pipeline_state, raw
from arxplore.domain.chunks import ChunkEmbedding, ChunkInput, StoredChunk
from arxplore.domain.jobs import ClaimedJob, ClaimToken, EnqueueResult, QueuePolicy
from arxplore.domain.papers import Paper, PaperUpsert

LEGACY_DB = "v2_test_repo_legacy"
NEW_DB = "v2_test_repo_new"
SCRATCH_PREFIX = "v2_test_repo"
UNUSED_PASSWORD = "unused-by-trust-auth"
CHANNEL = "arxplore_prepare_jobs"
SENTINEL_PREFIX = "diff-sentinel:"
READ_ONLY_OPTIONS = "-c default_transaction_read_only=on"
DIMENSIONS = 1536
EMBEDDING_MODEL = "text-embedding-3-large"
TIMESTAMP = "<timestamp>"
POLICY = QueuePolicy(max_attempts=3, stale_seconds=900)

VOLATILE_KEYS = {
    "jobs.claim": frozenset({"created_at", "claimed_at", "updated_at"}),
    "state.load": frozenset({"updated_at"}),
    "fulltexts.get": frozenset({"updated_at"}),
    "chunks.list": frozenset({"updated_at"}),
    "ai.get_overview": frozenset({"generated_at"}),
    "ai.get_summary": frozenset({"generated_at"}),
}

SNAPSHOT_QUERIES = {
    "papers": """
        SELECT arxiv_id, title, authors::text, abstract, primary_category, categories::text, pdf_url,
               published_at::text, updated_at::text, upvotes, github_url, github_stars, citation_count, source,
               updated_at_utc IS NOT NULL AS has_updated_at_utc, title_abstract_vector::text
        FROM papers ORDER BY arxiv_id
    """,
    "paper_fulltexts": """
        SELECT arxiv_id, text, sections::text, source, quality_metrics::text, artifacts::text,
               parser_metadata::text, content_hash, updated_at IS NOT NULL AS has_updated_at
        FROM paper_fulltexts ORDER BY arxiv_id
    """,
    "paper_chunks": """
        SELECT id, arxiv_id, chunk_index, chunk_text, section_title, token_count, metadata::text,
               chunk_vector::text, updated_at IS NOT NULL AS has_updated_at
        FROM paper_chunks ORDER BY id
    """,
    "paper_embeddings": "SELECT chunk_id, embedding::text, model_name FROM paper_embeddings ORDER BY chunk_id",
    "paper_ai_overviews": """
        SELECT arxiv_id, overview, key_findings::text, model, generated_at IS NOT NULL AS has_generated_at
        FROM paper_ai_overviews ORDER BY arxiv_id
    """,
    "paper_ai_detailed_summaries": """
        SELECT id, arxiv_id, model, summary, generated_at IS NOT NULL AS has_generated_at
        FROM paper_ai_detailed_summaries ORDER BY id
    """,
    "raw_daily_papers": """
        SELECT source, date::text, payload::text, payload_hash, revision, fetched_count,
               collected_at IS NOT NULL AS has_collected_at
        FROM raw_daily_papers ORDER BY source, date
    """,
    "pipeline_state": "SELECT key, value::text, updated_at IS NOT NULL AS has_updated_at FROM pipeline_state ORDER BY key",
    "prepare_jobs": """
        SELECT id, mode, target_date::text, source, payload::text, result::text, status, attempt_count, worker_id,
               error, claimed_at IS NOT NULL AS claimed, finished_at IS NOT NULL AS finished,
               heartbeat_at IS NOT NULL AS heartbeat,
               CASE WHEN next_attempt_at IS NULL THEN NULL
                    WHEN next_attempt_at > NOW() THEN 'future' ELSE 'past' END AS next_attempt,
               ceil(extract(epoch FROM next_attempt_at - updated_at) / 60) AS next_attempt_minutes,
               created_at = updated_at AS untouched, claim_generation, raw_revision, pending_refresh
        FROM prepare_jobs ORDER BY id
    """,
}

LEGACY_RUNNER = r"""
import json
import sys
from datetime import date, datetime

sys.path.insert(0, sys.argv[1])
from src.integrations.db import close_all_pools
from src.integrations.paper_repository import PaperRepository
from src.integrations.prepare_job_repository import PrepareJobRepository
from src.integrations.raw_store import RawPaperStore
from src.integrations.vector_repository import VectorRepository

paper_repository = PaperRepository()
raw_store = RawPaperStore()
queue = PrepareJobRepository()
vectors = VectorRepository()
tokens = {}


def encode(value):
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, date):
        return {"$date": value.isoformat()}
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def token(step):
    values = dict(tokens[step["token"]])
    values.update(step.get("override") or {})
    return values


def claim(step):
    job = queue.claim_prepare_job(mode=step["mode"], worker_id=step["worker_id"])
    if job is not None and step.get("as"):
        tokens[step["as"]] = {
            "job_id": job["job_id"],
            "worker_id": job["worker_id"],
            "claim_generation": job["claim_generation"],
        }
    return job


def collect(step):
    with raw_store.transaction() as connection:
        saved = raw_store.save_daily_papers_response(date=step["date"], payload=step["payload"], connection=connection)
        queued = queue.enqueue_prepare_job(
            target_date=step["date"],
            mode=step.get("mode", "auto"),
            source="collect",
            raw_revision=saved["revision"] + step.get("revision_offset", 0),
            connection=connection,
        )
        if step.get("fail_after_enqueue"):
            raise RuntimeError("enqueue failed after insert")
    return {"saved": saved, "queued": queued}


def list_dates(step):
    return raw_store.list_daily_papers_dates(
        date_gt=step.get("date_gt"),
        date_gte=step.get("date_gte"),
        date_lte=step.get("date_lte"),
        limit=step.get("limit"),
        ascending=step.get("ascending", True),
    )


HANDLERS = {
    "raw.save": lambda step: raw_store.save_daily_papers_response(date=step["date"], payload=step["payload"]),
    "raw.load": lambda step: raw_store.load_daily_papers_response(date=step["date"]),
    "raw.has": lambda step: raw_store.has_daily_papers_response(date=step["date"]),
    "raw.list_dates": list_dates,
    "state.save": lambda step: raw_store.save_pipeline_state(
        pipeline=step["pipeline"], name=step["name"], state=step["state"]
    ),
    "state.load": lambda step: raw_store.load_pipeline_state(pipeline=step["pipeline"], name=step["name"]),
    "collect": collect,
    "jobs.enqueue": lambda step: queue.enqueue_prepare_job(
        target_date=step["date"],
        mode=step["mode"],
        source=step.get("source", "collect"),
        payload=step.get("payload"),
        raw_revision=step.get("raw_revision"),
    ),
    "jobs.claim": claim,
    "jobs.heartbeat": lambda step: queue.heartbeat_prepare_job(**token(step)),
    "jobs.complete": lambda step: queue.complete_prepare_job(**token(step), result=step.get("result")),
    "jobs.fail": lambda step: queue.fail_prepare_job(**token(step), error=step["error"]),
    "jobs.reset_stale": lambda step: queue.reset_stale_prepare_jobs(
        mode=step["mode"], stale_seconds=step.get("stale_seconds")
    ),
    "jobs.requeue": lambda step: queue.requeue_failed_prepare_jobs(
        mode=step["mode"],
        since_date=step.get("since_date"),
        dry_run=step.get("dry_run", True),
        force=step.get("force", False),
    ),
    "papers.save": lambda step: paper_repository.save_paper(step["paper"]),
    "papers.get": lambda step: paper_repository.get_paper(step["arxiv_id"]),
    "papers.recent": lambda step: paper_repository.list_recent_papers(limit=step["limit"]),
    "papers.missing_metadata": lambda step: paper_repository.list_papers_missing_arxiv_metadata(limit=step["limit"]),
    "fulltexts.save": lambda step: paper_repository.save_paper_fulltext(step["arxiv_id"], **step["fields"]),
    "fulltexts.get": lambda step: paper_repository.get_paper_fulltext(step["arxiv_id"]),
    "fulltexts.state": lambda step: paper_repository.get_paper_fulltext_state(step["arxiv_id"]),
    "fulltexts.update_hash": lambda step: paper_repository.update_paper_fulltext_content_hash(
        step["arxiv_id"], step["content_hash"]
    ),
    "chunks.save": lambda step: paper_repository.save_paper_chunks(step["arxiv_id"], step["chunks"]),
    "chunks.list": lambda step: paper_repository.list_paper_chunks(step["arxiv_id"], limit=step.get("limit")),
    "embeddings.upsert": lambda step: vectors.upsert_paper_embeddings(step["rows"]),
    "embeddings.missing": lambda step: vectors.list_chunks_missing_embeddings(
        limit=step.get("limit", 200), arxiv_id=step.get("arxiv_id")
    ),
    "ai.save_overview": lambda step: paper_repository.upsert_paper_overview(
        step["arxiv_id"], step["overview"], step["key_findings"], step["model"]
    ),
    "ai.get_overview": lambda step: paper_repository.get_paper_overview(step["arxiv_id"]),
    "ai.save_summary": lambda step: paper_repository.upsert_detailed_summary(
        step["arxiv_id"], step["summary"], step["model"]
    ),
    "ai.get_summary": lambda step: paper_repository.get_detailed_summary(step["arxiv_id"], step["model"]),
}

try:
    for line in sys.stdin:
        step = json.loads(line)
        try:
            response = {"ok": encode(HANDLERS[step["op"]](step))}
        except Exception as exc:
            response = {"error": {"type": type(exc).__name__, "sqlstate": getattr(exc, "pgcode", None)}}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
finally:
    close_all_pools()
"""


@dataclass(frozen=True)
class Target:
    host: str
    port: int
    user: str

    def url(self, name: str, *, options: str | None = None) -> str:
        extra = {"options": options} if options else {}
        return make_conninfo(
            "", host=self.host, port=self.port, user=self.user, password=UNUSED_PASSWORD, dbname=name, **extra
        )


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def encode(value: Any) -> Any:
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, date):
        return {"$date": value.isoformat()}
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def iso(value: date | datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def to_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def required_date(value: str) -> date:
    return date.fromisoformat(value)


def paper_view(paper: Paper) -> dict[str, Any]:
    view = asdict(paper)
    view["published_at"] = iso(paper.published_at)
    view["updated_at"] = iso(paper.updated_at)
    return view


def chunk_view(chunk: StoredChunk) -> dict[str, Any]:
    view = asdict(chunk)
    view["updated_at"] = iso(chunk.updated_at)
    return view


def claimed_view(job: ClaimedJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "mode": job.mode,
        "date": job.target_date.isoformat(),
        "source": job.source,
        "payload": job.payload,
        "status": str(job.status),
        "attempt_count": job.attempt_count,
        "worker_id": job.worker_id,
        "created_at": job.created_at,
        "claimed_at": job.claimed_at,
        "updated_at": job.updated_at,
        "claim_generation": job.claim_generation,
        "raw_revision": job.raw_revision,
    }


def enqueued_view(result: EnqueueResult) -> dict[str, Any]:
    return {
        "enqueued": result.enqueued,
        "job_id": result.job_id,
        "mode": result.mode,
        "date": result.target_date.isoformat(),
        "status": str(result.status),
        "raw_revision": result.raw_revision,
        "pending_refresh": result.pending_refresh,
    }


class NewRunner:
    def __init__(self, pool: Pool) -> None:
        self._pool = pool
        self._tokens: dict[str, ClaimToken] = {}
        self._handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "raw.save": self._raw_save,
            "raw.load": self._raw_load,
            "raw.has": self._raw_has,
            "raw.list_dates": self._raw_list_dates,
            "state.save": self._state_save,
            "state.load": self._state_load,
            "collect": self._collect,
            "jobs.enqueue": self._enqueue,
            "jobs.claim": self._claim,
            "jobs.heartbeat": self._heartbeat,
            "jobs.complete": self._complete,
            "jobs.fail": self._fail,
            "jobs.reset_stale": self._reset_stale,
            "jobs.requeue": self._requeue,
            "papers.save": self._paper_save,
            "papers.get": self._paper_get,
            "papers.recent": self._papers_recent,
            "papers.missing_metadata": self._papers_missing,
            "fulltexts.save": self._fulltext_save,
            "fulltexts.get": self._fulltext_get,
            "fulltexts.state": self._fulltext_state,
            "fulltexts.update_hash": self._fulltext_hash,
            "chunks.save": self._chunks_save,
            "chunks.list": self._chunks_list,
            "embeddings.upsert": self._embeddings_upsert,
            "embeddings.missing": self._embeddings_missing,
            "ai.save_overview": self._overview_save,
            "ai.get_overview": self._overview_get,
            "ai.save_summary": self._summary_save,
            "ai.get_summary": self._summary_get,
        }

    async def run(self, step: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"ok": encode(await self._handlers[step["op"]](step))}
        except Exception as exc:
            return {"error": {"type": type(exc).__name__, "sqlstate": getattr(exc, "sqlstate", None)}}

    def _token(self, step: dict[str, Any]) -> ClaimToken:
        values = asdict(self._tokens[step["token"]])
        values.update(step.get("override") or {})
        return ClaimToken(**values)

    async def _raw_save(self, step: dict[str, Any]) -> Any:
        saved = await raw.save_daily_papers(
            self._pool, target_date=required_date(step["date"]), payload=step["payload"]
        )
        return asdict(saved)

    async def _raw_load(self, step: dict[str, Any]) -> Any:
        return await raw.load_daily_papers(self._pool, target_date=required_date(step["date"]))

    async def _raw_has(self, step: dict[str, Any]) -> Any:
        return await raw.has_daily_papers(self._pool, target_date=required_date(step["date"]))

    async def _raw_list_dates(self, step: dict[str, Any]) -> Any:
        found = await raw.list_daily_papers_dates(
            self._pool,
            date_gt=to_date(step.get("date_gt")),
            date_gte=to_date(step.get("date_gte")),
            date_lte=to_date(step.get("date_lte")),
            limit=step.get("limit"),
            ascending=step.get("ascending", True),
        )
        return [value.isoformat() for value in found]

    async def _state_save(self, step: dict[str, Any]) -> Any:
        await pipeline_state.save_pipeline_state(
            self._pool, pipeline=step["pipeline"], name=step["name"], state=step["state"]
        )

    async def _state_load(self, step: dict[str, Any]) -> Any:
        loaded = await pipeline_state.load_pipeline_state(self._pool, pipeline=step["pipeline"], name=step["name"])
        if loaded is None:
            return None
        return {**loaded.values, "pipeline": loaded.pipeline, "name": loaded.name, "updated_at": loaded.updated_at}

    async def _collect(self, step: dict[str, Any]) -> Any:
        target_date = required_date(step["date"])
        async with transaction(self._pool) as connection:
            saved = await raw.save_daily_papers(connection, target_date=target_date, payload=step["payload"])
            queued = await jobs.enqueue(
                connection,
                target_date=target_date,
                mode=step.get("mode", "auto"),
                source="collect",
                raw_revision=saved.revision + step.get("revision_offset", 0),
            )
            if step.get("fail_after_enqueue"):
                raise RuntimeError("enqueue failed after insert")
        return {"saved": asdict(saved), "queued": enqueued_view(queued)}

    async def _enqueue(self, step: dict[str, Any]) -> Any:
        result = await jobs.enqueue(
            self._pool,
            target_date=required_date(step["date"]),
            mode=step["mode"],
            source=step.get("source", "collect"),
            payload=step.get("payload"),
            raw_revision=step.get("raw_revision"),
        )
        return enqueued_view(result)

    async def _claim(self, step: dict[str, Any]) -> Any:
        job = await jobs.claim(self._pool, policy=POLICY, mode=step["mode"], worker_id=step["worker_id"])
        if job is None:
            return None
        if step.get("as"):
            self._tokens[step["as"]] = job.token
        return claimed_view(job)

    async def _heartbeat(self, step: dict[str, Any]) -> Any:
        return await jobs.heartbeat(self._pool, self._token(step))

    async def _complete(self, step: dict[str, Any]) -> Any:
        return await jobs.complete(self._pool, self._token(step), result=step.get("result"))

    async def _fail(self, step: dict[str, Any]) -> Any:
        return await jobs.fail(self._pool, self._token(step), policy=POLICY, error=step["error"])

    async def _reset_stale(self, step: dict[str, Any]) -> Any:
        return await jobs.reset_stale(
            self._pool, policy=POLICY, mode=step["mode"], stale_seconds=step.get("stale_seconds")
        )

    async def _requeue(self, step: dict[str, Any]) -> Any:
        requeued = await jobs.requeue_failed(
            self._pool,
            mode=step["mode"],
            since_date=to_date(step.get("since_date")),
            dry_run=step.get("dry_run", True),
            force=step.get("force", False),
        )
        return [
            {"id": job.job_id, "target_date": job.target_date.isoformat(), "attempt_count": job.attempt_count}
            for job in requeued
        ]

    async def _paper_save(self, step: dict[str, Any]) -> Any:
        return await papers.save_paper(self._pool, PaperUpsert.from_mapping(step["paper"]))

    async def _paper_get(self, step: dict[str, Any]) -> Any:
        paper = await papers.get_paper(self._pool, step["arxiv_id"])
        return paper_view(paper) if paper is not None else None

    async def _papers_recent(self, step: dict[str, Any]) -> Any:
        return [paper_view(paper) for paper in await papers.list_recent_papers(self._pool, limit=step["limit"])]

    async def _papers_missing(self, step: dict[str, Any]) -> Any:
        found = await papers.list_papers_missing_arxiv_metadata(self._pool, limit=step["limit"])
        return [paper_view(paper) for paper in found]

    async def _fulltext_save(self, step: dict[str, Any]) -> Any:
        await fulltexts.save_fulltext(self._pool, step["arxiv_id"], **step["fields"])

    async def _fulltext_get(self, step: dict[str, Any]) -> Any:
        found = await fulltexts.get_fulltext(self._pool, step["arxiv_id"])
        if found is None:
            return None
        view = asdict(found)
        view["updated_at"] = iso(found.updated_at)
        return view

    async def _fulltext_state(self, step: dict[str, Any]) -> Any:
        state = await fulltexts.get_fulltext_state(self._pool, step["arxiv_id"])
        return asdict(state) if state is not None else None

    async def _fulltext_hash(self, step: dict[str, Any]) -> Any:
        await fulltexts.update_content_hash(self._pool, step["arxiv_id"], step["content_hash"])

    async def _chunks_save(self, step: dict[str, Any]) -> Any:
        chunks = [ChunkInput.from_mapping(chunk) for chunk in step["chunks"]]
        return await fulltexts.save_chunks(self._pool, step["arxiv_id"], chunks)

    async def _chunks_list(self, step: dict[str, Any]) -> Any:
        return [
            chunk_view(chunk)
            for chunk in await fulltexts.list_chunks(self._pool, step["arxiv_id"], limit=step.get("limit"))
        ]

    async def _embeddings_upsert(self, step: dict[str, Any]) -> Any:
        rows = [
            ChunkEmbedding(chunk_id=row["chunk_id"], embedding=row["embedding"], model_name=row["model_name"])
            for row in step["rows"]
        ]
        await embeddings.upsert_embeddings(self._pool, rows)

    async def _embeddings_missing(self, step: dict[str, Any]) -> Any:
        found = await embeddings.list_chunks_missing_embeddings(
            self._pool, limit=step.get("limit", 200), arxiv_id=step.get("arxiv_id")
        )
        return [asdict(chunk) for chunk in found]

    async def _overview_save(self, step: dict[str, Any]) -> Any:
        await ai_cache.save_overview(
            self._pool,
            step["arxiv_id"],
            overview=step["overview"],
            key_findings=step["key_findings"],
            model=step["model"],
        )

    async def _overview_get(self, step: dict[str, Any]) -> Any:
        found = await ai_cache.get_overview(self._pool, step["arxiv_id"])
        if found is None:
            return None
        return {**asdict(found), "generated_at": iso(found.generated_at)}

    async def _summary_save(self, step: dict[str, Any]) -> Any:
        await ai_cache.save_detailed_summary(self._pool, step["arxiv_id"], summary=step["summary"], model=step["model"])

    async def _summary_get(self, step: dict[str, Any]) -> Any:
        found = await ai_cache.get_detailed_summary(self._pool, step["arxiv_id"], step["model"])
        if found is None:
            return None
        return {**asdict(found), "generated_at": iso(found.generated_at)}


class LegacyRunner:
    def __init__(self, process: subprocess.Popen[str], stderr: IO[str]) -> None:
        self._process = process
        self._stderr = stderr

    def run(self, step: dict[str, Any]) -> dict[str, Any]:
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write(json.dumps(step) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            self._stderr.seek(0)
            fail(f"legacy runner stopped at {step['op']}:\n{self._stderr.read()}")
        response: dict[str, Any] = json.loads(line)
        return response


@contextmanager
def start_legacy_runner(legacy_root: Path, env: dict[str, str], workdir: Path) -> Iterator[LegacyRunner]:
    with (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr,
        subprocess.Popen(
            [str(legacy_root / ".venv" / "bin" / "python"), "-c", LEGACY_RUNNER, str(legacy_root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env,
            cwd=workdir,
            text=True,
            encoding="utf-8",
        ) as process,
    ):
        yield LegacyRunner(process, stderr)


class Observer:
    def __init__(self, url: str) -> None:
        self._url = url
        self._probe: psycopg.AsyncConnection[Any] | None = None
        self._listener: psycopg.AsyncConnection[Any] | None = None
        self._sentinels = 0

    async def open(self) -> None:
        self._probe = await psycopg.AsyncConnection.connect(
            self._url, autocommit=True, row_factory=dict_row, options="-c TimeZone=UTC"
        )
        self._listener = await psycopg.AsyncConnection.connect(self._url, autocommit=True)
        await self._listener.execute(sql.SQL("LISTEN {}").format(sql.Identifier(CHANNEL)))

    async def close(self) -> None:
        for connection in (self._probe, self._listener):
            if connection is not None:
                await connection.close()

    @property
    def probe(self) -> psycopg.AsyncConnection[Any]:
        assert self._probe is not None
        return self._probe

    async def execute(self, query: str, params: Iterable[Any] = ()) -> None:
        await self.probe.execute(query, tuple(params))

    async def chunk_id(self, arxiv_id: str, chunk_index: int) -> int | None:
        cursor = await self.probe.execute(
            "SELECT id FROM paper_chunks WHERE arxiv_id = %s AND chunk_index = %s", (arxiv_id, chunk_index)
        )
        row = await cursor.fetchone()
        return int(row["id"]) if row is not None else None

    async def notifications(self) -> list[str]:
        assert self._listener is not None
        self._sentinels += 1
        sentinel = f"{SENTINEL_PREFIX}{self._sentinels}"
        await self.probe.execute("SELECT pg_notify(%s, %s)", (CHANNEL, sentinel))
        received: list[str] = []
        async with aclosing(self._listener.notifies(timeout=10.0)) as stream:
            async for notify in stream:
                if notify.payload == sentinel:
                    return received
                received.append(notify.payload)
        fail("sentinel notification did not arrive")
        return received

    async def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        tables: dict[str, list[dict[str, Any]]] = {}
        for table, query in SNAPSHOT_QUERIES.items():
            cursor = await self.probe.execute(query)
            tables[table] = [{key: _plain(value) for key, value in row.items()} for row in await cursor.fetchall()]
        return tables


def _plain(value: Any) -> Any:
    if isinstance(value, float):
        return repr(value)
    if hasattr(value, "is_finite"):
        return str(value)
    return value


def normalize(op: str, value: Any) -> Any:
    volatile = VOLATILE_KEYS.get(op, frozenset())

    def walk(item: Any, key: str | None = None) -> Any:
        if key in volatile and item is not None:
            return TIMESTAMP
        if isinstance(item, dict):
            return {name: walk(child, name) for name, child in item.items()}
        if isinstance(item, list):
            return [walk(child) for child in item]
        return item

    return walk(value)


def first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return f"{path}: {type(left).__name__} {_short(left)} != {type(right).__name__} {_short(right)}"
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                return f"{path}.{key}: present only in {'new' if key not in left else 'legacy'}"
            found = first_difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (one, other) in enumerate(zip(left, right, strict=True)):
            found = first_difference(one, other, f"{path}[{index}]")
            if found:
                return found
        return None
    return None if left == right else f"{path}: {_short(left)} != {_short(right)}"


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 160 else text[:157] + "..."


def vector(seed: int) -> list[float]:
    generator = random.Random(seed)
    values = [generator.gauss(0.0, 1.0) for _ in range(DIMENSIONS)]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def chunk(index: Any, text: str, **fields: Any) -> dict[str, Any]:
    return {"chunk_index": index, "chunk_text": text, **fields}


def queue_steps() -> list[dict[str, Any]]:
    day = "2026-04-07"
    steps: list[dict[str, Any]] = [
        {"op": "jobs.claim", "mode": "q0", "worker_id": "w1"},
        {"op": "jobs.enqueue", "mode": "q1", "date": day, "raw_revision": 1},
        {"op": "jobs.claim", "mode": "q1", "worker_id": "w1", "as": "q1"},
        {"op": "jobs.complete", "token": "q1", "override": {"claim_generation": 0}},
        {"op": "jobs.complete", "token": "q1", "override": {"worker_id": "w2"}},
        {"op": "jobs.fail", "token": "q1", "override": {"claim_generation": 2}, "error": "x"},
        {"op": "jobs.heartbeat", "token": "q1", "override": {"worker_id": "w2"}},
        {"op": "jobs.heartbeat", "token": "q1"},
        {"op": "jobs.complete", "token": "q1", "result": {"ok": 1, "failures": []}},
        {"op": "jobs.complete", "token": "q1"},
        {"op": "jobs.enqueue", "mode": "q2", "date": day},
        {"op": "jobs.claim", "mode": "q2", "worker_id": "w1", "as": "zombie"},
        {
            "op": "sql",
            "sql": "UPDATE prepare_jobs SET claimed_at = claimed_at - INTERVAL '2 hours',"
            " heartbeat_at = heartbeat_at - INTERVAL '1 hour' WHERE mode = 'q2'",
        },
        {"op": "jobs.claim", "mode": "q2", "worker_id": "w1", "as": "owner"},
        {"op": "jobs.heartbeat", "token": "zombie"},
        {"op": "jobs.complete", "token": "zombie"},
        {"op": "jobs.fail", "token": "zombie", "error": "late"},
        {"op": "jobs.complete", "token": "owner"},
        {"op": "jobs.enqueue", "mode": "q3", "date": day},
        {"op": "jobs.claim", "mode": "q3", "worker_id": "w1", "as": "q3"},
        {"op": "sql", "sql": "UPDATE prepare_jobs SET claimed_at = claimed_at - INTERVAL '2 hours' WHERE mode = 'q3'"},
        {"op": "jobs.heartbeat", "token": "q3"},
        {"op": "jobs.reset_stale", "mode": "q3", "stale_seconds": 900},
        {
            "op": "sql",
            "sql": "UPDATE prepare_jobs SET heartbeat_at = heartbeat_at - INTERVAL '20 minutes' WHERE mode = 'q3'",
        },
        {"op": "jobs.reset_stale", "mode": "q3", "stale_seconds": 900},
        {"op": "jobs.enqueue", "mode": "q4", "date": day},
        {"op": "jobs.claim", "mode": "q4", "worker_id": "w1"},
        {
            "op": "sql",
            "sql": "UPDATE prepare_jobs SET heartbeat_at = NULL, claimed_at = claimed_at - INTERVAL '20 minutes'"
            " WHERE mode = 'q4'",
        },
        {"op": "jobs.reset_stale", "mode": "q4", "stale_seconds": 900},
        {"op": "jobs.enqueue", "mode": "q5", "date": day},
        {"op": "jobs.claim", "mode": "q5", "worker_id": "w1"},
        {
            "op": "sql",
            "sql": "UPDATE prepare_jobs SET attempt_count = 3, heartbeat_at = heartbeat_at - INTERVAL '1 hour'"
            " WHERE mode = 'q5'",
        },
        {"op": "jobs.reset_stale", "mode": "q5", "stale_seconds": 900},
        {"op": "jobs.claim", "mode": "q5", "worker_id": "w1"},
        {"op": "jobs.enqueue", "mode": "q6", "date": day},
    ]
    make_due = {
        "op": "sql",
        "sql": "UPDATE prepare_jobs SET next_attempt_at = updated_at - INTERVAL '1 hour' WHERE mode = 'q6'",
    }
    for attempt in (1, 2, 3):
        steps += [
            {"op": "jobs.claim", "mode": "q6", "worker_id": "w1", "as": f"q6-{attempt}"},
            {"op": "jobs.fail", "token": f"q6-{attempt}", "error": f"boom {attempt}"},
            {"op": "jobs.claim", "mode": "q6", "worker_id": "w1"},
            make_due,
        ]
    steps += [
        {"op": "jobs.requeue", "mode": "q6"},
        {"op": "jobs.requeue", "mode": "q6", "dry_run": False},
        {"op": "jobs.claim", "mode": "q6", "worker_id": "w1"},
        {"op": "jobs.enqueue", "mode": "q7", "date": day, "raw_revision": 1},
        {"op": "jobs.claim", "mode": "q7", "worker_id": "w1", "as": "q7"},
        {"op": "jobs.complete", "token": "q7"},
        {"op": "jobs.enqueue", "mode": "q7", "date": day, "raw_revision": 1},
        {"op": "jobs.enqueue", "mode": "q7", "date": day},
        {"op": "jobs.enqueue", "mode": "q7", "date": day, "raw_revision": 2},
        {"op": "jobs.enqueue", "mode": "q8", "date": day, "raw_revision": 1},
        {"op": "jobs.claim", "mode": "q8", "worker_id": "w1", "as": "q8-1"},
        {"op": "jobs.enqueue", "mode": "q8", "date": day, "raw_revision": 2},
        {"op": "jobs.enqueue", "mode": "q8", "date": day, "raw_revision": 1},
        {"op": "jobs.heartbeat", "token": "q8-1"},
        {"op": "jobs.complete", "token": "q8-1"},
        {"op": "jobs.claim", "mode": "q8", "worker_id": "w1", "as": "q8-2"},
        {"op": "jobs.complete", "token": "q8-2"},
        {"op": "jobs.enqueue", "mode": "q9", "date": day, "raw_revision": 1},
        {"op": "sql", "sql": "UPDATE prepare_jobs SET attempt_count = 2 WHERE mode = 'q9'"},
        {"op": "jobs.claim", "mode": "q9", "worker_id": "w1", "as": "q9"},
        {"op": "jobs.enqueue", "mode": "q9", "date": day, "raw_revision": 2},
        {"op": "jobs.fail", "token": "q9", "error": "boom"},
        {"op": "jobs.enqueue", "mode": "q10", "date": day},
        {
            "op": "sql",
            "sql": "UPDATE prepare_jobs SET status = 'failed', attempt_count = 3, error = 'x' WHERE mode = 'q10'",
        },
        {"op": "jobs.enqueue", "mode": "q10", "date": day, "raw_revision": 1},
        {"op": "jobs.enqueue", "mode": "q11", "date": day, "payload": {"note": "x"}},
        {"op": "sql", "sql": "UPDATE prepare_jobs SET status = 'failed', attempt_count = 3 WHERE mode = 'q11'"},
        {"op": "jobs.requeue", "mode": "q11", "dry_run": False, "force": True},
        {"op": "jobs.claim", "mode": "q11", "worker_id": "w1"},
        {"op": "sql", "sql": "UPDATE prepare_jobs SET status = 'failed' WHERE mode = 'q11'"},
        {"op": "jobs.requeue", "mode": "q11", "dry_run": False},
        {"op": "jobs.claim", "mode": "q11", "worker_id": "w1"},
        {"op": "jobs.enqueue", "mode": "q12", "date": day, "payload": {"note": "x"}},
        {"op": "sql", "sql": "UPDATE prepare_jobs SET status = 'failed', attempt_count = 3 WHERE mode = 'q12'"},
        {"op": "jobs.requeue", "mode": "q12", "dry_run": False, "force": True},
        {"op": "jobs.enqueue", "mode": "q12", "date": day, "payload": {"collected": 2}, "raw_revision": 1},
        {"op": "jobs.claim", "mode": "q12", "worker_id": "w1", "as": "q12"},
        {"op": "jobs.complete", "token": "q12", "result": {"saved_papers": 1}},
    ]
    for target in ("2026-04-09", "2026-04-07", "2026-04-08"):
        steps.append({"op": "jobs.enqueue", "mode": "q13", "date": target, "source": "backfill"})
    steps += [{"op": "jobs.claim", "mode": "q13", "worker_id": f"w{index}"} for index in range(4)]
    steps += [
        {
            "op": "sql",
            "sql": "UPDATE prepare_jobs SET status = 'failed', attempt_count = 3, worker_id = NULL WHERE mode = 'q13'",
        },
        {"op": "jobs.requeue", "mode": "q13"},
        {"op": "jobs.requeue", "mode": "q13", "since_date": "2026-04-08", "dry_run": False},
        {"op": "jobs.requeue", "mode": "q13", "dry_run": False, "force": True},
        {"op": "jobs.enqueue", "mode": "q14", "date": day, "payload": {"force": True}},
        {"op": "jobs.claim", "mode": "q14", "worker_id": "w1", "as": "q14-1"},
        {"op": "jobs.fail", "token": "q14-1", "error": "retry"},
        {
            "op": "sql",
            "sql": "UPDATE prepare_jobs SET next_attempt_at = updated_at - INTERVAL '1 hour' WHERE mode = 'q14'",
        },
        {"op": "jobs.claim", "mode": "q14", "worker_id": "w1", "as": "q14-2"},
        {"op": "jobs.enqueue", "mode": "q14", "date": day, "raw_revision": 1},
        {
            "op": "sql",
            "sql": "UPDATE prepare_jobs SET heartbeat_at = heartbeat_at - INTERVAL '1 hour' WHERE mode = 'q14'",
        },
        {"op": "jobs.reset_stale", "mode": "q14"},
        {"op": "jobs.claim", "mode": "q14", "worker_id": "w1", "as": "q14-3"},
        {"op": "sql", "sql": "UPDATE prepare_jobs SET attempt_count = 3 WHERE mode = 'q14'"},
        {"op": "jobs.fail", "token": "q14-3", "error": "final"},
        {"op": "jobs.enqueue", "mode": "q15", "date": day},
        {"op": "jobs.claim", "mode": "q15", "worker_id": "w1"},
        {
            "op": "sql",
            "sql": "UPDATE prepare_jobs SET heartbeat_at = heartbeat_at - INTERVAL '1 day' WHERE mode = 'q15'",
        },
        {"op": "jobs.reset_stale", "mode": "q15", "stale_seconds": 0},
        {"op": "jobs.reset_stale", "mode": "q15", "stale_seconds": -5},
        {"op": "jobs.reset_stale", "mode": "q15"},
        {"op": "jobs.claim", "mode": "auto", "worker_id": "collector", "as": "auto"},
        {"op": "jobs.complete", "token": "auto", "result": {"saved": 1}},
        {
            "op": "collect",
            "date": "2026-04-12",
            "payload": [{"paper": {"id": "2604.00001"}}, {"paper": {"id": "2604.00002"}}],
        },
        {"op": "jobs.claim", "mode": "auto", "worker_id": "collector", "as": "auto-2"},
        {"op": "collect", "date": "2026-04-12", "payload": [{"paper": {"id": "2604.00003"}}]},
        {"op": "jobs.fail", "token": "auto-2", "error": "refresh during processing"},
    ]
    return steps


def raw_steps() -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {"op": "raw.save", "date": "2026-04-07", "payload": [{"paper": {"id": "a"}}]},
        {"op": "raw.has", "date": "2026-04-07"},
        {"op": "raw.has", "date": "2026-04-08"},
        {"op": "raw.save", "date": "2026-04-07", "payload": [{"paper": {"id": "a"}}]},
        {"op": "raw.save", "date": "2026-04-07", "payload": [{"paper": {"id": "a", "title": "T"}}]},
        {"op": "raw.save", "date": "2026-04-07", "payload": [{"paper": {"title": "T", "id": "a"}}]},
        {"op": "raw.save", "date": "2026-04-07", "payload": [{"paper": {"id": "a"}}, {"paper": {"id": "b"}}]},
        {"op": "raw.save", "date": "2026-04-08", "payload": []},
        {"op": "raw.save", "date": "2026-04-09", "payload": {"dailyPapers": [{"paper": {"id": "c"}}]}},
        {
            "op": "raw.save",
            "date": "2026-04-10",
            "payload": [
                {"title": "a\x00b", "k\x00": "\ud800x", "요약": "확산 모델 \U0001f600", "n": [1, 2.5, None, True]}
            ],
        },
        {
            "op": "raw.save",
            "date": "2026-04-10",
            "payload": [{"title": "ab", "k": "x", "요약": "확산 모델 \U0001f600", "n": [1, 2.5, None, True]}],
        },
    ]
    steps += [{"op": "raw.load", "date": value} for value in ("2026-04-07", "2026-04-09", "2026-04-10", "2026-04-30")]
    steps += [
        {"op": "raw.list_dates"},
        {"op": "raw.list_dates", "date_gt": "2026-04-07", "ascending": False},
        {"op": "raw.list_dates", "ascending": False, "limit": 1},
        {"op": "raw.list_dates", "date_gte": "2026-04-08", "date_lte": "2026-04-09", "limit": 0},
        {"op": "state.load", "pipeline": "hf_daily_papers_backfill", "name": "default"},
        {
            "op": "state.save",
            "pipeline": "hf_daily_papers_backfill",
            "name": "default",
            "state": {"cursor_date": "2026-04-06", "last_failure": {"date": "2026-04-07", "error": "429"}},
        },
        {
            "op": "state.save",
            "pipeline": "hf_daily_papers_backfill",
            "name": "default",
            "state": {
                "cursor_date": "2026-04-05",
                "last_processed_dates": ["2026-04-06"],
                "last_failure": None,
                "pipeline": "x",
                "name": "y",
                "updated_at": "z",
                "note": "ok\x00\ud800 수집",
            },
        },
        {"op": "state.load", "pipeline": "hf_daily_papers_backfill", "name": "default"},
        {"op": "state.load", "pipeline": "hf_daily_papers_backfill", "name": "other"},
        {"op": "state.save", "pipeline": "prepare_papers_backfill", "name": "gpu", "state": {"status": "success"}},
        {"op": "collect", "date": "2026-04-12", "payload": [{"paper": {"id": "2604.00001"}}]},
        {"op": "collect", "date": "2026-04-12", "payload": [{"paper": {"id": "2604.00001"}}]},
        {"op": "collect", "date": "2026-04-13", "payload": [{"paper": {"id": "x"}}], "fail_after_enqueue": True},
        {"op": "collect", "date": "2026-04-13", "payload": [{"paper": {"id": "x"}}], "revision_offset": 2**40},
    ]
    return steps


FULL_PAPER = {
    "arxiv_id": "2601.00001",
    "title": "Diffusion Transformers",
    "authors": [{"name": "Ada"}, {"name": "성민"}],
    "abstract": "We scale diffusion.",
    "primary_category": "cs.CV",
    "categories": ["cs.CV", "cs.LG"],
    "pdf_url": "https://arxiv.org/pdf/2601.00001",
    "published_at": "2026-01-02T03:04:05Z",
    "updated_at": "2026-01-03T00:00:00+09:00",
    "upvotes": 12,
    "github_url": "https://github.com/example/dit",
    "github_stars": 40,
    "citation_count": 3,
    "source": "hf_daily_papers+arxiv",
}


def paper_steps() -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [
        {"op": "papers.get", "arxiv_id": "2601.00001"},
        {"op": "papers.save", "paper": FULL_PAPER},
        {"op": "papers.get", "arxiv_id": "2601.00001"},
        {"op": "papers.save", "paper": {"arxiv_id": "2601.00001", "title": "HF title", "categories": []}},
        {"op": "papers.get", "arxiv_id": "2601.00001"},
        {
            "op": "papers.save",
            "paper": {
                **FULL_PAPER,
                "primary_category": "cs.AI",
                "categories": ["cs.AI"],
                "published_at": "2026-02-01T00:00:00Z",
                "upvotes": "0",
                "github_stars": "41",
                "citation_count": "x",
            },
        },
        {
            "op": "papers.save",
            "paper": {
                "arxiv_id": " 2601.00003 ",
                "title": "T\ud800",
                "authors": [{"na\udfffme": "A"}],
                "published_at": "2026-04-07T00:00:00",
                "updated_at": "",
            },
        },
        {"op": "papers.save", "paper": {"arxiv_id": "  "}},
        {"op": "papers.save", "paper": {"title": "no id"}},
        {
            "op": "papers.save",
            "paper": {
                "arxiv_id": "2601.00004",
                "title": "Raw",
                "published_at": "2026-04-07T09:00:00+09:00",
                "upvotes": None,
                "source": "hf_daily_papers_raw",
            },
        },
        {"op": "papers.save", "paper": {"arxiv_id": "2601.00005", "title": None}},
    ]
    for index in range(14):
        steps.append(
            {
                "op": "papers.save",
                "paper": {
                    "arxiv_id": f"2609.{(index * 37) % 101:05d}",
                    "title": f"Tie {index}",
                    "published_at": "2026-03-14T00:00:00Z",
                    "upvotes": index,
                },
            }
        )
    for index in range(0, 14, 3):
        steps.append(
            {"op": "papers.save", "paper": {"arxiv_id": f"2609.{(index * 37) % 101:05d}", "title": "Retitled"}}
        )
    steps += [
        {"op": "papers.save", "paper": {"arxiv_id": "undated-1", "title": "U1"}},
        {"op": "papers.save", "paper": {"arxiv_id": "undated-2", "title": "U2", "categories": ["cs.CL"]}},
    ]
    steps += [{"op": "papers.recent", "limit": limit} for limit in (5, 12, 100, 0)]
    steps += [{"op": "papers.missing_metadata", "limit": limit} for limit in (3, 100)]
    steps.append({"op": "papers.get", "arxiv_id": "2601.00003"})
    return steps


def fulltext_steps() -> list[dict[str, Any]]:
    first = [
        chunk(
            0,
            "Latent diffusion transformers generate long videos.",
            section_title="Abstract",
            token_count=6,
            metadata={"content_role": "body"},
        ),
        chunk(
            1,
            "Patch embeddings are denoised by a transformer.",
            section_title="3 Method",
            token_count=7,
            metadata={"content_role": "body"},
        ),
        chunk(
            2,
            "1 Introduction 2 Method 3 Experiments",
            section_title="Contents",
            token_count=6,
            metadata={"content_role": "toc"},
        ),
        chunk(
            3,
            "[1] A. Author. Paper. 2024.",
            section_title="References",
            token_count=5,
            metadata={"content_role": "references"},
        ),
    ]
    other = [
        chunk(
            0,
            "Reinforcement learning from human feedback is costly.",
            section_title="1 Introduction",
            metadata={"content_role": "body"},
        ),
        chunk(
            1,
            "The 50%_off baseline uses the reward model only.",
            section_title="4 Experiments",
            metadata={"content_role": "table_like"},
        ),
    ]
    relabeled = [dict(item, metadata={"content_role": "body", "section_index": 1}) for item in first]
    changed = [dict(item) for item in first]
    changed[1]["chunk_text"] = "Patch embeddings are denoised by a bigger transformer."
    rows = [
        {"chunk": ["2601.00001", 0], "vector": 1, "model_name": EMBEDDING_MODEL},
        {"chunk": ["2601.00001", 1], "vector": 2, "model_name": EMBEDDING_MODEL},
        {"chunk": ["2601.00001", 0], "vector": 3, "model_name": EMBEDDING_MODEL},
        {"chunk": ["2601.00004", 0], "vector": 4, "model_name": "old-model"},
    ]
    return [
        {"op": "fulltexts.state", "arxiv_id": "2601.00001"},
        {"op": "fulltexts.get", "arxiv_id": "2601.00001"},
        {
            "op": "fulltexts.save",
            "arxiv_id": "2601.00001",
            "fields": {
                "text": "Body \ud800text",
                "sections": [{"title": "Abstract\udfff", "text": "x"}],
                "source": "layout_pdf",
                "quality_metrics": {"text_length": 9, "fallback_used": False},
                "artifacts": {"tables": [{"page": 1, "caption": "T\ud800"}]},
                "parser_metadata": {"provider": "huridocs"},
                "content_hash": None,
            },
        },
        {"op": "fulltexts.get", "arxiv_id": "2601.00001"},
        {"op": "fulltexts.state", "arxiv_id": "2601.00001"},
        {"op": "chunks.save", "arxiv_id": "2601.00001", "chunks": [first[2], first[0], first[3], first[1]]},
        {"op": "chunks.list", "arxiv_id": "2601.00001"},
        {"op": "chunks.list", "arxiv_id": "2601.00001", "limit": 2},
        {"op": "chunks.list", "arxiv_id": "2601.00001", "limit": 0},
        {"op": "fulltexts.update_hash", "arxiv_id": "2601.00001", "content_hash": "hash-1"},
        {"op": "fulltexts.state", "arxiv_id": "2601.00001"},
        {"op": "chunks.save", "arxiv_id": "2601.00004", "chunks": other},
        {"op": "embeddings.missing"},
        {"op": "embeddings.missing", "limit": 2},
        {"op": "embeddings.missing", "arxiv_id": "2601.00004", "limit": 0},
        {"op": "embeddings.upsert", "rows": rows},
        {"op": "embeddings.missing"},
        {"op": "embeddings.upsert", "rows": [{"chunk": ["2601.00004", 0], "vector": 5, "model_name": EMBEDDING_MODEL}]},
        {"op": "chunks.save", "arxiv_id": "2601.00001", "chunks": relabeled},
        {"op": "chunks.save", "arxiv_id": "2601.00001", "chunks": list(reversed(relabeled))},
        {"op": "chunks.save", "arxiv_id": "2601.00001", "chunks": changed},
        {"op": "chunks.save", "arxiv_id": "2601.00001", "chunks": [first[0], dict(first[1], chunk_index=0)]},
        {"op": "chunks.save", "arxiv_id": "2601.00001", "chunks": []},
        {"op": "chunks.save", "arxiv_id": "2601.00001", "chunks": first[:2]},
        {
            "op": "chunks.save",
            "arxiv_id": "2601.00003",
            "chunks": [
                chunk("1", "b\ud800", token_count="3"),
                chunk(0, "a", section_title="Abs\udfff", metadata={"k\ud800": [1, "v\udc00"]}),
            ],
        },
        {"op": "chunks.list", "arxiv_id": "2601.00003"},
        {"op": "embeddings.missing", "limit": 50},
        {"op": "fulltexts.save", "arxiv_id": "2601.00001", "fields": {"text": "Plain"}},
        {"op": "fulltexts.get", "arxiv_id": "2601.00001"},
        {"op": "fulltexts.state", "arxiv_id": "2601.00001"},
        {"op": "fulltexts.save", "arxiv_id": "missing", "fields": {"text": "x"}},
        {"op": "chunks.save", "arxiv_id": "missing", "chunks": [chunk(0, "x")]},
        {"op": "ai.get_overview", "arxiv_id": "2601.00001"},
        {
            "op": "ai.save_overview",
            "arxiv_id": "2601.00001",
            "overview": "First",
            "key_findings": ["a", "b"],
            "model": "gpt-5-mini",
        },
        {
            "op": "ai.save_overview",
            "arxiv_id": "2601.00001",
            "overview": "Second 요약",
            "key_findings": ["c"],
            "model": "gpt-5",
        },
        {"op": "ai.get_overview", "arxiv_id": "2601.00001"},
        {"op": "ai.save_summary", "arxiv_id": "2601.00001", "summary": "mini", "model": "gpt-5-mini"},
        {"op": "ai.save_summary", "arxiv_id": "2601.00001", "summary": "full", "model": "gpt-5"},
        {"op": "ai.save_summary", "arxiv_id": "2601.00001", "summary": "mini 2", "model": "gpt-5-mini"},
        {"op": "ai.get_summary", "arxiv_id": "2601.00001", "model": "gpt-5-mini"},
        {"op": "ai.get_summary", "arxiv_id": "2601.00001", "model": "gpt-4o"},
        {"op": "ai.save_overview", "arxiv_id": "missing", "overview": "x", "key_findings": [], "model": "m"},
    ]


def scenario() -> list[dict[str, Any]]:
    return raw_steps() + queue_steps() + paper_steps() + fulltext_steps()


async def resolve(step: dict[str, Any], observer: Observer) -> dict[str, Any]:
    if step["op"] != "embeddings.upsert":
        return step
    rows = []
    for row in step["rows"]:
        chunk_id = await observer.chunk_id(*row["chunk"])
        rows.append({"chunk_id": chunk_id, "embedding": vector(row["vector"]), "model_name": row["model_name"]})
    return {**step, "rows": rows}


def database_env(target: Target, name: str) -> dict[str, str]:
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG") if key in os.environ}
    env.update(
        {
            "POSTGRES_HOST": f"{target.host}:{target.port}",
            "POSTGRES_DB": name,
            "APP_POSTGRES_DB": name,
            "POSTGRES_USER": target.user,
            "POSTGRES_PASSWORD": UNUSED_PASSWORD,
            "PREPARE_JOB_STALE_SECONDS": str(POLICY.stale_seconds),
            "PREPARE_JOB_MAX_ATTEMPTS": str(POLICY.max_attempts),
            "OPENAI_API_KEY": "sk-fake",
            "OPENAI_EMBEDDING_MODEL": EMBEDDING_MODEL,
            "LANGSMITH_TRACING": "false",
            "LANGSMITH_API_KEY": "",
            "LANGCHAIN_TRACING_V2": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env


def recreate_database(target: Target, name: str) -> None:
    if not name.startswith(SCRATCH_PREFIX):
        fail(f"refusing to touch database {name}")
    with psycopg.connect(target.url("postgres"), autocommit=True) as connection:
        connection.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def drop_database(target: Target, name: str) -> None:
    if not name.startswith(SCRATCH_PREFIX):
        fail(f"refusing to touch database {name}")
    with psycopg.connect(target.url("postgres"), autocommit=True) as connection:
        connection.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


def migrate_legacy(legacy_root: Path, env: dict[str, str], workdir: Path) -> None:
    result = subprocess.run(
        [str(legacy_root / ".venv" / "bin" / "python"), str(legacy_root / "scripts" / "migrate_schema.py")],
        env=env,
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        fail(f"legacy migrate_schema.py failed:\n{result.stdout}\n{result.stderr}")


async def migrate_new(target: Target) -> None:
    async with await psycopg.AsyncConnection.connect(target.url(NEW_DB), autocommit=True) as connection:
        await migrate(connection)


@dataclass
class Tally:
    steps: int = 0
    results: int = 0
    snapshots: int = 0
    notifications: int = 0
    notified: int = 0
    errors: int = 0
    operations: Counter[str] = field(default_factory=Counter)

    def report(self) -> str:
        operations = ", ".join(f"{name} {count}" for name, count in sorted(self.operations.items()))
        return (
            f"steps {self.steps}, compared results {self.results}, table snapshots {self.snapshots}, "
            f"notification lists {self.notifications} ({self.notified} payloads), "
            f"errors raised on both sides {self.errors}\noperations: {operations}"
        )


async def compare_writes(target: Target, legacy_root: Path, workdir: Path) -> list[str]:
    mismatches: list[str] = []
    tally = Tally()
    observers = {"legacy": Observer(target.url(LEGACY_DB)), "new": Observer(target.url(NEW_DB))}
    pool = create_pool(target.url(NEW_DB), role="diff", max_size=4, timeout=10, connect_timeout=5)
    try:
        for observer in observers.values():
            await observer.open()
        with start_legacy_runner(legacy_root, database_env(target, LEGACY_DB), workdir) as legacy:
            async with open_pool(pool):
                runner = NewRunner(pool)
                for index, step in enumerate(scenario(), start=1):
                    mismatches += await replay_step(
                        f"step {index} {step['op']}", step, legacy, runner, observers, tally
                    )
    finally:
        for observer in observers.values():
            await observer.close()
    print(f"write scenario: {tally.report()}")
    return mismatches


async def replay_step(
    label: str,
    step: dict[str, Any],
    legacy: LegacyRunner,
    runner: NewRunner,
    observers: dict[str, Observer],
    tally: Tally,
) -> list[str]:
    mismatches: list[str] = []
    tally.steps += 1
    tally.operations[step["op"]] += 1
    if step["op"] == "sql":
        for observer in observers.values():
            await observer.execute(step["sql"])
    else:
        legacy_result = legacy.run(await resolve(step, observers["legacy"]))
        new_result = await runner.run(await resolve(step, observers["new"]))
        tally.results += 1
        tally.errors += int("error" in legacy_result and "error" in new_result)
        difference = first_difference(normalize(step["op"], legacy_result), normalize(step["op"], new_result))
        if difference:
            mismatches.append(f"{label} result {difference}")
    notices = {side: await observer.notifications() for side, observer in observers.items()}
    tally.notifications += 1
    tally.notified += len(notices["legacy"])
    if notices["legacy"] != notices["new"]:
        mismatches.append(f"{label} notifications legacy={notices['legacy']} new={notices['new']}")
    snapshots = {side: await observer.snapshot() for side, observer in observers.items()}
    tally.snapshots += 1
    difference = first_difference(snapshots["legacy"], snapshots["new"])
    if difference:
        mismatches.append(f"{label} tables {difference}")
    return mismatches


async def golden_steps(target: Target, golden_db: str) -> list[dict[str, Any]]:
    url = target.url(golden_db, options=READ_ONLY_OPTIONS)
    async with await psycopg.AsyncConnection.connect(url, row_factory=dict_row) as connection:
        ids = [
            row["arxiv_id"]
            for row in await (await connection.execute("SELECT arxiv_id FROM papers ORDER BY arxiv_id")).fetchall()
        ]
        dates = [
            row["date"].isoformat()
            for row in await (await connection.execute("SELECT date FROM raw_daily_papers ORDER BY date")).fetchall()
        ]
        summaries = await (
            await connection.execute("SELECT arxiv_id, model FROM paper_ai_detailed_summaries")
        ).fetchall()
        missing_ids = [
            row["arxiv_id"]
            for row in await (
                await connection.execute(
                    "SELECT DISTINCT c.arxiv_id FROM paper_chunks c LEFT JOIN paper_embeddings e ON e.chunk_id = c.id "
                    "WHERE e.chunk_id IS NULL ORDER BY c.arxiv_id LIMIT 5"
                )
            ).fetchall()
        ]
    steps: list[dict[str, Any]] = [{"op": "papers.recent", "limit": limit} for limit in (1500, 30, 5)]
    steps += [{"op": "papers.missing_metadata", "limit": limit} for limit in (30, 200)]
    steps += [{"op": "papers.get", "arxiv_id": arxiv_id} for arxiv_id in [*ids, "missing"]]
    steps += [{"op": "fulltexts.state", "arxiv_id": arxiv_id} for arxiv_id in ids]
    steps += [{"op": "fulltexts.get", "arxiv_id": arxiv_id} for arxiv_id in ids[:25]]
    steps += [{"op": "chunks.list", "arxiv_id": arxiv_id} for arxiv_id in ids[:10]]
    steps += [{"op": "chunks.list", "arxiv_id": arxiv_id, "limit": 20} for arxiv_id in ids[:80]]
    steps += [{"op": "embeddings.missing", "limit": limit} for limit in (200, 5000)]
    steps += [{"op": "embeddings.missing", "arxiv_id": arxiv_id} for arxiv_id in missing_ids]
    steps += [{"op": "ai.get_overview", "arxiv_id": arxiv_id} for arxiv_id in ids]
    steps += [
        {"op": "ai.get_summary", "arxiv_id": row["arxiv_id"], "model": model}
        for row in summaries
        for model in {row["model"], "gpt-5", "gpt-5-mini"}
    ]
    steps += [{"op": "raw.load", "date": value} for value in dates]
    steps += [{"op": "raw.has", "date": value} for value in [*dates, "2000-01-01"]]
    steps += [{"op": "raw.list_dates"}, {"op": "raw.list_dates", "ascending": False, "limit": 3}]
    steps += [{"op": "state.load", "pipeline": "hf_daily_papers_backfill", "name": "default"}]
    steps += [{"op": "jobs.requeue", "mode": "auto"}]
    return steps


async def compare_golden_reads(target: Target, golden_db: str, legacy_root: Path, workdir: Path) -> list[str]:
    mismatches: list[str] = []
    steps = await golden_steps(target, golden_db)
    env = {**database_env(target, golden_db), "PGOPTIONS": READ_ONLY_OPTIONS}
    pool = create_pool(
        target.url(golden_db, options=READ_ONLY_OPTIONS), role="diff-read", max_size=2, timeout=10, connect_timeout=5
    )
    compared = 0
    rows = 0
    with start_legacy_runner(legacy_root, env, workdir) as legacy:
        async with open_pool(pool):
            runner = NewRunner(pool)
            for index, step in enumerate(steps, start=1):
                legacy_result = legacy.run(step)
                new_result = await runner.run(step)
                compared += 1
                value = legacy_result.get("ok")
                rows += len(value) if isinstance(value, list) else int(value is not None)
                if "error" in legacy_result or "error" in new_result:
                    mismatches.append(f"golden step {index} {step['op']} failed: {legacy_result} / {new_result}")
                    continue
                difference = first_difference(normalize(step["op"], legacy_result), normalize(step["op"], new_result))
                if difference:
                    mismatches.append(f"golden step {index} {step['op']} {difference}")
    print(f"golden read-only comparison on {golden_db}: {compared} calls, {rows} returned rows or objects")
    return mismatches


async def run(options: argparse.Namespace) -> int:
    target = Target(host=options.host, port=options.port, user=options.user)
    legacy_root: Path = options.legacy_root
    recreate_database(target, LEGACY_DB)
    recreate_database(target, NEW_DB)
    try:
        with tempfile.TemporaryDirectory(prefix="arxplore-diff-") as workdir:
            migrate_legacy(legacy_root, database_env(target, LEGACY_DB), Path(workdir))
            await migrate_new(target)
            mismatches = await compare_writes(target, legacy_root, Path(workdir))
            if not options.skip_golden:
                mismatches += await compare_golden_reads(target, options.golden_db, legacy_root, Path(workdir))
    finally:
        if not options.keep:
            drop_database(target, LEGACY_DB)
            drop_database(target, NEW_DB)
    for mismatch in mismatches:
        print(f"MISMATCH {mismatch}")
    print("identical" if not mismatches else f"{len(mismatches)} mismatches")
    return 0 if not mismatches else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay the same repository operations on the legacy and the v2 code and compare the results."
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=55432)
    parser.add_argument("--user", default="arxplore")
    parser.add_argument("--legacy-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--golden-db", default="arxplore", help="Database read (read-only) by both implementations.")
    parser.add_argument("--skip-golden", action="store_true", help="Skip the read-only comparison on the golden data.")
    parser.add_argument("--keep", action="store_true", help=f"Keep {LEGACY_DB} and {NEW_DB} for inspection.")
    options = parser.parse_args()
    if not (options.legacy_root / ".venv" / "bin" / "python").exists():
        fail(f"legacy virtualenv not found under {options.legacy_root}")
    sys.stdout.reconfigure(line_buffering=True)
    return asyncio.run(run(options))


if __name__ == "__main__":
    raise SystemExit(main())
