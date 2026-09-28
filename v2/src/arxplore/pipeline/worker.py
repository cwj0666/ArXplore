import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from arxplore.config import AppSettings
from arxplore.db.migrate import verify_schema
from arxplore.db.pool import Pool, create_pool_from_settings, open_pool
from arxplore.db.repositories import jobs
from arxplore.domain.jobs import DEFAULT_JOB_MODE, QueuePolicy
from arxplore.embedding import EmbeddingClient
from arxplore.parsing.layout_client import LayoutParserClient
from arxplore.pipeline.embed import Embedder, EmbeddingBudget, embed_after_prepare
from arxplore.pipeline.prepare.paper import HttpPdfSource, PdfFulltextParser, PrepareContext
from arxplore.pipeline.prepare.run import consume_prepare_queue

logger = logging.getLogger(__name__)

WORKER_ROLE = "worker"
DEFAULT_WORKER_NAME = "default"
DEFAULT_WAIT_TIMEOUT_SECONDS = 120.0
FAILED_WAIT_CAP_SECONDS = 30.0

type Emit = Callable[[dict[str, Any]], None]


class JobWaiter(Protocol):
    async def wait(self, timeout: float) -> list[str]: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerOptions:
    worker_id: str = DEFAULT_WORKER_NAME
    max_jobs_per_run: int = 1
    max_papers: int | None = None
    force: bool = False
    skip_embed: bool = False
    embedding: EmbeddingBudget = field(default_factory=EmbeddingBudget)
    wait_timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class WorkerRuntime:
    pool: Pool
    policy: QueuePolicy
    prepare: PrepareContext
    embedder: Embedder


def merge_embed_status(prepare_status: str, embed_status: str) -> str:
    if embed_status == "failed" and prepare_status != "failed":
        return "failed"
    if embed_status == "partial_failed" and prepare_status == "success":
        return "partial_failed"
    return prepare_status


async def run_worker_once(runtime: WorkerRuntime, options: WorkerOptions) -> dict[str, Any]:
    queue_run = await consume_prepare_queue(
        runtime.prepare,
        runtime.pool,
        policy=runtime.policy,
        worker_id=options.worker_id,
        mode=DEFAULT_JOB_MODE,
        max_jobs_per_run=options.max_jobs_per_run,
        max_papers=options.max_papers,
        force=options.force,
    )
    result = queue_run.as_dict()
    if options.skip_embed:
        result["embed"] = {"stage": "embed_after_prepare", "status": "skipped", "reason": "skip_embed_enabled"}
        return result
    try:
        embed_result = (
            await embed_after_prepare(runtime.pool, runtime.embedder, queue_run.prepared_arxiv_ids(), options.embedding)
        ).as_dict()
    except Exception as exc:
        embed_result = {"stage": "embed_after_prepare", "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    result["embed"] = embed_result
    result["status"] = merge_embed_status(str(result["status"]), str(embed_result["status"]))
    return result


def wait_seconds_after(status: str, wait_timeout_seconds: float) -> float | None:
    if status == "no_op":
        return wait_timeout_seconds
    if status == "failed":
        return min(wait_timeout_seconds, FAILED_WAIT_CAP_SECONDS)
    return None


async def run_worker_loop(runtime: WorkerRuntime, options: WorkerOptions, waiter: JobWaiter, emit: Emit) -> None:
    while True:
        result = await run_worker_once(runtime, options)
        emit(result)
        wait = wait_seconds_after(str(result.get("status") or ""), options.wait_timeout_seconds)
        if wait is not None:
            await waiter.wait(wait)


def emit_json(result: dict[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)


@asynccontextmanager
async def open_worker_runtime(settings: AppSettings) -> AsyncIterator[WorkerRuntime]:
    pool = create_pool_from_settings(settings, role=WORKER_ROLE)
    embedder = EmbeddingClient.from_settings(settings)
    async with (
        open_pool(pool),
        httpx.AsyncClient(follow_redirects=True) as download_http,
        httpx.AsyncClient() as layout_http,
    ):
        async with pool.connection() as connection:
            await verify_schema(connection, embedding_dimensions=settings.openai_embedding_dimensions)
        layout_client = LayoutParserClient.from_settings(settings, http_client=layout_http)
        try:
            yield WorkerRuntime(
                pool=pool,
                policy=QueuePolicy.from_settings(settings),
                prepare=PrepareContext(
                    db=pool,
                    pdfs=HttpPdfSource(download_http),
                    parser=PdfFulltextParser(layout_client=layout_client),
                ),
                embedder=embedder,
            )
        finally:
            await embedder.aclose()


async def run_worker(settings: AppSettings, options: WorkerOptions, *, loop: bool, emit: Emit = emit_json) -> int:
    async with open_worker_runtime(settings) as runtime:
        if not loop:
            result = await run_worker_once(runtime, options)
            emit(result)
            return 1 if result.get("status") == "failed" else 0
        async with jobs.job_listener(settings) as listener:
            await run_worker_loop(runtime, options, listener, emit)
    return 0
