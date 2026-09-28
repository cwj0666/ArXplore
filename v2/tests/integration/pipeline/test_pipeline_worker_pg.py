import asyncio
import time
from datetime import date
from typing import Any

import pytest
from psycopg.conninfo import conninfo_to_dict

from arxplore.config import AppSettings
from arxplore.db.repositories import jobs, raw
from arxplore.embedding import EmbeddingClient
from arxplore.pipeline.embed import EmbeddingBudget
from arxplore.pipeline.prepare.paper import PrepareContext
from arxplore.pipeline.worker import WorkerOptions, WorkerRuntime, run_worker_loop, run_worker_once

pytestmark = pytest.mark.integration

APRIL_7 = date(2026, 4, 7)
APRIL_8 = date(2026, 4, 8)
MODEL = "text-embedding-3-large"


def _embedder(fake_openai: Any, *, api_key: str | None = "sk-fake-arxplore") -> EmbeddingClient:
    return EmbeddingClient(model=MODEL, dimensions=1536, api_key=api_key, base_url=fake_openai.base_url)


def _runtime(pool: Any, policy: Any, kit: Any, embedder: Any, parser: Any = None) -> WorkerRuntime:
    context = PrepareContext(db=pool, pdfs=kit.FakePdfSource(), parser=parser or kit.FakeParser())
    return WorkerRuntime(pool=pool, policy=policy, prepare=context, embedder=embedder)


async def _seed(pool: Any, kit: Any, target_date: date, arxiv_ids: list[str]) -> int:
    saved = await raw.save_daily_papers(pool, target_date=target_date, payload=[kit.hf_item(i) for i in arxiv_ids])
    return (await jobs.enqueue(pool, target_date=target_date, raw_revision=saved.revision)).job_id


async def _missing_embeddings(probe: Any) -> int:
    return int(
        await probe.scalar(
            """
            SELECT count(*) FROM paper_chunks c LEFT JOIN paper_embeddings e ON e.chunk_id = c.id
            WHERE e.chunk_id IS NULL AND COALESCE(c.metadata->>'content_role', '') <> 'references'
            """
        )
    )


async def test_prepared_papers_are_embedded_after_the_run(pool, probe, policy, kit, fake_openai) -> None:
    await _seed(pool, kit, APRIL_7, ["2604.00001", "2604.00002"])
    embedder = _embedder(fake_openai)

    result = await run_worker_once(_runtime(pool, policy, kit, embedder), WorkerOptions(worker_id="w1"))
    await embedder.aclose()

    assert result["status"] == "success"
    embed = result["embed"]
    assert embed["status"] == "success"
    assert embed["target_arxiv_count"] == 2
    assert embed["embedded_arxiv_count"] == 2
    assert [entry["arxiv_id"] for entry in embed["per_arxiv"]] == ["2604.00001", "2604.00002"]
    assert embed["embedded_chunk_count"] == await probe.scalar("SELECT count(*) FROM paper_chunks")
    assert await _missing_embeddings(probe) == 0
    assert await probe.scalar("SELECT DISTINCT model_name FROM paper_embeddings") == MODEL
    requests = fake_openai.requests_for("/v1/embeddings")
    assert {request.body["dimensions"] for request in requests} == {1536}


async def test_reference_chunks_are_never_embedded(pool, probe, policy, kit, fake_openai) -> None:
    sections = [
        {"title": "1 Introduction", "text": "We study retrieval. It works well for long documents."},
        {"title": "References", "text": "[1] A. Author. A paper. In Proceedings, 2024."},
    ]
    await _seed(pool, kit, APRIL_7, ["2604.00001"])
    parser = kit.FakeParser(sections_by_id={"2604.00001": sections})
    embedder = _embedder(fake_openai)

    await run_worker_once(_runtime(pool, policy, kit, embedder, parser), WorkerOptions())
    await embedder.aclose()

    rows = await probe.rows(
        """
        SELECT c.metadata->>'content_role' AS role, e.chunk_id IS NOT NULL AS embedded
        FROM paper_chunks c LEFT JOIN paper_embeddings e ON e.chunk_id = c.id ORDER BY c.chunk_index
        """
    )
    assert {row["role"]: row["embedded"] for row in rows} == {"body": True, "references": False}


async def test_backlog_runs_after_a_no_op_prepare_within_its_budget(pool, probe, policy, kit, fake_openai) -> None:
    await _seed(pool, kit, APRIL_7, [f"2604.0000{number}" for number in range(1, 8)])
    await run_worker_once(_runtime(pool, policy, kit, None), WorkerOptions(skip_embed=True))
    missing_before = await _missing_embeddings(probe)
    assert missing_before >= 7
    embedder = _embedder(fake_openai)
    options = WorkerOptions(embedding=EmbeddingBudget(chunks_per_round=2, backlog_chunks=5))

    result = await run_worker_once(_runtime(pool, policy, kit, embedder), options)
    await embedder.aclose()

    assert result["status"] == "no_op"
    assert result["embed"]["status"] == "success"
    assert result["embed"]["backlog_selected_chunk_count"] == 5
    assert result["embed"]["backlog_embedded_chunk_count"] == 5
    assert [len(request.body["input"]) for request in fake_openai.requests_for("/v1/embeddings")] == [2, 2, 1]
    assert await _missing_embeddings(probe) == missing_before - 5


async def test_backlog_stops_when_a_round_comes_back_short(pool, probe, policy, kit, fake_openai) -> None:
    await _seed(pool, kit, APRIL_7, ["2604.00001"])
    await run_worker_once(_runtime(pool, policy, kit, None), WorkerOptions(skip_embed=True))
    missing = await _missing_embeddings(probe)
    embedder = _embedder(fake_openai)
    options = WorkerOptions(embedding=EmbeddingBudget(chunks_per_round=200, backlog_chunks=400))

    result = await run_worker_once(_runtime(pool, policy, kit, embedder), options)
    await embedder.aclose()

    assert result["embed"]["backlog_embedded_chunk_count"] == missing
    assert len(fake_openai.requests_for("/v1/embeddings")) == 1


async def test_skip_embed_calls_nothing(pool, policy, kit, fake_openai) -> None:
    await _seed(pool, kit, APRIL_7, ["2604.00001"])

    result = await run_worker_once(_runtime(pool, policy, kit, _embedder(fake_openai)), WorkerOptions(skip_embed=True))

    assert result["status"] == "success"
    assert result["embed"] == {"stage": "embed_after_prepare", "status": "skipped", "reason": "skip_embed_enabled"}
    assert fake_openai.requests == []


async def test_embedding_without_a_key_fails_the_run(pool, policy, kit, fake_openai) -> None:
    await _seed(pool, kit, APRIL_7, ["2604.00001"])

    result = await run_worker_once(_runtime(pool, policy, kit, _embedder(fake_openai, api_key=None)), WorkerOptions())

    assert result["status"] == "failed"
    embed = result["embed"]
    assert embed["status"] == "failed"
    assert embed["failures"][0]["arxiv_id"] == "2604.00001"
    assert "OpenAI API 키가 없습니다" in embed["failures"][0]["error"]
    assert embed["backlog_error"].startswith("EmbeddingKeyMissingError: ")


async def test_a_backlog_error_is_recorded_without_failing_a_no_op_run(pool, policy, kit, fake_openai) -> None:
    await _seed(pool, kit, APRIL_7, ["2604.00001"])
    await run_worker_once(_runtime(pool, policy, kit, None), WorkerOptions(skip_embed=True))

    result = await run_worker_once(_runtime(pool, policy, kit, _embedder(fake_openai, api_key=None)), WorkerOptions())

    assert result["status"] == "no_op"
    assert result["embed"]["status"] == "partial_failed"
    assert result["embed"]["backlog_error"].startswith("EmbeddingKeyMissingError: ")


async def test_a_paper_over_the_round_limit_is_reported(pool, probe, policy, kit, fake_openai) -> None:
    await _seed(pool, kit, APRIL_7, ["2604.00001"])
    embedder = _embedder(fake_openai)
    options = WorkerOptions(embedding=EmbeddingBudget(chunks_per_round=1, max_rounds_per_paper=2, backlog_chunks=0))

    result = await run_worker_once(_runtime(pool, policy, kit, embedder), options)
    await embedder.aclose()

    assert await probe.scalar("SELECT count(*) FROM paper_chunks") > 2
    assert result["embed"]["failures"] == [{"arxiv_id": "2604.00001", "error": "embed_loop_limit_exceeded"}]
    assert result["embed"]["per_arxiv"] == []
    assert result["embed"]["status"] == "failed"
    assert result["status"] == "failed"
    assert await probe.scalar("SELECT count(*) FROM paper_embeddings") == 2


async def test_papers_prepared_in_a_failed_job_are_embedded(pool, policy, kit, fake_openai) -> None:
    await _seed(pool, kit, APRIL_7, ["2604.00001", "2604.00002", "2604.00003"])
    embedder = _embedder(fake_openai)
    parser = kit.FakeParser(failing_ids={"2604.00002"})
    options = WorkerOptions(embedding=EmbeddingBudget(backlog_chunks=0))

    result = await run_worker_once(_runtime(pool, policy, kit, embedder, parser), options)
    await embedder.aclose()

    assert result["status"] == "failed"
    assert [entry["arxiv_id"] for entry in result["embed"]["per_arxiv"]] == ["2604.00001", "2604.00003"]
    assert result["embed"]["status"] == "success"


def _settings(database_url: str) -> AppSettings:
    params = conninfo_to_dict(database_url)
    return AppSettings.model_validate(
        {
            "POSTGRES_HOST": f"{params.get('host', 'localhost')}:{params.get('port', 5432)}",
            "APP_POSTGRES_DB": str(params["dbname"]),
            "POSTGRES_USER": str(params.get("user", "")),
            "POSTGRES_PASSWORD": str(params.get("password") or "unused-by-trust-auth"),
        }
    )


class StopWorker(Exception):
    pass


async def test_a_notification_sent_before_the_wait_starts_is_not_lost(
    pool, probe, policy, kit, pipeline_database
) -> None:
    emitted: list[dict[str, Any]] = []
    waits: list[float] = []

    def emit(result: dict[str, Any]) -> None:
        emitted.append(result)
        if len(emitted) >= 2:
            raise StopWorker

    class RecordingListener:
        def __init__(self, listener: Any) -> None:
            self.listener = listener

        async def wait(self, timeout: float) -> list[str]:
            if not waits:
                await _seed(pool, kit, APRIL_7, ["2604.00001"])
            started = time.monotonic()
            payloads = await self.listener.wait(timeout)
            waits.append(time.monotonic() - started)
            return payloads

    options = WorkerOptions(skip_embed=True, wait_timeout_seconds=30.0)
    async with jobs.job_listener(_settings(pipeline_database)) as listener:
        with pytest.raises(StopWorker):
            await asyncio.wait_for(
                run_worker_loop(_runtime(pool, policy, kit, None), options, RecordingListener(listener), emit),
                timeout=20,
            )

    assert [result["status"] for result in emitted] == ["no_op", "success"]
    assert waits[0] < 5
    assert await probe.scalar("SELECT status FROM prepare_jobs") == "done"


async def test_the_loop_waits_briefly_after_a_failure_and_not_after_a_success(pool, policy, kit) -> None:
    await _seed(pool, kit, APRIL_7, ["2604.00001"])
    await _seed(pool, kit, APRIL_8, ["2604.00002"])
    parser = kit.FakeParser(failing_ids={"2604.00002"})
    statuses: list[str] = []
    waits: list[float] = []

    def emit(result: dict[str, Any]) -> None:
        statuses.append(result["status"])
        if len(statuses) >= 3:
            raise StopWorker

    class Waiter:
        async def wait(self, timeout: float) -> list[str]:
            waits.append(timeout)
            return []

    options = WorkerOptions(skip_embed=True, wait_timeout_seconds=120.0)
    with pytest.raises(StopWorker):
        await run_worker_loop(_runtime(pool, policy, kit, None, parser), options, Waiter(), emit)

    assert statuses == ["success", "failed", "no_op"]
    assert waits == [30.0]
