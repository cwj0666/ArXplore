from collections.abc import AsyncIterator
from contextlib import aclosing
from datetime import date
from typing import Any

import psycopg
import pytest
from psycopg import sql

from arxplore.db.repositories import embeddings, jobs, raw
from arxplore.domain.chunks import ChunkEmbedding
from arxplore.pipeline.corpus import rechunk_corpus, reparse_dates
from arxplore.pipeline.prepare.mapping import select_candidates
from arxplore.pipeline.prepare.paper import PrepareContext, prepare_paper
from arxplore.pipeline.prepare.run import consume_prepare_queue

pytestmark = pytest.mark.integration

DAY = date(2026, 9, 15)
VECTOR = [1.0] + [0.0] * 1535


async def _prepare(pool: Any, kit: Any, arxiv_id: str) -> None:
    candidate = select_candidates(DAY, [kit.hf_item(arxiv_id)]).candidates[0]
    await prepare_paper(candidate, PrepareContext(db=pool, pdfs=kit.FakePdfSource(), parser=kit.FakeParser()))


async def _embed_all(pool: Any) -> None:
    missing = await embeddings.list_chunks_missing_embeddings(pool, limit=1000)
    await embeddings.upsert_embeddings(
        pool, [ChunkEmbedding(chunk_id=chunk.chunk_id, embedding=VECTOR, model_name="m") for chunk in missing]
    )


async def _chunks(probe: Any, arxiv_id: str) -> list[dict[str, Any]]:
    return await probe.rows(
        "SELECT id, chunk_index, chunk_text FROM paper_chunks WHERE arxiv_id = %s ORDER BY chunk_index", (arxiv_id,)
    )


async def _embedded(probe: Any, arxiv_id: str) -> int:
    return int(
        await probe.scalar(
            "SELECT count(*) FROM paper_embeddings e JOIN paper_chunks c ON c.id = e.chunk_id WHERE c.arxiv_id = %s",
            (arxiv_id,),
        )
    )


@pytest.fixture
async def outdated_corpus(pool: Any, probe: Any, kit: Any) -> dict[str, Any]:
    await _prepare(pool, kit, "2609.00001")
    await _prepare(pool, kit, "2609.00002")
    await probe.execute(
        "UPDATE paper_chunks SET chunk_text = 'old chunker ' || chunk_text WHERE arxiv_id = '2609.00001'"
    )
    await _embed_all(pool)
    await probe.execute(
        "INSERT INTO papers (arxiv_id, title) VALUES ('2609.00003', 'No fulltext yet')",
    )
    return {
        "stale": await _chunks(probe, "2609.00001"),
        "current": await _chunks(probe, "2609.00002"),
        "embedded": await _embedded(probe, "2609.00001"),
    }


async def test_rechunk_dry_run_reports_without_writing(pool, probe, outdated_corpus) -> None:
    result = (await rechunk_corpus(pool)).as_dict()

    assert result["applied"] is False
    assert (result["paper_count"], result["without_fulltext_count"], result["changed_paper_count"]) == (2, 1, 1)
    assert result["changed_arxiv_ids"] == ["2609.00001"]
    assert result["embeddings_dropped"] == outdated_corpus["embedded"] > 0
    assert result["estimated_embedding_tokens"] > 0
    assert await _chunks(probe, "2609.00001") == outdated_corpus["stale"]
    assert await _embedded(probe, "2609.00001") == outdated_corpus["embedded"]


async def test_rechunk_apply_replaces_only_changed_chunks(pool, probe, outdated_corpus) -> None:
    hash_before = await probe.scalar("SELECT content_hash FROM paper_fulltexts WHERE arxiv_id = '2609.00001'")

    result = (await rechunk_corpus(pool, apply=True)).as_dict()

    assert result["applied"] is True
    rebuilt = await _chunks(probe, "2609.00001")
    assert not any(row["chunk_text"].startswith("old chunker") for row in rebuilt)
    assert {row["id"] for row in rebuilt}.isdisjoint({row["id"] for row in outdated_corpus["stale"]})
    assert await _embedded(probe, "2609.00001") == 0
    assert await _chunks(probe, "2609.00002") == outdated_corpus["current"]
    assert await _embedded(probe, "2609.00002") > 0
    fulltext = await probe.row(
        "SELECT content_hash, quality_metrics FROM paper_fulltexts WHERE arxiv_id = '2609.00001'"
    )
    assert fulltext is not None
    assert fulltext["content_hash"] == hash_before
    assert fulltext["quality_metrics"]["chunk_count"] == len(rebuilt)
    assert (await rechunk_corpus(pool)).as_dict()["changed_paper_count"] == 0


async def test_rechunk_can_target_papers(pool, outdated_corpus) -> None:
    result = (await rechunk_corpus(pool, arxiv_ids=["2609.00002", "2609.09999"])).as_dict()
    assert (result["paper_count"], result["changed_paper_count"], result["without_fulltext_count"]) == (1, 0, 1)


async def _job_rows(probe: Any) -> dict[date, dict[str, Any]]:
    rows = await probe.rows("SELECT target_date, status, payload, attempt_count, source, error FROM prepare_jobs")
    return {row["target_date"]: row for row in rows}


@pytest.fixture
async def dated_jobs(pool: Any, probe: Any) -> dict[str, date]:
    days = {name: date(2026, 9, number) for number, name in enumerate(["done", "failed", "missing", "busy"], 10)}
    for day in days.values():
        await raw.save_daily_papers(pool, target_date=day, payload=[])
    for name in ("done", "failed", "busy"):
        await jobs.enqueue(pool, target_date=days[name], raw_revision=1)
    await probe.execute(
        "UPDATE prepare_jobs SET status = 'done', finished_at = NOW() WHERE target_date = %s", (days["done"],)
    )
    await probe.execute(
        "UPDATE prepare_jobs SET status = 'failed', attempt_count = 3, error = 'boom' WHERE target_date = %s",
        (days["failed"],),
    )
    await probe.execute(
        "UPDATE prepare_jobs SET status = 'processing', worker_id = 'w9' WHERE target_date = %s", (days["busy"],)
    )
    return days


@pytest.fixture
async def listener(pipeline_database: str) -> AsyncIterator[psycopg.AsyncConnection[Any]]:
    async with await psycopg.AsyncConnection.connect(pipeline_database, autocommit=True) as connection:
        await connection.execute(sql.SQL("LISTEN {}").format(sql.Identifier(jobs.JOB_CHANNEL)))
        yield connection


async def _notifications(connection: psycopg.AsyncConnection[Any]) -> list[str]:
    async with aclosing(connection.notifies(timeout=1.0)) as stream:
        return sorted([notify.payload async for notify in stream])


async def test_reparse_dry_run_plans_each_collected_date(pool, probe, dated_jobs) -> None:
    before = await _job_rows(probe)

    result = (await reparse_dates(pool)).as_dict()

    assert result["applied"] is False
    assert [(entry["date"], entry["action"]) for entry in result["dates"]] == [
        ("2026-09-10", "reopen"),
        ("2026-09-11", "reopen"),
        ("2026-09-12", "enqueue"),
        ("2026-09-13", "skip_processing"),
    ]
    assert (result["reopened"], result["enqueued"], result["skipped_processing"]) == (2, 1, 1)
    assert await _job_rows(probe) == before


async def test_reparse_apply_reopens_and_enqueues_with_force(pool, probe, dated_jobs, listener) -> None:
    await reparse_dates(pool, apply=True)

    rows = await _job_rows(probe)
    for name in ("done", "failed", "missing"):
        row = rows[dated_jobs[name]]
        assert (row["status"], row["payload"], row["attempt_count"], row["error"]) == (
            "pending",
            {"force": True},
            0,
            None,
        ), name
    assert rows[dated_jobs["missing"]]["source"] == "reparse"
    assert rows[dated_jobs["busy"]]["status"] == "processing"
    assert rows[dated_jobs["busy"]]["payload"] == {}
    assert await _notifications(listener) == ["auto:2026-09-10", "auto:2026-09-11", "auto:2026-09-12"]


async def test_reparse_without_force_clears_the_flag(pool, probe, dated_jobs) -> None:
    await probe.execute("UPDATE prepare_jobs SET payload = '{\"force\": true}'::jsonb")

    await reparse_dates(pool, force=False, apply=True)

    rows = await _job_rows(probe)
    assert rows[dated_jobs["done"]]["payload"] == {}
    assert rows[dated_jobs["missing"]]["payload"] == {}


async def test_reparse_limits_dates_and_modes(pool, probe, dated_jobs) -> None:
    result = (await reparse_dates(pool, since=date(2026, 9, 11), until=date(2026, 9, 12))).as_dict()
    assert [entry["date"] for entry in result["dates"]] == ["2026-09-11", "2026-09-12"]

    backfill = (await reparse_dates(pool, mode="backfill", until=date(2026, 9, 10), apply=True)).as_dict()
    assert backfill["dates"] == [{"date": "2026-09-10", "job_id": None, "status": None, "action": "enqueue"}]
    assert await probe.scalar("SELECT count(*) FROM prepare_jobs WHERE mode = 'backfill'") == 1


async def test_a_forced_reparse_parses_final_papers_again(pool, probe, policy, kit) -> None:
    saved = await raw.save_daily_papers(pool, target_date=DAY, payload=[kit.hf_item("2609.00001")])
    await jobs.enqueue(pool, target_date=DAY, raw_revision=saved.revision)
    parser = kit.FakeParser()
    context = PrepareContext(db=pool, pdfs=kit.FakePdfSource(), parser=parser)
    await consume_prepare_queue(context, pool, policy=policy, worker_id="w1")
    await consume_prepare_queue(context, pool, policy=policy, worker_id="w1")
    assert len(parser.parsed) == 1

    await reparse_dates(pool, apply=True)
    run = await consume_prepare_queue(context, pool, policy=policy, worker_id="w1")

    assert run.status == "success"
    assert len(parser.parsed) == 2
    assert await probe.scalar("SELECT payload FROM prepare_jobs") == {}
