from datetime import date
from typing import Any

import pytest

from arxplore.db.repositories import jobs, raw
from arxplore.pipeline.prepare import run as prepare_run
from arxplore.pipeline.prepare.paper import PrepareContext
from arxplore.pipeline.prepare.run import consume_prepare_queue

pytestmark = pytest.mark.integration

APRIL_7 = date(2026, 4, 7)
APRIL_8 = date(2026, 4, 8)


def _context(pool: Any, kit: Any, parser: Any = None) -> PrepareContext:
    return PrepareContext(db=pool, pdfs=kit.FakePdfSource(), parser=parser or kit.FakeParser())


async def _seed(pool: Any, kit: Any, target_date: date, arxiv_ids: list[str], **payload: Any) -> int:
    saved = await raw.save_daily_papers(pool, target_date=target_date, payload=[kit.hf_item(i) for i in arxiv_ids])
    queued = await jobs.enqueue(pool, target_date=target_date, raw_revision=saved.revision, payload=payload or None)
    return queued.job_id


async def _job(probe: Any, job_id: int) -> dict[str, Any]:
    row = await probe.row(
        "SELECT status, attempt_count, error, result, payload, next_attempt_at FROM prepare_jobs WHERE id = %s",
        (job_id,),
    )
    assert row is not None
    return row


async def test_a_clean_date_completes_the_job_with_the_legacy_result(pool, probe, policy, kit) -> None:
    job_id = await _seed(pool, kit, APRIL_7, ["2604.00001", "2604.00002"])

    run = await consume_prepare_queue(_context(pool, kit), pool, policy=policy, worker_id="w1")

    assert run.status == "success"
    assert run.claimed_dates == ["2026-04-07"]
    job = await _job(probe, job_id)
    assert job["status"] == "done"
    result = job["result"]
    assert result["success_count"] == 2
    assert result["failure_count"] == 0
    assert result["failures"] == []
    assert result["prepared_arxiv_count"] == 2
    assert result["saved_papers"] == 2
    assert result["saved_fulltexts"] == 2
    assert result["saved_chunks"] > 0
    success = run.as_dict()["successes"][0]
    assert success["date"] == "2026-04-07"
    assert success["prepared_arxiv_ids"] == ["2604.00001", "2604.00002"]
    assert (success["paper_success_count"], success["paper_failure_count"], success["paper_failures"]) == (2, 0, [])


async def test_a_paper_failure_fails_the_job_for_a_backoff_retry(pool, probe, policy, kit) -> None:
    job_id = await _seed(pool, kit, APRIL_7, ["2604.00001", "2604.00002", "2604.00003"])
    parser = kit.FakeParser(failing_ids={"2604.00002"})

    run = await consume_prepare_queue(_context(pool, kit, parser), pool, policy=policy, worker_id="w1")

    assert run.status == "failed"
    job = await _job(probe, job_id)
    assert (job["status"], job["attempt_count"], job["result"]) == ("pending", 1, {})
    assert job["next_attempt_at"] is not None
    assert job["error"] == (
        "1 of 3 paper(s) failed (success=2, failure=1); first error: RuntimeError: parser exploded on 2604.00002"
    )
    assert run.failures == [
        {
            "date": "2026-04-07",
            "mode": "auto",
            "error": job["error"],
            "prepared_arxiv_ids": ["2604.00001", "2604.00003"],
        }
    ]
    assert run.prepared_arxiv_ids() == ["2604.00001", "2604.00003"]


async def test_all_papers_failing_is_reported_as_all_failed(pool, probe, policy, kit) -> None:
    job_id = await _seed(pool, kit, APRIL_7, ["2604.00001", "2604.00002"])
    parser = kit.FakeParser(failing_ids={"2604.00001", "2604.00002"})

    run = await consume_prepare_queue(_context(pool, kit, parser), pool, policy=policy, worker_id="w1")

    assert run.status == "failed"
    assert (await _job(probe, job_id))["error"].startswith("all 2 paper(s) failed (success=0, failure=2)")


async def test_a_date_level_error_fails_the_job_with_the_message(pool, probe, policy, kit, monkeypatch) -> None:
    job_id = await _seed(pool, kit, APRIL_7, ["2604.00001"])

    async def broken_load(db: Any, *, target_date: date) -> list[Any]:
        raise RuntimeError("raw store down")

    monkeypatch.setattr(raw, "load_daily_papers", broken_load)

    run = await consume_prepare_queue(_context(pool, kit), pool, policy=policy, worker_id="w1")

    assert run.status == "failed"
    assert run.failures == [{"date": "2026-04-07", "mode": "auto", "error": "raw store down"}]
    assert (await _job(probe, job_id))["error"] == "raw store down"


@pytest.mark.parametrize(
    "payload,argument,expected",
    [({}, False, False), ({"force": True}, False, True), ({"force": "yes"}, False, False), ({}, True, True)],
)
async def test_force_comes_from_the_payload_or_the_argument(
    pool, policy, kit, monkeypatch, payload, argument, expected
) -> None:
    await _seed(pool, kit, APRIL_7, ["2604.00001"], **payload)
    forced: list[bool] = []
    original = prepare_run.prepare_paper

    async def tracking(candidate: Any, context: Any, *, force: bool = False) -> Any:
        forced.append(force)
        return await original(candidate, context, force=force)

    monkeypatch.setattr(prepare_run, "prepare_paper", tracking)

    run = await consume_prepare_queue(_context(pool, kit), pool, policy=policy, worker_id="w1", force=argument)

    assert run.status == "success"
    assert forced == [expected]


async def test_completing_drops_the_force_flag(pool, probe, policy, kit) -> None:
    job_id = await _seed(pool, kit, APRIL_7, ["2604.00001"], force=True)

    await consume_prepare_queue(_context(pool, kit), pool, policy=policy, worker_id="w1")

    assert (await _job(probe, job_id))["payload"] == {}


async def test_each_paper_is_preceded_by_a_heartbeat_with_the_claim_token(pool, policy, kit, monkeypatch) -> None:
    await _seed(pool, kit, APRIL_7, ["2604.00001", "2604.00002", "2604.00003"])
    tokens: list[Any] = []
    original = jobs.heartbeat

    async def tracking(db: Any, token: Any) -> bool:
        tokens.append(token)
        return await original(db, token)

    monkeypatch.setattr(jobs, "heartbeat", tracking)

    run = await consume_prepare_queue(_context(pool, kit), pool, policy=policy, worker_id="w1")

    assert run.status == "success"
    assert len(tokens) == 3
    assert {(token.worker_id, token.claim_generation) for token in tokens} == {("w1", 1)}


async def test_a_lost_claim_at_heartbeat_stops_before_any_paper(pool, probe, policy, kit, monkeypatch) -> None:
    job_id = await _seed(pool, kit, APRIL_7, ["2604.00001", "2604.00002"])
    parser = kit.FakeParser()
    original = jobs.heartbeat

    async def steal(db: Any, token: Any) -> bool:
        await probe.execute("UPDATE prepare_jobs SET claim_generation = claim_generation + 1 WHERE id = %s", (job_id,))
        return await original(db, token)

    monkeypatch.setattr(jobs, "heartbeat", steal)

    run = await consume_prepare_queue(_context(pool, kit, parser), pool, policy=policy, worker_id="w1")

    assert parser.parsed == []
    assert run.status == "claim_lost"
    assert run.lost_claims == [
        {"date": "2026-04-07", "mode": "auto", "job_id": job_id, "claim_generation": 1, "stage": "heartbeat"}
    ]
    assert (await _job(probe, job_id))["status"] == "processing"


async def test_heartbeat_errors_are_tolerated(pool, policy, kit, monkeypatch) -> None:
    await _seed(pool, kit, APRIL_7, ["2604.00001"])

    async def broken(db: Any, token: Any) -> bool:
        raise ConnectionError("db blip")

    monkeypatch.setattr(jobs, "heartbeat", broken)

    run = await consume_prepare_queue(_context(pool, kit), pool, policy=policy, worker_id="w1")

    assert run.status == "success"


async def test_a_claim_lost_before_complete_is_recorded_not_raised(pool, probe, policy, kit, monkeypatch) -> None:
    job_id = await _seed(pool, kit, APRIL_7, ["2604.00001"])
    original = jobs.complete

    async def stale_complete(db: Any, token: Any, *, result: Any = None) -> bool:
        await probe.execute("UPDATE prepare_jobs SET claim_generation = claim_generation + 1 WHERE id = %s", (job_id,))
        return await original(db, token, result=result)

    monkeypatch.setattr(jobs, "complete", stale_complete)

    run = await consume_prepare_queue(_context(pool, kit), pool, policy=policy, worker_id="w1")

    assert run.status == "claim_lost"
    assert run.successes == []
    assert run.lost_claims[0]["stage"] == "complete"


async def test_a_claim_lost_on_fail_still_reports_the_failure(pool, probe, policy, kit, monkeypatch) -> None:
    job_id = await _seed(pool, kit, APRIL_7, ["2604.00001"])
    original = jobs.fail

    async def stale_fail(db: Any, token: Any, *, policy: Any, error: str) -> bool:
        await probe.execute("UPDATE prepare_jobs SET claim_generation = claim_generation + 1 WHERE id = %s", (job_id,))
        return await original(db, token, policy=policy, error=error)

    monkeypatch.setattr(jobs, "fail", stale_fail)

    run = await consume_prepare_queue(
        _context(pool, kit, kit.FakeParser(failing_ids={"2604.00001"})), pool, policy=policy, worker_id="w1"
    )

    assert run.status == "failed"
    assert run.lost_claims[0]["stage"] == "fail"


async def test_a_lost_claim_next_to_a_success_is_partial(pool, probe, policy, kit, monkeypatch) -> None:
    first = await _seed(pool, kit, APRIL_7, ["2604.00001"])
    await _seed(pool, kit, APRIL_8, ["2604.00002"])
    original = jobs.complete

    async def lose_first(db: Any, token: Any, *, result: Any = None) -> bool:
        if token.job_id == first:
            await probe.execute("UPDATE prepare_jobs SET claim_generation = 99 WHERE id = %s", (first,))
        return await original(db, token, result=result)

    monkeypatch.setattr(jobs, "complete", lose_first)

    run = await consume_prepare_queue(_context(pool, kit), pool, policy=policy, worker_id="w1", max_jobs_per_run=2)

    assert run.status == "partial_failed"
    assert [item["date"] for item in run.successes] == ["2026-04-08"]
    assert [item["date"] for item in run.lost_claims] == ["2026-04-07"]


async def test_an_empty_queue_is_a_no_op(pool, policy, kit) -> None:
    run = await consume_prepare_queue(_context(pool, kit), pool, policy=policy, worker_id="w1", max_jobs_per_run=3)
    assert run.status == "no_op"
    assert run.as_dict()["claimed_count"] == 0


async def test_jobs_are_claimed_oldest_date_first_up_to_the_limit(pool, policy, kit) -> None:
    await _seed(pool, kit, APRIL_8, ["2604.00002"])
    await _seed(pool, kit, APRIL_7, ["2604.00001"])

    run = await consume_prepare_queue(_context(pool, kit), pool, policy=policy, worker_id="w1", max_jobs_per_run=1)

    assert run.claimed_dates == ["2026-04-07"]


async def _seed_backfill(pool: Any, kit: Any, target_date: date, arxiv_ids: list[str]) -> int:
    saved = await raw.save_daily_papers(pool, target_date=target_date, payload=[kit.hf_item(i) for i in arxiv_ids])
    queued = await jobs.enqueue(
        pool, target_date=target_date, mode="backfill", source="backfill", raw_revision=saved.revision
    )
    return queued.job_id


async def test_backfill_jobs_wait_unless_their_mode_is_consumed(pool, probe, policy, kit) -> None:
    job_id = await _seed_backfill(pool, kit, APRIL_7, ["2604.00001"])

    run = await consume_prepare_queue(_context(pool, kit), pool, policy=policy, worker_id="w1")

    assert run.status == "no_op"
    assert (await _job(probe, job_id))["status"] == "pending"


async def test_auto_jobs_are_claimed_before_older_backfill_jobs(pool, probe, policy, kit) -> None:
    backfill_id = await _seed_backfill(pool, kit, APRIL_7, ["2604.00001"])
    auto_id = await _seed(pool, kit, APRIL_8, ["2604.00002"])

    run = await consume_prepare_queue(
        _context(pool, kit), pool, policy=policy, worker_id="w1", modes=("auto", "backfill"), max_jobs_per_run=2
    )

    assert run.status == "success"
    assert [(entry["date"], entry["mode"]) for entry in run.successes] == [
        ("2026-04-08", "auto"),
        ("2026-04-07", "backfill"),
    ]
    assert run.as_dict()["modes"] == ["auto", "backfill"]
    assert (await _job(probe, auto_id))["status"] == "done"
    assert (await _job(probe, backfill_id))["status"] == "done"
