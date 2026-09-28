import asyncio
import dataclasses
from datetime import date
from typing import Any

import pytest

from arxplore.db.pool import transaction
from arxplore.db.repositories import jobs
from arxplore.domain.jobs import STALE_EXHAUSTED_ERROR, ClaimToken, JobStatus, QueuePolicy, RequeuedJob

pytestmark = pytest.mark.integration

APRIL_7 = date(2026, 4, 7)
APRIL_8 = date(2026, 4, 8)
APRIL_9 = date(2026, 4, 9)

JOB_COLUMNS = """
    status, attempt_count, claim_generation, worker_id, raw_revision, pending_refresh, error,
    heartbeat_at, claimed_at, next_attempt_at, finished_at, payload, result,
    EXTRACT(EPOCH FROM (next_attempt_at - NOW()))::float8 AS next_attempt_in_seconds
"""


async def _job(probe, job_id: int) -> dict[str, Any]:
    row = await probe.row(f"SELECT {JOB_COLUMNS} FROM prepare_jobs WHERE id = %s", (job_id,))
    assert row is not None
    return row


async def _make_due(probe, job_id: int) -> None:
    await probe.execute(
        "UPDATE prepare_jobs SET next_attempt_at = NOW() - INTERVAL '1 second' WHERE id = %s", (job_id,)
    )


def _with(token: ClaimToken, **changes: Any) -> ClaimToken:
    return dataclasses.replace(token, **changes)


async def _claim(pool, policy: QueuePolicy, worker_id: str = "w1", mode: str = "auto"):
    return await jobs.claim(pool, policy=policy, mode=mode, worker_id=worker_id)


async def test_wrong_claim_token_cannot_complete_or_fail(pool, probe, policy) -> None:
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7, raw_revision=1)).job_id
    claimed = await _claim(pool, policy)
    assert claimed is not None
    assert (claimed.job_id, claimed.claim_generation) == (job_id, 1)
    token = claimed.token

    assert await jobs.complete(pool, _with(token, claim_generation=0)) is False
    assert await jobs.complete(pool, _with(token, worker_id="w2")) is False
    assert await jobs.fail(pool, _with(token, claim_generation=2), policy=policy, error="x") is False
    assert await jobs.heartbeat(pool, _with(token, worker_id="w2")) is False
    assert (await _job(probe, job_id))["status"] == "processing"

    assert await jobs.complete(pool, token, result={"ok": 1}) is True
    assert (await _job(probe, job_id))["status"] == "done"
    assert await jobs.complete(pool, token) is False


async def test_stale_reclaim_fences_out_zombie_worker(pool, probe, policy) -> None:
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7)).job_id
    zombie = await _claim(pool, policy)
    assert zombie is not None
    await probe.execute(
        "UPDATE prepare_jobs SET claimed_at = NOW() - INTERVAL '2 hours', heartbeat_at = NOW() - INTERVAL '1 hour'"
        " WHERE id = %s",
        (job_id,),
    )

    owner = await _claim(pool, policy)
    assert owner is not None
    assert owner.job_id == job_id
    assert owner.claim_generation == zombie.claim_generation + 1

    assert await jobs.heartbeat(pool, zombie.token) is False
    assert await jobs.complete(pool, zombie.token) is False
    assert await jobs.fail(pool, zombie.token, policy=policy, error="late") is False
    assert (await _job(probe, job_id))["status"] == "processing"
    assert await jobs.complete(pool, owner.token) is True
    assert (await _job(probe, job_id))["status"] == "done"


async def test_stale_reset_uses_heartbeat_not_claimed_at(pool, probe, policy) -> None:
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7)).job_id
    claimed = await _claim(pool, policy)
    assert claimed is not None
    await probe.execute("UPDATE prepare_jobs SET claimed_at = NOW() - INTERVAL '2 hours' WHERE id = %s", (job_id,))

    assert await jobs.heartbeat(pool, claimed.token) is True
    assert await jobs.reset_stale(pool, policy=policy, stale_seconds=900) == 0
    assert (await _job(probe, job_id))["status"] == "processing"

    await probe.execute("UPDATE prepare_jobs SET heartbeat_at = NOW() - INTERVAL '20 minutes' WHERE id = %s", (job_id,))
    assert await jobs.reset_stale(pool, policy=policy, stale_seconds=900) == 1
    job = await _job(probe, job_id)
    assert job["status"] == "pending"
    assert job["worker_id"] is None
    assert job["heartbeat_at"] is None
    assert job["attempt_count"] == 1


async def test_stale_reset_falls_back_to_claimed_at_for_legacy_rows(pool, probe, policy) -> None:
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7)).job_id
    await _claim(pool, policy)
    await probe.execute(
        "UPDATE prepare_jobs SET heartbeat_at = NULL, claimed_at = NOW() - INTERVAL '20 minutes' WHERE id = %s",
        (job_id,),
    )

    assert await jobs.reset_stale(pool, policy=policy, stale_seconds=900) == 1
    assert (await _job(probe, job_id))["status"] == "pending"


async def test_stale_reset_closes_job_that_exhausted_attempts(pool, probe, policy) -> None:
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7)).job_id
    await _claim(pool, policy)
    await probe.execute(
        "UPDATE prepare_jobs SET attempt_count = 3, heartbeat_at = NOW() - INTERVAL '1 hour' WHERE id = %s",
        (job_id,),
    )

    assert await jobs.reset_stale(pool, policy=policy, stale_seconds=900) == 1
    job = await _job(probe, job_id)
    assert job["status"] == "failed"
    assert job["error"] == STALE_EXHAUSTED_ERROR
    assert job["finished_at"] is not None
    assert await _claim(pool, policy) is None


async def test_reset_stale_uses_the_policy_threshold_and_skips_non_positive_ones(pool, probe) -> None:
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7)).job_id
    await _claim(pool, QueuePolicy(max_attempts=3, stale_seconds=0))
    await probe.execute("UPDATE prepare_jobs SET heartbeat_at = NOW() - INTERVAL '1 day' WHERE id = %s", (job_id,))

    assert await jobs.reset_stale(pool, policy=QueuePolicy(max_attempts=3, stale_seconds=0)) == 0
    assert await jobs.reset_stale(pool, policy=QueuePolicy(max_attempts=3, stale_seconds=900), stale_seconds=-1) == 0
    assert await jobs.reset_stale(pool, policy=QueuePolicy(max_attempts=3, stale_seconds=90_000)) == 0
    assert (await _job(probe, job_id))["status"] == "processing"
    assert await jobs.reset_stale(pool, policy=QueuePolicy(max_attempts=3, stale_seconds=3600)) == 1


async def test_failures_back_off_then_fail_and_requeue_resets(pool, probe, policy) -> None:
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7)).job_id

    first = await _claim(pool, policy)
    assert first is not None
    assert first.attempt_count == 1
    assert await jobs.fail(pool, first.token, policy=policy, error="boom 1") is True
    job = await _job(probe, job_id)
    assert job["status"] == "pending"
    assert job["error"] == "boom 1"
    assert 55 <= job["next_attempt_in_seconds"] <= 61
    assert await _claim(pool, policy) is None

    await _make_due(probe, job_id)
    second = await _claim(pool, policy)
    assert second is not None
    assert (second.attempt_count, second.claim_generation) == (2, 2)
    assert (await _job(probe, job_id))["next_attempt_at"] is None
    assert await jobs.fail(pool, second.token, policy=policy, error="boom 2") is True
    assert 115 <= (await _job(probe, job_id))["next_attempt_in_seconds"] <= 121

    await _make_due(probe, job_id)
    third = await _claim(pool, policy)
    assert third is not None
    assert third.attempt_count == 3
    assert await jobs.fail(pool, third.token, policy=policy, error="boom 3") is True
    job = await _job(probe, job_id)
    assert job["status"] == "failed"
    assert job["next_attempt_at"] is None
    assert job["finished_at"] is not None
    assert await _claim(pool, policy) is None

    requeued = await jobs.requeue_failed(pool, mode="auto", dry_run=False)
    assert requeued == [RequeuedJob(job_id=job_id, target_date=APRIL_7, attempt_count=3)]
    job = await _job(probe, job_id)
    assert (job["status"], job["attempt_count"], job["next_attempt_at"]) == ("pending", 0, None)
    reclaimed = await _claim(pool, policy)
    assert reclaimed is not None
    assert reclaimed.attempt_count == 1


async def test_recollect_refreshes_done_job_only_for_newer_revision(pool, probe, policy) -> None:
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7, raw_revision=1)).job_id
    first = await _claim(pool, policy)
    assert first is not None
    assert first.raw_revision == 1
    assert await jobs.complete(pool, first.token) is True

    same = await jobs.enqueue(pool, target_date=APRIL_7, raw_revision=1)
    assert same.status == JobStatus.DONE
    assert (await jobs.enqueue(pool, target_date=APRIL_7)).status == JobStatus.DONE
    job = await _job(probe, job_id)
    assert job["worker_id"] == "w1"
    assert job["finished_at"] is not None

    refreshed = await jobs.enqueue(pool, target_date=APRIL_7, raw_revision=2)
    assert (refreshed.job_id, refreshed.status, refreshed.raw_revision) == (job_id, JobStatus.PENDING, 2)
    job = await _job(probe, job_id)
    assert (job["attempt_count"], job["worker_id"], job["finished_at"]) == (0, None, None)


async def test_recollect_during_processing_requeues_once_after_complete(pool, probe, policy) -> None:
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7, raw_revision=1)).job_id
    first = await _claim(pool, policy)
    assert first is not None

    during = await jobs.enqueue(pool, target_date=APRIL_7, raw_revision=2)
    assert (during.status, during.pending_refresh, during.raw_revision) == (JobStatus.PROCESSING, True, 2)
    older = await jobs.enqueue(pool, target_date=APRIL_7, raw_revision=1)
    assert (older.pending_refresh, older.raw_revision) == (True, 2)
    assert await jobs.heartbeat(pool, first.token) is True

    assert await jobs.complete(pool, first.token) is True
    job = await _job(probe, job_id)
    assert (job["status"], job["pending_refresh"], job["attempt_count"]) == ("pending", False, 0)

    second = await _claim(pool, policy)
    assert second is not None
    assert second.claim_generation == 2
    assert await jobs.complete(pool, second.token) is True
    assert (await _job(probe, job_id))["status"] == "done"


async def test_recollect_during_processing_gives_fresh_attempts_on_failure(pool, probe, policy) -> None:
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7, raw_revision=1)).job_id
    await probe.execute("UPDATE prepare_jobs SET attempt_count = 2 WHERE id = %s", (job_id,))
    claimed = await _claim(pool, policy)
    assert claimed is not None
    assert claimed.attempt_count == 3
    await jobs.enqueue(pool, target_date=APRIL_7, raw_revision=2)

    assert await jobs.fail(pool, claimed.token, policy=policy, error="boom") is True
    job = await _job(probe, job_id)
    assert (job["status"], job["attempt_count"], job["next_attempt_at"], job["pending_refresh"]) == (
        "pending",
        0,
        None,
        False,
    )


async def test_enqueue_failed_job_resets_attempts(pool, probe) -> None:
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7)).job_id
    await probe.execute(
        "UPDATE prepare_jobs SET status = 'failed', attempt_count = 3, error = 'x' WHERE id = %s", (job_id,)
    )

    result = await jobs.enqueue(pool, target_date=APRIL_7, raw_revision=1)
    assert result.status == JobStatus.PENDING
    job = await _job(probe, job_id)
    assert (job["attempt_count"], job["error"], job["raw_revision"]) == (0, None, 1)


async def test_requeue_force_sets_payload_flag_and_plain_requeue_clears_it(pool, probe, policy) -> None:
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7, payload={"note": "x"})).job_id
    await probe.execute("UPDATE prepare_jobs SET status = 'failed', attempt_count = 3 WHERE id = %s", (job_id,))

    await jobs.requeue_failed(pool, mode="auto", dry_run=False, force=True)
    claimed = await _claim(pool, policy)
    assert claimed is not None
    assert claimed.payload == {"note": "x", "force": True}
    assert claimed.force_requested is True

    await probe.execute("UPDATE prepare_jobs SET status = 'failed' WHERE id = %s", (job_id,))
    await jobs.requeue_failed(pool, mode="auto", dry_run=False)
    plain = await _claim(pool, policy)
    assert plain is not None
    assert plain.payload == {"note": "x"}
    assert plain.force_requested is False


async def test_collect_enqueue_keeps_requeued_force_until_job_completes(pool, probe, policy) -> None:
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7, payload={"note": "x"})).job_id
    await probe.execute("UPDATE prepare_jobs SET status = 'failed', attempt_count = 3 WHERE id = %s", (job_id,))
    await jobs.requeue_failed(pool, mode="auto", dry_run=False, force=True)

    await jobs.enqueue(pool, target_date=APRIL_7, payload={"collected": 2}, raw_revision=1)
    claimed = await _claim(pool, policy)
    assert claimed is not None
    assert claimed.payload == {"note": "x", "force": True, "collected": 2}

    assert await jobs.complete(
        pool, ClaimToken(job_id=job_id, worker_id="w1", claim_generation=claimed.claim_generation)
    )
    assert (await _job(probe, job_id))["payload"] == {"note": "x", "collected": 2}


async def test_force_survives_retries_and_is_dropped_when_the_job_finally_fails(pool, probe) -> None:
    single_retry = QueuePolicy(max_attempts=2, stale_seconds=900)
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7, payload={"force": True})).job_id

    first = await _claim(pool, single_retry)
    assert first is not None
    assert await jobs.fail(pool, first.token, policy=single_retry, error="boom 1") is True
    assert (await _job(probe, job_id))["payload"] == {"force": True}

    await _make_due(probe, job_id)
    second = await _claim(pool, single_retry)
    assert second is not None
    await probe.execute("UPDATE prepare_jobs SET heartbeat_at = NOW() - INTERVAL '1 hour' WHERE id = %s", (job_id,))
    assert await jobs.reset_stale(pool, policy=QueuePolicy(max_attempts=3, stale_seconds=900)) == 1
    assert (await _job(probe, job_id))["payload"] == {"force": True}

    third = await _claim(pool, single_retry)
    assert third is not None
    assert await jobs.fail(pool, third.token, policy=single_retry, error="boom 3") is True
    job = await _job(probe, job_id)
    assert (job["status"], job["payload"], job["error"]) == ("failed", {}, "boom 3")


async def test_complete_records_the_result_and_clears_the_error(pool, probe, policy) -> None:
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7)).job_id
    first = await _claim(pool, policy)
    assert first is not None
    await jobs.fail(pool, first.token, policy=policy, error="boom")
    await _make_due(probe, job_id)
    second = await _claim(pool, policy)
    assert second is not None

    assert await jobs.complete(pool, second.token, result={"saved_papers": 3, "failures": []}) is True
    job = await _job(probe, job_id)
    assert (job["status"], job["error"], job["result"]) == ("done", None, {"saved_papers": 3, "failures": []})
    assert (job["worker_id"], job["heartbeat_at"], job["next_attempt_at"]) == ("w1", None, None)
    assert job["claimed_at"] is not None
    assert job["finished_at"] is not None


async def test_claim_takes_the_oldest_date_of_its_mode_first(pool, policy) -> None:
    later = await jobs.enqueue(pool, target_date=APRIL_9)
    earlier = await jobs.enqueue(pool, target_date=APRIL_7)
    other_mode = await jobs.enqueue(pool, target_date=date(2026, 4, 1), mode="backfill", source="backfill")

    first = await _claim(pool, policy)
    second = await _claim(pool, policy)
    third = await _claim(pool, policy)
    backfill = await _claim(pool, policy, mode="backfill")

    assert first is not None and second is not None and backfill is not None
    assert (first.job_id, first.target_date) == (earlier.job_id, APRIL_7)
    assert (second.job_id, second.target_date) == (later.job_id, APRIL_9)
    assert third is None
    assert (backfill.job_id, backfill.mode, backfill.source) == (other_mode.job_id, "backfill", "backfill")


async def test_concurrent_claims_skip_locked_jobs(pool, policy) -> None:
    for day in (APRIL_7, APRIL_8, APRIL_9):
        await jobs.enqueue(pool, target_date=day)

    claimed = await asyncio.gather(*(_claim(pool, policy, worker_id=f"w{index}") for index in range(4)))

    taken = sorted(job.target_date for job in claimed if job is not None)
    assert taken == [APRIL_7, APRIL_8, APRIL_9]
    assert sum(job is None for job in claimed) == 1


async def test_claimed_job_carries_the_date_and_a_fencing_token(pool, policy) -> None:
    queued = await jobs.enqueue(pool, target_date=APRIL_7, payload={"note": "x"}, raw_revision=4)
    assert (queued.enqueued, queued.mode, queued.target_date, queued.status) == (True, "auto", APRIL_7, "pending")
    assert (await jobs.enqueue(pool, target_date=APRIL_7)).enqueued is False

    claimed = await _claim(pool, policy, worker_id="gpu-1")

    assert claimed is not None
    assert isinstance(claimed.target_date, date)
    assert (claimed.mode, claimed.source, claimed.status, claimed.raw_revision) == ("auto", "collect", "processing", 4)
    assert claimed.token == ClaimToken(job_id=queued.job_id, worker_id="gpu-1", claim_generation=1)
    assert claimed.claimed_at == claimed.updated_at


async def test_requeue_preview_changes_nothing_and_filters_by_date_and_mode(pool, probe) -> None:
    for day in (APRIL_9, APRIL_7, APRIL_8):
        await jobs.enqueue(pool, target_date=day)
    await jobs.enqueue(pool, target_date=APRIL_8, mode="backfill")
    await probe.execute("UPDATE prepare_jobs SET status = 'failed', attempt_count = 3")

    preview = await jobs.requeue_failed(pool, mode="auto")
    assert [job.target_date for job in preview] == [APRIL_7, APRIL_8, APRIL_9]
    assert await probe.scalar("SELECT count(*) FROM prepare_jobs WHERE status = 'failed'") == 4

    since = await jobs.requeue_failed(pool, mode="auto", since_date=APRIL_8, dry_run=False)
    assert [(job.target_date, job.attempt_count) for job in since] == [(APRIL_8, 3), (APRIL_9, 3)]
    rows = await probe.rows("SELECT mode, target_date, status FROM prepare_jobs ORDER BY mode, target_date")
    assert [(row["mode"], row["target_date"], row["status"]) for row in rows] == [
        ("auto", APRIL_7, "failed"),
        ("auto", APRIL_8, "pending"),
        ("auto", APRIL_9, "pending"),
        ("backfill", APRIL_8, "failed"),
    ]


async def test_transitions_notify_with_mode_and_date(pool, probe, policy, notifications) -> None:
    job_id = (await jobs.enqueue(pool, target_date=APRIL_7, raw_revision=1)).job_id
    assert await notifications.drain() == ["auto:2026-04-07"]

    first = await _claim(pool, policy)
    assert first is not None
    assert await jobs.heartbeat(pool, first.token) is True
    await jobs.enqueue(pool, target_date=APRIL_7, raw_revision=1)
    assert await notifications.drain() == ["auto:2026-04-07"]

    await jobs.fail(pool, first.token, policy=policy, error="backoff")
    assert await notifications.drain() == []

    await _make_due(probe, job_id)
    second = await _claim(pool, policy)
    assert second is not None
    await jobs.enqueue(pool, target_date=APRIL_7, raw_revision=2)
    await notifications.drain()
    await jobs.fail(pool, second.token, policy=policy, error="refresh")
    assert await notifications.drain() == ["auto:2026-04-07"]

    third = await _claim(pool, policy)
    assert third is not None
    await jobs.enqueue(pool, target_date=APRIL_7, raw_revision=3)
    await notifications.drain()
    await jobs.complete(pool, third.token)
    assert await notifications.drain() == ["auto:2026-04-07"]

    fourth = await _claim(pool, policy)
    assert fourth is not None
    await jobs.complete(pool, fourth.token)
    assert await notifications.drain() == []

    for day in (APRIL_9, APRIL_8):
        await jobs.enqueue(pool, target_date=day, mode="m")
    await notifications.drain()
    for worker in ("a", "b"):
        await _claim(pool, policy, worker_id=worker, mode="m")
    await probe.execute(
        "UPDATE prepare_jobs SET heartbeat_at = NOW() - INTERVAL '1 hour',"
        " attempt_count = CASE WHEN target_date = %s THEN 3 ELSE attempt_count END WHERE mode = 'm'",
        (APRIL_9,),
    )
    assert await jobs.reset_stale(pool, policy=policy, mode="m") == 2
    assert await notifications.drain() == ["m:2026-04-08"]

    await probe.execute("UPDATE prepare_jobs SET status = 'failed' WHERE mode = 'm'")
    await jobs.requeue_failed(pool, mode="m")
    assert await notifications.drain() == []
    await jobs.requeue_failed(pool, mode="m", dry_run=False)
    assert await notifications.drain() == ["m:2026-04-08", "m:2026-04-09"]


async def test_enqueue_in_a_caller_transaction_notifies_only_on_commit(pool, probe, notifications) -> None:
    async with transaction(pool) as connection:
        await jobs.enqueue(connection, target_date=APRIL_7)
        assert await notifications.drain() == []
        assert await probe.scalar("SELECT count(*) FROM prepare_jobs") == 0
    assert await notifications.drain() == ["auto:2026-04-07"]

    with pytest.raises(RuntimeError, match="abort"):
        async with transaction(pool) as connection:
            await jobs.enqueue(connection, target_date=APRIL_8)
            raise RuntimeError("abort")
    assert await notifications.drain() == []
    assert await probe.scalar("SELECT count(*) FROM prepare_jobs") == 1
