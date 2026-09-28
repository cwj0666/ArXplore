import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import psycopg
import pytest

from arxplore.config import AppSettings
from arxplore.db.repositories import scheduler_runs
from arxplore.db.repositories.scheduler_runs import RunStatus, RunTrigger
from arxplore.scheduler import jobs as scheduler_jobs
from arxplore.scheduler.jobs import (
    COLLECT_JOB,
    COLLECT_RETRY,
    MAINTENANCE_JOB,
    NO_RETRY,
    CollectParams,
    JobDefinition,
    JobResources,
    MaintenanceParams,
    RetryPolicy,
    job_definition,
)
from arxplore.scheduler.runner import Scheduler, SchedulerLockedError, run_manual, run_scheduled
from arxplore.sources.arxiv import ArxivClient

pytestmark = pytest.mark.integration

SEOUL = ZoneInfo("Asia/Seoul")
START = datetime(2026, 9, 29, 10, 5, tzinfo=SEOUL)


class Stop(BaseException):
    pass


@dataclass
class FakeDailyPapers:
    requested: list[date] = field(default_factory=list)

    async def fetch_daily_papers(self, target_date: date) -> list[Any]:
        self.requested.append(target_date)
        return [{"paper": {"id": "2609.00001"}}]


def _resources(pool: Any, source: Any = None) -> JobResources:
    return JobResources(
        settings=AppSettings.model_validate({}),
        pool=pool,
        hf=source or FakeDailyPapers(),
        arxiv=ArxivClient(base_url="http://127.0.0.1:9/never", timeout_seconds=1, delay_seconds=0),
        today=lambda: date(2026, 9, 29),
    )


def _definition(name: str, run: Any, *, cron: str = "0 18 * * *") -> JobDefinition:
    return JobDefinition(name=name, cron=cron, run=run, scheduled_params=lambda when: CollectParams())


async def _runs(probe: Any) -> list[dict[str, Any]]:
    return await probe.rows(
        "SELECT job, trigger, scheduled_for, status, params, result, error, attempts, finished_at IS NOT NULL AS done"
        " FROM scheduler_runs ORDER BY id"
    )


async def test_a_manual_collect_is_recorded_with_its_params_and_result(pool, probe) -> None:
    source = FakeDailyPapers()
    params = CollectParams(target_date=date(2026, 9, 15))

    outcome = await run_manual(_resources(pool, source), job_definition(COLLECT_JOB), params)

    assert outcome.status is RunStatus.SUCCESS
    assert source.requested == [date(2026, 9, 15)]
    [run] = await _runs(probe)
    assert run["job"] == "collect"
    assert (run["trigger"], run["scheduled_for"], run["status"], run["attempts"], run["done"]) == (
        "manual",
        None,
        "success",
        1,
        True,
    )
    assert run["params"] == {"target_date": "2026-09-15"}
    assert run["result"]["target_date"] == "2026-09-15"
    assert run["result"]["prepare_job_enqueued"] is True
    assert await probe.scalar("SELECT mode FROM prepare_jobs") == "auto"


async def test_a_manual_collect_without_a_date_uses_today(pool) -> None:
    source = FakeDailyPapers()
    await run_manual(_resources(pool, source), job_definition(COLLECT_JOB), CollectParams())
    assert source.requested == [date(2026, 9, 29)]


async def test_a_failing_job_is_recorded_as_failed(pool, probe) -> None:
    async def explode(resources: JobResources, params: Any) -> dict[str, Any]:
        raise RuntimeError("HF is down")

    outcome = await run_manual(_resources(pool), _definition("collect", explode), CollectParams())

    assert (outcome.status, outcome.error, outcome.attempts) == (RunStatus.FAILED, "RuntimeError: HF is down", 1)
    [run] = await _runs(probe)
    assert (run["status"], run["error"], run["result"]) == ("failed", "RuntimeError: HF is down", None)


async def test_maintenance_enriches_even_when_the_backfill_fails(pool, probe, monkeypatch) -> None:
    async def broken_backfill(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("state table locked")

    enriched: list[int] = []

    async def fake_enrich(pool: Any, arxiv: Any, *, batch_size: int, max_papers: int) -> Any:
        enriched.append(max_papers)

        class Result:
            def as_dict(self) -> dict[str, Any]:
                return {"stage": "enrich_papers_metadata", "status": "success"}

        return Result()

    monkeypatch.setattr(scheduler_jobs, "backfill_daily_papers", broken_backfill)
    monkeypatch.setattr(scheduler_jobs, "enrich_papers_metadata", fake_enrich)

    outcome = await run_manual(_resources(pool), job_definition(MAINTENANCE_JOB), MaintenanceParams())

    assert enriched == [30]
    assert outcome.status is RunStatus.FAILED
    assert outcome.error == "JobFailedError: backfill: RuntimeError: state table locked"
    assert outcome.result == {
        "backfill": {"status": "error", "error": "RuntimeError: state table locked"},
        "enrich": {"stage": "enrich_papers_metadata", "status": "success"},
    }
    [run] = await _runs(probe)
    assert run["params"] == {
        "backfill_cursor_date": "",
        "backfill_oldest_date": "",
        "backfill_batch_days": 30,
        "backfill_state_name": "default",
        "enrich_max_papers": 30,
    }
    assert run["result"] == outcome.result


async def test_a_scheduled_slot_runs_at_most_once(pool, probe) -> None:
    calls: list[Any] = []

    async def record(resources: JobResources, params: Any) -> dict[str, Any]:
        calls.append(params)
        return {"ok": True}

    definition = _definition("collect", record)
    slot = datetime(2026, 9, 28, 18, 0, tzinfo=SEOUL)

    first = await run_scheduled(_resources(pool), definition, slot)
    second = await run_scheduled(_resources(pool), definition, slot)

    assert first is not None and first.status is RunStatus.SUCCESS
    assert second is None
    assert len(calls) == 1
    [run] = await _runs(probe)
    assert (run["trigger"], run["scheduled_for"]) == ("scheduled", slot)


async def test_the_scheduled_collect_uses_the_date_of_its_slot(pool) -> None:
    source = FakeDailyPapers()
    slot = datetime(2026, 9, 28, 18, 0, tzinfo=SEOUL)

    await run_scheduled(_resources(pool, source), job_definition(COLLECT_JOB), slot)

    assert source.requested == [date(2026, 9, 28)]


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += timedelta(seconds=seconds)
        await asyncio.sleep(0)


def _scheduler(pool: Any, pipeline_database: str, definitions: list[JobDefinition], clock: FakeClock) -> Scheduler:
    async def connect_lock() -> psycopg.AsyncConnection[Any]:
        return await psycopg.AsyncConnection.connect(pipeline_database, autocommit=True)

    return Scheduler(
        _resources(pool),
        definitions,
        timezone=SEOUL,
        connect_lock=connect_lock,
        clock=clock,
        sleep=clock.sleep,
    )


async def test_the_scheduler_catches_up_once_then_follows_the_cron(pool, probe, pipeline_database) -> None:
    clock = FakeClock(START)
    fired: list[datetime] = []

    async def record(resources: JobResources, params: Any) -> dict[str, Any]:
        fired.append(clock.now)
        if len(fired) >= 3:
            raise Stop
        return {"ok": True}

    scheduler = _scheduler(pool, pipeline_database, [_definition("collect", record)], clock)

    with pytest.raises(Stop):
        await scheduler.run()

    assert fired == [START, datetime(2026, 9, 29, 18, 0, tzinfo=SEOUL), datetime(2026, 9, 30, 18, 0, tzinfo=SEOUL)]
    runs = await _runs(probe)
    assert [run["scheduled_for"] for run in runs] == [
        datetime(2026, 9, 28, 18, 0, tzinfo=SEOUL),
        datetime(2026, 9, 29, 18, 0, tzinfo=SEOUL),
        datetime(2026, 9, 30, 18, 0, tzinfo=SEOUL),
    ]
    assert [run["status"] for run in runs] == ["success", "success", "running"]
    assert max(clock.sleeps) <= 60


async def test_no_catch_up_when_the_previous_slot_already_ran(pool, probe, pipeline_database) -> None:
    await scheduler_runs.start_scheduled_run(
        pool, job="collect", scheduled_for=datetime(2026, 9, 28, 18, 0, tzinfo=SEOUL), params={}
    )
    clock = FakeClock(START)
    fired: list[datetime] = []

    async def record(resources: JobResources, params: Any) -> dict[str, Any]:
        fired.append(clock.now)
        raise Stop

    with pytest.raises(Stop):
        await _scheduler(pool, pipeline_database, [_definition("collect", record)], clock).run()

    assert fired == [datetime(2026, 9, 29, 18, 0, tzinfo=SEOUL)]


async def test_unfinished_scheduled_runs_are_marked_abandoned_at_start(pool, probe, pipeline_database) -> None:
    await scheduler_runs.start_scheduled_run(
        pool, job="collect", scheduled_for=datetime(2026, 9, 28, 18, 0, tzinfo=SEOUL), params={}
    )
    manual = await scheduler_runs.start_manual_run(pool, job="collect", params={})

    async def stop(resources: JobResources, params: Any) -> dict[str, Any]:
        raise Stop

    with pytest.raises(Stop):
        await _scheduler(pool, pipeline_database, [_definition("collect", stop)], FakeClock(START)).run()

    rows = await probe.rows("SELECT id, trigger, status, error FROM scheduler_runs ORDER BY id")
    assert rows[0]["status"] == "abandoned"
    assert rows[0]["error"] == scheduler_runs.ABANDONED_ERROR
    assert [row["status"] for row in rows if row["id"] == manual] == ["running"]


async def test_a_job_error_does_not_stop_the_scheduler(pool, probe, pipeline_database) -> None:
    clock = FakeClock(START)
    attempts: list[datetime] = []

    async def flaky(resources: JobResources, params: Any) -> dict[str, Any]:
        attempts.append(clock.now)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        raise Stop

    with pytest.raises(Stop):
        await _scheduler(pool, pipeline_database, [_definition("collect", flaky)], clock).run()

    assert len(attempts) == 2
    assert [run["status"] for run in await _runs(probe)] == ["failed", "running"]


async def test_only_one_scheduler_holds_the_lock(pool, pipeline_database) -> None:
    clock = FakeClock(START)
    first = _scheduler(pool, pipeline_database, [], clock)
    second = _scheduler(pool, pipeline_database, [], clock)

    await first.acquire_lock()
    try:
        with pytest.raises(SchedulerLockedError):
            await second.acquire_lock()
    finally:
        await first.release_lock()

    await second.acquire_lock()
    await second.release_lock()


async def test_history_lists_recent_runs_newest_first(pool) -> None:
    for job in ("collect", "maintenance", "collect"):
        run_id = await scheduler_runs.start_manual_run(pool, job=job, params={"n": 1})
        await scheduler_runs.finish_run(
            pool, run_id, status=RunStatus.SUCCESS, result={"ok": True}, error=None, attempts=1
        )

    runs = await scheduler_runs.list_recent_runs(pool, job="collect", limit=5)

    assert [run.run_id for run in runs] == [3, 1]
    assert runs[0].trigger is RunTrigger.MANUAL
    assert runs[0].status is RunStatus.SUCCESS
    assert runs[0].result == {"ok": True}
    assert len(await scheduler_runs.list_recent_runs(pool, limit=2)) == 2


async def test_a_failing_job_is_retried_until_it_succeeds(pool, probe) -> None:
    sleeps: list[float] = []
    seen_during_retry: list[dict[str, Any] | None] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        seen_during_retry.append(await probe.row("SELECT status, attempts, error FROM scheduler_runs"))

    calls = {"count": 0}

    async def flaky(resources: JobResources, params: Any) -> dict[str, Any]:
        calls["count"] += 1
        if calls["count"] < 3:
            raise RuntimeError(f"HF is down {calls['count']}")
        return {"ok": True}

    outcome = await run_manual(
        _resources(pool),
        _definition("collect", flaky),
        CollectParams(),
        retry=RetryPolicy(retries=3, delay_seconds=5.0),
        sleep=fake_sleep,
    )

    assert (outcome.status, outcome.attempts, outcome.error) == (RunStatus.SUCCESS, 3, None)
    assert sleeps == [5.0, 5.0]
    assert seen_during_retry == [
        {"status": "running", "attempts": 1, "error": "RuntimeError: HF is down 1"},
        {"status": "running", "attempts": 2, "error": "RuntimeError: HF is down 2"},
    ]
    [run] = await _runs(probe)
    assert (run["status"], run["attempts"], run["error"]) == ("success", 3, None)


async def test_a_job_fails_when_the_retries_run_out(pool, probe) -> None:
    async def fake_sleep(seconds: float) -> None:
        return None

    async def broken(resources: JobResources, params: Any) -> dict[str, Any]:
        raise RuntimeError("HF is down")

    outcome = await run_manual(
        _resources(pool),
        _definition("collect", broken),
        CollectParams(),
        retry=RetryPolicy(retries=2, delay_seconds=1.0),
        sleep=fake_sleep,
    )

    assert (outcome.status, outcome.attempts) == (RunStatus.FAILED, 3)
    [run] = await _runs(probe)
    assert (run["status"], run["attempts"], run["error"]) == ("failed", 3, "RuntimeError: HF is down")


def test_only_collect_retries_by_default() -> None:
    assert job_definition(COLLECT_JOB).retry == COLLECT_RETRY
    assert job_definition(MAINTENANCE_JOB).retry == NO_RETRY


async def test_scheduled_collect_uses_the_configured_retry(pool, probe, pipeline_database) -> None:
    clock = FakeClock(START)
    calls = {"count": 0}

    async def flaky(resources: JobResources, params: Any) -> dict[str, Any]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("blip")
        raise Stop

    definition = JobDefinition(
        name="collect",
        cron="0 18 * * *",
        run=flaky,
        scheduled_params=lambda when: CollectParams(),
        retry=RetryPolicy(retries=1, delay_seconds=120.0),
    )

    with pytest.raises(Stop):
        await _scheduler(pool, pipeline_database, [definition], clock).run()

    assert calls["count"] == 2
    assert clock.now == START + timedelta(seconds=120)
