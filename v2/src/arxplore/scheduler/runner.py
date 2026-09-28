import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from psycopg import AsyncConnection, OperationalError

from arxplore.db.repositories import scheduler_runs
from arxplore.db.repositories.scheduler_runs import RunStatus
from arxplore.scheduler.cron import next_fire_time, previous_fire_time
from arxplore.scheduler.jobs import NO_RETRY, JobDefinition, JobFailedError, JobResources, RetryPolicy, params_as_dict

logger = logging.getLogger(__name__)

MAX_SLEEP_SLICE_SECONDS = 60.0

type Clock = Callable[[], datetime]
type Sleep = Callable[[float], Awaitable[None]]
type LockConnector = Callable[[], Awaitable[AsyncConnection[Any]]]


class SchedulerLockedError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class RunOutcome:
    run_id: int
    status: RunStatus
    result: dict[str, Any] | None
    error: str | None
    attempts: int


def describe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


async def execute_run(
    resources: JobResources,
    definition: JobDefinition,
    params: Any,
    run_id: int,
    *,
    retry: RetryPolicy = NO_RETRY,
    sleep: Sleep = asyncio.sleep,
) -> RunOutcome:
    attempts = 0
    while True:
        attempts += 1
        try:
            result = await definition.run(resources, params)
        except Exception as exc:
            error = describe_error(exc)
            partial = exc.result if isinstance(exc, JobFailedError) else None
            if attempts > retry.retries:
                logger.exception("scheduler job %s failed after %s attempt(s)", definition.name, attempts)
                await scheduler_runs.finish_run(
                    resources.pool, run_id, status=RunStatus.FAILED, result=partial, error=error, attempts=attempts
                )
                return RunOutcome(
                    run_id=run_id, status=RunStatus.FAILED, result=partial, error=error, attempts=attempts
                )
            logger.warning(
                "scheduler job %s attempt %s failed, retrying in %ss: %s",
                definition.name,
                attempts,
                retry.delay_seconds,
                error,
            )
            await scheduler_runs.record_attempt(resources.pool, run_id, attempts=attempts, error=error)
            await sleep(retry.delay_seconds)
            continue
        await scheduler_runs.finish_run(
            resources.pool, run_id, status=RunStatus.SUCCESS, result=result, error=None, attempts=attempts
        )
        return RunOutcome(run_id=run_id, status=RunStatus.SUCCESS, result=result, error=None, attempts=attempts)


async def run_manual(
    resources: JobResources,
    definition: JobDefinition,
    params: Any,
    *,
    retry: RetryPolicy = NO_RETRY,
    sleep: Sleep = asyncio.sleep,
) -> RunOutcome:
    run_id = await scheduler_runs.start_manual_run(resources.pool, job=definition.name, params=params_as_dict(params))
    return await execute_run(resources, definition, params, run_id, retry=retry, sleep=sleep)


async def run_scheduled(
    resources: JobResources,
    definition: JobDefinition,
    scheduled_for: datetime,
    *,
    retry: RetryPolicy = NO_RETRY,
    sleep: Sleep = asyncio.sleep,
) -> RunOutcome | None:
    params = definition.scheduled_params(scheduled_for)
    run_id = await scheduler_runs.start_scheduled_run(
        resources.pool, job=definition.name, scheduled_for=scheduled_for, params=params_as_dict(params)
    )
    if run_id is None:
        logger.info("scheduler job %s at %s already has a run", definition.name, scheduled_for.isoformat())
        return None
    return await execute_run(resources, definition, params, run_id, retry=retry, sleep=sleep)


class Scheduler:
    def __init__(
        self,
        resources: JobResources,
        definitions: Sequence[JobDefinition],
        *,
        timezone: ZoneInfo,
        connect_lock: LockConnector,
        retries: dict[str, RetryPolicy] | None = None,
        clock: Clock | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.resources = resources
        self.definitions = tuple(definitions)
        self.timezone = timezone
        self.retries = dict(retries or {})
        self._connect_lock = connect_lock
        self._clock = clock or (lambda: datetime.now(timezone))
        self._sleep = sleep
        self._lock_connection: AsyncConnection[Any] | None = None

    def now(self) -> datetime:
        return self._clock().astimezone(self.timezone)

    def retry_for(self, definition: JobDefinition) -> RetryPolicy:
        return self.retries.get(definition.name, definition.retry)

    async def acquire_lock(self) -> None:
        connection = await self._connect_lock()
        try:
            locked = await scheduler_runs.try_acquire_scheduler_lock(connection)
        except BaseException:
            await connection.close()
            raise
        if not locked:
            await connection.close()
            raise SchedulerLockedError("another scheduler instance holds the scheduler lock")
        self._lock_connection = connection

    async def ensure_lock(self) -> None:
        connection = self._lock_connection
        if connection is not None and not connection.closed:
            try:
                await connection.execute("SELECT 1")
                return
            except OperationalError:
                logger.warning("scheduler lock connection was lost; reacquiring", exc_info=True)
        if connection is not None:
            await connection.close()
        self._lock_connection = None
        await self.acquire_lock()

    async def release_lock(self) -> None:
        connection, self._lock_connection = self._lock_connection, None
        if connection is None:
            return
        try:
            await scheduler_runs.release_scheduler_lock(connection)
        finally:
            await connection.close()

    async def sleep_until(self, target: datetime) -> None:
        while True:
            remaining = (target - self.now()).total_seconds()
            if remaining <= 0:
                return
            await self._sleep(min(remaining, MAX_SLEEP_SLICE_SECONDS))

    async def fire(self, definition: JobDefinition, scheduled_for: datetime) -> RunOutcome | None:
        await self.ensure_lock()
        return await run_scheduled(
            self.resources, definition, scheduled_for, retry=self.retry_for(definition), sleep=self._sleep
        )

    async def catch_up(self, definition: JobDefinition) -> RunOutcome | None:
        previous = previous_fire_time(definition.cron, self.now())
        if await scheduler_runs.has_scheduled_run(self.resources.pool, job=definition.name, scheduled_for=previous):
            return None
        logger.info("scheduler job %s missed %s; running it once", definition.name, previous.isoformat())
        return await self.fire(definition, previous)

    async def _guarded(self, definition: JobDefinition, operation: Awaitable[RunOutcome | None]) -> None:
        try:
            await operation
        except SchedulerLockedError:
            raise
        except Exception:
            logger.exception("scheduler job %s could not be run", definition.name)

    async def job_loop(self, definition: JobDefinition) -> None:
        await self._guarded(definition, self.catch_up(definition))
        while True:
            scheduled_for = next_fire_time(definition.cron, self.now())
            await self.sleep_until(scheduled_for)
            await self._guarded(definition, self.fire(definition, scheduled_for))

    async def run(self) -> None:
        await self.acquire_lock()
        try:
            abandoned = await scheduler_runs.abandon_running_runs(self.resources.pool)
            if abandoned:
                logger.warning("marked %s unfinished scheduled run(s) as abandoned", abandoned)
            tasks = [
                asyncio.create_task(self.job_loop(definition), name=f"scheduler:{definition.name}")
                for definition in self.definitions
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            await self.release_lock()
