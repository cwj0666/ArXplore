from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from zoneinfo import ZoneInfo

from psycopg import AsyncConnection

from arxplore.config import AppSettings
from arxplore.db.migrate import verify_schema
from arxplore.db.pool import connect_from_settings
from arxplore.scheduler.jobs import JOB_DEFINITIONS, JobDefinition, JobResources, open_job_resources
from arxplore.scheduler.runner import NO_RETRY, RetryPolicy, RunOutcome, Scheduler, run_manual

SCHEDULER_ROLE = "scheduler"
SCHEDULER_LOCK_ROLE = "scheduler-lock"
MANUAL_RUN_ROLE = "cli"


@asynccontextmanager
async def open_verified_resources(settings: AppSettings, *, role: str) -> AsyncIterator[JobResources]:
    async with open_job_resources(settings, role=role) as resources:
        async with resources.pool.connection() as connection:
            await verify_schema(connection, embedding_dimensions=settings.openai_embedding_dimensions)
        yield resources


async def run_scheduler(settings: AppSettings, *, retries: dict[str, RetryPolicy] | None = None) -> None:
    timezone = ZoneInfo(settings.scheduler_timezone)

    async def connect_lock() -> AsyncConnection[Any]:
        return await connect_from_settings(settings, role=SCHEDULER_LOCK_ROLE, autocommit=True)

    async with open_verified_resources(settings, role=SCHEDULER_ROLE) as resources:
        scheduler = Scheduler(
            resources,
            JOB_DEFINITIONS,
            timezone=timezone,
            connect_lock=connect_lock,
            retries=retries,
        )
        await scheduler.run()


async def run_job_manually(
    settings: AppSettings,
    definition: JobDefinition,
    params: Any,
    *,
    retry: RetryPolicy = NO_RETRY,
) -> RunOutcome:
    async with open_verified_resources(settings, role=MANUAL_RUN_ROLE) as resources:
        return await run_manual(resources, definition, params, retry=retry)
