import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

import httpx

from arxplore.config import AppSettings
from arxplore.db.pool import AnyPool, create_pool_from_settings, open_pool
from arxplore.db.repositories.pipeline_state import DEFAULT_STATE_NAME
from arxplore.pipeline.backfill import DEFAULT_BACKFILL_BATCH_DAYS, backfill_daily_papers
from arxplore.pipeline.collect import DailyPapersSource, collect_daily_papers
from arxplore.pipeline.dates import Today, container_today
from arxplore.pipeline.enrich import DEFAULT_ENRICH_MAX_PAPERS, enrich_papers_metadata
from arxplore.pipeline.langsmith_cleanup import DEFAULT_RETENTION_DAYS, CleanupOptions, cleanup_langsmith_runs
from arxplore.sources.arxiv import ArxivClient
from arxplore.sources.hf_daily import HfDailyPapersClient

COLLECT_JOB = "collect"
MAINTENANCE_JOB = "maintenance"
LANGSMITH_CLEANUP_JOB = "langsmith_cleanup"

type CleanupRunner = Callable[[AppSettings, CleanupOptions], dict[str, Any]]


class JobFailedError(Exception):
    def __init__(self, message: str, result: dict[str, Any]) -> None:
        super().__init__(message)
        self.result = result


def _isoformat(value: date | None) -> str:
    return value.isoformat() if value else ""


@dataclass(frozen=True, slots=True, kw_only=True)
class CollectParams:
    target_date: date | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"target_date": _isoformat(self.target_date)}


@dataclass(frozen=True, slots=True, kw_only=True)
class MaintenanceParams:
    backfill_cursor_date: date | None = None
    backfill_oldest_date: date | None = None
    backfill_batch_days: int = DEFAULT_BACKFILL_BATCH_DAYS
    backfill_state_name: str = DEFAULT_STATE_NAME
    enrich_max_papers: int = DEFAULT_ENRICH_MAX_PAPERS

    def as_dict(self) -> dict[str, Any]:
        return {
            "backfill_cursor_date": _isoformat(self.backfill_cursor_date),
            "backfill_oldest_date": _isoformat(self.backfill_oldest_date),
            "backfill_batch_days": self.backfill_batch_days,
            "backfill_state_name": self.backfill_state_name,
            "enrich_max_papers": self.enrich_max_papers,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class LangsmithCleanupParams:
    days: int = DEFAULT_RETENTION_DAYS
    project_name: str | None = None
    user: str | None = None
    limit: int | None = None
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def options(self) -> CleanupOptions:
        return CleanupOptions(
            days=self.days,
            project_name=self.project_name,
            user=self.user,
            limit=self.limit,
            dry_run=self.dry_run,
        )


@dataclass(slots=True)
class JobResources:
    settings: AppSettings
    pool: AnyPool
    hf: DailyPapersSource
    arxiv: ArxivClient
    today: Today = container_today
    cleanup: CleanupRunner = field(default=cleanup_langsmith_runs)


async def run_collect(resources: JobResources, params: CollectParams) -> dict[str, Any]:
    result = await collect_daily_papers(
        resources.pool,
        resources.hf,
        target_date=params.target_date or resources.today(),
    )
    return result.as_dict()


def _step_error(exc: Exception) -> dict[str, Any]:
    return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


async def run_maintenance(resources: JobResources, params: MaintenanceParams) -> dict[str, Any]:
    result: dict[str, Any] = {}
    errors: list[str] = []
    try:
        backfill = await backfill_daily_papers(
            resources.pool,
            resources.hf,
            today=resources.today(),
            cursor_date=params.backfill_cursor_date,
            oldest_date=params.backfill_oldest_date,
            batch_days=params.backfill_batch_days,
            state_name=params.backfill_state_name or DEFAULT_STATE_NAME,
        )
        result["backfill"] = backfill.as_dict()
    except Exception as exc:
        result["backfill"] = _step_error(exc)
        errors.append(f"backfill: {result['backfill']['error']}")
    try:
        enrich = await enrich_papers_metadata(
            resources.pool,
            resources.arxiv,
            batch_size=resources.settings.arxiv_request_batch_size,
            max_papers=params.enrich_max_papers,
        )
        result["enrich"] = enrich.as_dict()
    except Exception as exc:
        result["enrich"] = _step_error(exc)
        errors.append(f"enrich: {result['enrich']['error']}")
    if errors:
        raise JobFailedError("; ".join(errors), result)
    return result


async def run_langsmith_cleanup(resources: JobResources, params: LangsmithCleanupParams) -> dict[str, Any]:
    return await asyncio.to_thread(resources.cleanup, resources.settings, params.options())


type JobFunction = Callable[[JobResources, Any], Awaitable[dict[str, Any]]]
type ScheduledParams = Callable[[datetime], Any]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    retries: int = 0
    delay_seconds: float = 0.0


NO_RETRY = RetryPolicy()
COLLECT_RETRIES = 3
COLLECT_RETRY_DELAY_SECONDS = 600.0
COLLECT_RETRY = RetryPolicy(retries=COLLECT_RETRIES, delay_seconds=COLLECT_RETRY_DELAY_SECONDS)


@dataclass(frozen=True, slots=True, kw_only=True)
class JobDefinition:
    name: str
    cron: str
    run: JobFunction
    scheduled_params: ScheduledParams
    retry: RetryPolicy = NO_RETRY


def _collect_params_for(scheduled_for: datetime) -> CollectParams:
    return CollectParams(target_date=scheduled_for.date())


JOB_DEFINITIONS: tuple[JobDefinition, ...] = (
    JobDefinition(
        name=COLLECT_JOB,
        cron="0 18 * * *",
        run=run_collect,
        scheduled_params=_collect_params_for,
        retry=COLLECT_RETRY,
    ),
    JobDefinition(
        name=MAINTENANCE_JOB,
        cron="0 */3 * * *",
        run=run_maintenance,
        scheduled_params=lambda _: MaintenanceParams(),
    ),
    JobDefinition(
        name=LANGSMITH_CLEANUP_JOB,
        cron="0 3 * * *",
        run=run_langsmith_cleanup,
        scheduled_params=lambda _: LangsmithCleanupParams(),
    ),
)


def job_definition(name: str) -> JobDefinition:
    for definition in JOB_DEFINITIONS:
        if definition.name == name:
            return definition
    raise KeyError(name)


def params_as_dict(params: Any) -> dict[str, Any]:
    as_dict = getattr(params, "as_dict", None)
    return dict(as_dict()) if callable(as_dict) else {}


@asynccontextmanager
async def open_job_resources(settings: AppSettings, *, role: str) -> AsyncIterator[JobResources]:
    pool = create_pool_from_settings(settings, role=role)
    async with open_pool(pool), httpx.AsyncClient(follow_redirects=True) as http:
        yield JobResources(
            settings=settings,
            pool=pool,
            hf=HfDailyPapersClient.from_settings(settings, http=http),
            arxiv=ArxivClient.from_settings(settings, http=http),
        )
