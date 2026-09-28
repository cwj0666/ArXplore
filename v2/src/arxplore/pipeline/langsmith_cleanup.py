from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from langsmith import Client

from arxplore.config import AppSettings

DEFAULT_RETENTION_DAYS = 14
DISABLED_MESSAGE = "LangSmith tracing is not enabled. Check LANGSMITH_API_KEY / LANGSMITH_TRACING."


class RunStore(Protocol):
    def list_runs(self, *, project_name: str) -> Iterable[Any]: ...

    def delete_run(self, run_id: str) -> None: ...


class LangSmithRunStore:
    def __init__(self, client: Client) -> None:
        self._client = client

    def list_runs(self, *, project_name: str) -> Iterable[Any]:
        runs: Iterable[Any] = self._client.list_runs(project_name=project_name)
        return runs

    def delete_run(self, run_id: str) -> None:
        response = self._client.request_with_retries("DELETE", f"/runs/{run_id}")
        response.raise_for_status()


def langsmith_enabled(settings: AppSettings) -> bool:
    return bool(settings.langsmith_tracing and settings.langsmith_api_key)


def run_store_from_settings(settings: AppSettings) -> RunStore:
    return LangSmithRunStore(Client(api_key=settings.langsmith_api_key))


@dataclass(frozen=True, slots=True, kw_only=True)
class CleanupOptions:
    days: int = DEFAULT_RETENTION_DAYS
    project_name: str | None = None
    user: str | None = None
    limit: int | None = None
    dry_run: bool = False


def _started_at(run: Any) -> datetime | None:
    started_at = getattr(run, "start_time", None)
    if not isinstance(started_at, datetime):
        return None
    return started_at if started_at.tzinfo is not None else started_at.replace(tzinfo=UTC)


def _run_user(run: Any) -> Any:
    return (getattr(run, "metadata", {}) or {}).get("user")


def iter_expired_runs(runs: Iterable[Any], *, cutoff: datetime, user: str | None, limit: int | None) -> Iterator[Any]:
    inspected = 0
    for run in runs:
        if limit is not None and inspected >= limit:
            break
        inspected += 1
        started_at = _started_at(run)
        if started_at is None or started_at >= cutoff:
            continue
        if user and _run_user(run) != user:
            continue
        yield run


def _summary(run: Any) -> dict[str, Any]:
    started_at = getattr(run, "start_time", None)
    return {
        "run_id": str(run.id),
        "start_time": started_at.isoformat() if started_at else None,
        "user": _run_user(run),
    }


def cleanup_langsmith_runs(
    settings: AppSettings,
    options: CleanupOptions,
    *,
    store: RunStore | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if options.days < 1:
        raise ValueError("days는 1 이상이어야 합니다.")
    if not langsmith_enabled(settings):
        raise RuntimeError(DISABLED_MESSAGE)
    project_name = options.project_name or settings.langsmith_project
    run_store = store or run_store_from_settings(settings)
    cutoff = (now or datetime.now(UTC)) - timedelta(days=options.days)
    matched = list(
        iter_expired_runs(
            run_store.list_runs(project_name=project_name),
            cutoff=cutoff,
            user=options.user,
            limit=options.limit,
        )
    )
    deleted: list[str] = []
    if not options.dry_run:
        for run in matched:
            run_id = str(run.id)
            run_store.delete_run(run_id)
            deleted.append(run_id)
    return {
        "stage": "cleanup_langsmith",
        "status": "success",
        "project_name": project_name,
        "days": options.days,
        "cutoff": cutoff.isoformat(),
        "user": options.user,
        "limit": options.limit,
        "dry_run": options.dry_run,
        "matched_count": len(matched),
        "deleted_count": 0 if options.dry_run else len(deleted),
        "deleted_run_ids": deleted,
        "matched_runs": [_summary(run) for run in matched],
    }
