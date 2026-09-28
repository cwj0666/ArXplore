from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from arxplore.db.pool import AnyPool
from arxplore.db.repositories import pipeline_state, raw
from arxplore.db.repositories.pipeline_state import DEFAULT_STATE_NAME
from arxplore.pipeline.collect import DailyPapersSource, collect_daily_papers

BACKFILL_PIPELINE = "hf_daily_papers_backfill"
DEFAULT_BACKFILL_BATCH_DAYS = 30
DEFAULT_BACKFILL_LOOKBACK_DAYS = 365
RATE_LIMIT_MARKERS = ("429", "Too Many Requests")


def backfill_dates(*, cursor_date: date, oldest_date: date, batch_days: int) -> list[date]:
    dates: list[date] = []
    current = cursor_date
    while current >= oldest_date and len(dates) < batch_days:
        dates.append(current)
        current -= timedelta(days=1)
    return dates


def _state_date(state: Mapping[str, Any] | None, key: str) -> date | None:
    if state and state.get(key):
        return date.fromisoformat(str(state[key]))
    return None


def resolve_oldest_date(oldest_date: date | None, state: Mapping[str, Any] | None, today: date) -> date:
    return oldest_date or _state_date(state, "oldest_date") or today - timedelta(days=DEFAULT_BACKFILL_LOOKBACK_DAYS)


def resolve_cursor_date(cursor_date: date | None, state: Mapping[str, Any] | None, today: date) -> date:
    return cursor_date or _state_date(state, "cursor_date") or today - timedelta(days=1)


def is_rate_limited_error(exc: BaseException) -> bool:
    message = str(exc)
    return any(marker in message for marker in RATE_LIMIT_MARKERS)


def resolve_backfill_status(*, stopped_reason: str, next_cursor_date: date | None) -> str:
    if stopped_reason == "rate_limited":
        return "rate_limited"
    if stopped_reason == "collection_failed":
        return "failed"
    if next_cursor_date is None:
        return "completed"
    return "success"


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


@dataclass(slots=True, kw_only=True)
class BackfillResult:
    status: str
    state_name: str
    cursor_date: date
    next_cursor_date: date | None
    oldest_date: date
    batch_days: int
    stopped_reason: str
    successes: list[dict[str, Any]] = field(default_factory=list)
    skipped_existing: list[date] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)

    def state(self) -> dict[str, Any]:
        return {
            "cursor_date": _iso(self.next_cursor_date),
            "oldest_date": self.oldest_date.isoformat(),
            "batch_days": self.batch_days,
            "status": self.status,
            "last_processed_dates": [item["date"] for item in self.successes]
            + [value.isoformat() for value in self.skipped_existing],
            "last_failure": self.failures[0] if self.failures else None,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": "backfill_collect_papers",
            "status": self.status,
            "state_name": self.state_name,
            "cursor_date": self.cursor_date.isoformat(),
            "next_cursor_date": _iso(self.next_cursor_date),
            "oldest_date": self.oldest_date.isoformat(),
            "batch_days": self.batch_days,
            "success_count": len(self.successes),
            "skipped_existing_count": len(self.skipped_existing),
            "failure_count": len(self.failures),
            "successes": list(self.successes),
            "skipped_existing": [value.isoformat() for value in self.skipped_existing],
            "failures": list(self.failures),
            "stopped_reason": self.stopped_reason,
        }


async def backfill_daily_papers(
    pool: AnyPool,
    source: DailyPapersSource,
    *,
    today: date,
    cursor_date: date | None = None,
    oldest_date: date | None = None,
    batch_days: int = DEFAULT_BACKFILL_BATCH_DAYS,
    state_name: str = DEFAULT_STATE_NAME,
) -> BackfillResult:
    stored = await pipeline_state.load_pipeline_state(pool, pipeline=BACKFILL_PIPELINE, name=state_name)
    state = stored.values if stored else None
    oldest = resolve_oldest_date(oldest_date, state, today)
    cursor = resolve_cursor_date(cursor_date, state, today)
    days = max(1, int(batch_days or DEFAULT_BACKFILL_BATCH_DAYS))

    if cursor < oldest:
        result = BackfillResult(
            status="completed",
            state_name=state_name,
            cursor_date=cursor,
            next_cursor_date=None,
            oldest_date=oldest,
            batch_days=days,
            stopped_reason="cursor_exhausted",
        )
        await pipeline_state.save_pipeline_state(
            pool, pipeline=BACKFILL_PIPELINE, name=state_name, state=result.state()
        )
        return result

    successes: list[dict[str, Any]] = []
    skipped_existing: list[date] = []
    failures: list[dict[str, str]] = []
    stopped_reason = "batch_complete"
    next_cursor: date | None = None
    for target in backfill_dates(cursor_date=cursor, oldest_date=oldest, batch_days=days):
        candidate = target - timedelta(days=1)
        if await raw.has_daily_papers(pool, target_date=target):
            skipped_existing.append(target)
            next_cursor = candidate if candidate >= oldest else None
            continue
        try:
            collected = await collect_daily_papers(pool, source, target_date=target, enqueue_prepare=False)
        except Exception as exc:
            failures.append({"date": target.isoformat(), "error": str(exc)})
            stopped_reason = "rate_limited" if is_rate_limited_error(exc) else "collection_failed"
            next_cursor = target
            break
        successes.append(
            {
                "date": target.isoformat(),
                "fetched_count": collected.fetched_count,
                "stored_record_id": collected.stored_record_id,
            }
        )
        next_cursor = candidate if candidate >= oldest else None

    result = BackfillResult(
        status=resolve_backfill_status(stopped_reason=stopped_reason, next_cursor_date=next_cursor),
        state_name=state_name,
        cursor_date=cursor,
        next_cursor_date=next_cursor,
        oldest_date=oldest,
        batch_days=days,
        stopped_reason=stopped_reason,
        successes=successes,
        skipped_existing=skipped_existing,
        failures=failures,
    )
    await pipeline_state.save_pipeline_state(pool, pipeline=BACKFILL_PIPELINE, name=state_name, state=result.state())
    return result
