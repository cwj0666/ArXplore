from __future__ import annotations

from datetime import date as date_cls
from datetime import timedelta
from typing import Any

from src.integrations.paper_search import PaperSearchClient
from src.integrations.prepare_job_repository import PrepareJobRepository
from src.integrations.raw_store import RawPaperStore

from .tracing import build_pipeline_trace_config


def run_collect_papers(
    *,
    runtime: str = "airflow",
    user: str | None = None,
    target_date: str | None = None,
    enqueue_prepare: bool = True,
) -> dict[str, Any]:
    """HF Daily Papers를 수집하고 MongoDB에 원본을 저장한다."""
    normalized_target_date = (target_date or "").strip() or None
    normalized_date = (
        date_cls.fromisoformat(normalized_target_date).isoformat()
        if normalized_target_date
        else date_cls.today().isoformat()
    )

    search_client = PaperSearchClient()
    raw_store = RawPaperStore()
    prepare_job_repository: PrepareJobRepository | None = None
    if enqueue_prepare:
        prepare_job_repository = PrepareJobRepository()

    payload = search_client.fetch_daily_papers(normalized_date)
    saved = raw_store.save_daily_papers_response(date=normalized_date, payload=payload)
    record_id = saved["record_id"]
    raw_revision = saved["revision"]
    raw_payload_changed = bool(saved.get("changed", True))
    queue_result = (
        prepare_job_repository.enqueue_prepare_job(
            target_date=normalized_date,
            mode="auto",
            source="collect",
            raw_revision=raw_revision,
        )
        if prepare_job_repository is not None
        else {"enqueued": False, "job_id": None}
    )

    sample_arxiv_ids = [
        paper.get("paper", {}).get("id")
        for paper in payload
        if isinstance(paper, dict) and isinstance(paper.get("paper"), dict) and paper.get("paper", {}).get("id")
    ][:5]

    trace_config = build_pipeline_trace_config(
        stage="collect_papers",
        runtime=runtime,
        user=user,
        extra_metadata={
            "target_date": normalized_date,
            "fetched_count": len(payload),
            "stored_record_id": record_id,
            "raw_revision": raw_revision,
            "raw_payload_changed": raw_payload_changed,
            "prepare_job_enqueued": bool(queue_result.get("enqueued")),
            "prepare_queue_enqueued": bool(queue_result.get("enqueued")),
        },
    )

    return {
        "stage": "collect_papers",
        "status": "success",
        "target_date": normalized_date,
        "fetched_count": len(payload),
        "stored_record_id": record_id,
        "raw_revision": raw_revision,
        "raw_payload_changed": raw_payload_changed,
        "prepare_job_enqueued": bool(queue_result.get("enqueued")),
        "prepare_queue_enqueued": bool(queue_result.get("enqueued")),
        "prepare_job_id": queue_result.get("job_id"),
        "prepare_job_status": queue_result.get("status"),
        "sample_arxiv_ids": sample_arxiv_ids,
        "trace_config": trace_config,
    }


def run_backfill_collect_papers(
    *,
    runtime: str = "airflow",
    user: str | None = None,
    cursor_date: str | None = None,
    oldest_date: str | None = None,
    batch_days: int = 30,
    state_name: str = "default",
) -> dict[str, Any]:
    """과거 HF Daily Papers 원본을 하루 최대 batch_days개씩 순차 수집한다."""
    today = date_cls.today()
    raw_store = RawPaperStore()

    existing_state = raw_store.load_pipeline_state(
        pipeline="hf_daily_papers_backfill",
        name=state_name,
    )

    normalized_oldest_date = _resolve_backfill_oldest_date(
        oldest_date=oldest_date,
        existing_state=existing_state,
        today=today,
    )
    normalized_cursor_date = _resolve_backfill_cursor_date(
        cursor_date=cursor_date,
        existing_state=existing_state,
        today=today,
    )
    normalized_batch_days = max(1, int(batch_days or 30))

    if normalized_cursor_date < normalized_oldest_date:
        state = {
            "cursor_date": None,
            "oldest_date": normalized_oldest_date.isoformat(),
            "batch_days": normalized_batch_days,
            "status": "completed",
            "last_processed_dates": [],
            "last_failure": None,
        }
        raw_store.save_pipeline_state(
            pipeline="hf_daily_papers_backfill",
            name=state_name,
            state=state,
        )
        return _build_backfill_result(
            status="completed",
            state_name=state_name,
            oldest_date=normalized_oldest_date.isoformat(),
            cursor_date=normalized_cursor_date.isoformat(),
            next_cursor_date=None,
            batch_days=normalized_batch_days,
            successes=[],
            skipped_existing=[],
            failures=[],
            stopped_reason="cursor_exhausted",
            runtime=runtime,
            user=user,
        )

    target_dates = _build_backfill_dates(
        cursor_date=normalized_cursor_date,
        oldest_date=normalized_oldest_date,
        batch_days=normalized_batch_days,
    )
    successes: list[dict[str, Any]] = []
    skipped_existing: list[str] = []
    failures: list[dict[str, str]] = []
    stopped_reason = "batch_complete"
    next_cursor_date: str | None = None

    for target in target_dates:
        target_str = target.isoformat()
        next_cursor_candidate = target - timedelta(days=1)

        if raw_store.has_daily_papers_response(date=target_str):
            skipped_existing.append(target_str)
            next_cursor_date = (
                next_cursor_candidate.isoformat() if next_cursor_candidate >= normalized_oldest_date else None
            )
            continue

        try:
            result = run_collect_papers(
                runtime=runtime,
                user=user,
                target_date=target_str,
                enqueue_prepare=False,
            )
        except Exception as exc:
            failures.append({"date": target_str, "error": str(exc)})
            stopped_reason = "rate_limited" if _is_rate_limited_error(exc) else "collection_failed"
            next_cursor_date = target_str
            break

        successes.append(
            {
                "date": target_str,
                "fetched_count": int(result.get("fetched_count", 0) or 0),
                "stored_record_id": str(result.get("stored_record_id") or ""),
            }
        )
        next_cursor_date = (
            next_cursor_candidate.isoformat() if next_cursor_candidate >= normalized_oldest_date else None
        )

    status = _resolve_backfill_status(
        stopped_reason=stopped_reason,
        next_cursor_date=next_cursor_date,
    )
    state = {
        "cursor_date": next_cursor_date,
        "oldest_date": normalized_oldest_date.isoformat(),
        "batch_days": normalized_batch_days,
        "status": status,
        "last_processed_dates": [item["date"] for item in successes] + skipped_existing,
        "last_failure": failures[0] if failures else None,
    }
    raw_store.save_pipeline_state(
        pipeline="hf_daily_papers_backfill",
        name=state_name,
        state=state,
    )

    return _build_backfill_result(
        status=status,
        state_name=state_name,
        oldest_date=normalized_oldest_date.isoformat(),
        cursor_date=normalized_cursor_date.isoformat(),
        next_cursor_date=next_cursor_date,
        batch_days=normalized_batch_days,
        successes=successes,
        skipped_existing=skipped_existing,
        failures=failures,
        stopped_reason=stopped_reason,
        runtime=runtime,
        user=user,
    )


def _build_backfill_dates(
    *,
    cursor_date: date_cls,
    oldest_date: date_cls,
    batch_days: int,
) -> list[date_cls]:
    dates: list[date_cls] = []
    current = cursor_date
    while current >= oldest_date and len(dates) < batch_days:
        dates.append(current)
        current -= timedelta(days=1)
    return dates


def _resolve_backfill_oldest_date(
    *,
    oldest_date: str | None,
    existing_state: dict[str, Any] | None,
    today: date_cls,
) -> date_cls:
    if oldest_date:
        return date_cls.fromisoformat(oldest_date)
    if existing_state and existing_state.get("oldest_date"):
        return date_cls.fromisoformat(str(existing_state["oldest_date"]))
    return today - timedelta(days=365)


def _resolve_backfill_cursor_date(
    *,
    cursor_date: str | None,
    existing_state: dict[str, Any] | None,
    today: date_cls,
) -> date_cls:
    if cursor_date:
        return date_cls.fromisoformat(cursor_date)
    if existing_state and existing_state.get("cursor_date"):
        return date_cls.fromisoformat(str(existing_state["cursor_date"]))
    return today - timedelta(days=1)


def _is_rate_limited_error(exc: Exception) -> bool:
    message = str(exc)
    return "429" in message or "Too Many Requests" in message


def _resolve_backfill_status(
    *,
    stopped_reason: str,
    next_cursor_date: str | None,
) -> str:
    if stopped_reason == "rate_limited":
        return "rate_limited"
    if stopped_reason == "collection_failed":
        return "failed"
    if next_cursor_date is None:
        return "completed"
    return "success"


def _build_backfill_result(
    *,
    status: str,
    state_name: str,
    oldest_date: str,
    cursor_date: str,
    next_cursor_date: str | None,
    batch_days: int,
    successes: list[dict[str, Any]],
    skipped_existing: list[str],
    failures: list[dict[str, str]],
    stopped_reason: str,
    runtime: str,
    user: str | None,
) -> dict[str, Any]:
    trace_config = build_pipeline_trace_config(
        stage="backfill_collect_papers",
        runtime=runtime,
        user=user,
        extra_metadata={
            "state_name": state_name,
            "cursor_date": cursor_date,
            "next_cursor_date": next_cursor_date,
            "oldest_date": oldest_date,
            "batch_days": batch_days,
            "success_count": len(successes),
            "skipped_existing_count": len(skipped_existing),
            "failure_count": len(failures),
            "stopped_reason": stopped_reason,
        },
    )

    return {
        "stage": "backfill_collect_papers",
        "status": status,
        "state_name": state_name,
        "cursor_date": cursor_date,
        "next_cursor_date": next_cursor_date,
        "oldest_date": oldest_date,
        "batch_days": batch_days,
        "success_count": len(successes),
        "skipped_existing_count": len(skipped_existing),
        "failure_count": len(failures),
        "successes": successes,
        "skipped_existing": skipped_existing,
        "failures": failures,
        "stopped_reason": stopped_reason,
        "trace_config": trace_config,
    }
