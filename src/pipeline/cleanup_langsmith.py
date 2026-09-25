from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from langsmith import Client

from src.shared import get_settings, is_langsmith_enabled


def run_cleanup_langsmith(
    *,
    days: int = 14,
    project_name: str | None = None,
    user: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    settings = get_settings()

    if days < 1:
        raise ValueError("days는 1 이상이어야 합니다.")

    if not is_langsmith_enabled(settings):
        raise RuntimeError("LangSmith tracing is not enabled. Check LANGSMITH_API_KEY / LANGSMITH_TRACING.")

    resolved_project_name = project_name or settings.langsmith_project
    client = Client(api_key=settings.langsmith_api_key)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    matched_runs = list(
        _iter_target_runs(
            client,
            project_name=resolved_project_name,
            cutoff=cutoff,
            user=user,
            limit=limit,
        )
    )

    deleted_run_ids: list[str] = []
    matched_run_summaries = [
        {
            "run_id": str(getattr(run, "id")),
            "start_time": getattr(run, "start_time", None).isoformat()
            if getattr(run, "start_time", None)
            else None,
            "user": ((getattr(run, "metadata", {}) or {}).get("user")),
        }
        for run in matched_runs
    ]

    if not dry_run:
        for run in matched_runs:
            run_id = str(getattr(run, "id"))
            response = client.request_with_retries("DELETE", f"/runs/{run_id}")
            response.raise_for_status()
            deleted_run_ids.append(run_id)

    return {
        "stage": "cleanup_langsmith",
        "status": "success",
        "project_name": resolved_project_name,
        "days": days,
        "cutoff": cutoff.isoformat(),
        "user": user,
        "limit": limit,
        "dry_run": dry_run,
        "matched_count": len(matched_runs),
        "deleted_count": 0 if dry_run else len(deleted_run_ids),
        "deleted_run_ids": deleted_run_ids,
        "matched_runs": matched_run_summaries,
    }


def _iter_target_runs(
    client: Client,
    *,
    project_name: str,
    cutoff: datetime,
    user: str | None,
    limit: int | None,
) -> Iterable[object]:
    inspected = 0
    for run in client.list_runs(project_name=project_name):
        if limit is not None and inspected >= limit:
            break
        inspected += 1

        started_at = getattr(run, "start_time", None)
        if not isinstance(started_at, datetime):
            continue
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=timezone.utc)
        if started_at >= cutoff:
            continue

        metadata = getattr(run, "metadata", {}) or {}
        if user and metadata.get("user") != user:
            continue

        yield run


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete LangSmith runs older than the configured retention window.",
    )
    parser.add_argument("--days", type=int, default=14, help="Delete runs older than this many days. Default: 14")
    parser.add_argument("--project", default=None, help="LangSmith project name.")
    parser.add_argument("--user", default=None, help="Only delete runs whose metadata.user matches this value.")
    parser.add_argument("--limit", type=int, default=None, help="Only inspect up to this many runs.")
    parser.add_argument("--dry-run", action="store_true", help="Print matching runs without deleting them.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_cleanup_langsmith(
        days=args.days,
        project_name=(args.project or "").strip() or None,
        user=(args.user or "").strip() or None,
        limit=args.limit,
        dry_run=args.dry_run,
    )

    if result["matched_count"] == 0:
        print("No matching LangSmith runs found.")
        return 0

    print(
        f"Matched {result['matched_count']} run(s) older than {result['days']} day(s) "
        f"in project '{result['project_name']}'."
    )

    for run in result["matched_runs"]:
        print(f"- {run['run_id']} | {run['start_time']} | user={run['user'] or '-'}")

    if result["dry_run"]:
        print("Dry run only. No LangSmith runs were deleted.")
        return 0

    print(f"Deleted {result['deleted_count']} LangSmith run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
