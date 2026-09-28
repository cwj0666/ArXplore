import asyncio
import json
import logging
from typing import Annotated, Any

import typer

from arxplore.config import get_settings
from arxplore.db.migrate import MigrationError
from arxplore.db.pool import create_pool_from_settings, open_pool
from arxplore.db.repositories import scheduler_runs
from arxplore.scheduler.runner import SchedulerLockedError
from arxplore.scheduler.service import MANUAL_RUN_ROLE, run_scheduler

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=1)


def start_scheduler() -> None:
    configure_logging()
    settings = get_settings()
    try:
        asyncio.run(run_scheduler(settings))
    except (SchedulerLockedError, MigrationError, ValueError) as exc:
        raise _fail(str(exc)) from exc


async def _history(job: str | None, limit: int) -> list[dict[str, Any]]:
    settings = get_settings()
    pool = create_pool_from_settings(settings, role=MANUAL_RUN_ROLE)
    async with open_pool(pool):
        runs = await scheduler_runs.list_recent_runs(pool, job=job, limit=limit)
    return [
        {
            "run_id": run.run_id,
            "job": run.job,
            "trigger": run.trigger.value,
            "scheduled_for": run.scheduled_for,
            "status": run.status.value,
            "attempts": run.attempts,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "params": run.params,
            "error": run.error,
        }
        for run in runs
    ]


def register(app: typer.Typer) -> None:
    scheduler_app = typer.Typer(
        help="Run the collect, maintenance and LangSmith cleanup jobs on their cron schedules.",
        add_completion=False,
    )

    @scheduler_app.callback(invoke_without_command=True)
    def scheduler_command(ctx: typer.Context) -> None:
        if ctx.invoked_subcommand is None:
            start_scheduler()

    @scheduler_app.command("history", help="Show recent scheduler runs, newest first.")
    def history_command(
        job: Annotated[str | None, typer.Option("--job", help="Only show runs of this job.")] = None,
        limit: Annotated[int, typer.Option("--limit", min=1, help="Number of runs to show.")] = 20,
    ) -> None:
        rows = asyncio.run(_history(job, limit))
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2, default=str))

    app.add_typer(scheduler_app, name="scheduler")
