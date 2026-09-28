import asyncio
import json
import logging
from datetime import date
from typing import Annotated, Any

import typer

from arxplore.config import get_settings
from arxplore.db.migrate import MigrationError
from arxplore.pipeline.embed import (
    EMBED_BACKLOG_BUDGET,
    EMBED_CHUNKS_PER_ROUND,
    EMBED_MAX_ROUNDS_PER_PAPER,
    EmbeddingBudget,
)
from arxplore.pipeline.langsmith_cleanup import DEFAULT_RETENTION_DAYS
from arxplore.pipeline.worker import DEFAULT_WAIT_TIMEOUT_SECONDS, DEFAULT_WORKER_NAME, WorkerOptions, run_worker
from arxplore.scheduler.jobs import (
    COLLECT_JOB,
    LANGSMITH_CLEANUP_JOB,
    MAINTENANCE_JOB,
    CollectParams,
    LangsmithCleanupParams,
    MaintenanceParams,
    job_definition,
)
from arxplore.scheduler.runner import RunOutcome
from arxplore.scheduler.service import run_job_manually

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=1)


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def parse_date_option(value: str | None) -> date | None:
    normalized = (value or "").strip()
    if not normalized:
        return None
    try:
        return date.fromisoformat(normalized)
    except ValueError as exc:
        raise typer.BadParameter(f"expected YYYY-MM-DD, got {value!r}") from exc


def positive_or_default(value: int, default: int) -> int:
    return value if value > 0 else default


def _outcome_payload(job: str, outcome: RunOutcome) -> dict[str, Any]:
    return {
        "run_id": outcome.run_id,
        "job": job,
        "trigger": "manual",
        "status": outcome.status.value,
        "attempts": outcome.attempts,
        "error": outcome.error,
        "result": outcome.result,
    }


def _run_manual_job(job: str, params: Any) -> None:
    configure_logging()
    settings = get_settings()
    try:
        outcome = asyncio.run(run_job_manually(settings, job_definition(job), params))
    except (MigrationError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    _echo_json(_outcome_payload(job, outcome))
    if outcome.status.value != "success":
        raise typer.Exit(code=1)


def register_worker(app: typer.Typer) -> None:
    @app.command("worker", help="Consume prepare jobs: parse PDFs, store chunks and embed them.")
    def worker_command(
        worker_id: Annotated[
            str, typer.Option("--worker-id", help="Identifier recorded on claimed prepare jobs.")
        ] = DEFAULT_WORKER_NAME,
        max_jobs_per_run: Annotated[
            int, typer.Option("--max-jobs-per-run", help="Prepare jobs to claim before embedding.")
        ] = 1,
        max_papers: Annotated[
            int | None, typer.Option("--max-papers", help="Papers per date; unset or 0 means all.")
        ] = None,
        embed_chunks_per_round: Annotated[
            int,
            typer.Option(
                "--embed-chunks-per-round",
                "--embed-max-chunks",
                help="Chunks selected and embedded per round, per paper and for the backlog.",
            ),
        ] = EMBED_CHUNKS_PER_ROUND,
        embed_max_rounds_per_paper: Annotated[
            int,
            typer.Option(
                "--embed-max-rounds-per-paper",
                help="Rounds per prepared paper; a paper can embed chunks-per-round x this many chunks.",
            ),
        ] = EMBED_MAX_ROUNDS_PER_PAPER,
        embed_backlog_budget: Annotated[
            int,
            typer.Option(
                "--embed-backlog-budget",
                "--embed-backlog-max-chunks",
                help="Chunks of older papers to embed after each run; 0 turns the backlog off.",
            ),
        ] = EMBED_BACKLOG_BUDGET,
        skip_embed: Annotated[bool, typer.Option("--skip-embed", help="Do not embed after preparing.")] = False,
        force: Annotated[
            bool, typer.Option("--force", help="Replace fulltexts and chunks regardless of source rank and hash.")
        ] = False,
        loop: Annotated[bool, typer.Option("--loop", help="Keep running and wait for job notifications.")] = False,
        wait_timeout_seconds: Annotated[
            float,
            typer.Option("--wait-timeout-seconds", help="Longest wait for a job notification when the queue is idle."),
        ] = DEFAULT_WAIT_TIMEOUT_SECONDS,
    ) -> None:
        configure_logging()
        settings = get_settings()
        options = WorkerOptions(
            worker_id=worker_id.strip() or DEFAULT_WORKER_NAME,
            max_jobs_per_run=max(1, max_jobs_per_run),
            max_papers=max_papers if max_papers and max_papers > 0 else None,
            force=force,
            skip_embed=skip_embed,
            embedding=EmbeddingBudget(
                chunks_per_round=positive_or_default(embed_chunks_per_round, EMBED_CHUNKS_PER_ROUND),
                max_rounds_per_paper=positive_or_default(embed_max_rounds_per_paper, EMBED_MAX_ROUNDS_PER_PAPER),
                backlog_chunks=max(0, embed_backlog_budget),
            ),
            wait_timeout_seconds=max(0.0, wait_timeout_seconds),
            consume_backfill=settings.prepare_backfill_enabled,
        )
        try:
            exit_code = asyncio.run(run_worker(settings, options, loop=loop))
        except (MigrationError, ValueError) as exc:
            raise _fail(str(exc)) from exc
        raise typer.Exit(code=exit_code)


def register_run(app: typer.Typer) -> None:
    run_app = typer.Typer(help="Run a pipeline job once, recorded as a manual scheduler run.", add_completion=False)

    @run_app.command("collect", help="Collect one day of HF Daily Papers and enqueue its prepare job.")
    def collect_command(
        target_date: Annotated[str | None, typer.Option("--target-date", help="YYYY-MM-DD; defaults to today.")] = None,
    ) -> None:
        _run_manual_job(COLLECT_JOB, CollectParams(target_date=parse_date_option(target_date)))

    @run_app.command("maintenance", help="Backfill missing HF days, then enrich papers from arXiv.")
    def maintenance_command(
        backfill_cursor_date: Annotated[
            str | None, typer.Option("--backfill-cursor-date", help="YYYY-MM-DD to start from.")
        ] = None,
        backfill_oldest_date: Annotated[
            str | None, typer.Option("--backfill-oldest-date", help="YYYY-MM-DD to stop at.")
        ] = None,
        backfill_batch_days: Annotated[int, typer.Option("--backfill-batch-days", help="Days per run.")] = 30,
        backfill_state_name: Annotated[
            str, typer.Option("--backfill-state-name", help="pipeline_state cursor name.")
        ] = "default",
        enrich_max_papers: Annotated[int, typer.Option("--enrich-max-papers", help="Papers to enrich.")] = 30,
    ) -> None:
        params = MaintenanceParams(
            backfill_cursor_date=parse_date_option(backfill_cursor_date),
            backfill_oldest_date=parse_date_option(backfill_oldest_date),
            backfill_batch_days=backfill_batch_days,
            backfill_state_name=backfill_state_name.strip() or "default",
            enrich_max_papers=enrich_max_papers,
        )
        _run_manual_job(MAINTENANCE_JOB, params)

    @run_app.command("langsmith-cleanup", help="Delete LangSmith runs older than the retention window.")
    def langsmith_cleanup_command(
        days: Annotated[int, typer.Option("--days", help="Delete runs older than this many days.")] = (
            DEFAULT_RETENTION_DAYS
        ),
        project_name: Annotated[str | None, typer.Option("--project-name", help="LangSmith project.")] = None,
        user: Annotated[str | None, typer.Option("--user", help="Only runs whose metadata.user matches.")] = None,
        limit: Annotated[int | None, typer.Option("--limit", help="Upper bound on runs to process.")] = None,
        dry_run: Annotated[bool, typer.Option("--dry-run", help="List matching runs without deleting.")] = False,
    ) -> None:
        params = LangsmithCleanupParams(
            days=days,
            project_name=(project_name or "").strip() or None,
            user=(user or "").strip() or None,
            limit=limit,
            dry_run=dry_run,
        )
        _run_manual_job(LANGSMITH_CLEANUP_JOB, params)

    app.add_typer(run_app, name="run")


def register(app: typer.Typer) -> None:
    register_worker(app)
    register_run(app)
