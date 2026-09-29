import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import date
from enum import StrEnum
from typing import Annotated, Any

import typer

from arxplore.config import get_settings
from arxplore.db.migrate import MigrationError, verify_schema
from arxplore.db.pool import Pool, create_pool_from_settings, open_pool
from arxplore.domain.jobs import DEFAULT_JOB_MODE
from arxplore.pipeline.backfill import BACKFILL_JOB_MODE
from arxplore.pipeline.corpus import rechunk_corpus, reparse_dates
from arxplore.pipeline.embed import (
    EMBED_BACKLOG_BUDGET,
    EMBED_CHUNKS_PER_ROUND,
    EMBED_MAX_ROUNDS_PER_PAPER,
    EmbeddingBudget,
)
from arxplore.pipeline.langsmith_cleanup import DEFAULT_RETENTION_DAYS
from arxplore.pipeline.requeue import requeue_failed_jobs
from arxplore.pipeline.worker import DEFAULT_WAIT_TIMEOUT_SECONDS, DEFAULT_WORKER_NAME, WorkerOptions, run_worker
from arxplore.scheduler.jobs import (
    COLLECT_JOB,
    COLLECT_RETRY_DELAY_SECONDS,
    LANGSMITH_CLEANUP_JOB,
    MAINTENANCE_JOB,
    NO_RETRY,
    CollectParams,
    LangsmithCleanupParams,
    MaintenanceParams,
    RetryPolicy,
    job_definition,
)
from arxplore.scheduler.runner import RunOutcome
from arxplore.scheduler.service import run_job_manually

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
CORPUS_ROLE = "corpus"


class JobMode(StrEnum):
    AUTO = DEFAULT_JOB_MODE
    BACKFILL = BACKFILL_JOB_MODE


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


def _run_manual_job(job: str, params: Any, *, retry: RetryPolicy = NO_RETRY) -> None:
    configure_logging()
    settings = get_settings()
    try:
        outcome = asyncio.run(run_job_manually(settings, job_definition(job), params, retry=retry))
    except (MigrationError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    _echo_json(_outcome_payload(job, outcome))
    if outcome.status.value != "success":
        raise typer.Exit(code=1)


def _run_with_pool(operation: Callable[[Pool], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
    configure_logging()
    settings = get_settings()

    async def session() -> dict[str, Any]:
        pool = create_pool_from_settings(settings, role=CORPUS_ROLE)
        async with open_pool(pool):
            async with pool.connection() as connection:
                await verify_schema(connection, embedding_dimensions=settings.openai_embedding_dimensions)
            return await operation(pool)

    try:
        return asyncio.run(session())
    except (MigrationError, ValueError) as exc:
        raise _fail(str(exc)) from exc


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
        retries: Annotated[int, typer.Option("--retries", min=0, help="Extra attempts when collecting fails.")] = 0,
        retry_delay: Annotated[
            float, typer.Option("--retry-delay", min=0.0, help="Seconds to wait between attempts.")
        ] = COLLECT_RETRY_DELAY_SECONDS,
    ) -> None:
        _run_manual_job(
            COLLECT_JOB,
            CollectParams(target_date=parse_date_option(target_date)),
            retry=RetryPolicy(retries=retries, delay_seconds=retry_delay),
        )

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

    @run_app.command("rechunk", help="Rebuild chunks from stored sections with the current chunker.")
    def rechunk_command(
        arxiv_ids: Annotated[
            list[str] | None, typer.Option("--arxiv-id", help="Only this paper; repeat for more. Default: all.")
        ] = None,
        apply: Annotated[
            bool, typer.Option("--apply", help="Replace changed chunks. Without it nothing is written.")
        ] = False,
    ) -> None:
        async def operation(pool: Pool) -> dict[str, Any]:
            return (await rechunk_corpus(pool, arxiv_ids=arxiv_ids, apply=apply)).as_dict()

        _echo_json(_run_with_pool(operation))

    @run_app.command(
        "reparse",
        help=(
            "Queue every collected date in the range for another parse, forced by default: pending, done and "
            "failed jobs are reopened and dates without a job get one. To retry only failed jobs use "
            "requeue-failed."
        ),
    )
    def reparse_command(
        since: Annotated[str | None, typer.Option("--since", help="First date, YYYY-MM-DD.")] = None,
        until: Annotated[str | None, typer.Option("--until", help="Last date, YYYY-MM-DD.")] = None,
        mode: Annotated[str, typer.Option("--mode", help="prepare_jobs mode to queue.")] = DEFAULT_JOB_MODE,
        force: Annotated[
            bool, typer.Option("--force/--no-force", help="Replace fulltexts even when the stored one is final.")
        ] = True,
        apply: Annotated[bool, typer.Option("--apply", help="Queue the jobs. Without it nothing is written.")] = False,
    ) -> None:
        since_date, until_date = parse_date_option(since), parse_date_option(until)

        async def operation(pool: Pool) -> dict[str, Any]:
            result = await reparse_dates(
                pool,
                mode=mode.strip() or DEFAULT_JOB_MODE,
                since=since_date,
                until=until_date,
                force=force,
                apply=apply,
            )
            return result.as_dict()

        _echo_json(_run_with_pool(operation))

    @run_app.command(
        "requeue-failed",
        help=(
            "Reopen only failed prepare jobs, those that used up their attempts, as pending with the attempts "
            "reset. Done jobs are left alone; to parse finished dates again use reparse."
        ),
    )
    def requeue_failed_command(
        since: Annotated[
            str | None, typer.Option("--since", help="Only jobs whose date is YYYY-MM-DD or later.")
        ] = None,
        mode: Annotated[JobMode, typer.Option("--mode", help="prepare_jobs mode of the jobs.")] = JobMode.AUTO,
        apply: Annotated[
            bool, typer.Option("--apply", help="Reopen the jobs. Without it they are only listed.")
        ] = False,
        force: Annotated[
            bool, typer.Option("--force", help="With --apply, also ask for a forced parse of those dates.")
        ] = False,
    ) -> None:
        since_date = parse_date_option(since)

        async def operation(pool: Pool) -> dict[str, Any]:
            result = await requeue_failed_jobs(pool, mode=mode.value, since=since_date, force=force, apply=apply)
            return result.as_dict()

        _echo_json(_run_with_pool(operation))

    app.add_typer(run_app, name="run")


def register(app: typer.Typer) -> None:
    register_worker(app)
    register_run(app)
