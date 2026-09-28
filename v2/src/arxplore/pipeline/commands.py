import asyncio
import json
import logging
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
from arxplore.pipeline.worker import DEFAULT_WAIT_TIMEOUT_SECONDS, DEFAULT_WORKER_NAME, WorkerOptions, run_worker

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=1)


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def positive_or_default(value: int, default: int) -> int:
    return value if value > 0 else default


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
        )
        try:
            exit_code = asyncio.run(run_worker(get_settings(), options, loop=loop))
        except (MigrationError, ValueError) as exc:
            raise _fail(str(exc)) from exc
        raise typer.Exit(code=exit_code)


def register(app: typer.Typer) -> None:
    register_worker(app)
