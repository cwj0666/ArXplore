from datetime import date
from typing import Any

import pytest
from typer.testing import CliRunner

from arxplore.cli import app
from arxplore.db.repositories.scheduler_runs import RunStatus
from arxplore.pipeline import commands
from arxplore.pipeline.embed import EmbeddingBudget
from arxplore.pipeline.worker import WorkerOptions
from arxplore.scheduler.jobs import CollectParams, LangsmithCleanupParams, MaintenanceParams
from arxplore.scheduler.runner import RunOutcome

runner = CliRunner()


@pytest.fixture
def captured_worker(monkeypatch: pytest.MonkeyPatch, clean_env: Any) -> list[tuple[WorkerOptions, bool]]:
    calls: list[tuple[WorkerOptions, bool]] = []

    async def fake_run_worker(settings: Any, options: WorkerOptions, *, loop: bool) -> int:
        calls.append((options, loop))
        return 0

    monkeypatch.setattr(commands, "run_worker", fake_run_worker)
    return calls


def test_worker_defaults_match_the_legacy_cli(captured_worker: list[tuple[WorkerOptions, bool]]) -> None:
    result = runner.invoke(app, ["worker"])

    assert result.exit_code == 0, result.output
    options, loop = captured_worker[0]
    assert loop is False
    assert options == WorkerOptions(
        worker_id="default",
        max_jobs_per_run=1,
        max_papers=None,
        force=False,
        skip_embed=False,
        embedding=EmbeddingBudget(chunks_per_round=200, max_rounds_per_paper=100, backlog_chunks=400),
        wait_timeout_seconds=120.0,
        consume_backfill=False,
    )


def test_worker_options_are_normalized_like_the_legacy_worker(
    captured_worker: list[tuple[WorkerOptions, bool]],
) -> None:
    result = runner.invoke(
        app,
        [
            "worker",
            "--worker-id",
            "gpu-1",
            "--max-jobs-per-run",
            "0",
            "--max-papers",
            "0",
            "--embed-max-chunks",
            "0",
            "--embed-backlog-max-chunks",
            "-5",
            "--skip-embed",
            "--force",
            "--loop",
            "--wait-timeout-seconds",
            "15",
        ],
    )

    assert result.exit_code == 0, result.output
    options, loop = captured_worker[0]
    assert loop is True
    assert options.worker_id == "gpu-1"
    assert (options.max_jobs_per_run, options.max_papers, options.force, options.skip_embed) == (1, None, True, True)
    assert options.embedding == EmbeddingBudget(chunks_per_round=200, max_rounds_per_paper=100, backlog_chunks=0)
    assert options.wait_timeout_seconds == 15.0


def test_backfill_jobs_are_consumed_when_the_setting_is_on(
    captured_worker: list[tuple[WorkerOptions, bool]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PREPARE_BACKFILL_ENABLED", "true")

    result = runner.invoke(app, ["worker"])

    assert result.exit_code == 0, result.output
    assert captured_worker[0][0].consume_backfill is True
    assert captured_worker[0][0].modes == ("auto", "backfill")


def test_the_new_embedding_option_names_are_accepted(captured_worker: list[tuple[WorkerOptions, bool]]) -> None:
    result = runner.invoke(
        app,
        [
            "worker",
            "--embed-chunks-per-round",
            "50",
            "--embed-max-rounds-per-paper",
            "4",
            "--embed-backlog-budget",
            "9",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured_worker[0][0].embedding == EmbeddingBudget(
        chunks_per_round=50, max_rounds_per_paper=4, backlog_chunks=9
    )


@pytest.mark.parametrize("removed", ["--mode", "--cursor-date", "--oldest-date", "--batch-days", "--state-name"])
def test_removed_legacy_worker_options_are_rejected(
    captured_worker: list[tuple[WorkerOptions, bool]], removed: str
) -> None:
    result = runner.invoke(app, ["worker", removed, "x"])
    assert result.exit_code == 2
    assert captured_worker == []


@pytest.fixture
def captured_runs(monkeypatch: pytest.MonkeyPatch, clean_env: Any) -> list[tuple[str, Any]]:
    calls: list[tuple[str, Any]] = []

    async def fake_run_job_manually(settings: Any, definition: Any, params: Any) -> RunOutcome:
        calls.append((definition.name, params))
        return RunOutcome(run_id=7, status=RunStatus.SUCCESS, result={"ok": True}, error=None, attempts=1)

    monkeypatch.setattr(commands, "run_job_manually", fake_run_job_manually)
    return calls


def test_run_collect_accepts_the_dag_target_date(captured_runs: list[tuple[str, Any]]) -> None:
    result = runner.invoke(app, ["run", "collect", "--target-date", "2026-09-15"])

    assert result.exit_code == 0, result.output
    assert captured_runs == [("collect", CollectParams(target_date=date(2026, 9, 15)))]
    assert '"trigger": "manual"' in result.output
    assert '"run_id": 7' in result.output


def test_run_collect_rejects_a_malformed_date(captured_runs: list[tuple[str, Any]]) -> None:
    result = runner.invoke(app, ["run", "collect", "--target-date", "15/09/2026"])
    assert result.exit_code == 2
    assert captured_runs == []


def test_run_maintenance_accepts_the_dag_params(captured_runs: list[tuple[str, Any]]) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "maintenance",
            "--backfill-cursor-date",
            "2026-09-10",
            "--backfill-oldest-date",
            "2026-01-01",
            "--backfill-batch-days",
            "5",
            "--backfill-state-name",
            " ",
            "--enrich-max-papers",
            "12",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured_runs == [
        (
            "maintenance",
            MaintenanceParams(
                backfill_cursor_date=date(2026, 9, 10),
                backfill_oldest_date=date(2026, 1, 1),
                backfill_batch_days=5,
                backfill_state_name="default",
                enrich_max_papers=12,
            ),
        )
    ]


def test_run_langsmith_cleanup_accepts_the_dag_params(captured_runs: list[tuple[str, Any]]) -> None:
    result = runner.invoke(
        app,
        ["run", "langsmith-cleanup", "--days", "7", "--project-name", "P", "--user", " ", "--limit", "5", "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert captured_runs == [
        ("langsmith_cleanup", LangsmithCleanupParams(days=7, project_name="P", user=None, limit=5, dry_run=True))
    ]


def test_a_failed_manual_run_exits_with_one(monkeypatch: pytest.MonkeyPatch, clean_env: Any) -> None:
    async def failing(settings: Any, definition: Any, params: Any) -> RunOutcome:
        return RunOutcome(run_id=1, status=RunStatus.FAILED, result=None, error="RuntimeError: x", attempts=1)

    monkeypatch.setattr(commands, "run_job_manually", failing)

    result = runner.invoke(app, ["run", "collect"])

    assert result.exit_code == 1
    assert '"error": "RuntimeError: x"' in result.output
