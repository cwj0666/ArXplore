import asyncio
from datetime import date
from typing import Any

import pytest
from typer.testing import CliRunner

from arxplore.cli import app
from arxplore.db.repositories.scheduler_runs import RunStatus
from arxplore.pipeline import commands
from arxplore.pipeline.embed import EmbeddingBudget
from arxplore.pipeline.worker import WorkerOptions
from arxplore.scheduler import commands as scheduler_commands
from arxplore.scheduler.jobs import (
    COLLECT_RETRY,
    NO_RETRY,
    CollectParams,
    LangsmithCleanupParams,
    MaintenanceParams,
    RetryPolicy,
)
from arxplore.scheduler.runner import RunOutcome

runner = CliRunner()
retries: list[RetryPolicy] = []


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
    retries.clear()

    async def fake_run_job_manually(
        settings: Any, definition: Any, params: Any, *, retry: RetryPolicy = NO_RETRY
    ) -> RunOutcome:
        calls.append((definition.name, params))
        retries.append(retry)
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
    async def failing(settings: Any, definition: Any, params: Any, *, retry: RetryPolicy = NO_RETRY) -> RunOutcome:
        return RunOutcome(run_id=1, status=RunStatus.FAILED, result=None, error="RuntimeError: x", attempts=1)

    monkeypatch.setattr(commands, "run_job_manually", failing)

    result = runner.invoke(app, ["run", "collect"])

    assert result.exit_code == 1
    assert '"error": "RuntimeError: x"' in result.output


def test_a_manual_collect_does_not_retry_unless_asked(captured_runs: list[tuple[str, Any]]) -> None:
    assert runner.invoke(app, ["run", "collect"]).exit_code == 0
    assert runner.invoke(app, ["run", "collect", "--retries", "2", "--retry-delay", "1.5"]).exit_code == 0
    assert retries == [RetryPolicy(retries=0, delay_seconds=600.0), RetryPolicy(retries=2, delay_seconds=1.5)]


def test_the_scheduler_retries_collect_three_times_ten_minutes_apart_by_default(
    monkeypatch: pytest.MonkeyPatch, clean_env: Any
) -> None:
    received: list[dict[str, RetryPolicy]] = []

    async def fake_run_scheduler(settings: Any, *, retries: dict[str, RetryPolicy]) -> None:
        received.append(retries)

    monkeypatch.setattr(scheduler_commands, "run_scheduler", fake_run_scheduler)

    assert runner.invoke(app, ["scheduler"]).exit_code == 0
    assert runner.invoke(app, ["scheduler", "--collect-retries", "1", "--collect-retry-delay", "30"]).exit_code == 0
    assert received == [
        {"collect": COLLECT_RETRY},
        {"collect": RetryPolicy(retries=1, delay_seconds=30.0)},
    ]
    assert RetryPolicy(retries=3, delay_seconds=600.0) == COLLECT_RETRY


class _Result:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def as_dict(self) -> dict[str, Any]:
        return self.payload


@pytest.fixture
def corpus_calls(monkeypatch: pytest.MonkeyPatch, clean_env: Any) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    def run_with_fake_pool(operation: Any) -> dict[str, Any]:
        return asyncio.run(operation("pool"))

    async def fake_rechunk(pool: Any, **kwargs: Any) -> _Result:
        calls.append(("rechunk", kwargs))
        return _Result({"stage": "rechunk"})

    async def fake_reparse(pool: Any, **kwargs: Any) -> _Result:
        calls.append(("reparse", kwargs))
        return _Result({"stage": "reparse"})

    monkeypatch.setattr(commands, "_run_with_pool", run_with_fake_pool)
    monkeypatch.setattr(commands, "rechunk_corpus", fake_rechunk)
    monkeypatch.setattr(commands, "reparse_dates", fake_reparse)
    return calls


def test_rechunk_and_reparse_are_dry_runs_by_default(corpus_calls: list[tuple[str, dict[str, Any]]]) -> None:
    assert runner.invoke(app, ["run", "rechunk"]).exit_code == 0
    assert runner.invoke(app, ["run", "reparse"]).exit_code == 0
    assert corpus_calls == [
        ("rechunk", {"arxiv_ids": None, "apply": False}),
        ("reparse", {"mode": "auto", "since": None, "until": None, "force": True, "apply": False}),
    ]


def test_rechunk_and_reparse_options(corpus_calls: list[tuple[str, dict[str, Any]]]) -> None:
    result = runner.invoke(app, ["run", "rechunk", "--arxiv-id", "2609.00001", "--arxiv-id", "2609.00002", "--apply"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app,
        [
            "run",
            "reparse",
            "--since",
            "2026-09-11",
            "--until",
            "2026-09-24",
            "--mode",
            "backfill",
            "--no-force",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    assert corpus_calls == [
        ("rechunk", {"arxiv_ids": ["2609.00001", "2609.00002"], "apply": True}),
        (
            "reparse",
            {
                "mode": "backfill",
                "since": date(2026, 9, 11),
                "until": date(2026, 9, 24),
                "force": False,
                "apply": True,
            },
        ),
    ]


@pytest.fixture
def requeue_calls(monkeypatch: pytest.MonkeyPatch, clean_env: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def run_with_fake_pool(operation: Any) -> dict[str, Any]:
        return asyncio.run(operation("pool"))

    async def fake_requeue(pool: Any, **kwargs: Any) -> _Result:
        calls.append(kwargs)
        return _Result({"stage": "requeue_failed", "applied": kwargs["apply"]})

    monkeypatch.setattr(commands, "_run_with_pool", run_with_fake_pool)
    monkeypatch.setattr(commands, "requeue_failed_jobs", fake_requeue)
    return calls


def test_requeue_failed_is_a_dry_run_for_auto_jobs_by_default(requeue_calls: list[dict[str, Any]]) -> None:
    result = runner.invoke(app, ["run", "requeue-failed"])

    assert result.exit_code == 0, result.output
    assert requeue_calls == [{"mode": "auto", "since": None, "force": False, "apply": False}]
    assert '"stage": "requeue_failed"' in result.output


def test_requeue_failed_takes_the_legacy_script_options(requeue_calls: list[dict[str, Any]]) -> None:
    result = runner.invoke(
        app, ["run", "requeue-failed", "--since", "2026-04-07", "--mode", "backfill", "--apply", "--force"]
    )

    assert result.exit_code == 0, result.output
    assert requeue_calls == [{"mode": "backfill", "since": date(2026, 4, 7), "force": True, "apply": True}]


@pytest.mark.parametrize("arguments", [["--mode", "manual"], ["--since", "2026/04/07"]])
def test_requeue_failed_rejects_unknown_modes_and_dates(
    requeue_calls: list[dict[str, Any]], arguments: list[str]
) -> None:
    result = runner.invoke(app, ["run", "requeue-failed", *arguments])
    assert result.exit_code == 2
    assert requeue_calls == []
