from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from arxplore.config import AppSettings
from arxplore.pipeline.langsmith_cleanup import CleanupOptions, cleanup_langsmith_runs

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
ENABLED = AppSettings.model_validate({"LANGSMITH_API_KEY": "ls-key", "LANGSMITH_PROJECT": "ArXplore"})


def _run(run_id: str, days_ago: float, *, user: str | None = None, naive: bool = False) -> SimpleNamespace:
    started = NOW - timedelta(days=days_ago)
    if naive:
        started = started.replace(tzinfo=None)
    return SimpleNamespace(id=run_id, start_time=started, metadata={"user": user} if user else {})


@dataclass
class FakeStore:
    runs: list[Any]
    projects: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    def list_runs(self, *, project_name: str) -> list[Any]:
        self.projects.append(project_name)
        return self.runs

    def delete_run(self, run_id: str) -> None:
        self.deleted.append(run_id)


def test_runs_older_than_the_window_are_deleted() -> None:
    store = FakeStore([_run("new", 1), _run("old", 20), _run("naive-old", 15, naive=True), _run("edge", 14)])

    result = cleanup_langsmith_runs(ENABLED, CleanupOptions(), store=store, now=NOW)

    assert store.projects == ["ArXplore"]
    assert store.deleted == ["old", "naive-old"]
    assert result["matched_count"] == 2
    assert result["deleted_count"] == 2
    assert result["deleted_run_ids"] == ["old", "naive-old"]
    assert result["cutoff"] == (NOW - timedelta(days=14)).isoformat()
    assert [run["run_id"] for run in result["matched_runs"]] == ["old", "naive-old"]


def test_runs_without_a_start_time_are_skipped() -> None:
    store = FakeStore([SimpleNamespace(id="none", start_time=None, metadata={}), _run("old", 30)])
    cleanup_langsmith_runs(ENABLED, CleanupOptions(), store=store, now=NOW)
    assert store.deleted == ["old"]


def test_user_and_project_filters_apply() -> None:
    store = FakeStore([_run("mine", 30, user="me"), _run("theirs", 30, user="you"), _run("anon", 30)])

    result = cleanup_langsmith_runs(ENABLED, CleanupOptions(user="me", project_name="Other"), store=store, now=NOW)

    assert store.projects == ["Other"]
    assert store.deleted == ["mine"]
    assert result["matched_runs"] == [
        {"run_id": "mine", "start_time": (NOW - timedelta(days=30)).isoformat(), "user": "me"}
    ]


def test_dry_run_deletes_nothing() -> None:
    store = FakeStore([_run("old", 30)])

    result = cleanup_langsmith_runs(ENABLED, CleanupOptions(dry_run=True), store=store, now=NOW)

    assert store.deleted == []
    assert (result["matched_count"], result["deleted_count"], result["dry_run"]) == (1, 0, True)


def test_limit_counts_runs_to_delete() -> None:
    store = FakeStore([_run("new-1", 1), _run("old-1", 20), _run("new-2", 2), _run("old-2", 21), _run("old-3", 22)])

    result = cleanup_langsmith_runs(ENABLED, CleanupOptions(limit=2), store=store, now=NOW)

    assert store.deleted == ["old-1", "old-2"]
    assert (result["limit"], result["matched_count"]) == (2, 2)


def test_limit_counts_only_runs_of_the_requested_user() -> None:
    store = FakeStore([_run("theirs", 30, user="you"), _run("mine-1", 30, user="me"), _run("mine-2", 31, user="me")])

    cleanup_langsmith_runs(ENABLED, CleanupOptions(user="me", limit=1), store=store, now=NOW)

    assert store.deleted == ["mine-1"]


def test_days_below_one_are_rejected_first() -> None:
    with pytest.raises(ValueError):
        cleanup_langsmith_runs(AppSettings.model_validate({}), CleanupOptions(days=0), store=FakeStore([]), now=NOW)


@pytest.mark.parametrize("values", [{}, {"LANGSMITH_API_KEY": "  "}, {"LANGSMITH_TRACING": "false"}])
def test_without_an_api_key_the_cleanup_is_skipped(values: dict[str, str]) -> None:
    store = FakeStore([_run("old", 30)])

    result = cleanup_langsmith_runs(AppSettings.model_validate(values), CleanupOptions(), store=store, now=NOW)

    assert store.projects == []
    assert store.deleted == []
    assert (result["status"], result["reason"], result["matched_count"], result["deleted_count"]) == (
        "skipped",
        "no_langsmith_api_key",
        0,
        0,
    )


def test_a_key_is_enough_even_when_tracing_is_off() -> None:
    store = FakeStore([_run("old", 30)])
    settings = AppSettings.model_validate({"LANGSMITH_API_KEY": "ls-key", "LANGSMITH_TRACING": "false"})

    result = cleanup_langsmith_runs(settings, CleanupOptions(), store=store, now=NOW)

    assert result["status"] == "success"
    assert store.deleted == ["old"]
