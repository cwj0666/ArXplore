from __future__ import annotations

import argparse
from typing import Any


from src.pipeline import prepare_worker


def _args(**overrides: Any) -> argparse.Namespace:
    values = {
        "mode": "auto",
        "worker_id": "test",
        "state_name": "",
        "max_papers": "",
        "max_jobs_per_run": 1,
        "embed_max_chunks": 200,
        "embed_backlog_max_chunks": 400,
        "skip_embed": False,
        "cursor_date": "",
        "oldest_date": "",
        "batch_days": 3,
        "loop": False,
        "sleep_seconds": 0.0,
        "wait_timeout_seconds": 0.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _prepare_result(*, status: str = "no_op", successes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    successes = successes or []
    return {
        "stage": "consume_prepare_queue",
        "status": status,
        "success_count": len(successes),
        "failure_count": 0,
        "successes": successes,
        "failures": [],
    }


class FakeEmbed:
    def __init__(self, *, backlog_chunks: int = 0, backlog_error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.backlog_chunks = backlog_chunks
        self.backlog_error = backlog_error

    def __call__(self, *, runtime, user, max_chunks, arxiv_id):
        self.calls.append({"max_chunks": max_chunks, "arxiv_id": arxiv_id})
        if arxiv_id is None:
            if self.backlog_error is not None:
                raise self.backlog_error
            selected = min(max_chunks, self.backlog_chunks)
            self.backlog_chunks -= selected
            status = "success" if selected else "no_op"
            return {"status": status, "selected_chunk_count": selected, "embedded_chunk_count": selected}
        return {"status": "success", "selected_chunk_count": 5, "embedded_chunk_count": 5}


def _patch(monkeypatch, prepare_result: dict[str, Any], embed: FakeEmbed) -> None:
    monkeypatch.setattr(prepare_worker, "run_consume_prepare_queue", lambda **kwargs: prepare_result)
    monkeypatch.setattr(prepare_worker, "run_embed_papers", embed)


def test_backlog_runs_without_prepare_success(monkeypatch):
    embed = FakeEmbed(backlog_chunks=250)
    _patch(monkeypatch, _prepare_result(status="no_op"), embed)

    result = prepare_worker._run_once(_args(embed_max_chunks=200, embed_backlog_max_chunks=400))

    assert [call["arxiv_id"] for call in embed.calls] == [None, None]
    assert [call["max_chunks"] for call in embed.calls] == [200, 200]
    assert result["embed"]["backlog_embedded_chunk_count"] == 250
    assert result["embed"]["status"] == "success"
    assert result["embed"]["backlog_error"] is None
    assert result["status"] == "no_op"


def test_backlog_runs_after_failed_prepare(monkeypatch):
    embed = FakeEmbed(backlog_chunks=10)
    _patch(monkeypatch, _prepare_result(status="failed"), embed)

    result = prepare_worker._run_once(_args())

    assert embed.calls == [{"max_chunks": 200, "arxiv_id": None}]
    assert result["embed"]["backlog_embedded_chunk_count"] == 10


def test_skip_embed_calls_nothing(monkeypatch):
    embed = FakeEmbed(backlog_chunks=100)
    _patch(
        monkeypatch,
        _prepare_result(status="success", successes=[{"date": "2026-04-07", "prepared_arxiv_ids": ["a"]}]),
        embed,
    )

    result = prepare_worker._run_once(_args(skip_embed=True))

    assert embed.calls == []
    assert result["embed"]["status"] == "skipped"


def test_backlog_exception_is_recorded_not_raised(monkeypatch):
    embed = FakeEmbed(backlog_error=RuntimeError("openai 500"))
    _patch(monkeypatch, _prepare_result(status="no_op"), embed)

    result = prepare_worker._run_once(_args())

    assert result["embed"]["backlog_error"] == "RuntimeError: openai 500"
    assert result["embed"]["status"] == "partial_failed"
    assert result["status"] == "no_op"


def test_backlog_exception_after_prepare_success_keeps_per_paper_embeds(monkeypatch):
    embed = FakeEmbed(backlog_error=RuntimeError("openai 500"))
    _patch(
        monkeypatch,
        _prepare_result(status="success", successes=[{"date": "2026-04-07", "prepared_arxiv_ids": ["a", "b"]}]),
        embed,
    )

    result = prepare_worker._run_once(_args())

    assert [call["arxiv_id"] for call in embed.calls] == ["a", "b", None]
    assert result["embed"]["embedded_chunk_count"] == 10
    assert result["embed"]["backlog_error"] == "RuntimeError: openai 500"
    assert result["status"] == "partial_failed"


def test_no_backlog_budget_and_no_success_is_noop(monkeypatch):
    embed = FakeEmbed(backlog_chunks=100)
    _patch(monkeypatch, _prepare_result(status="no_op"), embed)

    result = prepare_worker._run_once(_args(embed_backlog_max_chunks=0))

    assert embed.calls == []
    assert result["embed"]["status"] == "no_op"


def test_unexpected_embed_error_does_not_propagate(monkeypatch):
    _patch(monkeypatch, _prepare_result(status="no_op"), FakeEmbed())

    def explode(**kwargs):
        raise KeyError("bug")

    monkeypatch.setattr(prepare_worker, "_run_embed_after_prepare", explode)

    result = prepare_worker._run_once(_args())

    assert result["embed"]["status"] == "failed"
    assert "KeyError" in result["embed"]["error"]


def test_main_ensures_schema_once_at_startup(monkeypatch):
    calls: list[str] = []

    class FakePaperRepository:
        def ensure_schema(self):
            calls.append("papers")

    class FakePrepareJobRepository:
        def ensure_schema(self):
            calls.append("prepare_jobs")

    monkeypatch.setattr(prepare_worker, "PaperRepository", FakePaperRepository)
    monkeypatch.setattr(prepare_worker, "PrepareJobRepository", FakePrepareJobRepository)
    monkeypatch.setattr(prepare_worker, "_run_once", lambda args: {"status": "no_op"})
    monkeypatch.setattr("sys.argv", ["prepare_worker", "--mode", "auto"])

    assert prepare_worker.main() == 0
    assert calls == ["papers", "prepare_jobs"]
