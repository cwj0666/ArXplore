from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.pipeline import prepare_papers


class FakeParser:
    def __init__(self, *, source: str, text: str = "parsed text") -> None:
        self.source = source
        self.text = text

    def parse_from_pdf_url(self, pdf_url: str, *, fallback_text: str = ""):
        return SimpleNamespace(
            text=self.text,
            sections=[{"title": "Introduction", "text": self.text}],
            source=self.source,
            quality_metrics={"fallback_used": self.source == "fallback_abstract", "text_length": len(self.text)},
            artifacts={},
            parser_metadata={},
        )

    def build_chunks(self, text: str, *, sections: list[dict[str, Any]]):
        return [{"chunk_index": 0, "chunk_text": text, "section_title": "Introduction", "metadata": {"content_role": "body"}}]

    def summarize_chunks(self, chunks):
        return {"chunk_count": len(chunks)}


class FakeRepository:
    def __init__(self, existing_source: str | None = None) -> None:
        self.existing_source = existing_source
        self.calls: list[str] = []

    def save_paper(self, prepared):
        self.calls.append("save_paper")
        return prepared["arxiv_id"]

    def get_paper_fulltext_source(self, arxiv_id: str):
        self.calls.append("get_paper_fulltext_source")
        return self.existing_source

    def save_paper_fulltext(self, arxiv_id, **kwargs):
        self.calls.append("save_paper_fulltext")

    def save_paper_chunks(self, arxiv_id, chunks):
        self.calls.append("save_paper_chunks")


def _candidate(arxiv_id: str = "2604.00001") -> dict[str, Any]:
    return {
        "arxiv_id": arxiv_id,
        "prepared": {
            "arxiv_id": arxiv_id,
            "title": "Paper",
            "abstract": "An abstract.",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        },
    }


@pytest.mark.parametrize("existing_source", ["layout_pdf", "pdf"])
def test_fallback_abstract_does_not_overwrite_existing_pdf_fulltext(existing_source):
    repository = FakeRepository(existing_source=existing_source)

    result = prepare_papers.prepare_single_paper(
        _candidate(),
        parser=FakeParser(source="fallback_abstract"),
        paper_repository=repository,
    )

    assert result["skipped_fallback_overwrite"] is True
    assert result["existing_fulltext_source"] == existing_source
    assert result["saved_fulltext"] == 0
    assert result["saved_chunks"] == 0
    assert "save_paper_fulltext" not in repository.calls
    assert "save_paper_chunks" not in repository.calls
    assert "save_paper" in repository.calls


@pytest.mark.parametrize("existing_source", [None, "fallback_abstract"])
def test_fallback_abstract_is_saved_when_no_pdf_fulltext_exists(existing_source):
    repository = FakeRepository(existing_source=existing_source)

    result = prepare_papers.prepare_single_paper(
        _candidate(),
        parser=FakeParser(source="fallback_abstract"),
        paper_repository=repository,
    )

    assert "skipped_fallback_overwrite" not in result
    assert result["saved_fulltext"] == 1
    assert result["saved_chunks"] == 1
    assert repository.calls.count("save_paper_fulltext") == 1
    assert repository.calls.count("save_paper_chunks") == 1


def test_pdf_result_overwrites_without_source_lookup():
    repository = FakeRepository(existing_source="layout_pdf")

    result = prepare_papers.prepare_single_paper(
        _candidate(),
        parser=FakeParser(source="pdf"),
        paper_repository=repository,
    )

    assert "get_paper_fulltext_source" not in repository.calls
    assert result["saved_fulltext"] == 1
    assert repository.calls.count("save_paper_chunks") == 1


def test_aggregate_counts_skipped_fallback_overwrites():
    results = [
        {"arxiv_id": "a", "saved_paper": 1, "skipped_fallback_overwrite": True, "fallback_used": True},
        {"arxiv_id": "b", "saved_paper": 1},
    ]
    aggregated = prepare_papers.aggregate_prepare_results(
        results,
        normalized_date="2026-04-07",
        raw_count=2,
        deduplicated_ids=["a", "b"],
        selected_ids=["a", "b"],
        enriched_count=0,
        skipped_by_category=0,
        runtime="test",
        user=None,
    )
    assert aggregated["skipped_fallback_overwrites"] == 1
    assert aggregated["status"] == "success"
    assert aggregated["success_count"] == 2
    assert aggregated["failure_count"] == 0


def _patch_prepare_run(monkeypatch, candidates, failing_ids: set[str]):
    monkeypatch.setattr(
        prepare_papers,
        "load_prepare_candidates",
        lambda **kwargs: {
            "normalized_date": "2026-04-07",
            "raw_count": len(candidates),
            "deduplicated_ids": [c["arxiv_id"] for c in candidates],
            "selected_ids": [c["arxiv_id"] for c in candidates],
            "enriched_count": 0,
            "skipped_by_category": 0,
            "candidates": candidates,
        },
    )
    monkeypatch.setattr(prepare_papers, "PaperRepository", lambda: FakeRepository())
    monkeypatch.setattr(prepare_papers, "FulltextParser", lambda: FakeParser(source="layout_pdf"))

    def fake_prepare_single_paper(candidate, *, parser, paper_repository):
        if candidate["arxiv_id"] in failing_ids:
            raise NameError("name 'FulltextParser' is not defined")
        return {"arxiv_id": candidate["arxiv_id"], "saved_paper": 1, "saved_fulltext": 1, "saved_chunks": 3}

    monkeypatch.setattr(prepare_papers, "prepare_single_paper", fake_prepare_single_paper)


def test_run_prepare_papers_isolates_per_paper_failures(monkeypatch):
    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]
    _patch_prepare_run(monkeypatch, candidates, failing_ids={"b"})

    result = prepare_papers.run_prepare_papers(runtime="test", target_date="2026-04-07")

    assert result["status"] == "partial_failed"
    assert result["success_count"] == 2
    assert result["failure_count"] == 1
    assert result["prepared_arxiv_ids"] == ["a", "c"]
    assert result["saved_chunks"] == 6
    assert result["failures"][0]["arxiv_id"] == "b"
    assert "NameError" in result["failures"][0]["error"]


def test_run_prepare_papers_all_failed(monkeypatch):
    candidates = [_candidate("a"), _candidate("b")]
    _patch_prepare_run(monkeypatch, candidates, failing_ids={"a", "b"})

    result = prepare_papers.run_prepare_papers(runtime="test", target_date="2026-04-07")

    assert result["status"] == "failed"
    assert result["success_count"] == 0
    assert result["failure_count"] == 2


def test_run_prepare_papers_empty_date_is_success(monkeypatch):
    _patch_prepare_run(monkeypatch, [], failing_ids=set())

    result = prepare_papers.run_prepare_papers(runtime="test", target_date="2026-04-07")

    assert result["status"] == "success"
    assert result["success_count"] == 0
    assert result["failure_count"] == 0


class FakeJobRepository:
    def __init__(self, dates: list[str]) -> None:
        self.dates = list(dates)
        self.completed: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []

    def claim_prepare_job(self, *, mode, worker_id):
        if not self.dates:
            return None
        return {"date": self.dates.pop(0)}

    def complete_prepare_job(self, *, mode, target_date, result=None):
        self.completed.append({"date": target_date, "result": result})

    def fail_prepare_job(self, *, mode, target_date, error):
        self.failed.append({"date": target_date, "error": error})


def _prepare_result(status: str, success_count: int, failure_count: int) -> dict[str, Any]:
    return {
        "status": status,
        "saved_papers": success_count,
        "saved_fulltexts": success_count,
        "saved_chunks": success_count * 3,
        "prepared_arxiv_ids": [f"id{i}" for i in range(success_count)],
        "success_count": success_count,
        "failure_count": failure_count,
        "failures": [{"arxiv_id": f"bad{i}", "error": "NameError: boom"} for i in range(failure_count)],
    }


def test_consume_queue_completes_job_on_partial_failure(monkeypatch):
    job_repository = FakeJobRepository(["2026-04-07"])
    monkeypatch.setattr(prepare_papers, "PrepareJobRepository", lambda: job_repository)
    monkeypatch.setattr(prepare_papers, "run_prepare_papers", lambda **kwargs: _prepare_result("partial_failed", 2, 1))

    result = prepare_papers.run_consume_prepare_queue(runtime="test")

    assert result["status"] == "success"
    assert job_repository.failed == []
    assert len(job_repository.completed) == 1
    job_result = job_repository.completed[0]["result"]
    assert job_result["success_count"] == 2
    assert job_result["failure_count"] == 1
    assert job_result["failures"][0]["arxiv_id"] == "bad0"
    assert result["successes"][0]["paper_failure_count"] == 1


def test_consume_queue_fails_job_when_no_paper_succeeded(monkeypatch):
    job_repository = FakeJobRepository(["2026-04-07"])
    monkeypatch.setattr(prepare_papers, "PrepareJobRepository", lambda: job_repository)
    monkeypatch.setattr(prepare_papers, "run_prepare_papers", lambda **kwargs: _prepare_result("failed", 0, 3))

    result = prepare_papers.run_consume_prepare_queue(runtime="test")

    assert result["status"] == "failed"
    assert job_repository.completed == []
    assert job_repository.failed[0]["date"] == "2026-04-07"
    assert "all 3 paper(s) failed" in job_repository.failed[0]["error"]


def test_consume_queue_still_fails_job_on_date_level_exception(monkeypatch):
    job_repository = FakeJobRepository(["2026-04-07"])
    monkeypatch.setattr(prepare_papers, "PrepareJobRepository", lambda: job_repository)

    def raise_error(**kwargs):
        raise RuntimeError("mongo down")

    monkeypatch.setattr(prepare_papers, "run_prepare_papers", raise_error)

    result = prepare_papers.run_consume_prepare_queue(runtime="test")

    assert result["status"] == "failed"
    assert job_repository.failed == [{"date": "2026-04-07", "error": "mongo down"}]


class FakeRawStore:
    def __init__(self) -> None:
        self.saved_state: dict[str, Any] | None = None

    def load_pipeline_state(self, *, pipeline, name):
        return None

    def save_pipeline_state(self, *, pipeline, name, state):
        self.saved_state = state

    def has_daily_papers_response(self, *, date):
        return True


def test_backfill_stops_when_all_papers_of_a_date_failed(monkeypatch):
    raw_store = FakeRawStore()
    monkeypatch.setattr(prepare_papers, "RawPaperStore", lambda: raw_store)
    monkeypatch.setattr(prepare_papers, "run_prepare_papers", lambda **kwargs: _prepare_result("failed", 0, 2))

    result = prepare_papers.run_backfill_prepare_papers(
        runtime="test",
        cursor_date="2026-04-07",
        oldest_date="2026-04-01",
        batch_days=2,
    )

    assert result["status"] == "failed"
    assert result["stopped_reason"] == "prepare_failed"
    assert result["next_cursor_date"] == "2026-04-07"
    assert result["failures"][0]["date"] == "2026-04-07"
