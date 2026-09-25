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
    def __init__(self, dates: list[str], *, claim_valid: bool = True, heartbeat_alive: bool = True) -> None:
        self.dates = list(dates)
        self.claim_valid = claim_valid
        self.heartbeat_alive = heartbeat_alive
        self.generation = 0
        self.completed: list[dict[str, Any]] = []
        self.failed: list[dict[str, Any]] = []
        self.heartbeats: list[dict[str, Any]] = []

    def claim_prepare_job(self, *, mode, worker_id):
        if not self.dates:
            return None
        self.generation += 1
        return {
            "job_id": 100 + self.generation,
            "worker_id": worker_id,
            "claim_generation": self.generation,
            "date": self.dates.pop(0),
        }

    def heartbeat_prepare_job(self, *, job_id, worker_id, claim_generation):
        self.heartbeats.append({"job_id": job_id, "worker_id": worker_id, "claim_generation": claim_generation})
        return self.heartbeat_alive

    def complete_prepare_job(self, *, job_id, worker_id, claim_generation, result=None):
        self.completed.append({"job_id": job_id, "claim_generation": claim_generation, "result": result})
        return self.claim_valid

    def fail_prepare_job(self, *, job_id, worker_id, claim_generation, error):
        self.failed.append({"job_id": job_id, "claim_generation": claim_generation, "error": error})
        return self.claim_valid


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
    assert job_repository.completed[0]["job_id"] == 101
    assert job_repository.completed[0]["claim_generation"] == 1
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
    assert result["failures"][0]["date"] == "2026-04-07"
    assert job_repository.failed[0]["job_id"] == 101
    assert "all 3 paper(s) failed" in job_repository.failed[0]["error"]


def test_consume_queue_still_fails_job_on_date_level_exception(monkeypatch):
    job_repository = FakeJobRepository(["2026-04-07"])
    monkeypatch.setattr(prepare_papers, "PrepareJobRepository", lambda: job_repository)

    def raise_error(**kwargs):
        raise RuntimeError("mongo down")

    monkeypatch.setattr(prepare_papers, "run_prepare_papers", raise_error)

    result = prepare_papers.run_consume_prepare_queue(runtime="test")

    assert result["status"] == "failed"
    assert job_repository.failed == [{"job_id": 101, "claim_generation": 1, "error": "mongo down"}]
    assert result["failures"] == [{"date": "2026-04-07", "error": "mongo down"}]


def test_consume_queue_lost_claim_on_complete_is_recorded_not_raised(monkeypatch):
    job_repository = FakeJobRepository(["2026-04-07"], claim_valid=False)
    monkeypatch.setattr(prepare_papers, "PrepareJobRepository", lambda: job_repository)
    monkeypatch.setattr(prepare_papers, "run_prepare_papers", lambda **kwargs: _prepare_result("success", 2, 0))

    result = prepare_papers.run_consume_prepare_queue(runtime="test", worker_id="w1")

    assert result["status"] == "claim_lost"
    assert result["successes"] == []
    assert result["lost_claim_count"] == 1
    assert result["lost_claims"] == [{"date": "2026-04-07", "job_id": 101, "claim_generation": 1, "stage": "complete"}]


def test_consume_queue_lost_claim_on_fail_still_reports_failure(monkeypatch):
    job_repository = FakeJobRepository(["2026-04-07"], claim_valid=False)
    monkeypatch.setattr(prepare_papers, "PrepareJobRepository", lambda: job_repository)
    monkeypatch.setattr(prepare_papers, "run_prepare_papers", lambda **kwargs: _prepare_result("failed", 0, 1))

    result = prepare_papers.run_consume_prepare_queue(runtime="test")

    assert result["status"] == "failed"
    assert result["lost_claims"][0]["stage"] == "fail"


def test_consume_queue_lost_claim_with_other_success_is_partial(monkeypatch):
    job_repository = FakeJobRepository(["2026-04-07", "2026-04-08"])
    outcomes = iter([False, True])
    original_complete = job_repository.complete_prepare_job

    def complete(**kwargs):
        original_complete(**kwargs)
        return next(outcomes)

    job_repository.complete_prepare_job = complete  # type: ignore[method-assign]
    monkeypatch.setattr(prepare_papers, "PrepareJobRepository", lambda: job_repository)
    monkeypatch.setattr(prepare_papers, "run_prepare_papers", lambda **kwargs: _prepare_result("success", 1, 0))

    result = prepare_papers.run_consume_prepare_queue(runtime="test", max_jobs_per_run=2)

    assert result["status"] == "partial_failed"
    assert [item["date"] for item in result["successes"]] == ["2026-04-08"]
    assert [item["date"] for item in result["lost_claims"]] == ["2026-04-07"]


def test_consume_queue_heartbeats_each_paper_with_claim_token(monkeypatch):
    job_repository = FakeJobRepository(["2026-04-07"])
    monkeypatch.setattr(prepare_papers, "PrepareJobRepository", lambda: job_repository)
    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]
    _patch_prepare_run(monkeypatch, candidates, failing_ids=set())

    result = prepare_papers.run_consume_prepare_queue(runtime="test", worker_id="w1")

    assert result["status"] == "success"
    assert job_repository.heartbeats == [{"job_id": 101, "worker_id": "w1", "claim_generation": 1}] * 3
    assert len(job_repository.completed) == 1


def test_consume_queue_stops_processing_when_heartbeat_reports_lost_claim(monkeypatch):
    job_repository = FakeJobRepository(["2026-04-07"], heartbeat_alive=False)
    monkeypatch.setattr(prepare_papers, "PrepareJobRepository", lambda: job_repository)
    processed: list[str] = []
    candidates = [_candidate("a"), _candidate("b")]
    _patch_prepare_run(monkeypatch, candidates, failing_ids=set())
    original = prepare_papers.prepare_single_paper

    def tracking_prepare(candidate, **kwargs):
        processed.append(candidate["arxiv_id"])
        return original(candidate, **kwargs)

    monkeypatch.setattr(prepare_papers, "prepare_single_paper", tracking_prepare)

    result = prepare_papers.run_consume_prepare_queue(runtime="test")

    assert processed == []
    assert job_repository.completed == []
    assert job_repository.failed == []
    assert result["status"] == "claim_lost"
    assert result["lost_claims"][0]["stage"] == "heartbeat"


def test_consume_queue_tolerates_heartbeat_errors(monkeypatch):
    job_repository = FakeJobRepository(["2026-04-07"])

    def broken_heartbeat(**kwargs):
        raise ConnectionError("db blip")

    job_repository.heartbeat_prepare_job = broken_heartbeat  # type: ignore[method-assign]
    monkeypatch.setattr(prepare_papers, "PrepareJobRepository", lambda: job_repository)
    _patch_prepare_run(monkeypatch, [_candidate("a")], failing_ids=set())

    result = prepare_papers.run_consume_prepare_queue(runtime="test")

    assert result["status"] == "success"
    assert len(job_repository.completed) == 1


def test_prepare_candidates_without_heartbeat_is_unchanged():
    results, failures = prepare_papers.prepare_candidates(
        [_candidate("a")],
        parser=FakeParser(source="layout_pdf"),
        paper_repository=FakeRepository(),
    )
    assert [result["arxiv_id"] for result in results] == ["a"]
    assert failures == []


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
