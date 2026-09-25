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
        return [
            {
                "chunk_index": 0,
                "chunk_text": text,
                "section_title": "Introduction",
                "metadata": {"content_role": "body"},
            }
        ]

    def summarize_chunks(self, chunks):
        return {"chunk_count": len(chunks)}


class FakeRepository:
    def __init__(
        self,
        existing_source: str | None = None,
        *,
        existing_hash: str | None = None,
        existing_chunk_count: int = 1,
        chunks_replaced: bool = True,
        chunk_failures: int = 0,
    ) -> None:
        self.existing = (
            {"source": existing_source, "content_hash": existing_hash, "chunk_count": existing_chunk_count}
            if existing_source is not None
            else None
        )
        self.chunks_replaced = chunks_replaced
        self.chunk_failures = chunk_failures
        self.calls: list[str] = []
        self.saved_fulltexts: list[dict[str, Any]] = []

    def save_paper(self, prepared):
        self.calls.append("save_paper")
        return prepared["arxiv_id"]

    def get_paper_fulltext_state(self, arxiv_id: str):
        self.calls.append("get_paper_fulltext_state")
        return self.existing

    def save_paper_fulltext(self, arxiv_id, **kwargs):
        self.calls.append("save_paper_fulltext")
        self.saved_fulltexts.append(kwargs)
        chunk_count = self.existing["chunk_count"] if self.existing else 0
        self.existing = {"source": kwargs["source"], "content_hash": kwargs["content_hash"], "chunk_count": chunk_count}

    def save_paper_chunks(self, arxiv_id, chunks):
        self.calls.append("save_paper_chunks")
        if self.chunk_failures:
            self.chunk_failures -= 1
            raise RuntimeError("chunk write failed")
        if self.existing is not None:
            self.existing["chunk_count"] = len(chunks)
        return self.chunks_replaced

    def update_paper_fulltext_content_hash(self, arxiv_id, content_hash):
        self.calls.append("update_paper_fulltext_content_hash")
        if self.existing is not None:
            self.existing["content_hash"] = content_hash


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


PARSED_HASH = prepare_papers.compute_fulltext_content_hash("parsed text", [{"title": "Introduction"}])


@pytest.mark.parametrize(
    "existing_source,new_source",
    [
        ("layout_pdf", "fallback_abstract"),
        ("pdf", "fallback_abstract"),
        ("layout_pdf", "pdf"),
    ],
)
def test_lower_ranked_source_does_not_overwrite_existing_fulltext(existing_source, new_source):
    repository = FakeRepository(existing_source=existing_source, existing_hash="old")

    result = prepare_papers.prepare_single_paper(
        _candidate(),
        parser=FakeParser(source=new_source),
        paper_repository=repository,
    )

    assert result["skipped_lower_rank_overwrite"] is True
    assert result["existing_fulltext_source"] == existing_source
    assert result["saved_fulltext"] == 0
    assert result["saved_chunks"] == 0
    assert "save_paper_fulltext" not in repository.calls
    assert "save_paper_chunks" not in repository.calls
    assert "save_paper" in repository.calls


def test_lower_ranked_source_is_saved_when_existing_fulltext_has_no_chunks():
    repository = FakeRepository(existing_source="layout_pdf", existing_hash=None, existing_chunk_count=0)

    result = prepare_papers.prepare_single_paper(
        _candidate(),
        parser=FakeParser(source="pdf"),
        paper_repository=repository,
    )

    assert "skipped_lower_rank_overwrite" not in result
    assert result["saved_fulltext"] == 1
    assert result["saved_chunks"] > 0
    assert repository.existing["source"] == "pdf"


def test_force_overwrites_with_lower_ranked_source():
    repository = FakeRepository(existing_source="layout_pdf", existing_hash="old")

    result = prepare_papers.prepare_single_paper(
        _candidate(),
        parser=FakeParser(source="pdf"),
        paper_repository=repository,
        force=True,
    )

    assert "skipped_lower_rank_overwrite" not in result
    assert result["saved_fulltext"] == 1
    assert result["saved_chunks"] == 1
    assert repository.saved_fulltexts[0]["source"] == "pdf"


@pytest.mark.parametrize(
    "existing_source,new_source",
    [
        (None, "fallback_abstract"),
        ("fallback_abstract", "fallback_abstract"),
        ("fallback_abstract", "pdf"),
        ("pdf", "layout_pdf"),
        ("layout_pdf", "layout_pdf"),
        ("legacy_source", "fallback_abstract"),
    ],
)
def test_equal_or_higher_ranked_changed_content_is_saved(existing_source, new_source):
    repository = FakeRepository(existing_source=existing_source, existing_hash="different")

    result = prepare_papers.prepare_single_paper(
        _candidate(),
        parser=FakeParser(source=new_source),
        paper_repository=repository,
    )

    assert "skipped_lower_rank_overwrite" not in result
    assert "skipped_unchanged" not in result
    assert result["saved_fulltext"] == 1
    assert result["saved_chunks"] == 1
    assert repository.saved_fulltexts[0]["content_hash"] is None
    assert repository.existing["content_hash"] == PARSED_HASH
    assert result["content_hash"] == PARSED_HASH
    assert repository.calls[-2:] == ["save_paper_chunks", "update_paper_fulltext_content_hash"]


def test_unchanged_content_skips_fulltext_and_chunk_replacement():
    repository = FakeRepository(existing_source="layout_pdf", existing_hash=PARSED_HASH)

    result = prepare_papers.prepare_single_paper(
        _candidate(),
        parser=FakeParser(source="layout_pdf"),
        paper_repository=repository,
    )

    assert result["skipped_unchanged"] is True
    assert result["saved_fulltext"] == 0
    assert result["saved_chunks"] == 0
    assert repository.calls == ["save_paper", "get_paper_fulltext_state"]


def test_chunk_write_failure_does_not_mark_content_unchanged_on_retry():
    repository = FakeRepository(chunk_failures=1)
    parser = FakeParser(source="layout_pdf")

    with pytest.raises(RuntimeError):
        prepare_papers.prepare_single_paper(_candidate(), parser=parser, paper_repository=repository)
    assert repository.existing["content_hash"] is None

    retried = prepare_papers.prepare_single_paper(_candidate(), parser=parser, paper_repository=repository)

    assert "skipped_unchanged" not in retried
    assert retried["saved_chunks"] == 1
    assert repository.calls.count("save_paper_chunks") == 2
    assert repository.existing == {"source": "layout_pdf", "content_hash": PARSED_HASH, "chunk_count": 1}

    third = prepare_papers.prepare_single_paper(_candidate(), parser=parser, paper_repository=repository)
    assert third["skipped_unchanged"] is True


def test_matching_hash_without_stored_chunks_is_rewritten():
    repository = FakeRepository(existing_source="layout_pdf", existing_hash=PARSED_HASH, existing_chunk_count=0)

    result = prepare_papers.prepare_single_paper(
        _candidate(),
        parser=FakeParser(source="layout_pdf"),
        paper_repository=repository,
    )

    assert "skipped_unchanged" not in result
    assert result["saved_chunks"] == 1


def test_same_hash_from_different_source_is_not_treated_as_unchanged():
    repository = FakeRepository(existing_source="pdf", existing_hash=PARSED_HASH)

    result = prepare_papers.prepare_single_paper(
        _candidate(),
        parser=FakeParser(source="layout_pdf"),
        paper_repository=repository,
    )

    assert "skipped_unchanged" not in result
    assert result["saved_fulltext"] == 1


def test_force_rewrites_unchanged_content():
    repository = FakeRepository(existing_source="layout_pdf", existing_hash=PARSED_HASH)

    result = prepare_papers.prepare_single_paper(
        _candidate(),
        parser=FakeParser(source="layout_pdf"),
        paper_repository=repository,
        force=True,
    )

    assert "skipped_unchanged" not in result
    assert repository.calls.count("save_paper_fulltext") == 1
    assert repository.calls.count("save_paper_chunks") == 1


def test_identical_chunks_are_reported_as_kept():
    repository = FakeRepository(existing_source="layout_pdf", existing_hash=None, chunks_replaced=False)

    result = prepare_papers.prepare_single_paper(
        _candidate(),
        parser=FakeParser(source="layout_pdf"),
        paper_repository=repository,
    )

    assert result["saved_fulltext"] == 1
    assert result["saved_chunks"] == 0
    assert result["chunks_unchanged"] is True


def test_content_hash_normalizes_whitespace_and_tracks_section_titles():
    base = prepare_papers.compute_fulltext_content_hash("a  b\nc", [{"title": "Intro"}])
    assert base == prepare_papers.compute_fulltext_content_hash(" a b c ", [{"title": " Intro "}])
    assert base != prepare_papers.compute_fulltext_content_hash("a b c", [{"title": "Method"}])
    assert base != prepare_papers.compute_fulltext_content_hash("a b d", [{"title": "Intro"}])


def test_fulltext_source_rank_order():
    ranks = [prepare_papers.fulltext_source_rank(source) for source in ("layout_pdf", "pdf", "fallback_abstract", None)]
    assert ranks == sorted(ranks, reverse=True)
    assert len(set(ranks)) == 4


def test_aggregate_counts_skip_reasons():
    results = [
        {"arxiv_id": "a", "saved_paper": 1, "skipped_lower_rank_overwrite": True, "fallback_used": True},
        {"arxiv_id": "b", "saved_paper": 1, "skipped_unchanged": True},
        {"arxiv_id": "c", "saved_paper": 1, "saved_fulltext": 1, "chunks_unchanged": True},
    ]
    aggregated = prepare_papers.aggregate_prepare_results(
        results,
        normalized_date="2026-04-07",
        raw_count=3,
        deduplicated_ids=["a", "b", "c"],
        selected_ids=["a", "b", "c"],
        enriched_count=0,
        skipped_by_category=0,
        runtime="test",
        user=None,
    )
    assert aggregated["skipped_lower_rank_overwrites"] == 1
    assert aggregated["skipped_unchanged"] == 1
    assert aggregated["unchanged_chunk_sets"] == 1
    assert aggregated["status"] == "success"
    assert aggregated["success_count"] == 3
    assert aggregated["failure_count"] == 0


def _patch_prepare_run(monkeypatch, candidates, failing_ids: set[str]) -> list[bool]:
    forced: list[bool] = []
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

    def fake_prepare_single_paper(candidate, *, parser, paper_repository, force=False):
        forced.append(force)
        if candidate["arxiv_id"] in failing_ids:
            raise NameError("name 'FulltextParser' is not defined")
        return {"arxiv_id": candidate["arxiv_id"], "saved_paper": 1, "saved_fulltext": 1, "saved_chunks": 3}

    monkeypatch.setattr(prepare_papers, "prepare_single_paper", fake_prepare_single_paper)
    return forced


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
    def __init__(
        self,
        dates: list[str],
        *,
        claim_valid: bool = True,
        heartbeat_alive: bool = True,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.dates = list(dates)
        self.payload = payload or {}
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
            "payload": self.payload,
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


def test_consume_queue_retries_job_on_partial_failure(monkeypatch):
    job_repository = FakeJobRepository(["2026-04-07"])
    monkeypatch.setattr(prepare_papers, "PrepareJobRepository", lambda: job_repository)
    monkeypatch.setattr(prepare_papers, "run_prepare_papers", lambda **kwargs: _prepare_result("partial_failed", 2, 1))

    result = prepare_papers.run_consume_prepare_queue(runtime="test")

    assert result["status"] == "failed"
    assert job_repository.completed == []
    assert len(job_repository.failed) == 1
    assert job_repository.failed[0]["job_id"] == 101
    assert job_repository.failed[0]["claim_generation"] == 1
    error = job_repository.failed[0]["error"]
    assert "1 of 3 paper(s) failed" in error
    assert "success=2, failure=1" in error
    assert "NameError: boom" in error
    assert result["successes"] == []
    assert result["failures"] == [{"date": "2026-04-07", "error": error, "prepared_arxiv_ids": ["id0", "id1"]}]


def test_consume_queue_completes_job_only_without_paper_failures(monkeypatch):
    job_repository = FakeJobRepository(["2026-04-07"])
    monkeypatch.setattr(prepare_papers, "PrepareJobRepository", lambda: job_repository)
    monkeypatch.setattr(prepare_papers, "run_prepare_papers", lambda **kwargs: _prepare_result("success", 2, 0))

    result = prepare_papers.run_consume_prepare_queue(runtime="test")

    assert result["status"] == "success"
    assert job_repository.failed == []
    assert job_repository.completed[0]["result"]["success_count"] == 2
    assert job_repository.completed[0]["result"]["failure_count"] == 0
    assert result["successes"][0]["paper_failure_count"] == 0


def test_consume_queue_counts_job_without_target_date_as_failure(monkeypatch):
    job_repository = FakeJobRepository([""])
    monkeypatch.setattr(prepare_papers, "PrepareJobRepository", lambda: job_repository)
    monkeypatch.setattr(prepare_papers, "run_prepare_papers", lambda **kwargs: pytest.fail("must not run"))

    result = prepare_papers.run_consume_prepare_queue(runtime="test")

    assert result["status"] == "failed"
    assert result["failure_count"] == 1
    assert result["failures"] == [{"date": "", "job_id": 101, "error": "missing target_date"}]
    assert job_repository.failed[0]["error"] == "missing target_date"


@pytest.mark.parametrize(
    "payload,argument,expected",
    [
        ({}, False, False),
        ({"force": True}, False, True),
        ({"force": "yes"}, False, False),
        ({}, True, True),
    ],
)
def test_consume_queue_plumbs_force_from_payload_or_argument(monkeypatch, payload, argument, expected):
    job_repository = FakeJobRepository(["2026-04-07"], payload=payload)
    monkeypatch.setattr(prepare_papers, "PrepareJobRepository", lambda: job_repository)
    forced = _patch_prepare_run(monkeypatch, [_candidate("a")], failing_ids=set())

    result = prepare_papers.run_consume_prepare_queue(runtime="test", force=argument)

    assert result["status"] == "success"
    assert forced == [expected]


def test_run_prepare_papers_defaults_to_not_forced(monkeypatch):
    forced = _patch_prepare_run(monkeypatch, [_candidate("a")], failing_ids=set())
    prepare_papers.run_prepare_papers(runtime="test", target_date="2026-04-07")
    prepare_papers.run_prepare_papers(runtime="test", target_date="2026-04-07", force=True)
    assert forced == [False, True]


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


def test_backfill_does_not_advance_cursor_past_partially_failed_date(monkeypatch):
    raw_store = FakeRawStore()
    monkeypatch.setattr(prepare_papers, "RawPaperStore", lambda: raw_store)
    results = {
        "2026-04-07": _prepare_result("success", 2, 0),
        "2026-04-06": _prepare_result("partial_failed", 1, 1),
    }
    monkeypatch.setattr(prepare_papers, "run_prepare_papers", lambda **kwargs: results[kwargs["target_date"]])

    result = prepare_papers.run_backfill_prepare_papers(
        runtime="test",
        cursor_date="2026-04-07",
        oldest_date="2026-04-01",
        batch_days=3,
    )

    assert result["status"] == "failed"
    assert result["stopped_reason"] == "prepare_failed"
    assert result["next_cursor_date"] == "2026-04-06"
    assert raw_store.saved_state["cursor_date"] == "2026-04-06"
    assert raw_store.saved_state["last_processed_dates"] == ["2026-04-07"]
    assert result["success_count"] == 1
    assert result["failure_count"] == 1
    failure = result["failures"][0]
    assert failure["date"] == "2026-04-06"
    assert failure["status"] == "partial_failed"
    assert failure["paper_success_count"] == 1
    assert failure["paper_failure_count"] == 1
    assert failure["paper_failures"] == [{"arxiv_id": "bad0", "error": "NameError: boom"}]
    assert failure["prepared_arxiv_ids"] == ["id0"]
    assert "1 of 2 paper(s) failed" in failure["error"]
