from datetime import date
from typing import Any

import pytest

from arxplore.pipeline.embed import EmbedAfterPrepareResult, EmbeddingBudget
from arxplore.pipeline.prepare.mapping import CandidateSelection
from arxplore.pipeline.prepare.paper import PaperOutcome
from arxplore.pipeline.prepare.run import (
    MAX_FAILURE_ERROR_CHARS,
    MAX_RECORDED_FAILURES,
    DatePrepareResult,
    PaperFailure,
    QueueRunResult,
    format_failure_error,
)
from arxplore.pipeline.worker import merge_embed_status, wait_seconds_after

DAY = date(2026, 4, 7)


def _outcome(arxiv_id: str, **flags: Any) -> PaperOutcome:
    values: dict[str, Any] = {
        "arxiv_id": arxiv_id,
        "title": f"T {arxiv_id}",
        "primary_category": None,
        "chunk_count": 3,
        "fulltext_source": "layout_pdf",
        "fallback_used": False,
        "section_count": 2,
        "text_length": 10,
        "saved_fulltext": 1,
        "saved_chunks": 3,
        "content_hash": "h",
    }
    values.update(flags)
    return PaperOutcome(**values)


def _result(outcomes: list[PaperOutcome], failures: list[PaperFailure], selected: int = 3) -> DatePrepareResult:
    ids = tuple(f"id{number}" for number in range(selected))
    selection = CandidateSelection(
        target_date=DAY, raw_count=selected, deduplicated_ids=ids, selected_ids=ids, candidates=()
    )
    return DatePrepareResult(selection=selection, outcomes=tuple(outcomes), failures=tuple(failures))


def test_aggregates_count_skip_reasons_and_saved_rows() -> None:
    result = _result(
        [
            _outcome("a", skipped_lower_rank_overwrite=True, fallback_used=True, saved_fulltext=0, saved_chunks=0),
            _outcome("b", skipped_unchanged=True, saved_fulltext=0, saved_chunks=0),
            _outcome("c", chunks_unchanged=True, saved_chunks=0),
        ],
        [],
    )
    assert result.status == "success"
    assert result.counts() == {
        "saved_papers": 3,
        "saved_fulltexts": 1,
        "saved_chunks": 0,
        "fallback_fulltexts": 1,
        "skipped_lower_rank_overwrites": 1,
        "skipped_unchanged": 1,
        "unchanged_chunk_sets": 1,
    }
    assert result.as_dict()["sample_prepared"][0] == {
        "arxiv_id": "a",
        "title": "T a",
        "primary_category": None,
        "chunk_count": 3,
        "fulltext_source": "layout_pdf",
        "fallback_used": True,
        "section_count": 2,
        "text_length": 10,
    }


@pytest.mark.parametrize(
    "outcomes,failures,status",
    [(0, 0, "success"), (2, 1, "partial_failed"), (0, 2, "failed"), (3, 0, "success")],
)
def test_date_status_follows_successes_and_failures(outcomes: int, failures: int, status: str) -> None:
    result = _result(
        [_outcome(f"ok{number}") for number in range(outcomes)],
        [PaperFailure(arxiv_id=f"bad{number}", error="E") for number in range(failures)],
    )
    assert result.status == status


def test_only_fifty_failures_are_recorded_but_all_are_counted() -> None:
    failures = [PaperFailure(arxiv_id=f"bad{number}", error=f"E{number}") for number in range(60)]
    result = _result([], failures, selected=60)
    assert len(result.job_result()["failures"]) == MAX_RECORDED_FAILURES
    assert result.job_result()["failure_count"] == 60
    assert result.failure_summary() == "all 60 paper(s) failed (success=0, failure=60); first error: E0"


def test_failure_summaries_and_errors_are_capped() -> None:
    long_error = "x" * 800
    result = _result([_outcome("a")], [PaperFailure(arxiv_id="b", error=long_error)])
    assert len(result.failure_summary()) == MAX_FAILURE_ERROR_CHARS
    assert result.failure_summary().startswith("1 of 2 paper(s) failed (success=1, failure=1); first error: xxx")
    assert format_failure_error(NameError("boom")) == "NameError: boom"
    assert len(format_failure_error(ValueError("y" * 900))) == MAX_FAILURE_ERROR_CHARS


@pytest.mark.parametrize(
    "claimed,successes,failures,lost,status",
    [
        (0, 0, 0, 0, "no_op"),
        (1, 1, 0, 0, "success"),
        (1, 0, 1, 0, "failed"),
        (2, 1, 1, 0, "partial_failed"),
        (1, 0, 0, 1, "claim_lost"),
        (2, 1, 0, 1, "partial_failed"),
        (1, 0, 1, 1, "failed"),
    ],
)
def test_queue_status_matrix(claimed: int, successes: int, failures: int, lost: int, status: str) -> None:
    run = QueueRunResult(
        mode="auto",
        worker_id="w",
        claimed_dates=["d"] * claimed,
        successes=[{"prepared_arxiv_ids": []}] * successes,
        failures=[{"date": "d", "error": "e"}] * failures,
        lost_claims=[{"stage": "complete"}] * lost,
    )
    assert run.status == status


def test_prepared_ids_come_from_successes_and_failures_once_each() -> None:
    run = QueueRunResult(
        mode="auto",
        worker_id="w",
        successes=[{"prepared_arxiv_ids": ["a", " b ", ""]}],
        failures=[{"prepared_arxiv_ids": ["b", "c"]}, {"error": "no ids"}],
    )
    assert run.prepared_arxiv_ids() == ["a", "b", "c"]


@pytest.mark.parametrize(
    "failures,embedded_papers,backlog_error,embedded,backlog,status",
    [
        (1, 0, None, 0, 0, "failed"),
        (1, 1, None, 5, 0, "partial_failed"),
        (0, 0, "RuntimeError: x", 0, 0, "partial_failed"),
        (0, 0, None, 0, 0, "no_op"),
        (0, 0, None, 0, 4, "success"),
        (0, 1, None, 5, 0, "success"),
    ],
)
def test_embed_status_matrix(
    failures: int, embedded_papers: int, backlog_error: str | None, embedded: int, backlog: int, status: str
) -> None:
    result = EmbedAfterPrepareResult(
        target_arxiv_count=failures + embedded_papers,
        embedded_arxiv_count=embedded_papers,
        embedded_chunk_count=embedded,
        backlog_embedded_chunk_count=backlog,
        backlog_error=backlog_error,
        failures=[{"arxiv_id": "x", "error": "e"}] * failures,
    )
    assert result.status == status


@pytest.mark.parametrize(
    "prepare,embed,merged",
    [
        ("success", "failed", "failed"),
        ("no_op", "failed", "failed"),
        ("failed", "failed", "failed"),
        ("success", "partial_failed", "partial_failed"),
        ("no_op", "partial_failed", "no_op"),
        ("partial_failed", "success", "partial_failed"),
        ("success", "skipped", "success"),
    ],
)
def test_embed_status_is_merged_into_the_run_status(prepare: str, embed: str, merged: str) -> None:
    assert merge_embed_status(prepare, embed) == merged


def test_the_loop_waits_only_after_no_op_or_failed_runs() -> None:
    assert wait_seconds_after("no_op", 120.0) == 120.0
    assert wait_seconds_after("failed", 120.0) == 30.0
    assert wait_seconds_after("failed", 10.0) == 10.0
    for status in ("success", "partial_failed", "claim_lost"):
        assert wait_seconds_after(status, 120.0) is None


def test_the_per_paper_embedding_cap_is_rounds_times_chunks() -> None:
    assert EmbeddingBudget().max_chunks_per_paper == 200 * 100
    assert EmbeddingBudget(chunks_per_round=10, max_rounds_per_paper=3).max_chunks_per_paper == 30
