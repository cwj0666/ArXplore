import hashlib
import json
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

import pytest

from arxplore.config import AppSettings
from arxplore.db.repositories.embeddings import vector_literal
from arxplore.db.repositories.pipeline_state import pipeline_state_key
from arxplore.db.repositories.raw import compute_payload_hash, record_id
from arxplore.domain.chunks import ChunkInput, chunk_text_hash
from arxplore.domain.jobs import (
    ClaimedJob,
    ClaimToken,
    FailureTransition,
    JobStatus,
    QueuePolicy,
    compute_retry_backoff_seconds,
    job_notification_payload,
    resolve_failure_transition,
)
from arxplore.domain.papers import PaperUpsert, int_or_none, parse_timestamp

NOW = datetime(2026, 4, 7, tzinfo=UTC)


@pytest.mark.parametrize(
    ("attempt_count", "expected"),
    [(0, 60), (1, 60), (2, 120), (3, 240), (6, 1920), (7, 3600), (50, 3600), (10_000, 3600)],
)
def test_retry_backoff_doubles_from_a_minute_up_to_an_hour(attempt_count: int, expected: int) -> None:
    assert compute_retry_backoff_seconds(attempt_count) == expected


def test_failure_transition_retries_with_backoff_below_the_limit() -> None:
    assert resolve_failure_transition(attempt_count=1, pending_refresh=False, max_attempts=3) == FailureTransition(
        status=JobStatus.PENDING, attempt_count=1, backoff_seconds=60
    )
    assert resolve_failure_transition(attempt_count=2, pending_refresh=False, max_attempts=3).backoff_seconds == 120


def test_failure_transition_fails_when_attempts_are_exhausted() -> None:
    transition = resolve_failure_transition(attempt_count=3, pending_refresh=False, max_attempts=3)

    assert transition == FailureTransition(status=JobStatus.FAILED, attempt_count=3, backoff_seconds=None)
    assert (transition.is_final, transition.retries_now) == (True, False)


def test_failure_transition_with_refresh_retries_now_with_fresh_attempts() -> None:
    transition = resolve_failure_transition(attempt_count=3, pending_refresh=True, max_attempts=3)

    assert transition == FailureTransition(status=JobStatus.PENDING, attempt_count=0, backoff_seconds=None)
    assert (transition.is_final, transition.retries_now) == (False, True)


def test_queue_policy_reads_the_settings_with_a_floor_of_one_attempt() -> None:
    settings = AppSettings.model_validate({"PREPARE_JOB_MAX_ATTEMPTS": "0", "PREPARE_JOB_STALE_SECONDS": "30"})
    assert QueuePolicy.from_settings(settings) == QueuePolicy(max_attempts=1, stale_seconds=30)
    defaults = AppSettings.model_validate({})
    assert QueuePolicy.from_settings(defaults) == QueuePolicy(max_attempts=3, stale_seconds=900)


def test_notification_payload_keeps_the_legacy_mode_and_iso_date() -> None:
    assert job_notification_payload("auto", date(2026, 4, 7)) == "auto:2026-04-07"
    assert job_notification_payload("backfill", date(987, 1, 2)) == "backfill:0987-01-02"


def _claimed(payload: dict[str, Any]) -> ClaimedJob:
    return ClaimedJob(
        job_id=7,
        mode="auto",
        target_date=date(2026, 4, 7),
        source="collect",
        payload=payload,
        status=JobStatus.PROCESSING,
        attempt_count=1,
        worker_id="w1",
        created_at=NOW,
        claimed_at=NOW,
        updated_at=NOW,
        claim_generation=2,
        raw_revision=1,
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [({"force": True}, True), ({"force": "true"}, False), ({"force": 1}, False), ({}, False)],
)
def test_claimed_job_force_needs_a_literal_true(payload: dict[str, Any], expected: bool) -> None:
    job = _claimed(payload)

    assert job.force_requested is expected
    assert job.token == ClaimToken(job_id=7, worker_id="w1", claim_generation=2)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("2026-04-07T00:00:00Z", datetime(2026, 4, 7, tzinfo=UTC)),
        (" 2026-04-07T09:00:00+09:00 ", datetime(2026, 4, 7, 9, tzinfo=timezone(timedelta(hours=9)))),
        ("2026-04-07T00:00:00", datetime(2026, 4, 7)),
        ("2026-04-07", datetime(2026, 4, 7)),
        (NOW, NOW),
        (date(2026, 4, 7), None),
        (1712448000, None),
    ],
)
def test_parse_timestamp_follows_the_legacy_rules(value: Any, expected: datetime | None) -> None:
    assert parse_timestamp(value) == expected


def test_parse_timestamp_rejects_malformed_text() -> None:
    with pytest.raises(ValueError):
        parse_timestamp("yesterday")


@pytest.mark.parametrize(
    ("value", "expected"), [(None, None), ("12", 12), (12.7, 12), ("x", None), ("3.5", None), (True, 1), ([], None)]
)
def test_int_or_none_follows_the_legacy_rules(value: Any, expected: int | None) -> None:
    assert int_or_none(value) == expected


def test_paper_from_mapping_applies_defaults_and_cleans_text() -> None:
    paper = PaperUpsert.from_mapping(
        {
            "arxiv_id": " 2601.00001\ud800 ",
            "authors": [{"na\ud800me": "Ada"}],
            "published_at": "2026-04-07T00:00:00Z",
            "upvotes": "7",
            "github_stars": "many",
        }
    )

    assert paper == PaperUpsert(
        arxiv_id="2601.00001",
        title="",
        authors=[{"name": "Ada"}],
        abstract="",
        categories=[],
        published_at=datetime(2026, 4, 7, tzinfo=UTC),
        upvotes=7,
        github_stars=None,
        source="hf_daily_papers",
    )


def test_paper_from_mapping_needs_an_arxiv_id() -> None:
    with pytest.raises(KeyError):
        PaperUpsert.from_mapping({"title": "T"})


def test_chunk_from_mapping_coerces_like_the_legacy_writer() -> None:
    chunk = ChunkInput.from_mapping({"chunk_index": "3", "chunk_text": "b\ud800", "metadata": {"k\ud800": ["v\udfff"]}})

    assert chunk == ChunkInput(chunk_index=3, chunk_text="b", section_title=None, token_count=0, metadata={"k": ["v"]})
    assert ChunkInput.from_mapping({}) == ChunkInput(chunk_index=0, chunk_text="")


def test_chunk_text_hash_is_the_postgres_md5() -> None:
    assert chunk_text_hash("논문 text") == hashlib.md5("논문 text".encode()).hexdigest()
    assert chunk_text_hash("") == "d41d8cd98f00b204e9800998ecf8427e"


def test_payload_hash_ignores_key_order_but_not_values() -> None:
    assert compute_payload_hash({"a": 1, "b": [1, 2]}) == compute_payload_hash({"b": [1, 2], "a": 1})
    assert compute_payload_hash({"a": 1}) != compute_payload_hash({"a": 2})


def test_payload_hash_uses_unescaped_unicode_and_stringifies_unknown_types() -> None:
    payload = [{"title": "논문", "at": NOW}]
    serialized = json.dumps([{"at": str(NOW), "title": "논문"}], sort_keys=True, ensure_ascii=False)

    assert compute_payload_hash(payload) == hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def test_record_ids_and_state_keys_keep_the_legacy_format() -> None:
    assert record_id(date(2026, 4, 7)) == "hf_daily_papers:2026-04-07"
    assert pipeline_state_key("prepare_papers_backfill", "gpu") == "prepare_papers_backfill:gpu"
    assert pipeline_state_key("hf_daily_papers_backfill") == "hf_daily_papers_backfill:default"


def test_vector_literal_uses_twelve_decimals() -> None:
    assert vector_literal([1, 0.5, -1 / 3, 1e-13]) == "[1.000000000000,0.500000000000,-0.333333333333,0.000000000000]"
    assert vector_literal([]) == "[]"
