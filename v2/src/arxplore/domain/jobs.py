from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Self

from arxplore.config import AppSettings

RETRY_BACKOFF_BASE_SECONDS = 60
RETRY_BACKOFF_MAX_SECONDS = 3600
STALE_EXHAUSTED_ERROR = "stale claim reset after max attempts"
FORCE_PAYLOAD_KEY = "force"
DEFAULT_JOB_MODE = "auto"
DEFAULT_JOB_SOURCE = "collect"
DEFAULT_WORKER_ID = "local_prepare_worker"


class JobStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


def compute_retry_backoff_seconds(
    attempt_count: int,
    *,
    base_seconds: int = RETRY_BACKOFF_BASE_SECONDS,
    max_seconds: int = RETRY_BACKOFF_MAX_SECONDS,
) -> int:
    exponent = max(0, int(attempt_count) - 1)
    if exponent >= 32:
        return int(max_seconds)
    return int(min(max_seconds, base_seconds * (2**exponent)))


@dataclass(frozen=True, slots=True, kw_only=True)
class FailureTransition:
    status: JobStatus
    attempt_count: int
    backoff_seconds: int | None

    @property
    def is_final(self) -> bool:
        return self.status is JobStatus.FAILED

    @property
    def retries_now(self) -> bool:
        return self.status is JobStatus.PENDING and self.backoff_seconds is None


def resolve_failure_transition(*, attempt_count: int, pending_refresh: bool, max_attempts: int) -> FailureTransition:
    if pending_refresh:
        return FailureTransition(status=JobStatus.PENDING, attempt_count=0, backoff_seconds=None)
    if int(attempt_count) < int(max_attempts):
        return FailureTransition(
            status=JobStatus.PENDING,
            attempt_count=int(attempt_count),
            backoff_seconds=compute_retry_backoff_seconds(attempt_count),
        )
    return FailureTransition(status=JobStatus.FAILED, attempt_count=int(attempt_count), backoff_seconds=None)


@dataclass(frozen=True, slots=True, kw_only=True)
class QueuePolicy:
    max_attempts: int
    stale_seconds: int

    @classmethod
    def from_settings(cls, settings: AppSettings) -> Self:
        return cls(
            max_attempts=max(1, int(settings.prepare_job_max_attempts)),
            stale_seconds=int(settings.prepare_job_stale_seconds),
        )


def job_notification_payload(mode: str, target_date: date) -> str:
    return f"{mode}:{target_date.isoformat()}"


@dataclass(frozen=True, slots=True, kw_only=True)
class ClaimToken:
    job_id: int
    worker_id: str
    claim_generation: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ClaimedJob:
    job_id: int
    mode: str
    target_date: date
    source: str
    payload: dict[str, Any]
    status: JobStatus
    attempt_count: int
    worker_id: str
    created_at: datetime
    claimed_at: datetime
    updated_at: datetime
    claim_generation: int
    raw_revision: int

    @property
    def token(self) -> ClaimToken:
        return ClaimToken(job_id=self.job_id, worker_id=self.worker_id, claim_generation=self.claim_generation)

    @property
    def force_requested(self) -> bool:
        return self.payload.get(FORCE_PAYLOAD_KEY) is True


@dataclass(frozen=True, slots=True, kw_only=True)
class EnqueueResult:
    enqueued: bool
    job_id: int
    mode: str
    target_date: date
    status: JobStatus
    raw_revision: int
    pending_refresh: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class RequeuedJob:
    job_id: int
    target_date: date
    attempt_count: int
