import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from arxplore.db.pool import AnyPool
from arxplore.db.repositories import jobs, raw
from arxplore.domain.jobs import DEFAULT_JOB_MODE, ClaimedJob, ClaimToken, QueuePolicy
from arxplore.pipeline.prepare.mapping import CandidateSelection, PrepareCandidate, select_candidates
from arxplore.pipeline.prepare.paper import PaperOutcome, PrepareContext, prepare_paper

logger = logging.getLogger(__name__)

MAX_RECORDED_FAILURES = 50
MAX_FAILURE_ERROR_CHARS = 500
SAMPLE_PREPARED_LIMIT = 5

type Heartbeat = Callable[[], Awaitable[None]]


class ClaimLostError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperFailure:
    arxiv_id: str
    error: str

    def as_dict(self) -> dict[str, str]:
        return {"arxiv_id": self.arxiv_id, "error": self.error}


def format_failure_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:MAX_FAILURE_ERROR_CHARS]


def _sample(outcome: PaperOutcome) -> dict[str, Any]:
    return {
        "arxiv_id": outcome.arxiv_id,
        "title": outcome.title,
        "primary_category": outcome.primary_category,
        "chunk_count": outcome.chunk_count,
        "fulltext_source": outcome.fulltext_source,
        "fallback_used": outcome.fallback_used,
        "section_count": outcome.section_count,
        "text_length": outcome.text_length,
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class DatePrepareResult:
    selection: CandidateSelection
    outcomes: tuple[PaperOutcome, ...]
    failures: tuple[PaperFailure, ...]

    @property
    def target_date(self) -> date:
        return self.selection.target_date

    @property
    def success_count(self) -> int:
        return len(self.outcomes)

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def status(self) -> str:
        if self.failures and not self.outcomes:
            return "failed"
        if self.failures:
            return "partial_failed"
        return "success"

    @property
    def prepared_arxiv_ids(self) -> list[str]:
        return [outcome.arxiv_id.strip() for outcome in self.outcomes if outcome.arxiv_id]

    @property
    def recorded_failures(self) -> list[dict[str, str]]:
        return [failure.as_dict() for failure in self.failures[:MAX_RECORDED_FAILURES]]

    def _sum(self, name: str) -> int:
        return sum(int(getattr(outcome, name) or 0) for outcome in self.outcomes)

    def _count(self, name: str) -> int:
        return sum(1 for outcome in self.outcomes if getattr(outcome, name))

    def counts(self) -> dict[str, int]:
        return {
            "saved_papers": self._sum("saved_paper"),
            "saved_fulltexts": self._sum("saved_fulltext"),
            "saved_chunks": self._sum("saved_chunks"),
            "fallback_fulltexts": self._count("fallback_used"),
            "skipped_lower_rank_overwrites": self._count("skipped_lower_rank_overwrite"),
            "skipped_unchanged": self._count("skipped_unchanged"),
            "unchanged_chunk_sets": self._count("chunks_unchanged"),
        }

    def job_result(self) -> dict[str, Any]:
        return {
            **self.counts(),
            "selected_candidate_count": len(self.selection.selected_ids),
            "prepared_arxiv_count": len(self.prepared_arxiv_ids),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "failures": self.recorded_failures,
        }

    def failure_summary(self) -> str:
        recorded = self.recorded_failures
        first_error = recorded[0]["error"] if recorded else ""
        if self.success_count:
            headline = f"{self.failure_count} of {self.failure_count + self.success_count} paper(s) failed"
        else:
            headline = f"all {self.failure_count} paper(s) failed"
        summary = f"{headline} (success={self.success_count}, failure={self.failure_count}); first error: {first_error}"
        return summary[:MAX_FAILURE_ERROR_CHARS]

    def as_dict(self) -> dict[str, Any]:
        counts = self.counts()
        return {
            "stage": "prepare_papers",
            "status": self.status,
            "target_date": self.target_date.isoformat(),
            "raw_payload_count": self.selection.raw_count,
            "arxiv_candidate_count": len(self.selection.deduplicated_ids),
            "selected_candidate_count": len(self.selection.selected_ids),
            "saved_papers": counts["saved_papers"],
            "saved_fulltexts": counts["saved_fulltexts"],
            "saved_chunks": counts["saved_chunks"],
            "prepared_arxiv_ids": self.prepared_arxiv_ids,
            "fallback_fulltexts": counts["fallback_fulltexts"],
            "skipped_lower_rank_overwrites": counts["skipped_lower_rank_overwrites"],
            "skipped_unchanged": counts["skipped_unchanged"],
            "unchanged_chunk_sets": counts["unchanged_chunk_sets"],
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "failures": self.recorded_failures,
            "sample_prepared": [_sample(outcome) for outcome in self.outcomes[:SAMPLE_PREPARED_LIMIT]],
        }


async def prepare_candidates(
    candidates: Sequence[PrepareCandidate],
    context: PrepareContext,
    *,
    force: bool = False,
    heartbeat: Heartbeat | None = None,
) -> tuple[list[PaperOutcome], list[PaperFailure]]:
    outcomes: list[PaperOutcome] = []
    failures: list[PaperFailure] = []
    for candidate in candidates:
        if heartbeat is not None:
            await heartbeat()
        try:
            outcomes.append(await prepare_paper(candidate, context, force=force))
        except Exception as exc:
            logger.exception("prepare_paper failed: arxiv_id=%s", candidate.arxiv_id)
            failures.append(PaperFailure(arxiv_id=candidate.arxiv_id.strip(), error=format_failure_error(exc)))
    return outcomes, failures


async def prepare_date(
    target_date: date,
    context: PrepareContext,
    *,
    max_papers: int | None = None,
    force: bool = False,
    heartbeat: Heartbeat | None = None,
) -> DatePrepareResult:
    payload = await raw.load_daily_papers(context.db, target_date=target_date)
    selection = select_candidates(target_date, payload, max_papers=max_papers)
    outcomes, failures = await prepare_candidates(selection.candidates, context, force=force, heartbeat=heartbeat)
    return DatePrepareResult(selection=selection, outcomes=tuple(outcomes), failures=tuple(failures))


def job_heartbeat(pool: AnyPool, token: ClaimToken) -> Heartbeat:
    async def beat() -> None:
        try:
            alive = await jobs.heartbeat(pool, token)
        except Exception:
            logger.warning("prepare job heartbeat failed: job_id=%s", token.job_id, exc_info=True)
            return
        if not alive:
            raise ClaimLostError(f"prepare job claim lost: job_id={token.job_id}")

    return beat


@dataclass(slots=True)
class QueueRunResult:
    mode: str
    worker_id: str
    claimed_dates: list[str] = field(default_factory=list)
    successes: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    lost_claims: list[dict[str, Any]] = field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.claimed_dates and not self.failures and not self.lost_claims:
            return "no_op"
        if self.failures and not self.successes:
            return "failed"
        if self.failures or (self.lost_claims and self.successes):
            return "partial_failed"
        if self.lost_claims:
            return "claim_lost"
        return "success"

    def prepared_arxiv_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for entry in [*self.successes, *self.failures]:
            for value in entry.get("prepared_arxiv_ids", []):
                arxiv_id = str(value or "").strip()
                if arxiv_id:
                    seen.setdefault(arxiv_id, None)
        return list(seen)

    def record_lost_claim(self, job: ClaimedJob, *, stage: str) -> None:
        logger.warning(
            "prepare job claim lost at %s: job_id=%s date=%s generation=%s",
            stage,
            job.job_id,
            job.target_date.isoformat(),
            job.claim_generation,
        )
        self.lost_claims.append(
            {
                "date": job.target_date.isoformat(),
                "job_id": job.job_id,
                "claim_generation": job.claim_generation,
                "stage": stage,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": "consume_prepare_queue",
            "status": self.status,
            "mode": self.mode,
            "worker_id": self.worker_id,
            "claimed_dates": list(self.claimed_dates),
            "claimed_count": len(self.claimed_dates),
            "success_count": len(self.successes),
            "failure_count": len(self.failures),
            "lost_claim_count": len(self.lost_claims),
            "successes": list(self.successes),
            "failures": list(self.failures),
            "lost_claims": list(self.lost_claims),
        }


def _success_entry(result: DatePrepareResult) -> dict[str, Any]:
    counts = result.counts()
    return {
        "date": result.target_date.isoformat(),
        "saved_papers": counts["saved_papers"],
        "saved_fulltexts": counts["saved_fulltexts"],
        "saved_chunks": counts["saved_chunks"],
        "fallback_fulltexts": counts["fallback_fulltexts"],
        "selected_candidate_count": len(result.selection.selected_ids),
        "prepared_arxiv_ids": result.prepared_arxiv_ids,
        "paper_success_count": result.success_count,
        "paper_failure_count": result.failure_count,
        "paper_failures": result.recorded_failures,
    }


async def run_claimed_job(
    job: ClaimedJob,
    context: PrepareContext,
    pool: AnyPool,
    run: QueueRunResult,
    *,
    policy: QueuePolicy,
    max_papers: int | None,
    force: bool,
) -> None:
    token = job.token
    target_date = job.target_date.isoformat()
    try:
        result = await prepare_date(
            job.target_date,
            context,
            max_papers=max_papers,
            force=force or job.force_requested,
            heartbeat=job_heartbeat(pool, token),
        )
        if result.failure_count > 0 or result.status == "failed":
            error_summary = result.failure_summary()
            if not await jobs.fail(pool, token, policy=policy, error=error_summary):
                run.record_lost_claim(job, stage="fail")
            run.failures.append(
                {"date": target_date, "error": error_summary, "prepared_arxiv_ids": result.prepared_arxiv_ids}
            )
            return
        if not await jobs.complete(pool, token, result=result.job_result()):
            run.record_lost_claim(job, stage="complete")
            return
    except ClaimLostError:
        run.record_lost_claim(job, stage="heartbeat")
        return
    except Exception as exc:
        if not await jobs.fail(pool, token, policy=policy, error=str(exc)):
            run.record_lost_claim(job, stage="fail")
        run.failures.append({"date": target_date, "error": str(exc)})
        return
    run.successes.append(_success_entry(result))


async def consume_prepare_queue(
    context: PrepareContext,
    pool: AnyPool,
    *,
    policy: QueuePolicy,
    worker_id: str,
    mode: str = DEFAULT_JOB_MODE,
    max_jobs_per_run: int = 1,
    max_papers: int | None = None,
    force: bool = False,
) -> QueueRunResult:
    run = QueueRunResult(mode=mode, worker_id=worker_id)
    for _ in range(max(1, max_jobs_per_run)):
        job = await jobs.claim(pool, policy=policy, mode=mode, worker_id=worker_id)
        if job is None:
            break
        run.claimed_dates.append(job.target_date.isoformat())
        await run_claimed_job(job, context, pool, run, policy=policy, max_papers=max_papers, force=force)
    return run
