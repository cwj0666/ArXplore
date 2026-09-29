from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from arxplore.db.pool import ConnectionSource
from arxplore.db.repositories import embeddings
from arxplore.domain.chunks import ChunkEmbedding

EMBED_CHUNKS_PER_ROUND = 200
EMBED_MAX_ROUNDS_PER_PAPER = 100
EMBED_BACKLOG_BUDGET = 400
ROUND_LIMIT_EXCEEDED = "embed_loop_limit_exceeded"
SAMPLE_EMBEDDED_LIMIT = 5


class Embedder(Protocol):
    @property
    def model(self) -> str: ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class EmbeddingBudget:
    chunks_per_round: int = EMBED_CHUNKS_PER_ROUND
    max_rounds_per_paper: int = EMBED_MAX_ROUNDS_PER_PAPER
    backlog_chunks: int = EMBED_BACKLOG_BUDGET

    @property
    def max_chunks_per_paper(self) -> int:
        return self.chunks_per_round * self.max_rounds_per_paper


@dataclass(frozen=True, slots=True, kw_only=True)
class EmbedRound:
    status: str
    selected_chunk_count: int
    embedded_chunk_count: int
    model_name: str
    sample_embedded: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": "embed_papers",
            "status": self.status,
            "selected_chunk_count": self.selected_chunk_count,
            "embedded_chunk_count": self.embedded_chunk_count,
            "model_name": self.model_name,
            "sample_embedded": list(self.sample_embedded),
        }


async def embed_missing_chunks(
    db: ConnectionSource,
    embedder: Embedder,
    *,
    limit: int = EMBED_CHUNKS_PER_ROUND,
    arxiv_id: str | None = None,
) -> EmbedRound:
    candidates = await embeddings.list_chunks_missing_embeddings(db, limit=max(1, limit), arxiv_id=arxiv_id)
    if not candidates:
        return EmbedRound(status="no_op", selected_chunk_count=0, embedded_chunk_count=0, model_name=embedder.model)
    vectors = await embedder.embed([candidate.chunk_text for candidate in candidates])
    rows = [
        ChunkEmbedding(chunk_id=candidate.chunk_id, embedding=vector, model_name=embedder.model)
        for candidate, vector in zip(candidates, vectors, strict=True)
    ]
    await embeddings.upsert_embeddings(db, rows)
    return EmbedRound(
        status="success",
        selected_chunk_count=len(candidates),
        embedded_chunk_count=len(rows),
        model_name=embedder.model,
        sample_embedded=tuple(
            {
                "chunk_id": candidate.chunk_id,
                "arxiv_id": candidate.arxiv_id,
                "chunk_index": candidate.chunk_index,
                "section_title": candidate.section_title,
            }
            for candidate in candidates[:SAMPLE_EMBEDDED_LIMIT]
        ),
    )


@dataclass(slots=True)
class EmbedAfterPrepareResult:
    target_arxiv_count: int
    embedded_arxiv_count: int = 0
    selected_chunk_count: int = 0
    embedded_chunk_count: int = 0
    backlog_selected_chunk_count: int = 0
    backlog_embedded_chunk_count: int = 0
    backlog_error: str | None = None
    per_arxiv: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, str]] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.failures and self.embedded_arxiv_count == 0:
            return "failed"
        if self.failures or self.backlog_error:
            return "partial_failed"
        if self.embedded_chunk_count == 0 and self.backlog_embedded_chunk_count == 0:
            return "no_op"
        return "success"

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": "embed_after_prepare",
            "status": self.status,
            "target_arxiv_count": self.target_arxiv_count,
            "embedded_arxiv_count": self.embedded_arxiv_count,
            "selected_chunk_count": self.selected_chunk_count,
            "embedded_chunk_count": self.embedded_chunk_count,
            "backlog_selected_chunk_count": self.backlog_selected_chunk_count,
            "backlog_embedded_chunk_count": self.backlog_embedded_chunk_count,
            "backlog_error": self.backlog_error,
            "per_arxiv": list(self.per_arxiv),
            "failures": list(self.failures),
        }


class RoundLimitExceededError(RuntimeError):
    pass


async def embed_paper(
    db: ConnectionSource, embedder: Embedder, arxiv_id: str, budget: EmbeddingBudget
) -> tuple[int, int]:
    selected_total = 0
    embedded_total = 0
    for _ in range(budget.max_rounds_per_paper):
        round_result = await embed_missing_chunks(db, embedder, limit=budget.chunks_per_round, arxiv_id=arxiv_id)
        selected_total += round_result.selected_chunk_count
        embedded_total += round_result.embedded_chunk_count
        if round_result.status == "no_op" or round_result.selected_chunk_count < budget.chunks_per_round:
            return selected_total, embedded_total
    raise RoundLimitExceededError(ROUND_LIMIT_EXCEEDED)


async def embed_backlog(
    db: ConnectionSource, embedder: Embedder, budget: EmbeddingBudget, result: EmbedAfterPrepareResult
) -> None:
    remaining = max(0, budget.backlog_chunks)
    while remaining > 0:
        limit = min(budget.chunks_per_round, remaining)
        round_result = await embed_missing_chunks(db, embedder, limit=limit)
        result.backlog_selected_chunk_count += round_result.selected_chunk_count
        result.backlog_embedded_chunk_count += round_result.embedded_chunk_count
        if (
            round_result.status == "no_op"
            or round_result.selected_chunk_count <= 0
            or round_result.embedded_chunk_count <= 0
        ):
            break
        if round_result.selected_chunk_count < limit:
            break
        remaining -= round_result.selected_chunk_count


async def embed_after_prepare(
    db: ConnectionSource,
    embedder: Embedder,
    arxiv_ids: Sequence[str],
    budget: EmbeddingBudget,
) -> EmbedAfterPrepareResult:
    result = EmbedAfterPrepareResult(target_arxiv_count=len(arxiv_ids))
    for arxiv_id in arxiv_ids:
        try:
            selected, embedded = await embed_paper(db, embedder, arxiv_id, budget)
        except RoundLimitExceededError:
            result.failures.append({"arxiv_id": arxiv_id, "error": ROUND_LIMIT_EXCEEDED})
            continue
        except Exception as exc:
            result.failures.append({"arxiv_id": arxiv_id, "error": str(exc)})
            continue
        result.selected_chunk_count += selected
        result.embedded_chunk_count += embedded
        if embedded > 0:
            result.embedded_arxiv_count += 1
        result.per_arxiv.append(
            {"arxiv_id": arxiv_id, "selected_chunk_count": selected, "embedded_chunk_count": embedded}
        )
    if budget.backlog_chunks > 0:
        try:
            await embed_backlog(db, embedder, budget, result)
        except Exception as exc:
            result.backlog_error = f"{type(exc).__name__}: {exc}"
    return result
