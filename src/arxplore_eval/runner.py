import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from arxplore.domain.retrieval import RetrievalContext, RetrievalHit
from arxplore.retrieval.diversity import apply_paper_diversity
from arxplore.retrieval.fusion import fuse_hybrid_hits
from arxplore.retrieval.lexical import fetch_lexical_candidates, filter_lexical_hits, rerank_lexical_hits
from arxplore.retrieval.query import candidate_fetch_limit, hybrid_branch_limit, normalize_search_query
from arxplore.retrieval.service import EmbedderMissingError, QueryEmbedder, Retriever, run_hybrid_channels
from arxplore.retrieval.vector import fetch_vector_candidates, rerank_vector_hits
from arxplore_eval.dataset import EvalQuery
from arxplore_eval.metrics import hit_at_k, mean, mrr_at_k, ndcg_at_k, percentile, precision_at_k, recall_at_k
from arxplore_eval.research_fusion import LEGACY_RULES_FUSION, STANDARD_RRF_FUSION, HybridFusionConfig, fuse_hits
from arxplore_eval.subsets import build_subsets, in_subset

NOISE_ROLES = frozenset({"references", "toc", "front_matter"})
HIT_KS = (1, 5, 10)

type SearchFn = Callable[[Retriever, str, int, int], Awaitable[list[RetrievalContext]]]
type Clock = Callable[[], float]
type ProgressFn = Callable[[int, int, str, EvalQuery], None]


@dataclass(frozen=True, slots=True)
class RankedHit:
    chunk_id: int | None
    arxiv_id: str
    content_role: str

    @classmethod
    def from_hit(cls, hit: RetrievalHit) -> Self:
        return cls(chunk_id=hit.chunk_id, arxiv_id=hit.arxiv_id, content_role=hit.content_role)

    @classmethod
    def from_context(cls, context: RetrievalContext) -> Self:
        return cls.from_hit(context.hit)

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> Self:
        chunk_id = item.get("chunk_id")
        return cls(
            chunk_id=None if chunk_id is None else int(chunk_id),
            arxiv_id=str(item.get("arxiv_id") or ""),
            content_role=str(item.get("content_role") or ""),
        )


def query_embedder(retriever: Retriever) -> QueryEmbedder:
    if retriever.embedder is None:
        raise EmbedderMissingError("vector search needs a query embedder")
    return retriever.embedder


async def lexical_hits(
    retriever: Retriever, query: str, limit: int, *, apply_filter: bool = True
) -> list[RetrievalHit]:
    normalized = normalize_search_query(query)
    if not normalized:
        return []
    hits = await fetch_lexical_candidates(retriever.pool, normalized, limit=candidate_fetch_limit(limit))
    hits = rerank_lexical_hits(normalized, hits)
    return filter_lexical_hits(normalized, hits) if apply_filter else hits


async def vector_hits(retriever: Retriever, query: str, limit: int, *, apply_rerank: bool = True) -> list[RetrievalHit]:
    normalized = normalize_search_query(query)
    if not normalized:
        return []
    embedding = (await query_embedder(retriever).embed([normalized]))[0]
    hits = await fetch_vector_candidates(
        retriever.pool,
        embedding,
        query=normalized,
        limit=candidate_fetch_limit(limit),
        model_name=retriever.config.embedding_model,
        min_similarity=retriever.config.vector_min_similarity,
    )
    return rerank_vector_hits(normalized, hits) if apply_rerank else hits


async def search_lexical(retriever: Retriever, query: str, k: int, window: int) -> list[RetrievalContext]:
    return await retriever.search_lexical_contexts(query, limit=k, adjacency_window=window)


async def search_vector(retriever: Retriever, query: str, k: int, window: int) -> list[RetrievalContext]:
    return await retriever.search_vector_contexts(query, limit=k, adjacency_window=window)


async def search_hybrid(retriever: Retriever, query: str, k: int, window: int) -> list[RetrievalContext]:
    return await retriever.search_hybrid_contexts(query, limit=k, adjacency_window=window)


async def search_lexical_nodiv(retriever: Retriever, query: str, k: int, window: int) -> list[RetrievalContext]:
    hits = await lexical_hits(retriever, query, k)
    return await retriever.build_contexts(hits[:k], adjacency_window=window)


async def search_vector_nodiv(retriever: Retriever, query: str, k: int, window: int) -> list[RetrievalContext]:
    hits = await vector_hits(retriever, query, k)
    return await retriever.build_contexts(hits[:k], adjacency_window=window)


async def search_hybrid_nodiv(retriever: Retriever, query: str, k: int, window: int) -> list[RetrievalContext]:
    branch_limit = hybrid_branch_limit(k)
    lexical, vector = await run_hybrid_channels(
        lexical_hits(retriever, query, branch_limit), vector_hits(retriever, query, branch_limit)
    )
    fused = fuse_hybrid_hits(lexical[:branch_limit], vector[:branch_limit])
    return await retriever.build_contexts(fused[:k], adjacency_window=window)


async def search_lexical_nofilter(retriever: Retriever, query: str, k: int, window: int) -> list[RetrievalContext]:
    hits = await lexical_hits(retriever, query, k, apply_filter=False)
    return await retriever.build_contexts(apply_paper_diversity(hits, limit=k), adjacency_window=window)


async def search_vector_norerank(retriever: Retriever, query: str, k: int, window: int) -> list[RetrievalContext]:
    hits = await vector_hits(retriever, query, k, apply_rerank=False)
    return await retriever.build_contexts(apply_paper_diversity(hits, limit=k), adjacency_window=window)


def research_fusion_search(config: HybridFusionConfig) -> SearchFn:
    async def search(retriever: Retriever, query: str, k: int, window: int) -> list[RetrievalContext]:
        inputs = await retriever.hybrid_inputs(query, limit=k)
        return await retriever.build_contexts(fuse_hits(inputs, config, limit=k), adjacency_window=window)

    return search


@dataclass(frozen=True)
class MethodSpec:
    name: str
    search: SearchFn
    needs_embeddings: bool
    description: str


METHODS: dict[str, MethodSpec] = {
    spec.name: spec
    for spec in (
        MethodSpec("lexical", search_lexical, False, "Retriever.search_lexical_contexts (제품 경로)"),
        MethodSpec("vector", search_vector, True, "Retriever.search_vector_contexts"),
        MethodSpec("hybrid", search_hybrid, True, "Retriever.search_hybrid_contexts"),
        MethodSpec("lexical_nodiv", search_lexical_nodiv, False, "lexical, 논문당 2청크 diversity 제거"),
        MethodSpec("vector_nodiv", search_vector_nodiv, True, "vector, diversity 제거"),
        MethodSpec("hybrid_nodiv", search_hybrid_nodiv, True, "hybrid, 입력·출력 diversity 모두 제거"),
        MethodSpec(
            "lexical_nofilter", search_lexical_nofilter, False, "lexical, references/front_matter/outline 필터 제거"
        ),
        MethodSpec("vector_norerank", search_vector_norerank, True, "vector, Python 재정렬 제거(SQL 감점은 유지)"),
        MethodSpec(
            "hybrid_plainrrf",
            research_fusion_search(STANDARD_RRF_FUSION),
            True,
            "hybrid, 표준 RRF(research_fusion.STANDARD_RRF_FUSION)",
        ),
        MethodSpec(
            "hybrid_rules",
            research_fusion_search(LEGACY_RULES_FUSION),
            True,
            "hybrid, 2026-09-29까지의 가중 RRF 규칙(research_fusion.LEGACY_RULES_FUSION)",
        ),
    )
}

BASE_METHODS = ("lexical", "vector", "hybrid")
ABLATIONS: dict[str, tuple[str, ...]] = {
    "nodiv": ("lexical_nodiv", "vector_nodiv", "hybrid_nodiv"),
    "nofilter": ("lexical_nofilter",),
    "norerank": ("vector_norerank",),
    "plainrrf": ("hybrid_plainrrf",),
    "rules": ("hybrid_rules",),
}
ALL_ABLATIONS = "all"


def resolve_methods(methods: Iterable[str], ablations: Iterable[str] = ()) -> list[str]:
    resolved = [name.strip() for name in methods if name.strip()]
    ablation_names = [name.strip() for name in ablations if name.strip()]
    if ALL_ABLATIONS in ablation_names:
        ablation_names = list(ABLATIONS)
    for ablation in ablation_names:
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation {ablation!r}; choose from {sorted(ABLATIONS)} or 'all'")
        resolved.extend(ABLATIONS[ablation])
    for name in resolved:
        if name not in METHODS:
            raise ValueError(f"unknown method {name!r}; choose from {sorted(METHODS)}")
    return list(dict.fromkeys(resolved))


def needs_embeddings(methods: Iterable[str]) -> bool:
    return any(METHODS[method].needs_embeddings for method in methods)


@dataclass(frozen=True)
class QueryResult:
    query_id: str
    lang: str
    source: str
    method: str
    k: int
    latency_ms: float | None
    retrieved_arxiv_ids: list[str] = field(default_factory=list)
    retrieved_chunk_ids: list[int] = field(default_factory=list)
    retrieved_roles: list[str] = field(default_factory=list)
    paper_metrics: dict[str, float | None] = field(default_factory=dict)
    chunk_metrics: dict[str, float | None] = field(default_factory=dict)
    noise_count: int = 0
    error: str | None = None
    category: str = ""
    expected_behavior: str = "answer"
    paper_extended: dict[str, float | None] = field(default_factory=dict)
    chunk_extended: dict[str, float | None] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


def hit_cutoffs(k: int) -> list[int]:
    return [cutoff for cutoff in HIT_KS if cutoff <= k] or [k]


def metric_names(k: int) -> list[str]:
    return [f"hit@{cutoff}" for cutoff in hit_cutoffs(k)] + [f"mrr@{k}", f"recall@{k}"]


def extended_metric_names(k: int) -> list[str]:
    return [f"precision@{cutoff}" for cutoff in hit_cutoffs(k)] + [f"ndcg@{k}"]


def all_metric_names(k: int) -> list[str]:
    return [*metric_names(k), *extended_metric_names(k)]


def _metric_block(ranked: Sequence[Any], relevant: Sequence[Any], k: int) -> dict[str, float | None]:
    block: dict[str, float | None] = {f"hit@{cutoff}": hit_at_k(ranked, relevant, cutoff) for cutoff in hit_cutoffs(k)}
    block[f"mrr@{k}"] = mrr_at_k(ranked, relevant, k)
    block[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
    return block


def _extended_block(ranked: Sequence[Any], relevant: Sequence[Any], k: int) -> dict[str, float | None]:
    block: dict[str, float | None] = {
        f"precision@{cutoff}": precision_at_k(ranked, relevant, cutoff) for cutoff in hit_cutoffs(k)
    }
    block[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
    return block


def score_hits(
    query: EvalQuery, method: str, hits: Sequence[RankedHit], *, k: int, latency_ms: float | None
) -> QueryResult:
    top = list(hits)[:k]
    arxiv_ids = [hit.arxiv_id for hit in top]
    chunk_ids = [hit.chunk_id for hit in top if hit.chunk_id is not None]
    roles = [hit.content_role for hit in top]
    return QueryResult(
        query_id=query.id,
        lang=query.lang,
        source=query.source,
        category=query.category,
        expected_behavior=query.expected_behavior,
        method=method,
        k=k,
        latency_ms=latency_ms,
        retrieved_arxiv_ids=arxiv_ids,
        retrieved_chunk_ids=chunk_ids,
        retrieved_roles=roles,
        paper_metrics=_metric_block(arxiv_ids, query.relevant_arxiv_ids, k),
        chunk_metrics=_metric_block(chunk_ids, query.relevant_chunk_ids, k) if query.relevant_chunk_ids else {},
        paper_extended=_extended_block(arxiv_ids, query.relevant_arxiv_ids, k),
        chunk_extended=_extended_block(chunk_ids, query.relevant_chunk_ids, k) if query.relevant_chunk_ids else {},
        noise_count=sum(1 for role in roles if role in NOISE_ROLES),
    )


def failed_result(query: EvalQuery, method: str, k: int, error: BaseException) -> QueryResult:
    return QueryResult(
        query_id=query.id,
        lang=query.lang,
        source=query.source,
        method=method,
        k=k,
        latency_ms=None,
        error=f"{type(error).__name__}: {error}",
        category=query.category,
        expected_behavior=query.expected_behavior,
    )


async def run_query(
    retriever: Retriever,
    method: str,
    query: EvalQuery,
    *,
    k: int = 10,
    adjacency_window: int = 1,
    clock: Clock = time.perf_counter,
    keep_going: bool = False,
) -> QueryResult:
    spec = METHODS[method]
    started = clock()
    try:
        contexts = await spec.search(retriever, query.query, k, adjacency_window)
    except Exception as exc:
        if not keep_going:
            raise
        return failed_result(query, method, k, exc)
    latency_ms = (clock() - started) * 1000.0
    return score_hits(
        query, method, [RankedHit.from_context(context) for context in contexts], k=k, latency_ms=latency_ms
    )


def split_retrieval_queries(queries: Sequence[EvalQuery]) -> tuple[list[EvalQuery], list[EvalQuery]]:
    kept = [query for query in queries if query.retrieval_eligible]
    skipped = [query for query in queries if not query.retrieval_eligible]
    return kept, skipped


async def run_evaluation(
    retriever: Retriever,
    queries: Sequence[EvalQuery],
    methods: Sequence[str],
    *,
    k: int = 10,
    adjacency_window: int = 1,
    clock: Clock = time.perf_counter,
    keep_going: bool = False,
    progress: ProgressFn | None = None,
) -> list[QueryResult]:
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
    eligible, _ = split_retrieval_queries(queries)
    total = len(eligible) * len(methods)
    results: list[QueryResult] = []
    for query in eligible:
        for method in methods:
            if progress is not None:
                progress(len(results) + 1, total, method, query)
            results.append(
                await run_query(
                    retriever,
                    method,
                    query,
                    k=k,
                    adjacency_window=adjacency_window,
                    clock=clock,
                    keep_going=keep_going,
                )
            )
    return results


@dataclass(frozen=True)
class AggregateRow:
    method: str
    subset: str
    k: int
    n_queries: int
    n_errors: int
    paper: dict[str, float | None]
    n_chunk_queries: int
    chunk: dict[str, float | None]
    noise_rate: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    paper_extended: dict[str, float | None] = field(default_factory=dict)
    chunk_extended: dict[str, float | None] = field(default_factory=dict)


def _aggregate_group(method: str, subset: str, k: int, rows: Sequence[QueryResult]) -> AggregateRow:
    ok_rows = [row for row in rows if row.ok]
    chunk_rows = [row for row in ok_rows if row.chunk_metrics]
    retrieved_total = sum(len(row.retrieved_roles) for row in ok_rows)
    noise_total = sum(row.noise_count for row in ok_rows)
    latencies = [row.latency_ms for row in ok_rows if row.latency_ms is not None]
    names = metric_names(k)
    extended = extended_metric_names(k)
    return AggregateRow(
        method=method,
        subset=subset,
        k=k,
        n_queries=len(ok_rows),
        n_errors=len(rows) - len(ok_rows),
        paper={name: mean(row.paper_metrics.get(name) for row in ok_rows) for name in names},
        n_chunk_queries=len(chunk_rows),
        chunk={name: mean(row.chunk_metrics.get(name) for row in chunk_rows) for name in names},
        noise_rate=(noise_total / retrieved_total) if retrieved_total else None,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        paper_extended={name: mean(row.paper_extended.get(name) for row in ok_rows) for name in extended},
        chunk_extended={name: mean(row.chunk_extended.get(name) for row in chunk_rows) for name in extended},
    )


def aggregate(results: Sequence[QueryResult], *, subsets: Sequence[str] | None = None) -> list[AggregateRow]:
    methods = list(dict.fromkeys(row.method for row in results))
    aggregates: list[AggregateRow] = []
    for subset in build_subsets(results) if subsets is None else subsets:
        for method in methods:
            rows = [row for row in results if row.method == method and in_subset(row, subset)]
            if rows:
                aggregates.append(_aggregate_group(method, subset, rows[0].k, rows))
    return aggregates
