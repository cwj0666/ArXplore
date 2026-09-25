"""검색 방식별 질의 실행과 지표 집계.

retriever는 주입받는다. 기본 방식(lexical/vector/hybrid)은 공개 `search_paper_contexts*`만 호출하고,
ablation은 `PaperRetriever`의 하위 단계를 코드 수정 없이 다시 조합한다(ABLATIONS 참고).
정답 논문이 없는 케이스(거절·입력 거부 기대, 상세 챗 전용, 대화 이력)는 검색 평가에서 뺀다(`split_retrieval_queries`).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from eval.dataset import EvalQuery
from eval.metrics import hit_at_k, mean, mrr_at_k, percentile, recall_at_k
from src.integrations.paper_retriever import candidate_fetch_limit, hybrid_branch_limit

NOISE_ROLES = frozenset({"references", "toc", "front_matter"})
HIT_KS = (1, 5, 10)
RRF_RANK_CONSTANT = 60.0


class SupportsContextSearch(Protocol):
    def search_paper_contexts(self, query: str, *, limit: int, adjacency_window: int) -> list[dict]: ...

    def search_paper_contexts_by_vector(self, query: str, *, limit: int, adjacency_window: int) -> list[dict]: ...

    def search_paper_contexts_by_hybrid(self, query: str, *, limit: int, adjacency_window: int) -> list[dict]: ...


SearchFn = Callable[[Any, str, int, int], list[dict]]


def _lexical_pipeline(retriever: Any, query: str, limit: int, *, apply_filter: bool = True) -> list[dict]:
    candidates = retriever.repository.list_chunk_candidates_by_query(
        query, limit=candidate_fetch_limit(limit), arxiv_id=None
    )
    candidates = retriever._normalize_candidates(query, candidates, retrieval_method="lexical")
    candidates = retriever._rerank_lexical_candidates(query, candidates)
    if apply_filter:
        candidates = retriever._filter_lexical_candidates(query, candidates)
    return candidates


def _vector_pipeline(retriever: Any, query: str, limit: int, *, apply_rerank: bool = True) -> list[dict]:
    embedding = retriever.embedding_client.embed_texts([query])[0]
    candidates = retriever.vector_repository.search_paper_chunks(
        embedding, limit=candidate_fetch_limit(limit), arxiv_id=None
    )
    candidates = retriever._normalize_candidates(query, candidates, retrieval_method="vector")
    if apply_rerank:
        candidates = retriever._rerank_vector_candidates(query, candidates)
    return candidates


def _diversify(retriever: Any, candidates: list[dict], limit: int) -> list[dict]:
    return retriever._apply_paper_diversity(candidates, limit=limit, arxiv_id=None)


def _with_contexts(retriever: Any, candidates: list[dict], adjacency_window: int) -> list[dict]:
    return retriever._build_contexts(candidates, adjacency_window=adjacency_window)


def plain_rrf(ranked_lists: Sequence[Sequence[dict]], *, rank_constant: float = RRF_RANK_CONSTANT) -> list[dict]:
    """가중치·품질 보정·교차 보너스 없는 표준 RRF. 동점은 chunk_id 내림차순(retriever와 같은 규칙)."""
    merged: dict[int, dict] = {}
    for candidates in ranked_lists:
        for rank, candidate in enumerate(candidates, start=1):
            chunk_id = int(candidate.get("chunk_id") or 0)
            entry = merged.setdefault(chunk_id, {**candidate, "retrieval_method": "hybrid_plain_rrf", "score": 0.0})
            entry["score"] += 1.0 / (rank_constant + rank)
    return sorted(merged.values(), key=lambda item: (item["score"], int(item.get("chunk_id") or 0)), reverse=True)


def _search_lexical(retriever: Any, query: str, k: int, window: int) -> list[dict]:
    return retriever.search_paper_contexts(query, limit=k, adjacency_window=window)


def _search_vector(retriever: Any, query: str, k: int, window: int) -> list[dict]:
    return retriever.search_paper_contexts_by_vector(query, limit=k, adjacency_window=window)


def _search_hybrid(retriever: Any, query: str, k: int, window: int) -> list[dict]:
    return retriever.search_paper_contexts_by_hybrid(query, limit=k, adjacency_window=window)


def _search_lexical_nodiv(retriever: Any, query: str, k: int, window: int) -> list[dict]:
    return _with_contexts(retriever, _lexical_pipeline(retriever, query, k)[:k], window)


def _search_vector_nodiv(retriever: Any, query: str, k: int, window: int) -> list[dict]:
    return _with_contexts(retriever, _vector_pipeline(retriever, query, k)[:k], window)


def _search_hybrid_nodiv(retriever: Any, query: str, k: int, window: int) -> list[dict]:
    sub_limit = hybrid_branch_limit(k)
    lexical = _lexical_pipeline(retriever, query, sub_limit)[:sub_limit]
    vector = _vector_pipeline(retriever, query, sub_limit)[:sub_limit]
    merged = retriever._merge_hybrid_candidates(
        query, lexical, vector, arxiv_id=None, limit=max(1, len(lexical) + len(vector))
    )
    return _with_contexts(retriever, merged[:k], window)


def _search_lexical_nofilter(retriever: Any, query: str, k: int, window: int) -> list[dict]:
    candidates = _lexical_pipeline(retriever, query, k, apply_filter=False)
    return _with_contexts(retriever, _diversify(retriever, candidates, k), window)


def _search_vector_norerank(retriever: Any, query: str, k: int, window: int) -> list[dict]:
    candidates = _vector_pipeline(retriever, query, k, apply_rerank=False)
    return _with_contexts(retriever, _diversify(retriever, candidates, k), window)


def _search_hybrid_plainrrf(retriever: Any, query: str, k: int, window: int) -> list[dict]:
    sub_limit = hybrid_branch_limit(k)
    lexical = retriever.search_paper_chunks(query, limit=sub_limit)
    vector = retriever.search_paper_chunks_by_vector(query, limit=sub_limit)
    return _with_contexts(retriever, _diversify(retriever, plain_rrf([lexical, vector]), k), window)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    search: SearchFn
    needs_embeddings: bool
    description: str


METHODS: dict[str, MethodSpec] = {
    spec.name: spec
    for spec in (
        MethodSpec("lexical", _search_lexical, False, "search_paper_contexts (제품 경로)"),
        MethodSpec("vector", _search_vector, True, "search_paper_contexts_by_vector"),
        MethodSpec("hybrid", _search_hybrid, True, "search_paper_contexts_by_hybrid"),
        MethodSpec("lexical_nodiv", _search_lexical_nodiv, False, "lexical, 논문당 2청크 diversity 제거"),
        MethodSpec("vector_nodiv", _search_vector_nodiv, True, "vector, diversity 제거"),
        MethodSpec("hybrid_nodiv", _search_hybrid_nodiv, True, "hybrid, 입력·출력 diversity 모두 제거"),
        MethodSpec(
            "lexical_nofilter", _search_lexical_nofilter, False, "lexical, references/front_matter/outline 필터 제거"
        ),
        MethodSpec("vector_norerank", _search_vector_norerank, True, "vector, Python 재정렬 제거(SQL 감점은 유지)"),
        MethodSpec("hybrid_plainrrf", _search_hybrid_plainrrf, True, "hybrid, 가중치·품질 보정 없는 표준 RRF"),
    )
}

BASE_METHODS = ("lexical", "vector", "hybrid")
ABLATIONS: dict[str, tuple[str, ...]] = {
    "nodiv": ("lexical_nodiv", "vector_nodiv", "hybrid_nodiv"),
    "nofilter": ("lexical_nofilter",),
    "norerank": ("vector_norerank",),
    "plainrrf": ("hybrid_plainrrf",),
}


def resolve_methods(methods: Iterable[str], ablations: Iterable[str] = ()) -> list[str]:
    """기본 방식 + ablation 이름을 METHODS 키 목록으로 푼다. 'all'은 모든 ablation."""
    resolved = [name.strip() for name in methods if name.strip()]
    ablation_names = [name.strip() for name in ablations if name.strip()]
    if "all" in ablation_names:
        ablation_names = list(ABLATIONS)
    for ablation in ablation_names:
        if ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation {ablation!r}; choose from {sorted(ABLATIONS)} or 'all'")
        resolved.extend(ABLATIONS[ablation])
    for name in resolved:
        if name not in METHODS:
            raise ValueError(f"unknown method {name!r}; choose from {sorted(METHODS)}")
    return list(dict.fromkeys(resolved))


@dataclass
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

    @property
    def ok(self) -> bool:
        return self.error is None


def hit_cutoffs(k: int) -> list[int]:
    return [cutoff for cutoff in HIT_KS if cutoff <= k] or [k]


def _metric_block(ranked: Sequence[Any], relevant: Sequence[Any], k: int) -> dict[str, float | None]:
    block: dict[str, float | None] = {f"hit@{cutoff}": hit_at_k(ranked, relevant, cutoff) for cutoff in hit_cutoffs(k)}
    block[f"mrr@{k}"] = mrr_at_k(ranked, relevant, k)
    block[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
    return block


def score_hits(query: EvalQuery, method: str, hits: Sequence[dict], *, k: int, latency_ms: float | None) -> QueryResult:
    top = list(hits)[:k]
    arxiv_ids = [str(hit.get("arxiv_id") or "") for hit in top]
    chunk_ids = [int(hit["chunk_id"]) for hit in top if hit.get("chunk_id") is not None]
    roles = [str(hit.get("content_role") or "") for hit in top]
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
        noise_count=sum(1 for role in roles if role in NOISE_ROLES),
    )


def run_query(
    retriever: Any,
    method: str,
    query: EvalQuery,
    *,
    k: int = 10,
    adjacency_window: int = 1,
    clock: Callable[[], float] = time.perf_counter,
    keep_going: bool = False,
) -> QueryResult:
    spec = METHODS[method]
    started = clock()
    try:
        hits = spec.search(retriever, query.query, k, adjacency_window)
    except Exception as exc:
        if not keep_going:
            raise
        return QueryResult(
            query.id,
            query.lang,
            query.source,
            method,
            k,
            None,
            error=f"{type(exc).__name__}: {exc}",
            category=query.category,
            expected_behavior=query.expected_behavior,
        )
    latency_ms = (clock() - started) * 1000.0
    return score_hits(query, method, hits, k=k, latency_ms=latency_ms)


def split_retrieval_queries(queries: Sequence[EvalQuery]) -> tuple[list[EvalQuery], list[EvalQuery]]:
    """검색 평가 대상(`EvalQuery.retrieval_eligible`)과 제외 대상으로 나눈다. 순서는 유지한다."""
    kept = [query for query in queries if query.retrieval_eligible]
    skipped = [query for query in queries if not query.retrieval_eligible]
    return kept, skipped


def run_evaluation(
    retriever: Any,
    queries: Sequence[EvalQuery],
    methods: Sequence[str],
    *,
    k: int = 10,
    adjacency_window: int = 1,
    clock: Callable[[], float] = time.perf_counter,
    keep_going: bool = False,
    progress: Callable[[int, int, str, EvalQuery], None] | None = None,
) -> list[QueryResult]:
    """질의 순서대로 모든 방식을 실행한다(방식 간 캐시 영향이 한쪽에 몰리지 않도록 질의 단위로 교차).

    검색 평가 대상이 아닌 케이스(`split_retrieval_queries`의 제외 대상)는 실행하지 않는다.
    """
    for method in methods:
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
    queries, _ = split_retrieval_queries(queries)
    total = len(queries) * len(methods)
    results: list[QueryResult] = []
    for query in queries:
        for method in methods:
            if progress is not None:
                progress(len(results) + 1, total, method, query)
            results.append(
                run_query(
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


def metric_names(k: int) -> list[str]:
    return [f"hit@{cutoff}" for cutoff in hit_cutoffs(k)] + [f"mrr@{k}", f"recall@{k}"]


def _aggregate_group(method: str, subset: str, k: int, rows: list[QueryResult]) -> AggregateRow:
    ok_rows = [row for row in rows if row.ok]
    chunk_rows = [row for row in ok_rows if row.chunk_metrics]
    retrieved_total = sum(len(row.retrieved_roles) for row in ok_rows)
    noise_total = sum(row.noise_count for row in ok_rows)
    latencies = [row.latency_ms for row in ok_rows if row.latency_ms is not None]
    names = metric_names(k)
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
    )


SUBSETS = ("all", "ko", "en", "known_item", "llm_synth", "manual")
CATEGORY_PREFIX = "category:"
BEHAVIOR_PREFIX = "behavior:"


def in_subset(result: QueryResult, subset: str) -> bool:
    if subset == "all":
        return True
    if subset in ("ko", "en"):
        return result.lang == subset
    if subset.startswith(CATEGORY_PREFIX):
        return result.category == subset[len(CATEGORY_PREFIX) :]
    if subset.startswith(BEHAVIOR_PREFIX):
        return result.expected_behavior == subset[len(BEHAVIOR_PREFIX) :]
    return result.source == subset


def build_subsets(results: Sequence[QueryResult], base: Sequence[str] = SUBSETS) -> list[str]:
    """기본 부분집합 + 비어 있지 않은 `category:<이름>` + `behavior:<기대 행동>`(두 종류 이상일 때만)."""
    categories = sorted({row.category for row in results if row.category})
    behaviors = sorted({row.expected_behavior for row in results})
    subsets = [*base, *(f"{CATEGORY_PREFIX}{name}" for name in categories)]
    if len(behaviors) > 1:
        subsets.extend(f"{BEHAVIOR_PREFIX}{name}" for name in behaviors)
    return subsets


def aggregate(results: Sequence[QueryResult], *, subsets: Sequence[str] | None = None) -> list[AggregateRow]:
    """방식 × 부분집합(전체, 언어, 질의 출처, 케이스 카테고리, 기대 행동)별 평균.

    `subsets`를 주지 않으면 `build_subsets`로 정한다. 해당 질의가 없는 부분집합은 생략한다.
    """
    methods = list(dict.fromkeys(row.method for row in results))
    aggregates: list[AggregateRow] = []
    for subset in build_subsets(results) if subsets is None else subsets:
        for method in methods:
            rows = [row for row in results if row.method == method and in_subset(row, subset)]
            if rows:
                aggregates.append(_aggregate_group(method, subset, rows[0].k, rows))
    return aggregates
