import csv
import statistics
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arxplore.db.pool import AnyPool
from arxplore.domain.retrieval import HybridInputs, RetrievalContext, RetrievalHit
from arxplore.retrieval.query import hybrid_branch_limit, normalize_search_query
from arxplore.retrieval.service import DEFAULT_SEARCH_LIMIT, QueryEmbedder, RetrievalConfig, Retriever
from arxplore.retrieval.windows import DEFAULT_ADJACENCY_WINDOW
from arxplore_eval.dataset import EvalQuery
from arxplore_eval.fusion_sweep import bootstrap_draws, cluster_key, clusters_of, paired_bootstrap
from arxplore_eval.metrics import percentile
from arxplore_eval.tables import markdown_table

SEQUENTIAL = "sequential"
PARALLEL = "parallel"
CONTROL = "control"
VARIANTS = (SEQUENTIAL, PARALLEL)
HYBRID_EVAL = "hybrid_eval"
PRODUCT = "product"
LEXICAL_ONLY = "lexical_only"
VECTOR_ONLY = "vector_only"
PAIRED_PATHS = (HYBRID_EVAL, PRODUCT)
CONTROL_PATHS = (LEXICAL_ONLY, VECTOR_ONLY)
EMBEDDING_STAGE = "embedding"
LEXICAL_STAGE = "lexical_channel"
VECTOR_STAGE = "vector_channel"
CONTEXT_STAGE = "context_window"
STAGES = (EMBEDDING_STAGE, LEXICAL_STAGE, VECTOR_STAGE, CONTEXT_STAGE)
STAGE_LABELS = {
    EMBEDDING_STAGE: "임베딩 API",
    LEXICAL_STAGE: "lexical 채널 전체",
    VECTOR_STAGE: "vector 채널 전체",
    CONTEXT_STAGE: "문맥 창 쿼리",
}
PATH_LABELS = {
    HYBRID_EVAL: "hybrid (eval 경로, k=10, window=1)",
    PRODUCT: "제품 경로 `retrieve_contexts` (limit=5)",
    LEXICAL_ONLY: "대조군: lexical 단독 (k=10)",
    VECTOR_ONLY: "대조군: vector 단독 (k=10)",
}
SUBSETS = ("all", "ko", "en")
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260928
EMBEDDING_PRICE_PER_MTOK = 0.13
TOKENS_PER_MILLION = 1_000_000
MISMATCH_PREVIEW = 8
ERROR_PREVIEW = 10

type Clock = Callable[[], float]
type Signature = tuple[tuple[int, str], ...]
type PathFn = Callable[[Retriever, str], Awaitable[tuple[list[RetrievalContext], str]]]


class StageRecorder:
    def __init__(self, clock: Clock = time.perf_counter) -> None:
        self.clock = clock
        self._totals: dict[str, float] = {}

    def reset(self) -> None:
        self._totals = {}

    def add(self, stage: str, seconds: float) -> None:
        self._totals[stage] = self._totals.get(stage, 0.0) + seconds

    def snapshot_ms(self) -> dict[str, float]:
        return {stage: seconds * 1000.0 for stage, seconds in self._totals.items()}

    async def measure[T](self, stage: str, awaitable: Awaitable[T]) -> T:
        started = self.clock()
        try:
            return await awaitable
        finally:
            self.add(stage, self.clock() - started)


class TimedEmbedder:
    def __init__(self, inner: QueryEmbedder, recorder: StageRecorder) -> None:
        self.inner = inner
        self.recorder = recorder

    def is_available(self) -> bool:
        return self.inner.is_available()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.recorder.measure(EMBEDDING_STAGE, self.inner.embed(texts))


class BenchRetriever(Retriever):
    def __init__(
        self,
        pool: AnyPool,
        embedder: QueryEmbedder | None = None,
        config: RetrievalConfig | None = None,
        *,
        sequential: bool = False,
        recorder: StageRecorder | None = None,
    ) -> None:
        if recorder is not None and embedder is not None:
            embedder = TimedEmbedder(embedder, recorder)
        super().__init__(pool, embedder, config)
        self.sequential = sequential
        self.recorder = recorder

    async def _measure[T](self, stage: str, awaitable: Awaitable[T]) -> T:
        if self.recorder is None:
            return await awaitable
        return await self.recorder.measure(stage, awaitable)

    async def search_lexical(
        self, query: str, *, limit: int = DEFAULT_SEARCH_LIMIT, arxiv_id: str | None = None
    ) -> list[RetrievalHit]:
        return await self._measure(LEXICAL_STAGE, super().search_lexical(query, limit=limit, arxiv_id=arxiv_id))

    async def search_vector(
        self, query: str, *, limit: int = DEFAULT_SEARCH_LIMIT, arxiv_id: str | None = None
    ) -> list[RetrievalHit]:
        return await self._measure(VECTOR_STAGE, super().search_vector(query, limit=limit, arxiv_id=arxiv_id))

    async def build_contexts(
        self, hits: Sequence[RetrievalHit], *, adjacency_window: int = DEFAULT_ADJACENCY_WINDOW
    ) -> list[RetrievalContext]:
        return await self._measure(CONTEXT_STAGE, super().build_contexts(hits, adjacency_window=adjacency_window))

    async def hybrid_inputs(
        self,
        query: str,
        *,
        limit: int = DEFAULT_SEARCH_LIMIT,
        arxiv_id: str | None = None,
        lexical_limit: int | None = None,
        vector_limit: int | None = None,
    ) -> HybridInputs:
        if not self.sequential:
            return await super().hybrid_inputs(
                query, limit=limit, arxiv_id=arxiv_id, lexical_limit=lexical_limit, vector_limit=vector_limit
            )
        normalized_query = normalize_search_query(query)
        if not normalized_query:
            return HybridInputs(query=normalized_query, lexical=(), vector=())
        branch_limit = hybrid_branch_limit(max(1, limit))
        lexical = await self.search_lexical(normalized_query, limit=lexical_limit or branch_limit, arxiv_id=arxiv_id)
        vector = await self.search_vector(normalized_query, limit=vector_limit or branch_limit, arxiv_id=arxiv_id)
        return HybridInputs(query=normalized_query, lexical=tuple(lexical), vector=tuple(vector))


def build_paths(
    *, k: int, adjacency_window: int, product_limit: int, include_product: bool = True
) -> tuple[dict[str, PathFn], dict[str, PathFn]]:
    async def hybrid_eval(retriever: Retriever, query: str) -> tuple[list[RetrievalContext], str]:
        return await retriever.search_hybrid_contexts(query, limit=k, adjacency_window=adjacency_window), "hybrid"

    async def product(retriever: Retriever, query: str) -> tuple[list[RetrievalContext], str]:
        result = await retriever.retrieve_contexts(query, limit=product_limit)
        return list(result.contexts), result.mode

    async def lexical_only(retriever: Retriever, query: str) -> tuple[list[RetrievalContext], str]:
        return await retriever.search_lexical_contexts(query, limit=k, adjacency_window=adjacency_window), "lexical"

    async def vector_only(retriever: Retriever, query: str) -> tuple[list[RetrievalContext], str]:
        return await retriever.search_vector_contexts(query, limit=k, adjacency_window=adjacency_window), "vector"

    paired: dict[str, PathFn] = {HYBRID_EVAL: hybrid_eval}
    if include_product:
        paired[PRODUCT] = product
    return paired, {LEXICAL_ONLY: lexical_only, VECTOR_ONLY: vector_only}


def signature(contexts: Sequence[RetrievalContext]) -> Signature:
    return tuple((context.hit.chunk_id, context.hit.arxiv_id) for context in contexts)


def pair_order(repeat: int) -> tuple[str, str]:
    return (SEQUENTIAL, PARALLEL) if repeat % 2 == 0 else (PARALLEL, SEQUENTIAL)


@dataclass(frozen=True)
class Sample:
    query_id: str
    lang: str
    repeat: int
    path: str
    variant: str
    position: int
    total_ms: float
    stages_ms: Mapping[str, float]
    signature: Signature
    mode: str = ""
    error: str = ""


@dataclass
class BenchmarkPlan:
    queries: Sequence[EvalQuery]
    retrievers: Mapping[str, Retriever]
    recorder: StageRecorder
    paired_paths: Mapping[str, PathFn]
    control_paths: Mapping[str, PathFn] = field(default_factory=dict)
    repeats: int = 3
    warmup: int = 1
    clock: Clock = time.perf_counter
    control_variant: str = PARALLEL


@dataclass(frozen=True)
class _Run:
    total_ms: float
    stages_ms: dict[str, float]
    signature: Signature
    mode: str
    error: str


async def _run_once(plan: BenchmarkPlan, fn: PathFn, retriever: Retriever, query: EvalQuery) -> _Run:
    plan.recorder.reset()
    started = plan.clock()
    contexts: list[RetrievalContext] = []
    mode = ""
    error = ""
    try:
        contexts, mode = await fn(retriever, query.query)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (plan.clock() - started) * 1000.0
    return _Run(elapsed_ms, plan.recorder.snapshot_ms(), signature(contexts), mode, error)


type BenchProgressFn = Callable[[int, int, EvalQuery], None]


async def run_benchmark(plan: BenchmarkPlan, *, progress: BenchProgressFn | None = None) -> list[Sample]:
    samples: list[Sample] = []
    total = (plan.warmup + plan.repeats) * len(plan.queries)
    done = 0
    for repeat in range(-plan.warmup, plan.repeats):
        order = pair_order(repeat)
        for query in plan.queries:
            block: list[tuple[str, str, PathFn]] = [
                (path, variant, fn) for path, fn in plan.paired_paths.items() for variant in order
            ]
            block += [(path, CONTROL, fn) for path, fn in plan.control_paths.items()]
            positions: dict[str, int] = {}
            for path, variant, fn in block:
                retriever = plan.retrievers[plan.control_variant if variant == CONTROL else variant]
                run = await _run_once(plan, fn, retriever, query)
                position = positions.get(path, 0)
                positions[path] = position + 1
                if repeat >= 0:
                    samples.append(
                        Sample(
                            query_id=query.id,
                            lang=query.lang,
                            repeat=repeat,
                            path=path,
                            variant=variant,
                            position=position,
                            total_ms=run.total_ms,
                            stages_ms=run.stages_ms,
                            signature=run.signature,
                            mode=run.mode,
                            error=run.error,
                        )
                    )
            done += 1
            if progress is not None:
                progress(done, total, query)
    return samples


async def memo_equality(
    paths: Mapping[str, PathFn], retrievers: Mapping[str, Retriever], queries: Sequence[EvalQuery]
) -> dict[str, tuple[int, int]]:
    equality: dict[str, tuple[int, int]] = {}
    for path, fn in paths.items():
        same = 0
        for query in queries:
            sequential = await fn(retrievers[SEQUENTIAL], query.query)
            parallel = await fn(retrievers[PARALLEL], query.query)
            same += int(sequential == parallel)
        equality[path] = (same, len(queries))
    return equality


def select(samples: Sequence[Sample], **criteria: Any) -> list[Sample]:
    return [sample for sample in samples if all(getattr(sample, key) == value for key, value in criteria.items())]


def summarize(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "mean": statistics.fmean(values) if values else None,
    }


def per_query_medians(samples: Sequence[Sample], query_ids: Sequence[str]) -> list[float]:
    by_query: dict[str, list[float]] = {}
    for sample in samples:
        by_query.setdefault(sample.query_id, []).append(sample.total_ms)
    return [statistics.median(by_query[query_id]) for query_id in query_ids]


@dataclass(frozen=True)
class PairedDelta:
    path: str
    subset: str
    n: int
    n_clusters: int
    median_a: float
    median_b: float
    delta_ms: float
    low_ms: float
    high_ms: float
    relative: float
    relative_low: float
    relative_high: float


def paired_delta(
    samples: Sequence[Sample],
    queries: Sequence[EvalQuery],
    path: str,
    *,
    subset: str = "all",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> PairedDelta:
    chosen = [query for query in queries if subset == "all" or query.lang == subset]
    ids = [query.id for query in chosen]
    a = per_query_medians(select(samples, path=path, variant=SEQUENTIAL), ids)
    b = per_query_medians(select(samples, path=path, variant=PARALLEL), ids)
    clusters = clusters_of([cluster_key(query_id) for query_id in ids])
    draws = bootstrap_draws(len(clusters), resamples, seed)
    delta, low, high = paired_bootstrap(b, a, draws, clusters=clusters)
    ratios = [right / left - 1.0 for left, right in zip(a, b, strict=True)]
    relative, relative_low, relative_high = paired_bootstrap(ratios, [0.0] * len(ratios), draws, clusters=clusters)
    return PairedDelta(
        path=path,
        subset=subset,
        n=len(ids),
        n_clusters=len(clusters),
        median_a=statistics.median(a),
        median_b=statistics.median(b),
        delta_ms=delta,
        low_ms=low,
        high_ms=high,
        relative=relative,
        relative_low=relative_low,
        relative_high=relative_high,
    )


@dataclass(frozen=True)
class EqualityRow:
    path: str
    comparison: str
    compared: int
    mismatches: int
    same_set_reordered: int
    mismatch_ids: tuple[str, ...]


def _compare(label: str, path: str, left: Mapping[str, Sample], right: Mapping[str, Sample]) -> EqualityRow:
    keys = sorted(set(left) & set(right))
    mismatched = [
        key for key in keys if left[key].signature != right[key].signature or left[key].mode != right[key].mode
    ]
    reordered = [key for key in mismatched if set(left[key].signature) == set(right[key].signature)]
    return EqualityRow(path, label, len(keys), len(mismatched), len(reordered), tuple(mismatched))


def equality_rows(samples: Sequence[Sample], path: str, repeats: int) -> list[EqualityRow]:
    def index(variant: str, repeat: int) -> dict[str, Sample]:
        return {sample.query_id: sample for sample in select(samples, path=path, variant=variant, repeat=repeat)}

    rows = [_compare(f"A vs B (반복 {r})", path, index(SEQUENTIAL, r), index(PARALLEL, r)) for r in range(repeats)]
    for variant, name in ((SEQUENTIAL, "A"), (PARALLEL, "B")):
        rows += [
            _compare(f"{name} vs {name} (반복 {r}→{r + 1})", path, index(variant, r), index(variant, r + 1))
            for r in range(repeats - 1)
        ]
    return rows


def stage_medians(samples: Sequence[Sample]) -> dict[str, float | None]:
    return {
        stage: (statistics.median([sample.stages_ms.get(stage, 0.0) for sample in samples]) if samples else None)
        for stage in STAGES
    }


def embedding_cost(tokens: int, price_per_mtok: float = EMBEDDING_PRICE_PER_MTOK) -> float:
    return tokens / TOKENS_PER_MILLION * price_per_mtok


def write_samples_csv(samples: Sequence[Sample], path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "query_id",
                "lang",
                "repeat",
                "path",
                "variant",
                "position",
                "total_ms",
                *[f"{stage}_ms" for stage in STAGES],
                "mode",
                "result",
                "error",
            ]
        )
        for sample in samples:
            writer.writerow(
                [sample.query_id, sample.lang, sample.repeat, sample.path, sample.variant, sample.position]
                + [f"{sample.total_ms:.3f}"]
                + [f"{sample.stages_ms.get(stage, 0.0):.3f}" for stage in STAGES]
                + [sample.mode, " ".join(f"{chunk_id}:{arxiv_id}" for chunk_id, arxiv_id in sample.signature)]
                + [sample.error]
            )


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


def _variants(path: str) -> tuple[str, ...]:
    return VARIANTS if path in PAIRED_PATHS else (CONTROL,)


def latency_rows(samples: Sequence[Sample], paths: Sequence[str]) -> list[list[str]]:
    rows = []
    for path in paths:
        for variant in _variants(path):
            chosen = select(samples, path=path, variant=variant)
            overall = summarize([sample.total_ms for sample in chosen])
            cells = [PATH_LABELS.get(path, path), variant, str(overall["n"]), _ms(overall["p50"]), _ms(overall["p95"])]
            cells.append(_ms(overall["mean"]))
            for lang in ("ko", "en"):
                stats = summarize([sample.total_ms for sample in chosen if sample.lang == lang])
                cells += [_ms(stats["p50"]), _ms(stats["p95"])]
            rows.append(cells)
    return rows


def _stage_rows(samples: Sequence[Sample], paths: Sequence[str]) -> list[list[str]]:
    rows = []
    for path in paths:
        for variant in _variants(path):
            chosen = select(samples, path=path, variant=variant)
            medians = stage_medians(chosen)
            total = statistics.median([sample.total_ms for sample in chosen]) if chosen else None
            rows.append([PATH_LABELS.get(path, path), variant, _ms(total), *[_ms(medians[stage]) for stage in STAGES]])
    return rows


def render_markdown(
    samples: Sequence[Sample],
    *,
    title: str,
    meta: Mapping[str, str],
    deltas: Sequence[PairedDelta],
    equality: Sequence[EqualityRow],
    memo: Mapping[str, tuple[int, int]],
    embedding_usage: Mapping[str, Any],
    notes: Sequence[str] = (),
) -> str:
    paths = [path for path in (*PAIRED_PATHS, *CONTROL_PATHS) if select(samples, path=path)]
    lines = [f"# {title}", ""]
    lines += markdown_table(["항목", "값"], [[key, value] for key, value in meta.items()])
    lines += ["", "## 지연 (ms, 모든 질의 x 반복 표본)", ""]
    lines += markdown_table(
        ["경로", "변형", "n", "p50", "p95", "mean", "ko p50", "ko p95", "en p50", "en p95"],
        latency_rows(samples, paths),
    )
    lines += [
        "",
        "## 질의별 차이 B-A (병렬 - 순차)",
        "",
        "질의마다 반복 중앙값을 낸 뒤 차이를 평균한다. 95% 구간은 ko/en 쌍 클러스터 재표집(paired bootstrap)이다.",
        "상대 변화는 질의별 B/A-1의 평균이다.",
        "",
    ]
    lines += markdown_table(
        ["경로", "부분집합", "질의", "클러스터", "A 중앙값", "B 중앙값", "Δ ms (95% CI)", "상대 (95% CI)"],
        [
            [
                PATH_LABELS.get(row.path, row.path),
                row.subset,
                str(row.n),
                str(row.n_clusters),
                _ms(row.median_a),
                _ms(row.median_b),
                f"{row.delta_ms:+.1f} [{row.low_ms:+.1f}, {row.high_ms:+.1f}]",
                f"{_pct(row.relative)} [{_pct(row.relative_low)}, {_pct(row.relative_high)}]",
            ]
            for row in deltas
        ],
    )
    lines += [
        "",
        "## 단계별 시간 (ms, 표본 중앙값)",
        "",
        "계측 retriever(`arxplore_eval.latency.BenchRetriever`) 기준. 병렬 변형에서는 lexical·vector 채널이 겹치므로 "
        "단계 합이 전체보다 크다. vector 채널 시간에는 임베딩 API 시간이 들어 있다.",
        "",
    ]
    lines += markdown_table(
        ["경로", "변형", "전체", *[STAGE_LABELS[stage] for stage in STAGES]], _stage_rows(samples, paths)
    )
    lines += [
        "",
        "## 결과 동일성",
        "",
        "순위 (chunk_id, arxiv_id) 목록과 검색 방식 라벨 비교. A vs A / B vs B는 같은 코드를 다시 실행한 기준선이다"
        "(임베딩 API가 호출마다 조금 다른 벡터를 돌려줄 수 있다).",
        "",
    ]
    lines += markdown_table(
        ["경로", "비교", "질의", "불일치", "같은 집합·순서만 다름", "불일치 질의"],
        [
            [
                PATH_LABELS.get(row.path, row.path),
                row.comparison,
                str(row.compared),
                str(row.mismatches),
                str(row.same_set_reordered),
                ", ".join(row.mismatch_ids[:MISMATCH_PREVIEW])
                + (" ..." if len(row.mismatch_ids) > MISMATCH_PREVIEW else ""),
            ]
            for row in equality
        ],
    )
    if memo:
        lines += [
            "",
            "임베딩 고정 패스(질의당 임베딩 1회를 A와 B가 공유): 반환 문맥 전체(점수·점수 내역·문맥 창 포함) 비교.",
            "",
        ]
        lines += markdown_table(
            ["경로", "동일 / 전체"],
            [[PATH_LABELS.get(path, path), f"{same} / {total}"] for path, (same, total) in memo.items()],
        )
    lines += ["", "## 임베딩 호출", ""]
    lines += markdown_table(["항목", "값"], [[key, str(value)] for key, value in embedding_usage.items()])
    errors = [sample for sample in samples if sample.error]
    lines += ["", f"오류 표본: {len(errors)}개"]
    for sample in errors[:ERROR_PREVIEW]:
        lines.append(f"- {sample.query_id} {sample.path}/{sample.variant} r{sample.repeat}: {sample.error}")
    if notes:
        lines += ["", "## 메모", ""] + [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"
