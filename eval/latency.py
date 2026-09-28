"""hybrid 검색 지연 A/B 측정: lexical/vector 채널 순차 실행(A) vs 병렬 실행(B).

절차(`scripts/eval_hybrid_latency.py`가 DB·키를 붙여 실행한다)

1. 같은 프로세스에서 같은 저장소·임베딩 클라이언트를 공유하는 retriever 두 개를 만든다. A는 `parallel_channels=False`
   (병렬화 전 순차 경로), B는 기본값(병렬)이다. `instrument`가 두 retriever에 같은 계측 래퍼를 씌운다.
2. 워밍업 1회 뒤 반복마다 질의 순서대로 블록을 실행한다. 블록 = 경로별 A/B 쌍(eval 하네스 경로, 제품 경로)과
   대조군(lexical 단독, vector 단독). 쌍의 실행 순서는 반복마다 A→B / B→A로 번갈아(`pair_order`) API 시간 변동을 상쇄한다.
   임베딩은 캐시하지 않는다. 매 호출이 질의 임베딩 API를 부르며, 그 왕복이 병렬화로 숨기려는 지연이다.
3. 질의별 반복 중앙값의 차이(B−A)를 ko/en 쌍 클러스터 단위로 재표집해 95% 구간을 낸다(`eval.fusion_sweep` 도구 재사용).
4. 결과 동일성: 같은 반복의 A와 B 순위 (chunk_id, arxiv_id)를 비교하고, 같은 변형의 반복 간 비교(A vs A, B vs B)를
   임베딩 비결정성 기준선으로 함께 낸다. 임베딩을 메모이즈한 별도 패스에서는 A와 B 결과 전체(dict)가 같아야 한다.
"""

from __future__ import annotations

import csv
import statistics
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.fusion_sweep import bootstrap_draws, cluster_key, clusters_of, paired_bootstrap
from eval.metrics import percentile

SEQUENTIAL = "sequential"
PARALLEL = "parallel"
CONTROL = "control"
HYBRID_EVAL = "hybrid_eval"
PRODUCT = "product"
LEXICAL_ONLY = "lexical_only"
VECTOR_ONLY = "vector_only"
PAIRED_PATHS = (HYBRID_EVAL, PRODUCT)
CONTROL_PATHS = (LEXICAL_ONLY, VECTOR_ONLY)
STAGES = ("embedding", "lexical_sql", "vector_sql", "lexical_channel", "vector_channel", "fusion", "context_window")
STAGE_LABELS = {
    "embedding": "임베딩 API",
    "lexical_sql": "lexical SQL",
    "vector_sql": "vector SQL",
    "lexical_channel": "lexical 채널 전체",
    "vector_channel": "vector 채널 전체",
    "fusion": "융합",
    "context_window": "문맥 창 쿼리",
}
PATH_LABELS = {
    HYBRID_EVAL: "hybrid (eval 경로, k=10, window=1)",
    PRODUCT: "제품 경로 `retrieve_contexts` (limit=5)",
    LEXICAL_ONLY: "대조군: lexical 단독 (k=10)",
    VECTOR_ONLY: "대조군: vector 단독 (k=10)",
}
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260928
EMBEDDING_PRICE_PER_MTOK = 0.13


class StageRecorder:
    """단계별 소요 시간(초)을 누적한다. 병렬 경로에서 두 스레드가 동시에 기록하므로 잠금을 쓴다."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self.clock = clock
        self._lock = threading.Lock()
        self._totals: dict[str, float] = {}

    def reset(self) -> None:
        with self._lock:
            self._totals = {}

    def add(self, stage: str, seconds: float) -> None:
        with self._lock:
            self._totals[stage] = self._totals.get(stage, 0.0) + seconds

    def snapshot_ms(self) -> dict[str, float]:
        with self._lock:
            return {stage: seconds * 1000.0 for stage, seconds in self._totals.items()}

    def wrap(self, stage: str, function: Callable[..., Any]) -> Callable[..., Any]:
        def timed(*args: Any, **kwargs: Any) -> Any:
            started = self.clock()
            try:
                return function(*args, **kwargs)
            finally:
                self.add(stage, self.clock() - started)

        return timed


class TimedProxy:
    """대상 객체의 지정 메서드만 `StageRecorder`로 감싸고 나머지 속성은 그대로 넘긴다."""

    def __init__(self, target: Any, stages: Mapping[str, str], recorder: StageRecorder) -> None:
        self._target = target
        self._wrapped = {name: recorder.wrap(stage, getattr(target, name)) for name, stage in stages.items()}

    def __getattr__(self, name: str) -> Any:
        wrapped = self.__dict__.get("_wrapped", {})
        if name in wrapped:
            return wrapped[name]
        return getattr(self._target, name)


def instrument(retriever: Any, recorder: StageRecorder) -> Any:
    """retriever의 저장소·임베딩 클라이언트를 계측 프록시로 바꾸고 채널·융합 메서드를 인스턴스 속성으로 감싼다.
    제품 코드는 건드리지 않는다. 두 변형(A/B)에 같은 래퍼를 씌워 계측 비용이 양쪽에 같게 들어가게 한다."""
    retriever.repository = TimedProxy(
        retriever.repository,
        {"list_chunk_candidates_by_query": "lexical_sql", "list_chunk_windows": "context_window"},
        recorder,
    )
    retriever.embedding_client = TimedProxy(retriever.embedding_client, {"embed_texts": "embedding"}, recorder)
    retriever.vector_repository = TimedProxy(
        retriever.vector_repository, {"search_paper_chunks": "vector_sql"}, recorder
    )
    retriever.search_paper_chunks = recorder.wrap("lexical_channel", retriever.search_paper_chunks)
    retriever.search_paper_chunks_by_vector = recorder.wrap("vector_channel", retriever.search_paper_chunks_by_vector)
    retriever._merge_hybrid_candidates = recorder.wrap("fusion", retriever._merge_hybrid_candidates)
    return retriever


def signature(contexts: Sequence[Mapping[str, Any]]) -> tuple[tuple[Any, Any], ...]:
    """순위 비교용 (chunk_id, arxiv_id) 목록."""
    return tuple((context.get("chunk_id"), context.get("arxiv_id")) for context in contexts)


def pair_order(repeat: int) -> tuple[str, str]:
    """반복 번호(0부터)가 짝수면 A→B, 홀수면 B→A."""
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
    signature: tuple[tuple[Any, Any], ...]
    mode: str = ""
    error: str = ""


# (retriever, 질의) -> (문맥 목록, 검색 방식 라벨)
PathFn = Callable[[Any, str], tuple[list[dict], str]]


@dataclass
class BenchmarkPlan:
    queries: Sequence[Any]
    retrievers: Mapping[str, Any]
    recorder: StageRecorder
    paired_paths: Mapping[str, PathFn]
    control_paths: Mapping[str, PathFn] = field(default_factory=dict)
    repeats: int = 3
    warmup: int = 1
    clock: Callable[[], float] = time.perf_counter
    control_variant: str = PARALLEL


def _run_once(plan: BenchmarkPlan, fn: PathFn, retriever: Any, query: Any) -> tuple[float, dict, tuple, str, str]:
    plan.recorder.reset()
    started = plan.clock()
    try:
        contexts, mode = fn(retriever, query.query)
        error = ""
    except Exception as exc:  # noqa: BLE001 - 측정은 계속하고 오류를 기록한다
        contexts, mode, error = [], "", f"{type(exc).__name__}: {exc}"
    elapsed_ms = (plan.clock() - started) * 1000.0
    return elapsed_ms, plan.recorder.snapshot_ms(), signature(contexts), mode, error


def run_benchmark(plan: BenchmarkPlan, *, progress: Callable[[int, int, Any], None] | None = None) -> list[Sample]:
    """워밍업(반복 번호 음수, 기록하지 않음) 뒤 `plan.repeats`회 반복한다. 반복 안에서 질의마다 블록을 실행한다."""
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
                elapsed_ms, stages_ms, sig, mode, error = _run_once(plan, fn, retriever, query)
                position = positions.get(path, 0)
                positions[path] = position + 1
                if repeat >= 0:
                    samples.append(
                        Sample(
                            query.id,
                            query.lang,
                            repeat,
                            path,
                            variant,
                            position,
                            elapsed_ms,
                            stages_ms,
                            sig,
                            mode,
                            error,
                        )
                    )
            done += 1
            if progress is not None:
                progress(done, total, query)
    return samples


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
    """질의 순서대로 반복 중앙값(ms)."""
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
    queries: Sequence[Any],
    path: str,
    *,
    subset: str = "all",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> PairedDelta:
    """질의별 반복 중앙값 차이(B−A, ms)와 상대 변화(B/A−1)의 평균, ko/en 쌍 클러스터 재표집 95% 구간."""
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
        path,
        subset,
        len(ids),
        len(clusters),
        statistics.median(a),
        statistics.median(b),
        delta,
        low,
        high,
        relative,
        relative_low,
        relative_high,
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
    """같은 반복의 A vs B, 그리고 기준선으로 연속 반복의 A vs A, B vs B."""

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
    """단계별 중앙값(ms). 그 단계가 없는 호출은 0으로 친다."""
    return {
        stage: (statistics.median([sample.stages_ms.get(stage, 0.0) for sample in samples]) if samples else None)
        for stage in STAGES
    }


def embedding_cost(tokens: int, price_per_mtok: float = EMBEDDING_PRICE_PER_MTOK) -> float:
    return tokens / 1_000_000 * price_per_mtok


def write_samples_csv(samples: Sequence[Sample], path: str | Path) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["query_id", "lang", "repeat", "path", "variant", "position", "total_ms", *[f"{s}_ms" for s in STAGES]]
            + ["mode", "result", "error"]
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


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def latency_rows(samples: Sequence[Sample], paths: Sequence[str]) -> list[list[str]]:
    rows = []
    for path in paths:
        variants = (SEQUENTIAL, PARALLEL) if path in PAIRED_PATHS else (CONTROL,)
        for variant in variants:
            cells = [PATH_LABELS.get(path, path), variant]
            for lang in (None, "ko", "en"):
                chosen = select(samples, path=path, variant=variant)
                if lang:
                    chosen = [sample for sample in chosen if sample.lang == lang]
                stats = summarize([sample.total_ms for sample in chosen])
                cells += [_ms(stats["p50"]), _ms(stats["p95"])]
                if lang is None:
                    cells.insert(2, str(stats["n"]))
                    cells.append(_ms(stats["mean"]))
            rows.append(cells)
    return rows


def render_markdown(
    samples: Sequence[Sample],
    queries: Sequence[Any],
    *,
    title: str,
    meta: Mapping[str, str],
    deltas: Sequence[PairedDelta],
    equality: Sequence[EqualityRow],
    memo_equality: Mapping[str, tuple[int, int]],
    embedding_usage: Mapping[str, Any],
    notes: Sequence[str] = (),
) -> str:
    paths = [path for path in (*PAIRED_PATHS, *CONTROL_PATHS) if select(samples, path=path)]
    lines = [f"# {title}", ""]
    lines += _table(["항목", "값"], [[key, value] for key, value in meta.items()])
    lines += ["", "## 지연 (ms, 모든 질의 × 반복 표본)", ""]
    lines += _table(
        ["경로", "변형", "n", "p50", "p95", "mean", "ko p50", "ko p95", "en p50", "en p95"],
        latency_rows(samples, paths),
    )
    lines += [
        "",
        "## 질의별 차이 B−A (병렬 − 순차)",
        "",
        "질의마다 반복 중앙값을 낸 뒤 차이를 평균한다. 95% 구간은 ko/en 쌍 클러스터 재표집(paired bootstrap)이다.",
        "상대 변화는 질의별 B/A−1의 평균이다.",
        "",
    ]
    lines += _table(
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
        "계측 래퍼(eval/latency.py `instrument`) 기준. 병렬 변형에서는 lexical·vector 채널이 겹치므로 단계 합이 전체보다 크다.",
        "",
    ]
    stage_rows = []
    for path in paths:
        variants = (SEQUENTIAL, PARALLEL) if path in PAIRED_PATHS else (CONTROL,)
        for variant in variants:
            chosen = select(samples, path=path, variant=variant)
            medians = stage_medians(chosen)
            total = statistics.median([sample.total_ms for sample in chosen])
            stage_rows.append(
                [PATH_LABELS.get(path, path), variant, _ms(total), *[_ms(medians[stage]) for stage in STAGES]]
            )
    lines += _table(["경로", "변형", "전체", *[STAGE_LABELS[stage] for stage in STAGES]], stage_rows)
    lines += [
        "",
        "## 결과 동일성",
        "",
        "순위 (chunk_id, arxiv_id) 목록과 검색 방식 라벨 비교. A vs A / B vs B는 같은 코드를 다시 실행한 기준선이다"
        "(임베딩 API가 호출마다 조금 다른 벡터를 돌려줄 수 있다).",
        "",
    ]
    lines += _table(
        ["경로", "비교", "질의", "불일치", "같은 집합·순서만 다름", "불일치 질의"],
        [
            [
                PATH_LABELS.get(row.path, row.path),
                row.comparison,
                str(row.compared),
                str(row.mismatches),
                str(row.same_set_reordered),
                ", ".join(row.mismatch_ids[:8]) + (" ..." if len(row.mismatch_ids) > 8 else ""),
            ]
            for row in equality
        ],
    )
    if memo_equality:
        lines += [
            "",
            "임베딩 고정 패스(질의당 임베딩 1회를 A와 B가 공유): 반환 문맥 dict 전체(점수·score_breakdown·context 포함) 비교.",
            "",
        ]
        lines += _table(
            ["경로", "동일 / 전체"],
            [[PATH_LABELS.get(path, path), f"{same} / {total}"] for path, (same, total) in memo_equality.items()],
        )
    lines += ["", "## 임베딩 호출", ""]
    lines += _table(["항목", "값"], [[key, str(value)] for key, value in embedding_usage.items()])
    errors = [sample for sample in samples if sample.error]
    lines += ["", f"오류 표본: {len(errors)}개"]
    for sample in errors[:10]:
        lines.append(f"- {sample.query_id} {sample.path}/{sample.variant} r{sample.repeat}: {sample.error}")
    if notes:
        lines += ["", "## 메모", ""] + [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"
