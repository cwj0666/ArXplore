import csv
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from arxplore_eval.dataset import EvalQuery
from arxplore_eval.fusion_sweep import DEFAULT_CONFIDENCE
from arxplore_eval.paired_stats import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    DrawsFn,
    draws_cache,
    interval_text,
    paired_delta,
)
from arxplore_eval.subsets import build_subsets, in_subset
from arxplore_eval.tables import csv_number, format_rate, markdown_table

PAPER = "paper"
CHUNK = "chunk"
LEVELS = (PAPER, CHUNK)
CHUNK_SUBSETS = ("all", "ko", "en")
OUTCOME_SUBSETS = ("all", "ko", "en")
LEXICAL_PREFIX = "lexical"
LEXICAL_METHOD = "lexical"
VECTOR_METHOD = "vector"
IMPROVED = "improved"
TIED = "tied"
WORSE = "worse"
OUTCOMES = (IMPROVED, TIED, WORSE)
BOTH = "both"
LEXICAL_ONLY = "lexical_only"
VECTOR_ONLY = "vector_only"
NEITHER = "neither"
COMPLEMENT_CELLS = (BOTH, LEXICAL_ONLY, VECTOR_ONLY, NEITHER)
COMPLEMENT_LABELS = {
    BOTH: "둘 다 맞힘",
    LEXICAL_ONLY: "lexical만 맞힘",
    VECTOR_ONLY: "vector만 맞힘",
    NEITHER: "둘 다 실패",
}
MISMATCH_PREVIEW = 10
RANK_TOLERANCE = 1e-3
MD_FILE = "compare.md"
METRICS_CSV_FILE = "compare_metrics.csv"
QUERIES_CSV_FILE = "compare_queries.csv"


class CompareError(ValueError):
    pass


@dataclass(frozen=True)
class RunRow:
    query_id: str
    lang: str
    source: str
    category: str
    expected_behavior: str
    method: str
    k: int
    error: str
    paper: dict[str, float | None]
    chunk: dict[str, float | None]
    retrieved_arxiv_ids: tuple[str, ...]
    retrieved_chunk_ids: tuple[int, ...]

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass(frozen=True)
class RunTable:
    path: Path
    metrics: tuple[str, ...]
    rows: dict[tuple[str, str], RunRow]

    @property
    def methods(self) -> list[str]:
        return list(dict.fromkeys(method for method, _ in self.rows))

    @property
    def k(self) -> int:
        values = {row.k for row in self.rows.values()}
        if len(values) != 1:
            raise CompareError(f"{self.path}: k가 하나가 아닙니다 {sorted(values)}")
        return values.pop()


def _optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _metric_columns(header: Sequence[str], prefix: str) -> list[str]:
    return [name[len(prefix) :] for name in header if name.startswith(prefix)]


def read_run_csv(path: Path) -> RunTable:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        paper_names = _metric_columns(header, f"{PAPER}_")
        chunk_names = _metric_columns(header, f"{CHUNK}_")
        rows: dict[tuple[str, str], RunRow] = {}
        for record in reader:
            row = RunRow(
                query_id=record["query_id"],
                lang=record.get("lang", ""),
                source=record.get("source", ""),
                category=record.get("category", ""),
                expected_behavior=record.get("expected_behavior") or "answer",
                method=record["method"],
                k=int(record["k"]),
                error=record.get("error", ""),
                paper={name: _optional_float(record.get(f"{PAPER}_{name}")) for name in paper_names},
                chunk={name: _optional_float(record.get(f"{CHUNK}_{name}")) for name in chunk_names},
                retrieved_arxiv_ids=tuple((record.get("retrieved_arxiv_ids") or "").split()),
                retrieved_chunk_ids=tuple(int(value) for value in (record.get("retrieved_chunk_ids") or "").split()),
            )
            key = (row.method, row.query_id)
            if key in rows:
                raise CompareError(f"{path}: 같은 (method, query_id) 행이 두 번 있습니다: {key}")
            rows[key] = row
    if not rows:
        raise CompareError(f"{path}: 행이 없습니다")
    return RunTable(path=path, metrics=tuple(paper_names), rows=rows)


def first_relevant_rank(ranked: Sequence[Hashable], relevant: Sequence[Hashable], k: int) -> int | None:
    wanted = set(relevant)
    for position, item in enumerate(ranked[:k], start=1):
        if item in wanted:
            return position
    return None


@dataclass(frozen=True)
class QueryPair:
    method: str
    query: EvalQuery
    a: RunRow
    b: RunRow
    k: int

    def values(self, level: str, metric: str) -> tuple[float | None, float | None]:
        left = self.a.paper if level == PAPER else self.a.chunk
        right = self.b.paper if level == PAPER else self.b.chunk
        return left.get(metric), right.get(metric)

    def paper_ranks(self) -> tuple[int | None, int | None]:
        relevant = self.query.relevant_arxiv_ids
        return (
            first_relevant_rank(self.a.retrieved_arxiv_ids, relevant, self.k),
            first_relevant_rank(self.b.retrieved_arxiv_ids, relevant, self.k),
        )

    def chunk_ranks(self) -> tuple[int | None, int | None]:
        relevant = self.query.relevant_chunk_ids
        if not relevant:
            return None, None
        return (
            first_relevant_rank(self.a.retrieved_chunk_ids, relevant, self.k),
            first_relevant_rank(self.b.retrieved_chunk_ids, relevant, self.k),
        )

    def matches_queries(self, side: int) -> bool:
        recorded = (self.a if side == 0 else self.b).paper.get(f"mrr@{self.k}")
        if recorded is None:
            return True
        rank = self.paper_ranks()[side]
        expected = 0.0 if rank is None else 1.0 / rank
        return abs(recorded - expected) < RANK_TOLERANCE

    @property
    def lang(self) -> str:
        return self.query.lang

    @property
    def source(self) -> str:
        return self.query.source

    @property
    def category(self) -> str:
        return self.query.category

    @property
    def expected_behavior(self) -> str:
        return self.query.expected_behavior


@dataclass(frozen=True)
class PairedRuns:
    label_a: str
    label_b: str
    k: int
    methods: tuple[str, ...]
    metrics: tuple[str, ...]
    pairs: dict[str, list[QueryPair]]
    warnings: tuple[str, ...] = ()

    def subsets(self) -> list[str]:
        members = [pair for pairs in self.pairs.values() for pair in pairs]
        return build_subsets(members)


def pair_runs(
    a: RunTable,
    b: RunTable,
    queries: Sequence[EvalQuery],
    *,
    label_a: str,
    label_b: str,
    methods: Sequence[str] = (),
) -> PairedRuns:
    if a.k != b.k:
        raise CompareError(f"두 실행의 k가 다릅니다: {a.k} vs {b.k}")
    by_id = {query.id: query for query in queries}
    order = {query.id: index for index, query in enumerate(queries)}
    common_methods = [method for method in a.methods if method in b.methods]
    chosen = list(methods) or common_methods
    unknown = [method for method in chosen if method not in common_methods]
    if unknown:
        raise CompareError(f"두 실행에 모두 있는 방식이 아닙니다: {unknown}")
    if not chosen:
        raise CompareError("두 실행에 공통 방식이 없습니다")
    missing_queries = sorted(
        {query_id for method, query_id in [*a.rows, *b.rows] if method in chosen and query_id not in by_id}
    )
    if missing_queries:
        raise CompareError(f"질의셋에 없는 query_id: {missing_queries[:MISMATCH_PREVIEW]}")
    warnings: list[str] = []
    metrics = tuple(name for name in a.metrics if name in b.metrics)
    pairs: dict[str, list[QueryPair]] = {}
    for method in chosen:
        a_ids = {query_id for row_method, query_id in a.rows if row_method == method}
        b_ids = {query_id for row_method, query_id in b.rows if row_method == method}
        only = sorted(a_ids ^ b_ids)
        if only:
            warnings.append(
                f"{method}: 한쪽 실행에만 있는 질의 {len(only)}개(제외): {', '.join(only[:MISMATCH_PREVIEW])}"
            )
        failed = sorted(
            query_id
            for query_id in a_ids & b_ids
            if not a.rows[(method, query_id)].ok or not b.rows[(method, query_id)].ok
        )
        if failed:
            warnings.append(
                f"{method}: 오류 행이 있는 질의 {len(failed)}개(제외): {', '.join(failed[:MISMATCH_PREVIEW])}"
            )
        kept = sorted((a_ids & b_ids) - set(failed), key=lambda query_id: order[query_id])
        pairs[method] = [
            QueryPair(method, by_id[query_id], a.rows[(method, query_id)], b.rows[(method, query_id)], a.k)
            for query_id in kept
        ]
        for label, side in ((label_a, 0), (label_b, 1)):
            stale = [pair.query.id for pair in pairs[method] if not pair.matches_queries(side)]
            if stale:
                warnings.append(
                    f"{method}: {label} CSV의 MRR@{a.k}가 질의셋 정답으로 다시 센 순위와 다른 질의 {len(stale)}개"
                    f"(다른 정답 라벨로 돈 실행일 수 있다): {', '.join(stale[:MISMATCH_PREVIEW])}"
                )
    return PairedRuns(
        label_a=label_a,
        label_b=label_b,
        k=a.k,
        methods=tuple(chosen),
        metrics=metrics,
        pairs=pairs,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class MetricDelta:
    method: str
    level: str
    subset: str
    metric: str
    n: int
    n_clusters: int
    mean_a: float
    mean_b: float
    delta: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def metric_delta(
    pairs: Sequence[QueryPair],
    *,
    method: str,
    level: str,
    subset: str,
    metric: str,
    draws: DrawsFn,
    confidence: float = DEFAULT_CONFIDENCE,
) -> MetricDelta | None:
    kept: list[tuple[str, float, float]] = []
    for pair in pairs:
        left, right = pair.values(level, metric)
        if left is not None and right is not None:
            kept.append((pair.query.id, left, right))
    result = paired_delta(kept, draws=draws, confidence=confidence)
    if result is None:
        return None
    return MetricDelta(
        method=method,
        level=level,
        subset=subset,
        metric=metric,
        n=result.n,
        n_clusters=result.n_clusters,
        mean_a=result.mean_a,
        mean_b=result.mean_b,
        delta=result.delta,
        low=result.low,
        high=result.high,
    )


def subset_pairs(pairs: Sequence[QueryPair], subset: str) -> list[QueryPair]:
    return [pair for pair in pairs if in_subset(pair, subset)]


def metric_deltas(
    paired: PairedRuns,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[MetricDelta]:
    draws = draws_cache(resamples, seed)
    rows: list[MetricDelta] = []
    subsets = paired.subsets()
    for method in paired.methods:
        for level in LEVELS:
            for subset in subsets if level == PAPER else CHUNK_SUBSETS:
                chosen = subset_pairs(paired.pairs[method], subset)
                for metric in paired.metrics:
                    row = metric_delta(
                        chosen,
                        method=method,
                        level=level,
                        subset=subset,
                        metric=metric,
                        draws=draws,
                        confidence=confidence,
                    )
                    if row is not None:
                        rows.append(row)
    return rows


def outcome_of(a_value: float | None, b_value: float | None) -> str:
    left = a_value or 0.0
    right = b_value or 0.0
    if right > left:
        return IMPROVED
    if right < left:
        return WORSE
    return TIED


@dataclass(frozen=True)
class OutcomeCount:
    method: str
    subset: str
    metric: str
    counts: dict[str, int]
    query_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)


def outcome_counts(paired: PairedRuns, metrics: Sequence[str]) -> list[OutcomeCount]:
    rows: list[OutcomeCount] = []
    for method in paired.methods:
        for subset in OUTCOME_SUBSETS:
            chosen = subset_pairs(paired.pairs[method], subset)
            if not chosen:
                continue
            for metric in metrics:
                grouped: dict[str, list[str]] = {outcome: [] for outcome in OUTCOMES}
                for pair in chosen:
                    grouped[outcome_of(*pair.values(PAPER, metric))].append(pair.query.id)
                rows.append(
                    OutcomeCount(
                        method=method,
                        subset=subset,
                        metric=metric,
                        counts={outcome: len(ids) for outcome, ids in grouped.items()},
                        query_ids={outcome: tuple(ids) for outcome, ids in grouped.items()},
                    )
                )
    return rows


def complement_cell(lexical_hit: float | None, vector_hit: float | None) -> str:
    lexical = bool(lexical_hit)
    vector = bool(vector_hit)
    if lexical and vector:
        return BOTH
    if lexical:
        return LEXICAL_ONLY
    if vector:
        return VECTOR_ONLY
    return NEITHER


def complement_table(paired: PairedRuns, metric: str) -> dict[str, dict[str, int]]:
    if LEXICAL_METHOD not in paired.pairs or VECTOR_METHOD not in paired.pairs:
        return {}
    lexical = {pair.query.id: pair for pair in paired.pairs[LEXICAL_METHOD]}
    vector = {pair.query.id: pair for pair in paired.pairs[VECTOR_METHOD]}
    shared = [query_id for query_id in lexical if query_id in vector]
    table: dict[str, dict[str, int]] = {}
    for label, side in ((paired.label_a, 0), (paired.label_b, 1)):
        counts = dict.fromkeys(COMPLEMENT_CELLS, 0)
        for query_id in shared:
            lexical_hit = lexical[query_id].values(PAPER, metric)[side]
            vector_hit = vector[query_id].values(PAPER, metric)[side]
            counts[complement_cell(lexical_hit, vector_hit)] += 1
        table[label] = counts
    return table


@dataclass(frozen=True)
class IdentityCheck:
    method: str
    compared: int
    mismatched: tuple[str, ...]

    @property
    def identical(self) -> bool:
        return not self.mismatched


def lexical_identity(paired: PairedRuns) -> list[IdentityCheck]:
    checks: list[IdentityCheck] = []
    for method in paired.methods:
        if not method.startswith(LEXICAL_PREFIX):
            continue
        pairs = paired.pairs[method]
        mismatched = tuple(
            pair.query.id
            for pair in pairs
            if pair.a.retrieved_chunk_ids != pair.b.retrieved_chunk_ids
            or pair.a.retrieved_arxiv_ids != pair.b.retrieved_arxiv_ids
        )
        checks.append(IdentityCheck(method, len(pairs), mismatched))
    return checks


def identity_lines(checks: Sequence[IdentityCheck], label_a: str, label_b: str) -> list[str]:
    if not checks:
        return ["lexical 계열 방식이 없어 동일성 검사를 하지 않았습니다."]
    lines: list[str] = []
    for check in checks:
        if check.identical:
            lines.append(
                f"{check.method}: {check.compared}/{check.compared} 질의의 top-k가 {label_a}·{label_b}에서 같다."
            )
        else:
            preview = ", ".join(check.mismatched[:MISMATCH_PREVIEW])
            lines.append(
                f"경고: {check.method} top-k가 {len(check.mismatched)}/{check.compared} 질의에서 다르다 "
                f"(임베딩 외 조건이 바뀌었을 수 있다): {preview}"
            )
    return lines


@dataclass(frozen=True)
class Comparison:
    paired: PairedRuns
    deltas: list[MetricDelta]
    outcomes: list[OutcomeCount]
    complement: dict[str, dict[str, int]]
    identity: list[IdentityCheck]
    resamples: int
    seed: int
    confidence: float

    @property
    def identity_warnings(self) -> list[str]:
        return [
            line
            for line in identity_lines(self.identity, self.paired.label_a, self.paired.label_b)
            if line.startswith("경고")
        ]


def outcome_metrics(k: int) -> list[str]:
    return [f"hit@{min(5, k)}", f"mrr@{k}"]


def compare_runs(
    paired: PairedRuns,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Comparison:
    return Comparison(
        paired=paired,
        deltas=metric_deltas(paired, resamples=resamples, seed=seed, confidence=confidence),
        outcomes=outcome_counts(paired, [name for name in outcome_metrics(paired.k) if name in paired.metrics]),
        complement=complement_table(paired, f"hit@{paired.k}"),
        identity=lexical_identity(paired),
        resamples=resamples,
        seed=seed,
        confidence=confidence,
    )


def _interval(row: MetricDelta) -> str:
    return interval_text(row.delta, row.low, row.high, row.excludes_zero)


def _label(metric: str) -> str:
    return metric.replace("mrr@", "MRR@").replace("ndcg@", "nDCG@")


def _delta_table(rows: Sequence[MetricDelta], label_a: str, label_b: str) -> list[str]:
    return markdown_table(
        ["method", "metric", "n", "clusters", label_a, label_b, f"Δ {label_b}-{label_a} [95% CI]"],
        [
            [
                row.method,
                _label(row.metric),
                str(row.n),
                str(row.n_clusters),
                format_rate(row.mean_a),
                format_rate(row.mean_b),
                _interval(row),
            ]
            for row in rows
        ],
    )


def _subset_table(comparison: Comparison, metrics: Sequence[str]) -> list[str]:
    paired = comparison.paired
    index = {(row.method, row.subset, row.metric): row for row in comparison.deltas if row.level == PAPER}
    body: list[list[str]] = []
    for method in paired.methods:
        for subset in paired.subsets():
            first = next((index.get((method, subset, metric)) for metric in metrics), None)
            if first is None:
                continue
            cells = [method, subset, str(first.n)]
            for metric in metrics:
                row = index.get((method, subset, metric))
                cells.append("n/a" if row is None else _interval(row))
            body.append(cells)
    return markdown_table(["method", "subset", "n", *(f"Δ {_label(metric)}" for metric in metrics)], body)


def render_markdown(comparison: Comparison, *, title: str, meta: Mapping[str, str]) -> str:
    paired = comparison.paired
    k = paired.k
    lines = [f"# {title}", ""]
    lines += markdown_table(["항목", "값"], [[key, value] for key, value in meta.items()])
    lines += ["", "## lexical 동일성 (임베딩만 바뀌었는지)", ""]
    lines += [f"- {line}" for line in identity_lines(comparison.identity, paired.label_a, paired.label_b)]
    if paired.warnings:
        lines += ["", "## 짝짓기 경고", ""]
        lines += [f"- {line}" for line in paired.warnings]

    paper_all = [row for row in comparison.deltas if row.level == PAPER and row.subset == "all"]
    lines += ["", "## 논문 단위 (전체)", ""]
    lines += _delta_table(paper_all, paired.label_a, paired.label_b)

    chunk_rows = [row for row in comparison.deltas if row.level == CHUNK]
    lines += ["", "## 청크 단위 (relevant_chunk_ids가 있는 질의만)", ""]
    if chunk_rows:
        lines += _delta_table([row for row in chunk_rows if row.subset == "all"], paired.label_a, paired.label_b)
        by_lang = [row for row in chunk_rows if row.subset != "all"]
        if by_lang:
            lines += ["", "언어별", ""]
            lines += markdown_table(
                ["method", "subset", "metric", "n", f"Δ {paired.label_b}-{paired.label_a} [95% CI]"],
                [[row.method, row.subset, _label(row.metric), str(row.n), _interval(row)] for row in by_lang],
            )
    else:
        lines.append("청크 정답이 있는 질의가 없습니다.")

    subset_metrics = [name for name in (f"hit@{min(5, k)}", f"mrr@{k}", f"ndcg@{k}") if name in paired.metrics]
    lines += ["", "## 부분집합 (논문 단위)", ""]
    lines += _subset_table(comparison, subset_metrics)

    lines += ["", f"## 질의별 개선·동률·악화 ({paired.label_b} 기준, {paired.label_a} 대비)", ""]
    lines += markdown_table(
        ["method", "subset", "metric", "개선", "동률", "악화"],
        [
            [row.method, row.subset, _label(row.metric), *(str(row.counts[outcome]) for outcome in OUTCOMES)]
            for row in comparison.outcomes
        ],
    )

    lines += ["", f"## lexical·vector 상호 보완성 (hit@{k}, 모델별)", ""]
    if comparison.complement:
        lines += markdown_table(
            ["실행", *(COMPLEMENT_LABELS[cell] for cell in COMPLEMENT_CELLS)],
            [
                [label, *(str(counts[cell]) for cell in COMPLEMENT_CELLS)]
                for label, counts in comparison.complement.items()
            ],
        )
    else:
        lines.append("lexical과 vector가 두 실행에 모두 있어야 계산한다.")

    lines += [
        "",
        "## 정의",
        "",
        f"- Δ는 질의별 ({paired.label_b} - {paired.label_a})의 평균. 구간은 한영 번역쌍(`fusion_sweep.cluster_key`)을 "
        f"군집으로 재표집한 paired bootstrap {comparison.resamples}회, seed {comparison.seed}, "
        f"{comparison.confidence * 100:.0f}% 백분위 구간. `*`는 구간이 0을 포함하지 않는 행.",
        f"- 개선·동률·악화는 질의별 지표 값 비교(hit@{min(5, k)}은 맞힘 여부, MRR@{k}는 첫 정답 순위).",
        "- 오류 행이 있는 질의와 한쪽 실행에만 있는 질의는 그 방식의 비교에서 뺀다.",
    ]
    return "\n".join(lines) + "\n"


METRICS_CSV_FIELDS = [
    "method",
    "level",
    "subset",
    "metric",
    "n",
    "n_clusters",
    "mean_a",
    "mean_b",
    "delta",
    "low",
    "high",
    "excludes_zero",
]


def write_metrics_csv(rows: Sequence[MetricDelta], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.method,
                    "level": row.level,
                    "subset": row.subset,
                    "metric": row.metric,
                    "n": row.n,
                    "n_clusters": row.n_clusters,
                    "mean_a": csv_number(row.mean_a),
                    "mean_b": csv_number(row.mean_b),
                    "delta": csv_number(row.delta),
                    "low": csv_number(row.low),
                    "high": csv_number(row.high),
                    "excludes_zero": int(row.excludes_zero),
                }
            )


def _rank_cell(rank: int | None) -> str:
    return "" if rank is None else str(rank)


def _rank_change(a_rank: int | None, b_rank: int | None, k: int) -> str:
    left = a_rank or k + 1
    right = b_rank or k + 1
    if right < left:
        return IMPROVED
    if right > left:
        return WORSE
    return TIED


def write_queries_csv(paired: PairedRuns, path: Path) -> None:
    metrics = list(paired.metrics)
    fields = [
        "method",
        "query_id",
        "lang",
        "source",
        "category",
        "a_rank",
        "b_rank",
        "rank_change",
        "a_chunk_rank",
        "b_chunk_rank",
        *(f"{side}_{metric}" for metric in metrics for side in ("a", "b", "delta")),
        "a_retrieved_arxiv_ids",
        "b_retrieved_arxiv_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method in paired.methods:
            for pair in paired.pairs[method]:
                a_rank, b_rank = pair.paper_ranks()
                a_chunk, b_chunk = pair.chunk_ranks()
                record: dict[str, object] = {
                    "method": method,
                    "query_id": pair.query.id,
                    "lang": pair.lang,
                    "source": pair.source,
                    "category": pair.category,
                    "a_rank": _rank_cell(a_rank),
                    "b_rank": _rank_cell(b_rank),
                    "rank_change": _rank_change(a_rank, b_rank, paired.k),
                    "a_chunk_rank": _rank_cell(a_chunk),
                    "b_chunk_rank": _rank_cell(b_chunk),
                    "a_retrieved_arxiv_ids": " ".join(pair.a.retrieved_arxiv_ids),
                    "b_retrieved_arxiv_ids": " ".join(pair.b.retrieved_arxiv_ids),
                }
                for metric in metrics:
                    left, right = pair.values(PAPER, metric)
                    record[f"a_{metric}"] = csv_number(left)
                    record[f"b_{metric}"] = csv_number(right)
                    record[f"delta_{metric}"] = "" if left is None or right is None else csv_number(right - left)
                writer.writerow(record)
