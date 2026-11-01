import csv
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from arxplore_eval.answers import PAPER_CHAT_MODE, AnswersError, load_answer_records
from arxplore_eval.behavior import (
    DEFAULT_REFUSAL_PHRASES,
    OUTCOME_ANSWERED,
    OUTCOME_REFUSED,
    answer_checks,
    classify_outcome,
)
from arxplore_eval.compare_runs import OUTCOMES, outcome_of
from arxplore_eval.dataset import EvalQuery
from arxplore_eval.fusion_sweep import DEFAULT_CONFIDENCE
from arxplore_eval.paired_stats import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    PairedDelta,
    draws_cache,
    interval_text,
    paired_delta,
)
from arxplore_eval.subsets import build_subsets, in_subset
from arxplore_eval.tables import MISSING, csv_number, format_rate, markdown_table

RAGAS_METRICS = ("faithfulness", "answer_relevancy")
REFUSED = "refused"
ANSWER_CHECKS = ("lang_match", "citation_numbers_valid", "cited_relevant")
GENERATION_METRICS = (*RAGAS_METRICS, "lang_match", REFUSED, "citation_numbers_valid", "cited_relevant")
CSV_METRICS = (*RAGAS_METRICS, *ANSWER_CHECKS)
STRATUM_PREFIX = "stratum:"
SAMPLES_KIND = "samples"
ANSWERS_KIND = "answers"
OUTCOME_SUBSETS = ("all", "ko", "en")
REFUSE = "refuse"
ANSWER = "answer"
MAJORITY = 0.5
MISMATCH_PREVIEW = 10
REFUSAL_GROUP_LABELS = {
    REFUSE: "거절 기대(거절 유지율)",
    ANSWER: "답변 기대(과잉 거절률)",
}
TRANSITIONS = ("both_refused", "a_only", "b_only", "neither")
MD_FILE = "compare_generation.md"
METRICS_CSV_FILE = "compare_generation_metrics.csv"
QUERIES_CSV_FILE = "compare_generation_queries.csv"
VARIANCE_CSV_FILE = "compare_generation_variance.csv"
REFUSAL_CSV_FILE = "compare_generation_refusal.csv"


class CompareGenerationError(ValueError):
    pass


type MetricValues = dict[str, float | None]


@dataclass(frozen=True)
class RunValues:
    path: Path
    kind: str
    values: dict[str, MetricValues]
    outcomes: dict[str, str]
    prompt_variants: tuple[str, ...] = ()

    def mean(self, metric: str, query_ids: Sequence[str] | None = None) -> float | None:
        ids = (
            list(self.values) if query_ids is None else [query_id for query_id in query_ids if query_id in self.values]
        )
        present = [value for query_id in ids if (value := self.values[query_id].get(metric)) is not None]
        return sum(present) / len(present) if present else None


def refused_value(outcome: str) -> float | None:
    if outcome == OUTCOME_REFUSED:
        return 1.0
    if outcome == OUTCOME_ANSWERED:
        return 0.0
    return None


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def read_sample_csv(path: Path, *, mode: str = PAPER_CHAT_MODE) -> RunValues:
    values: dict[str, MetricValues] = {}
    outcomes: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        if not {"id", "mode", "outcome"} <= header:
            raise CompareGenerationError(f"{path}: generation 샘플 CSV가 아닙니다(id, mode, outcome 열 필요)")
        for row in reader:
            if row["mode"] != mode:
                continue
            query_id = row["id"]
            if query_id in values:
                raise CompareGenerationError(f"{path}: 같은 질의가 두 번 있습니다: {query_id}")
            outcome = row["outcome"]
            metrics: MetricValues = {
                metric: _optional_float(row.get(metric)) if metric in header else None for metric in CSV_METRICS
            }
            metrics[REFUSED] = refused_value(outcome)
            values[query_id] = metrics
            outcomes[query_id] = outcome
    return RunValues(path=path, kind=SAMPLES_KIND, values=values, outcomes=outcomes)


def read_answers_jsonl(
    path: Path, *, mode: str = PAPER_CHAT_MODE, refusal_phrases: Sequence[str] = DEFAULT_REFUSAL_PHRASES
) -> RunValues:
    values: dict[str, MetricValues] = {}
    outcomes: dict[str, str] = {}
    variants: list[str] = []
    for record in load_answer_records(path):
        if record["mode"] != mode:
            continue
        if record.get("sources_only"):
            raise CompareGenerationError(f"{path}: 검색 전용(--sources-only) 레코드는 비교할 답변이 없습니다")
        outcome = classify_outcome(record, refusal_phrases)
        _, checks = answer_checks(record, outcome)
        metrics: MetricValues = dict.fromkeys(RAGAS_METRICS)
        metrics.update({metric: checks.get(metric) for metric in ANSWER_CHECKS})
        metrics[REFUSED] = refused_value(outcome)
        query_id = str(record["id"])
        values[query_id] = metrics
        outcomes[query_id] = outcome
        variant = record.get("prompt_variant")
        if isinstance(variant, str) and variant not in variants:
            variants.append(variant)
    return RunValues(path=path, kind=ANSWERS_KIND, values=values, outcomes=outcomes, prompt_variants=tuple(variants))


def read_run(
    path: Path, *, mode: str = PAPER_CHAT_MODE, refusal_phrases: Sequence[str] = DEFAULT_REFUSAL_PHRASES
) -> RunValues:
    if path.suffix == ".csv":
        run = read_sample_csv(path, mode=mode)
    elif path.suffix == ".jsonl":
        try:
            run = read_answers_jsonl(path, mode=mode, refusal_phrases=refusal_phrases)
        except AnswersError as exc:
            raise CompareGenerationError(str(exc)) from exc
    else:
        raise CompareGenerationError(f"{path}: 샘플 CSV(.csv) 또는 답변 JSONL(.jsonl)만 읽습니다")
    if not run.values:
        raise CompareGenerationError(f"{path}: mode {mode} 레코드가 없습니다")
    return run


@dataclass(frozen=True)
class QueryCondition:
    values: dict[str, tuple[float, ...]]
    outcomes: tuple[str, ...]

    def mean(self, metric: str) -> float | None:
        present = self.values.get(metric, ())
        return sum(present) / len(present) if present else None

    def stdev(self, metric: str) -> float | None:
        present = self.values.get(metric, ())
        return statistics.stdev(present) if len(present) > 1 else None

    def count(self, metric: str) -> int:
        return len(self.values.get(metric, ()))


def query_condition(runs: Sequence[RunValues], query_id: str) -> QueryCondition:
    collected: dict[str, list[float]] = {metric: [] for metric in GENERATION_METRICS}
    outcomes: list[str] = []
    for run in runs:
        if query_id not in run.values:
            continue
        outcomes.append(run.outcomes[query_id])
        for metric in GENERATION_METRICS:
            value = run.values[query_id].get(metric)
            if value is not None:
                collected[metric].append(value)
    return QueryCondition(
        values={metric: tuple(values) for metric, values in collected.items()}, outcomes=tuple(outcomes)
    )


@dataclass(frozen=True)
class GenerationPair:
    query: EvalQuery
    a: QueryCondition
    b: QueryCondition

    def values(self, metric: str) -> tuple[float | None, float | None]:
        return self.a.mean(metric), self.b.mean(metric)

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
class PairedGeneration:
    label_a: str
    label_b: str
    runs_a: tuple[RunValues, ...]
    runs_b: tuple[RunValues, ...]
    metrics: tuple[str, ...]
    pairs: tuple[GenerationPair, ...]
    warnings: tuple[str, ...] = ()
    strata: Mapping[str, str] = field(default_factory=dict)

    def subsets(self) -> list[str]:
        names = sorted({self.strata[pair.query.id] for pair in self.pairs if pair.query.id in self.strata})
        return [*build_subsets(list(self.pairs)), *(f"{STRATUM_PREFIX}{name}" for name in names)]

    def select(self, subset: str) -> list[GenerationPair]:
        if subset.startswith(STRATUM_PREFIX):
            name = subset[len(STRATUM_PREFIX) :]
            return [pair for pair in self.pairs if self.strata.get(pair.query.id) == name]
        return subset_pairs(self.pairs, subset)


def _partial_reps(runs: Sequence[RunValues], ids: set[str]) -> list[str]:
    return sorted(query_id for query_id in ids if any(query_id not in run.values for run in runs))


def _present_metrics(pairs: Sequence[GenerationPair]) -> tuple[str, ...]:
    return tuple(
        metric for metric in GENERATION_METRICS if any(pair.a.count(metric) and pair.b.count(metric) for pair in pairs)
    )


def pair_generation(
    runs_a: Sequence[RunValues],
    runs_b: Sequence[RunValues],
    queries: Sequence[EvalQuery],
    *,
    label_a: str,
    label_b: str,
    strata: Mapping[str, str] | None = None,
) -> PairedGeneration:
    if not runs_a or not runs_b:
        raise CompareGenerationError("두 조건에 실행 파일이 하나 이상씩 있어야 합니다")
    by_id = {query.id: query for query in queries}
    ids_a = {query_id for run in runs_a for query_id in run.values}
    ids_b = {query_id for run in runs_b for query_id in run.values}
    unknown = sorted((ids_a | ids_b) - set(by_id))
    if unknown:
        raise CompareGenerationError(f"질의셋에 없는 id: {unknown[:MISMATCH_PREVIEW]}")
    warnings: list[str] = []
    only = sorted(ids_a ^ ids_b)
    if only:
        warnings.append(f"한쪽 조건에만 있는 질의 {len(only)}개(제외): {', '.join(only[:MISMATCH_PREVIEW])}")
    for label, runs, ids in ((label_a, runs_a, ids_a), (label_b, runs_b, ids_b)):
        partial = _partial_reps(runs, ids & ids_a & ids_b)
        if partial:
            warnings.append(
                f"{label}: 일부 반복에만 있는 질의 {len(partial)}개(있는 반복만 평균): "
                f"{', '.join(partial[:MISMATCH_PREVIEW])}"
            )
    pairs = tuple(
        GenerationPair(query, query_condition(runs_a, query.id), query_condition(runs_b, query.id))
        for query in queries
        if query.id in ids_a and query.id in ids_b
    )
    if not pairs:
        raise CompareGenerationError("두 조건에 공통 질의가 없습니다")
    for label, side in ((label_a, "a"), (label_b, "b")):
        failed = [
            pair.query.id
            for pair in pairs
            if any(refused_value(outcome) is None for outcome in getattr(pair, side).outcomes)
        ]
        if failed:
            warnings.append(
                f"{label}: rejected·error 결과가 있는 질의 {len(failed)}개(그 반복은 지표에서 빠짐): "
                f"{', '.join(failed[:MISMATCH_PREVIEW])}"
            )
    return PairedGeneration(
        label_a=label_a,
        label_b=label_b,
        runs_a=tuple(runs_a),
        runs_b=tuple(runs_b),
        metrics=_present_metrics(pairs),
        pairs=pairs,
        warnings=tuple(warnings),
        strata=dict(strata or {}),
    )


@dataclass(frozen=True)
class GenerationDelta:
    subset: str
    metric: str
    result: PairedDelta


def subset_pairs(pairs: Sequence[GenerationPair], subset: str) -> list[GenerationPair]:
    return [pair for pair in pairs if in_subset(pair, subset)]


def metric_values(pairs: Sequence[GenerationPair], metric: str) -> list[tuple[str, float, float]]:
    kept: list[tuple[str, float, float]] = []
    for pair in pairs:
        left, right = pair.values(metric)
        if left is not None and right is not None:
            kept.append((pair.query.id, left, right))
    return kept


def generation_deltas(
    paired: PairedGeneration,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> list[GenerationDelta]:
    draws = draws_cache(resamples, seed)
    rows: list[GenerationDelta] = []
    for subset in paired.subsets():
        chosen = paired.select(subset)
        for metric in paired.metrics:
            result = paired_delta(metric_values(chosen, metric), draws=draws, confidence=confidence)
            if result is not None:
                rows.append(GenerationDelta(subset=subset, metric=metric, result=result))
    return rows


@dataclass(frozen=True)
class RefusalRow:
    behavior: str
    n: int
    rate_a: float | None
    rate_b: float | None
    transitions: dict[str, int]
    query_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)


def transition_of(a_rate: float, b_rate: float) -> str:
    a_refused = a_rate >= MAJORITY
    b_refused = b_rate >= MAJORITY
    if a_refused and b_refused:
        return "both_refused"
    if a_refused:
        return "a_only"
    if b_refused:
        return "b_only"
    return "neither"


def refusal_table(paired: PairedGeneration) -> list[RefusalRow]:
    rows: list[RefusalRow] = []
    behaviors = sorted({pair.expected_behavior for pair in paired.pairs}, key=lambda name: (name != REFUSE, name))
    for behavior in behaviors:
        kept = metric_values([pair for pair in paired.pairs if pair.expected_behavior == behavior], REFUSED)
        if not kept:
            continue
        grouped: dict[str, list[str]] = {name: [] for name in TRANSITIONS}
        for query_id, left, right in kept:
            grouped[transition_of(left, right)].append(query_id)
        rows.append(
            RefusalRow(
                behavior=behavior,
                n=len(kept),
                rate_a=sum(left for _, left, _ in kept) / len(kept),
                rate_b=sum(right for _, _, right in kept) / len(kept),
                transitions={name: len(ids) for name, ids in grouped.items()},
                query_ids={name: tuple(ids) for name, ids in grouped.items()},
            )
        )
    return rows


@dataclass(frozen=True)
class VarianceRow:
    label: str
    metric: str
    reps: int
    n_queries: int
    mean_stdev: float | None
    max_stdev: float | None
    rep_means: tuple[float | None, ...]


def variance_rows(paired: PairedGeneration) -> list[VarianceRow]:
    ids = [pair.query.id for pair in paired.pairs]
    rows: list[VarianceRow] = []
    for label, runs, side in ((paired.label_a, paired.runs_a, "a"), (paired.label_b, paired.runs_b, "b")):
        for metric in paired.metrics:
            stdevs = [value for pair in paired.pairs if (value := getattr(pair, side).stdev(metric)) is not None]
            rows.append(
                VarianceRow(
                    label=label,
                    metric=metric,
                    reps=len(runs),
                    n_queries=len(stdevs),
                    mean_stdev=sum(stdevs) / len(stdevs) if stdevs else None,
                    max_stdev=max(stdevs) if stdevs else None,
                    rep_means=tuple(run.mean(metric, ids) for run in runs),
                )
            )
    return rows


@dataclass(frozen=True)
class ChangeCount:
    subset: str
    metric: str
    counts: dict[str, int]


def change_counts(paired: PairedGeneration) -> list[ChangeCount]:
    rows: list[ChangeCount] = []
    strata = [subset for subset in paired.subsets() if subset.startswith(STRATUM_PREFIX)]
    for subset in (*OUTCOME_SUBSETS, *strata):
        chosen = paired.select(subset)
        for metric in paired.metrics:
            kept = metric_values(chosen, metric)
            if not kept:
                continue
            counts = dict.fromkeys(OUTCOMES, 0)
            for _, left, right in kept:
                counts[outcome_of(left, right)] += 1
            rows.append(ChangeCount(subset=subset, metric=metric, counts=counts))
    return rows


@dataclass(frozen=True)
class GenerationComparison:
    paired: PairedGeneration
    deltas: list[GenerationDelta]
    refusal: list[RefusalRow]
    variance: list[VarianceRow]
    changes: list[ChangeCount]
    resamples: int
    seed: int
    confidence: float


def compare_generation(
    paired: PairedGeneration,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> GenerationComparison:
    return GenerationComparison(
        paired=paired,
        deltas=generation_deltas(paired, resamples=resamples, seed=seed, confidence=confidence),
        refusal=refusal_table(paired),
        variance=variance_rows(paired),
        changes=change_counts(paired),
        resamples=resamples,
        seed=seed,
        confidence=confidence,
    )


def _interval(row: GenerationDelta) -> str:
    result = row.result
    return interval_text(result.delta, result.low, result.high, result.excludes_zero)


def _overall_table(comparison: GenerationComparison) -> list[str]:
    paired = comparison.paired
    rows = [row for row in comparison.deltas if row.subset == "all"]
    return markdown_table(
        ["metric", "n", "clusters", paired.label_a, paired.label_b, f"Δ {paired.label_b}-{paired.label_a} [95% CI]"],
        [
            [
                row.metric,
                str(row.result.n),
                str(row.result.n_clusters),
                format_rate(row.result.mean_a),
                format_rate(row.result.mean_b),
                _interval(row),
            ]
            for row in rows
        ],
    )


def _subset_table(comparison: GenerationComparison) -> list[str]:
    paired = comparison.paired
    index = {(row.subset, row.metric): row for row in comparison.deltas}
    body: list[list[str]] = []
    for subset in paired.subsets():
        if not any((subset, metric) in index for metric in paired.metrics):
            continue
        cells = [subset]
        for metric in paired.metrics:
            row = index.get((subset, metric))
            cells.append(
                MISSING
                if row is None
                else f"{format_rate(row.result.mean_a)} → {format_rate(row.result.mean_b)} "
                f"(n={row.result.n}) {_interval(row)}"
            )
        body.append(cells)
    return markdown_table(["subset", *paired.metrics], body)


def _rep_means(values: Sequence[float | None]) -> str:
    return ", ".join(format_rate(value) for value in values)


def render_markdown(comparison: GenerationComparison, *, title: str, meta: Mapping[str, str]) -> str:
    paired = comparison.paired
    label_a, label_b = paired.label_a, paired.label_b
    lines = [f"# {title}", ""]
    lines += markdown_table(["항목", "값"], [[key, value] for key, value in meta.items()])
    if paired.warnings:
        lines += ["", "## 짝짓기 경고", ""]
        lines += [f"- {line}" for line in paired.warnings]

    lines += ["", "## 전체", ""]
    lines += _overall_table(comparison)

    lines += ["", f"## 부분집합 ({label_a} → {label_b}, Δ [95% CI])", ""]
    lines += _subset_table(comparison)

    lines += ["", "## 거절: 기대 행동별", ""]
    if comparison.refusal:
        lines += markdown_table(
            [
                "기대 행동",
                "n",
                f"{label_a} 거절률",
                f"{label_b} 거절률",
                "둘 다 거절",
                f"{label_a}만 거절",
                f"{label_b}만 거절",
                "둘 다 답변",
            ],
            [
                [
                    REFUSAL_GROUP_LABELS.get(row.behavior, row.behavior),
                    str(row.n),
                    format_rate(row.rate_a),
                    format_rate(row.rate_b),
                    *(str(row.transitions[name]) for name in TRANSITIONS),
                ]
                for row in comparison.refusal
            ],
        )
    else:
        lines.append("refused를 계산한 질의가 없습니다.")

    lines += ["", "## 반복 간 분산", ""]
    lines += markdown_table(
        ["조건", "metric", "반복 수", "질의 수(값 2개 이상)", "질의별 표준편차 평균", "최대", "반복별 평균"],
        [
            [
                row.label,
                row.metric,
                str(row.reps),
                str(row.n_queries),
                format_rate(row.mean_stdev),
                format_rate(row.max_stdev),
                _rep_means(row.rep_means),
            ]
            for row in comparison.variance
        ],
    )

    lines += ["", f"## 질의별 개선·동률·악화 ({label_b} 기준, {label_a} 대비)", ""]
    lines += markdown_table(
        ["subset", "metric", "개선", "동률", "악화"],
        [[row.subset, row.metric, *(str(row.counts[outcome]) for outcome in OUTCOMES)] for row in comparison.changes],
    )

    lines += [
        "",
        "## 정의",
        "",
        "- 질의별 값은 조건 안 반복들의 평균이다(값이 없는 반복은 빼고 평균). `refused`는 refused=1, answered=0이고 "
        "rejected·error는 값이 없다.",
        f"- Δ는 질의별 ({label_b} - {label_a})의 평균. 구간은 한영 번역쌍(`fusion_sweep.cluster_key`)을 군집으로 재표집한 "
        f"paired bootstrap {comparison.resamples}회, seed {comparison.seed}, {comparison.confidence * 100:.0f}% 백분위 구간. "
        "`*`는 구간이 0을 포함하지 않는 행.",
        "- 거절 표의 거절률은 질의별 반복 평균 refused의 평균이다. 둘 다 거절·한쪽만 거절·둘 다 답변은 질의별 반복 평균이 "
        f"{MAJORITY} 이상이면 거절로 센 건수다.",
        "- 반복 간 분산은 조건마다 값이 2개 이상인 질의의 표본 표준편차를 평균한 값이다. 반복별 평균은 짝지어진 질의만 센다.",
        "- 개선·동률·악화는 질의별 반복 평균 비교다.",
    ]
    return "\n".join(lines) + "\n"


METRICS_CSV_FIELDS = [
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


def write_metrics_csv(rows: Sequence[GenerationDelta], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRICS_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            result = row.result
            writer.writerow(
                {
                    "subset": row.subset,
                    "metric": row.metric,
                    "n": result.n,
                    "n_clusters": result.n_clusters,
                    "mean_a": csv_number(result.mean_a),
                    "mean_b": csv_number(result.mean_b),
                    "delta": csv_number(result.delta),
                    "low": csv_number(result.low),
                    "high": csv_number(result.high),
                    "excludes_zero": int(result.excludes_zero),
                }
            )


def write_queries_csv(paired: PairedGeneration, path: Path) -> None:
    metrics = list(paired.metrics)
    fields = [
        "query_id",
        "lang",
        "source",
        "category",
        "expected_behavior",
        "a_outcomes",
        "b_outcomes",
        *(f"{prefix}_{metric}" for metric in metrics for prefix in ("a", "b", "delta", "a_n", "b_n")),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pair in paired.pairs:
            record: dict[str, object] = {
                "query_id": pair.query.id,
                "lang": pair.lang,
                "source": pair.source,
                "category": pair.category,
                "expected_behavior": pair.expected_behavior,
                "a_outcomes": " ".join(pair.a.outcomes),
                "b_outcomes": " ".join(pair.b.outcomes),
            }
            for metric in metrics:
                left, right = pair.values(metric)
                record[f"a_{metric}"] = csv_number(left)
                record[f"b_{metric}"] = csv_number(right)
                record[f"delta_{metric}"] = "" if left is None or right is None else csv_number(right - left)
                record[f"a_n_{metric}"] = pair.a.count(metric)
                record[f"b_n_{metric}"] = pair.b.count(metric)
            writer.writerow(record)


def write_variance_csv(rows: Sequence[VarianceRow], path: Path) -> None:
    reps = max((len(row.rep_means) for row in rows), default=0)
    fields = [
        "label",
        "metric",
        "reps",
        "n_queries",
        "mean_stdev",
        "max_stdev",
        *(f"rep{index}_mean" for index in range(1, reps + 1)),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record: dict[str, object] = {
                "label": row.label,
                "metric": row.metric,
                "reps": row.reps,
                "n_queries": row.n_queries,
                "mean_stdev": csv_number(row.mean_stdev),
                "max_stdev": csv_number(row.max_stdev),
            }
            for index, value in enumerate(row.rep_means, start=1):
                record[f"rep{index}_mean"] = csv_number(value)
            writer.writerow(record)


def write_refusal_csv(rows: Sequence[RefusalRow], path: Path) -> None:
    fields = ["expected_behavior", "n", "rate_a", "rate_b", *TRANSITIONS, *(f"{name}_ids" for name in TRANSITIONS)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record: dict[str, object] = {
                "expected_behavior": row.behavior,
                "n": row.n,
                "rate_a": csv_number(row.rate_a),
                "rate_b": csv_number(row.rate_b),
            }
            for name in TRANSITIONS:
                record[name] = row.transitions[name]
                record[f"{name}_ids"] = " ".join(row.query_ids.get(name, ()))
            writer.writerow(record)
