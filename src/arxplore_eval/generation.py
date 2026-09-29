import asyncio
import csv
import importlib
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

from openai import AsyncOpenAI

from arxplore_eval.behavior import (
    BEHAVIOR_METRIC_DESCRIPTIONS,
    BEHAVIOR_METRICS,
    DEFAULT_REFUSAL_PHRASES,
    OUTCOME_REJECTED,
    OUTCOMES,
    behavior_scores,
)
from arxplore_eval.metrics import mean
from arxplore_eval.subsets import build_subsets, in_subset
from arxplore_eval.tables import MISSING, csv_number, format_rate, markdown_table

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
METRIC_DESCRIPTIONS = {
    "faithfulness": "답변의 주장 중 LLM에 넘긴 발췌문으로 뒷받침되는 비율",
    "answer_relevancy": "답변에서 거꾸로 만든 질문과 원래 질문의 임베딩 유사도(질문에 맞는 답인가)",
    "context_precision": "기준(reference_answer, 없으면 정답 청크 본문)에 쓸모 있는 발췌문이 앞 순위에 있는 정도",
    "context_recall": "reference_answer의 문장 중 발췌문으로 뒷받침되는 비율",
}
ALL_METRICS = "all"
DEFAULT_JUDGE_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CONCURRENCY = 4
REASONING_JUDGE_MAX_TOKENS = 8192
MODE_ORDER = ("agent", "paper_chat")
KOREAN = "ko"
KOREAN_LANGUAGE = "korean"
ERROR_TEXT_LIMIT = 300

SKIP_GENERATION_ERROR = "generation_error"
SKIP_EMPTY_ANSWER = "empty_answer"
SKIP_NO_CONTEXTS = "no_contexts"
SKIP_NO_REFERENCE = "no_reference"
SKIP_NO_REFERENCE_ANSWER = "no_reference_answer"
SKIP_UNDEFINED = "undefined"
SKIP_REJECTED = "rejected"
SKIP_NON_ANSWER_CASE = "expected_non_answer"

REFERENCE_ANSWER = "answer"
REFERENCE_GOLD_CHUNKS = "gold_chunks"

README_TABLE_HEADER = "| 모드 | faithfulness | answer_relevancy | context_precision | context_recall |"

RAGAS_TRACKING_ENV = "RAGAS_DO_NOT_TRACK"
RAGAS_METRIC_CLASSES = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "AnswerRelevancy",
    "context_precision": "ContextPrecisionWithReference",
    "context_recall": "ContextRecall",
}
RAGAS_EMBEDDING_METRICS = frozenset({"answer_relevancy"})

type ScoreResult = float | BaseException | None


def resolve_metrics(values: Iterable[str]) -> list[str]:
    metrics: list[str] = []
    for value in values:
        name = value.strip()
        if not name:
            continue
        expanded = list(METRICS) if name == ALL_METRICS else [name]
        for metric in expanded:
            if metric not in METRICS:
                raise ValueError(f"알 수 없는 지표: {metric!r}. 사용 가능: {', '.join(METRICS)}, {ALL_METRICS}")
            if metric not in metrics:
                metrics.append(metric)
    if not metrics:
        raise ValueError("지표를 하나 이상 지정하세요.")
    return metrics


@dataclass(frozen=True)
class MetricRequest:
    metric: str
    user_input: str
    response: str
    retrieved_contexts: tuple[str, ...]
    reference: str | None
    lang: str
    record_index: int = 0


class Scorer(Protocol):
    async def score(self, requests: Sequence[MetricRequest]) -> list[ScoreResult]: ...


@dataclass
class SampleScore:
    id: str
    mode: str
    lang: str
    source: str
    n_contexts: int
    reference_kind: str
    generation_error: str | None
    scores: dict[str, float] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    category: str = ""
    expected_behavior: str = "answer"
    outcome: str = ""
    behavior: dict[str, float] = field(default_factory=dict)


def resolve_reference(record: Mapping[str, Any], chunk_texts: Mapping[int, str]) -> tuple[str | None, str]:
    reference_answer = str(record.get("reference_answer") or "").strip()
    if reference_answer:
        return reference_answer, REFERENCE_ANSWER
    texts = [str(chunk_texts.get(int(chunk_id)) or "").strip() for chunk_id in record.get("relevant_chunk_ids") or []]
    texts = [text for text in texts if text]
    if texts:
        return "\n\n".join(texts), REFERENCE_GOLD_CHUNKS
    return None, ""


def _skip_reason(
    record: Mapping[str, Any],
    metric: str,
    *,
    answer: str,
    contexts: tuple[str, ...],
    reference: str | None,
    reference_answer: str,
) -> str | None:
    if record.get("error"):
        return SKIP_GENERATION_ERROR
    if record.get("outcome") == OUTCOME_REJECTED or record.get("rejection"):
        return SKIP_REJECTED
    if str(record.get("expected_behavior") or "answer") != "answer":
        return SKIP_NON_ANSWER_CASE
    if metric in ("faithfulness", "answer_relevancy") and not answer:
        return SKIP_EMPTY_ANSWER
    if metric != "answer_relevancy" and not contexts:
        return SKIP_NO_CONTEXTS
    if metric == "context_precision" and reference is None:
        return SKIP_NO_REFERENCE
    if metric == "context_recall" and not reference_answer:
        return SKIP_NO_REFERENCE_ANSWER
    return None


def plan_record(
    record: Mapping[str, Any],
    metrics: Sequence[str],
    *,
    chunk_texts: Mapping[int, str] | None = None,
    record_index: int = 0,
) -> tuple[list[MetricRequest], dict[str, str], str]:
    contexts = tuple(str(text) for text in record.get("contexts") or [] if str(text).strip())
    answer = str(record.get("answer") or "").strip()
    reference, reference_kind = resolve_reference(record, chunk_texts or {})
    reference_answer = str(record.get("reference_answer") or "").strip()
    references = {"context_precision": reference, "context_recall": reference_answer or None}

    requests: list[MetricRequest] = []
    skipped: dict[str, str] = {}
    for metric in metrics:
        reason = _skip_reason(
            record, metric, answer=answer, contexts=contexts, reference=reference, reference_answer=reference_answer
        )
        if reason is not None:
            skipped[metric] = reason
            continue
        requests.append(
            MetricRequest(
                metric=metric,
                user_input=str(record.get("query") or ""),
                response=answer,
                retrieved_contexts=contexts,
                reference=references.get(metric),
                lang=str(record.get("lang") or ""),
                record_index=record_index,
            )
        )
    return requests, skipped, reference_kind if "context_precision" in metrics else ""


def describe_error(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}".strip()
    return text[:ERROR_TEXT_LIMIT]


def _sample(
    record: Mapping[str, Any], reference_kind: str, skipped: dict[str, str], phrases: Sequence[str]
) -> SampleScore:
    outcome, behavior = behavior_scores(record, phrases)
    return SampleScore(
        id=str(record.get("id")),
        mode=str(record.get("mode")),
        lang=str(record.get("lang") or ""),
        source=str(record.get("source") or ""),
        n_contexts=len([text for text in record.get("contexts") or [] if str(text).strip()]),
        reference_kind=reference_kind,
        generation_error=record.get("error") or None,
        skipped=skipped,
        category=str(record.get("category") or ""),
        expected_behavior=str(record.get("expected_behavior") or "answer"),
        outcome=outcome,
        behavior=behavior,
    )


async def score_records(
    records: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    scorer: Scorer,
    *,
    chunk_texts: Mapping[int, str] | None = None,
    refusal_phrases: Sequence[str] = DEFAULT_REFUSAL_PHRASES,
) -> list[SampleScore]:
    samples: list[SampleScore] = []
    requests: list[MetricRequest] = []
    for index, record in enumerate(records):
        planned, skipped, reference_kind = plan_record(record, metrics, chunk_texts=chunk_texts, record_index=index)
        requests.extend(planned)
        samples.append(_sample(record, reference_kind, skipped, refusal_phrases))

    results = await scorer.score(requests) if requests else []
    if len(results) != len(requests):
        raise RuntimeError(f"점수기가 요청 {len(requests)}개에 결과 {len(results)}개를 돌려줬습니다.")
    for request, result in zip(requests, results, strict=True):
        sample = samples[request.record_index]
        if isinstance(result, BaseException):
            sample.errors[request.metric] = describe_error(result)
        elif result is None or (isinstance(result, float) and math.isnan(result)):
            sample.skipped[request.metric] = SKIP_UNDEFINED
        else:
            sample.scores[request.metric] = float(result)
    return samples


class NullScorer:
    async def score(self, requests: Sequence[MetricRequest]) -> list[ScoreResult]:
        if requests:
            raise RuntimeError("행동 지표만 계산할 때는 RAGAS 판정 요청을 만들지 않습니다.")
        return []


@dataclass(frozen=True)
class GenerationAggregate:
    mode: str
    subset: str
    n_records: int
    n_generation_errors: int
    means: dict[str, float | None]
    counts: dict[str, int]
    n_metric_errors: int
    behavior_means: dict[str, float | None] = field(default_factory=dict)
    behavior_counts: dict[str, int] = field(default_factory=dict)
    outcomes: dict[str, int] = field(default_factory=dict)


def _ordered_modes(samples: Sequence[SampleScore]) -> list[str]:
    present = list(dict.fromkeys(sample.mode for sample in samples))
    return [mode for mode in MODE_ORDER if mode in present] + [mode for mode in present if mode not in MODE_ORDER]


def _aggregate_group(
    mode: str, subset: str, group: Sequence[SampleScore], metrics: Sequence[str]
) -> GenerationAggregate:
    return GenerationAggregate(
        mode=mode,
        subset=subset,
        n_records=len(group),
        n_generation_errors=sum(1 for sample in group if sample.generation_error),
        means={metric: mean(sample.scores.get(metric) for sample in group) for metric in metrics},
        counts={metric: sum(1 for sample in group if metric in sample.scores) for metric in metrics},
        n_metric_errors=sum(len(sample.errors) for sample in group),
        behavior_means={metric: mean(sample.behavior.get(metric) for sample in group) for metric in BEHAVIOR_METRICS},
        behavior_counts={
            metric: sum(1 for sample in group if metric in sample.behavior) for metric in BEHAVIOR_METRICS
        },
        outcomes={outcome: sum(1 for sample in group if sample.outcome == outcome) for outcome in OUTCOMES},
    )


def aggregate_scores(
    samples: Sequence[SampleScore], metrics: Sequence[str], *, subsets: Sequence[str] | None = None
) -> list[GenerationAggregate]:
    rows: list[GenerationAggregate] = []
    for subset in build_subsets(samples) if subsets is None else subsets:
        for mode in _ordered_modes(samples):
            group = [sample for sample in samples if sample.mode == mode and in_subset(sample, subset)]
            if group:
                rows.append(_aggregate_group(mode, subset, group, metrics))
    return rows


def _cell(row: GenerationAggregate, metric: str) -> str:
    if metric not in row.means:
        return "-"
    value = row.means[metric]
    return MISSING if value is None else f"{value:.3f} ({row.counts[metric]})"


def _behavior_cell(row: GenerationAggregate, metric: str) -> str:
    if not row.behavior_counts.get(metric):
        return "-"
    return format_rate(row.behavior_means.get(metric)) + f" ({row.behavior_counts[metric]})"


def render_behavior_table(aggregates: Sequence[GenerationAggregate]) -> list[str]:
    return markdown_table(
        ["mode", "subset", "n", *OUTCOMES, *BEHAVIOR_METRICS],
        [
            [
                row.mode,
                row.subset,
                str(row.n_records),
                *(str(row.outcomes.get(outcome, 0)) for outcome in OUTCOMES),
                *(_behavior_cell(row, metric) for metric in BEHAVIOR_METRICS),
            ]
            for row in aggregates
        ],
    )


def render_readme_table(aggregates: Sequence[GenerationAggregate]) -> str:
    lines = [README_TABLE_HEADER, "| --- | --- | --- | --- | --- |"]
    for row in aggregates:
        if row.subset != "all":
            continue
        cells = [format_rate(row.means[metric]) if metric in row.means else "-" for metric in METRICS]
        lines.append(f"| {row.mode} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _skip_counts(samples: Sequence[SampleScore]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for sample in samples:
        for metric, reason in sample.skipped.items():
            counts[(metric, reason)] = counts.get((metric, reason), 0) + 1
    return counts


def render_markdown(
    aggregates: Sequence[GenerationAggregate],
    samples: Sequence[SampleScore],
    metrics: Sequence[str],
    *,
    title: str,
    meta: Mapping[str, str],
) -> str:
    lines = [f"# {title}", ""]
    lines.extend(f"- {key}: {value}" for key, value in meta.items())
    lines.append("")
    if not aggregates:
        lines.append("집계할 결과가 없습니다.")
        return "\n".join(lines) + "\n"

    lines += ["## README 붙여넣기용 (전체 언어)", "", render_readme_table(aggregates), ""]
    lines += ["## 모드 x 부분집합", "", "값은 `평균 (계산된 질의 수)`입니다.", ""]
    lines += markdown_table(
        ["mode", "subset", "n", *metrics, "생성 실패", "판정 실패"],
        [
            [
                row.mode,
                row.subset,
                str(row.n_records),
                *(_cell(row, metric) for metric in metrics),
                str(row.n_generation_errors),
                str(row.n_metric_errors),
            ]
            for row in aggregates
        ],
    )
    lines.append("")

    lines += [
        "## 행동 지표 (LLM 판정 없음)",
        "",
        "answered/refused/rejected/error는 결과 종류별 건수, 지표 값은 `평균 (해당 케이스 수)`이고 해당 케이스가 없으면 `-`입니다.",
        "",
    ]
    lines += render_behavior_table(aggregates)
    lines.append("")

    skip_counts = _skip_counts(samples)
    lines += ["## 건너뛴 지표", ""]
    if skip_counts:
        lines += markdown_table(
            ["metric", "사유", "건수"],
            [[metric, reason, str(count)] for (metric, reason), count in sorted(skip_counts.items())],
        )
    else:
        lines.append("건너뛴 지표가 없습니다.")
    lines.append("")

    lines += ["## 지표", ""]
    lines.extend(f"- `{metric}`: {METRIC_DESCRIPTIONS[metric]}" for metric in metrics)
    lines.extend(f"- `{metric}`: {BEHAVIOR_METRIC_DESCRIPTIONS[metric]}" for metric in BEHAVIOR_METRICS)
    lines += [
        "",
        "LLM 판정 점수이므로 같은 판정 모델·같은 답변 파일로 잰 값끼리 상대 비교에만 씁니다.",
    ]
    return "\n".join(lines) + "\n"


SAMPLE_CSV_BASE_FIELDS = [
    "id",
    "mode",
    "lang",
    "source",
    "category",
    "expected_behavior",
    "outcome",
    "n_contexts",
    "reference_kind",
    "generation_error",
]


def write_sample_csv(samples: Sequence[SampleScore], metrics: Sequence[str], path: str | Path) -> None:
    fields = [*SAMPLE_CSV_BASE_FIELDS]
    for metric in metrics:
        fields += [metric, f"{metric}_skipped", f"{metric}_error"]
    fields += list(BEHAVIOR_METRICS)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            record: dict[str, object] = {
                "id": sample.id,
                "mode": sample.mode,
                "lang": sample.lang,
                "source": sample.source,
                "category": sample.category,
                "expected_behavior": sample.expected_behavior,
                "outcome": sample.outcome,
                "n_contexts": sample.n_contexts,
                "reference_kind": sample.reference_kind,
                "generation_error": sample.generation_error or "",
            }
            for metric in metrics:
                record[metric] = csv_number(sample.scores.get(metric))
                record[f"{metric}_skipped"] = sample.skipped.get(metric, "")
                record[f"{metric}_error"] = sample.errors.get(metric, "")
            for metric in BEHAVIOR_METRICS:
                record[metric] = csv_number(sample.behavior.get(metric))
            writer.writerow(record)


def write_summary_csv(aggregates: Sequence[GenerationAggregate], metrics: Sequence[str], path: str | Path) -> None:
    fields = ["mode", "subset", "n_records", "n_generation_errors", "n_metric_errors"]
    for metric in metrics:
        fields += [metric, f"{metric}_n"]
    fields += [f"n_{outcome}" for outcome in OUTCOMES]
    for metric in BEHAVIOR_METRICS:
        fields += [metric, f"{metric}_n"]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in aggregates:
            record: dict[str, object] = {
                "mode": row.mode,
                "subset": row.subset,
                "n_records": row.n_records,
                "n_generation_errors": row.n_generation_errors,
                "n_metric_errors": row.n_metric_errors,
            }
            for metric in metrics:
                record[metric] = csv_number(row.means.get(metric))
                record[f"{metric}_n"] = row.counts.get(metric, 0)
            for outcome in OUTCOMES:
                record[f"n_{outcome}"] = row.outcomes.get(outcome, 0)
            for metric in BEHAVIOR_METRICS:
                record[metric] = csv_number(row.behavior_means.get(metric))
                record[f"{metric}_n"] = row.behavior_counts.get(metric, 0)
            writer.writerow(record)


def is_reasoning_model(model: str) -> bool:
    name = model.strip().lower()
    return name.startswith("gpt-5") or (len(name) > 1 and name[0] == "o" and name[1].isdigit())


def judge_model_kwargs(model: str, *, max_tokens: int | None = None) -> dict[str, Any]:
    if max_tokens is not None:
        return {"max_tokens": max_tokens}
    if is_reasoning_model(model):
        return {"max_tokens": REASONING_JUDGE_MAX_TOKENS}
    return {}


def request_kwargs(request: MetricRequest) -> dict[str, Any]:
    contexts = list(request.retrieved_contexts)
    if request.metric == "faithfulness":
        return {"user_input": request.user_input, "response": request.response, "retrieved_contexts": contexts}
    if request.metric == "answer_relevancy":
        return {"user_input": request.user_input, "response": request.response}
    return {"user_input": request.user_input, "reference": request.reference, "retrieved_contexts": contexts}


class RagasUnavailableError(RuntimeError):
    pass


def ragas_module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise RagasUnavailableError("ragas가 설치되어 있지 않습니다. `uv sync --group eval`로 설치하세요.") from exc


RAGAS_PACKAGE = "ragas"


def require_ragas() -> None:
    ragas_module(RAGAS_PACKAGE)


type JudgeProgressFn = Callable[[int, int, MetricRequest], None]


class RagasScorer:
    def __init__(
        self,
        *,
        api_key: str,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        metrics: Sequence[str] = METRICS,
        adapt_korean: bool = True,
        concurrency: int = DEFAULT_CONCURRENCY,
        max_tokens: int | None = None,
        base_url: str | None = None,
        progress: JudgeProgressFn | None = None,
    ) -> None:
        self.api_key = api_key
        self.judge_model = judge_model
        self.embedding_model = embedding_model
        self.metrics = list(metrics)
        self.adapt_korean = adapt_korean
        self.concurrency = max(1, concurrency)
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.progress = progress

    async def score(self, requests: Sequence[MetricRequest]) -> list[ScoreResult]:
        if not requests:
            return []
        os.environ.setdefault(RAGAS_TRACKING_ENV, "true")
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        try:
            llm, embeddings = self.judge(client)
            metric_sets = {"default": self.build_metrics(llm, embeddings)}
            if self.adapt_korean and any(request.lang == KOREAN for request in requests):
                metric_sets[KOREAN] = await self.adapt(self.build_metrics(llm, embeddings), llm, KOREAN_LANGUAGE)
            return await self.run(list(requests), metric_sets)
        finally:
            await client.close()

    def judge(self, client: AsyncOpenAI) -> tuple[Any, Any]:
        llm_factory = ragas_module("ragas.llms").llm_factory
        embeddings_class = ragas_module("ragas.embeddings").OpenAIEmbeddings
        llm = llm_factory(
            self.judge_model, client=client, **judge_model_kwargs(self.judge_model, max_tokens=self.max_tokens)
        )
        return llm, embeddings_class(client=client, model=self.embedding_model)

    def build_metrics(self, llm: Any, embeddings: Any) -> dict[str, Any]:
        collections = ragas_module("ragas.metrics.collections")
        built: dict[str, Any] = {}
        for metric in self.metrics:
            metric_class = getattr(collections, RAGAS_METRIC_CLASSES[metric])
            if metric in RAGAS_EMBEDDING_METRICS:
                built[metric] = metric_class(llm=llm, embeddings=embeddings)
            else:
                built[metric] = metric_class(llm=llm)
        return built

    async def adapt(self, metrics: dict[str, Any], llm: Any, language: str) -> dict[str, Any]:
        base_prompt = ragas_module("ragas.prompt.metrics.base_prompt").BasePrompt
        for metric in metrics.values():
            for name, value in list(vars(metric).items()):
                if isinstance(value, base_prompt):
                    setattr(metric, name, await value.adapt(language, llm))
        return metrics

    async def run(
        self, requests: Sequence[MetricRequest], metric_sets: Mapping[str, Mapping[str, Any]]
    ) -> list[ScoreResult]:
        semaphore = asyncio.Semaphore(self.concurrency)
        done = 0

        async def run_one(request: MetricRequest) -> float | None:
            nonlocal done
            metrics = metric_sets.get(request.lang, metric_sets["default"])
            async with semaphore:
                try:
                    result = await metrics[request.metric].ascore(**request_kwargs(request))
                finally:
                    done += 1
                    if self.progress is not None:
                        self.progress(done, len(requests), request)
            value = getattr(result, "value", result)
            return None if value is None else float(value)

        return list(await asyncio.gather(*(run_one(request) for request in requests), return_exceptions=True))
