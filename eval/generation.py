"""RAGAS 기반 생성 품질 평가.

답변 레코드(`eval/answers.py`)마다 계산할 수 있는 지표만 판정 요청으로 만들고, 점수기(`Scorer`)에 한 번에 넘긴 뒤
질의별 점수와 모드 × 부분집합 평균을 CSV·Markdown으로 기록한다. ragas 호출은 `RagasScorer` 하나에 모여 있어
테스트에서는 고정 점수를 돌려주는 가짜 점수기로 바꿔 끼운다. LLM 판정 없이 계산하는 행동 지표(`eval/behavior.py`)도
레코드마다 함께 계산해 별도 표로 기록한다.
"""

from __future__ import annotations

import asyncio
import csv
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from eval.behavior import (
    BEHAVIOR_METRIC_DESCRIPTIONS,
    BEHAVIOR_METRICS,
    DEFAULT_REFUSAL_PHRASES,
    OUTCOME_REJECTED,
    OUTCOMES,
    behavior_scores,
)
from eval.metrics import mean

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
METRIC_DESCRIPTIONS = {
    "faithfulness": "답변의 주장 중 LLM에 넘긴 발췌문으로 뒷받침되는 비율",
    "answer_relevancy": "답변에서 거꾸로 만든 질문과 원래 질문의 임베딩 유사도(질문에 맞는 답인가)",
    "context_precision": "기준(reference_answer, 없으면 정답 청크 본문)에 쓸모 있는 발췌문이 앞 순위에 있는 정도",
    "context_recall": "reference_answer의 문장 중 발췌문으로 뒷받침되는 비율",
}
DEFAULT_JUDGE_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
REASONING_JUDGE_MAX_TOKENS = 8192
SUBSETS = ("all", "ko", "en", "known_item", "llm_synth", "manual")
CATEGORY_PREFIX = "category:"
BEHAVIOR_PREFIX = "behavior:"
MODE_ORDER = ("agent", "paper_chat")

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
MISSING = "n/a"


def resolve_metrics(values: Iterable[str]) -> list[str]:
    """쉼표로 나눈 지표 이름을 검증한다. `all`은 전체. 모르는 이름이면 ValueError."""
    metrics: list[str] = []
    for value in values:
        name = value.strip()
        if not name:
            continue
        expanded = list(METRICS) if name == "all" else [name]
        for metric in expanded:
            if metric not in METRICS:
                raise ValueError(f"알 수 없는 지표: {metric!r}. 사용 가능: {', '.join(METRICS)}, all")
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
    def score(self, requests: Sequence[MetricRequest]) -> list[float | None | BaseException]: ...


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
    """context_precision 기준 텍스트와 종류. reference_answer가 우선이고, 없으면 정답 청크 본문을 이어 붙인다."""
    reference_answer = str(record.get("reference_answer") or "").strip()
    if reference_answer:
        return reference_answer, REFERENCE_ANSWER
    texts = [str(chunk_texts.get(int(chunk_id)) or "").strip() for chunk_id in record.get("relevant_chunk_ids") or []]
    texts = [text for text in texts if text]
    if texts:
        return "\n\n".join(texts), REFERENCE_GOLD_CHUNKS
    return None, ""


def plan_record(
    record: Mapping[str, Any],
    metrics: Sequence[str],
    *,
    chunk_texts: Mapping[int, str] | None = None,
    record_index: int = 0,
) -> tuple[list[MetricRequest], dict[str, str], str]:
    """레코드 하나에서 계산할 판정 요청, 건너뛴 지표와 사유, context_precision 기준 종류를 돌려준다.

    입력 거부 레코드(`rejected`)와 `expected_behavior`가 answer가 아닌 케이스(거절·되묻기 기대)는 RAGAS 판정을
    하지 않는다.
    """
    contexts = tuple(text for text in record.get("contexts") or [] if str(text).strip())
    answer = str(record.get("answer") or "").strip()
    reference, reference_kind = resolve_reference(record, chunk_texts or {})
    reference_answer = str(record.get("reference_answer") or "").strip()
    base = {
        "user_input": str(record.get("query") or ""),
        "response": answer,
        "retrieved_contexts": contexts,
        "lang": str(record.get("lang") or ""),
        "record_index": record_index,
    }

    requests: list[MetricRequest] = []
    skipped: dict[str, str] = {}
    for metric in metrics:
        reason: str | None = None
        metric_reference: str | None = None
        if record.get("error"):
            reason = SKIP_GENERATION_ERROR
        elif record.get("outcome") == OUTCOME_REJECTED or record.get("rejection"):
            reason = SKIP_REJECTED
        elif str(record.get("expected_behavior") or "answer") != "answer":
            reason = SKIP_NON_ANSWER_CASE
        elif metric in ("faithfulness", "answer_relevancy") and not answer:
            reason = SKIP_EMPTY_ANSWER
        elif metric != "answer_relevancy" and not contexts:
            reason = SKIP_NO_CONTEXTS
        elif metric == "context_precision":
            if reference is None:
                reason = SKIP_NO_REFERENCE
            metric_reference = reference
        elif metric == "context_recall":
            if not reference_answer:
                reason = SKIP_NO_REFERENCE_ANSWER
            metric_reference = reference_answer or None
        if reason is not None:
            skipped[metric] = reason
            continue
        requests.append(MetricRequest(metric=metric, reference=metric_reference, **base))
    return requests, skipped, reference_kind if "context_precision" in metrics else ""


def _describe_error(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}".strip()
    return text[:300]


def score_records(
    records: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    scorer: Scorer,
    *,
    chunk_texts: Mapping[int, str] | None = None,
    refusal_phrases: Sequence[str] = DEFAULT_REFUSAL_PHRASES,
) -> list[SampleScore]:
    """모든 레코드의 판정 요청을 모아 점수기를 한 번 호출하고 레코드별 점수로 되돌린다.

    점수기가 예외 객체를 돌려준 지표는 `errors`, NaN·None은 `skipped`(`undefined`)로 남기며 평균에서 빠진다.
    결과 종류(`outcome`)와 행동 지표는 `refusal_phrases`로 레코드마다 다시 계산한다(판정 요청이 없어도 계산).
    """
    samples: list[SampleScore] = []
    requests: list[MetricRequest] = []
    for index, record in enumerate(records):
        planned, skipped, reference_kind = plan_record(record, metrics, chunk_texts=chunk_texts, record_index=index)
        requests.extend(planned)
        outcome, behavior = behavior_scores(record, refusal_phrases)
        samples.append(
            SampleScore(
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
        )

    results = scorer.score(requests) if requests else []
    if len(results) != len(requests):
        raise RuntimeError(f"점수기가 요청 {len(requests)}개에 결과 {len(results)}개를 돌려줬습니다.")
    for request, result in zip(requests, results, strict=True):
        sample = samples[request.record_index]
        if isinstance(result, BaseException):
            sample.errors[request.metric] = _describe_error(result)
        elif result is None or (isinstance(result, float) and math.isnan(result)):
            sample.skipped[request.metric] = SKIP_UNDEFINED
        else:
            sample.scores[request.metric] = float(result)
    return samples


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


def in_subset(sample: SampleScore, subset: str) -> bool:
    if subset == "all":
        return True
    if subset in ("ko", "en"):
        return sample.lang == subset
    if subset.startswith(CATEGORY_PREFIX):
        return sample.category == subset[len(CATEGORY_PREFIX) :]
    if subset.startswith(BEHAVIOR_PREFIX):
        return sample.expected_behavior == subset[len(BEHAVIOR_PREFIX) :]
    return sample.source == subset


def build_subsets(samples: Sequence[SampleScore], base: Sequence[str] = SUBSETS) -> list[str]:
    """기본 부분집합 + 비어 있지 않은 `category:<이름>` + `behavior:<기대 행동>`(두 종류 이상일 때만)."""
    categories = sorted({sample.category for sample in samples if sample.category})
    behaviors = sorted({sample.expected_behavior for sample in samples})
    subsets = [*base, *(f"{CATEGORY_PREFIX}{name}" for name in categories)]
    if len(behaviors) > 1:
        subsets.extend(f"{BEHAVIOR_PREFIX}{name}" for name in behaviors)
    return subsets


def _ordered_modes(samples: Sequence[SampleScore]) -> list[str]:
    present = list(dict.fromkeys(sample.mode for sample in samples))
    return [mode for mode in MODE_ORDER if mode in present] + [mode for mode in present if mode not in MODE_ORDER]


def aggregate_scores(
    samples: Sequence[SampleScore], metrics: Sequence[str], *, subsets: Sequence[str] | None = None
) -> list[GenerationAggregate]:
    """모드 × 부분집합(전체, 언어, 질의 출처, 케이스 카테고리, 기대 행동)별 지표 평균과 표본 수.

    `subsets`를 주지 않으면 `build_subsets`로 정한다. 해당 레코드가 없는 부분집합은 생략한다.
    행동 지표 평균·표본 수와 결과 종류별 건수도 같은 행에 담는다.
    """
    rows: list[GenerationAggregate] = []
    for subset in build_subsets(samples) if subsets is None else subsets:
        for mode in _ordered_modes(samples):
            group = [sample for sample in samples if sample.mode == mode and in_subset(sample, subset)]
            if not group:
                continue
            rows.append(
                GenerationAggregate(
                    mode=mode,
                    subset=subset,
                    n_records=len(group),
                    n_generation_errors=sum(1 for sample in group if sample.generation_error),
                    means={metric: mean(sample.scores.get(metric) for sample in group) for metric in metrics},
                    counts={metric: sum(1 for sample in group if metric in sample.scores) for metric in metrics},
                    n_metric_errors=sum(len(sample.errors) for sample in group),
                    behavior_means={
                        metric: mean(sample.behavior.get(metric) for sample in group) for metric in BEHAVIOR_METRICS
                    },
                    behavior_counts={
                        metric: sum(1 for sample in group if metric in sample.behavior) for metric in BEHAVIOR_METRICS
                    },
                    outcomes={outcome: sum(1 for sample in group if sample.outcome == outcome) for outcome in OUTCOMES},
                )
            )
    return rows


def format_score(value: float | None) -> str:
    return MISSING if value is None else f"{value:.3f}"


def _cell(row: GenerationAggregate, metric: str) -> str:
    if metric not in row.means:
        return "-"
    value = row.means[metric]
    return MISSING if value is None else f"{value:.3f} ({row.counts[metric]})"


def _behavior_cell(row: GenerationAggregate, metric: str) -> str:
    if not row.behavior_counts.get(metric):
        return "-"
    return format_score(row.behavior_means.get(metric)) + f" ({row.behavior_counts[metric]})"


def render_behavior_table(aggregates: Sequence[GenerationAggregate]) -> list[str]:
    """행동 지표 표(모드 × 부분집합). 값은 `평균 (해당 케이스 수)`, 해당 케이스가 없으면 `-`."""
    return _table(
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


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_readme_table(aggregates: Sequence[GenerationAggregate]) -> str:
    """README Evaluation의 생성 품질 표와 같은 모양(전체 언어, 모드별 평균). 계산하지 않은 지표는 `-`."""
    lines = [README_TABLE_HEADER, "| --- | --- | --- | --- | --- |"]
    for row in aggregates:
        if row.subset != "all":
            continue
        cells = [format_score(row.means[metric]) if metric in row.means else "-" for metric in METRICS]
        lines.append(f"| {row.mode} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


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
    lines += ["## 모드 × 부분집합", "", "값은 `평균 (계산된 질의 수)`입니다.", ""]
    lines += _table(
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

    skip_counts: dict[tuple[str, str], int] = {}
    for sample in samples:
        for metric, reason in sample.skipped.items():
            skip_counts[(metric, reason)] = skip_counts.get((metric, reason), 0) + 1
    lines += ["## 건너뛴 지표", ""]
    if skip_counts:
        lines += _table(
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
                value = sample.scores.get(metric)
                record[metric] = "" if value is None else f"{value:.4f}"
                record[f"{metric}_skipped"] = sample.skipped.get(metric, "")
                record[f"{metric}_error"] = sample.errors.get(metric, "")
            for metric in BEHAVIOR_METRICS:
                value = sample.behavior.get(metric)
                record[metric] = "" if value is None else f"{value:.4f}"
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
                value = row.means.get(metric)
                record[metric] = "" if value is None else f"{value:.4f}"
                record[f"{metric}_n"] = row.counts.get(metric, 0)
            for outcome in OUTCOMES:
                record[f"n_{outcome}"] = row.outcomes.get(outcome, 0)
            for metric in BEHAVIOR_METRICS:
                value = row.behavior_means.get(metric)
                record[metric] = "" if value is None else f"{value:.4f}"
                record[f"{metric}_n"] = row.behavior_counts.get(metric, 0)
            writer.writerow(record)


def is_reasoning_model(model: str) -> bool:
    name = model.strip().lower()
    return name.startswith("gpt-5") or (len(name) > 1 and name[0] == "o" and name[1].isdigit())


def judge_model_kwargs(model: str, *, max_tokens: int | None = None) -> dict[str, Any]:
    """판정 LLM 추가 인자. gpt-5·o 계열은 추론 토큰이 출력 한도를 같이 쓰므로 한도를 넉넉히 준다.

    temperature는 ragas가 gpt-5·o 계열이면 1.0으로 고정하고 top_p를 빼므로 여기서는 넘기지 않는다.
    """
    if max_tokens is not None:
        return {"max_tokens": max_tokens}
    if is_reasoning_model(model):
        return {"max_tokens": REASONING_JUDGE_MAX_TOKENS}
    return {}


def _request_kwargs(request: MetricRequest) -> dict[str, Any]:
    contexts = list(request.retrieved_contexts)
    if request.metric == "faithfulness":
        return {"user_input": request.user_input, "response": request.response, "retrieved_contexts": contexts}
    if request.metric == "answer_relevancy":
        return {"user_input": request.user_input, "response": request.response}
    return {"user_input": request.user_input, "reference": request.reference, "retrieved_contexts": contexts}


ProgressFn = Callable[[int, int, MetricRequest], None]


class RagasScorer:
    """ragas `metrics.collections` 지표를 OpenAI 판정 LLM·임베딩으로 비동기 실행하는 점수기.

    `score()` 한 번이 이벤트 루프 하나를 쓰고, 그 안에서 클라이언트·지표 객체를 만든다. `adapt_korean=True`이고
    한국어 요청이 있으면 지표 프롬프트의 few-shot 예시를 판정 LLM으로 한국어로 번역한 지표 묶음을 한 번 만들어
    `lang == "ko"` 요청에 쓴다(지시문은 영어 그대로). ragas 사용 통계 전송은 끈다.
    """

    def __init__(
        self,
        *,
        api_key: str,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        metrics: Sequence[str] = METRICS,
        adapt_korean: bool = True,
        concurrency: int = 4,
        max_tokens: int | None = None,
        progress: ProgressFn | None = None,
    ) -> None:
        self.api_key = api_key
        self.judge_model = judge_model
        self.embedding_model = embedding_model
        self.metrics = list(metrics)
        self.adapt_korean = adapt_korean
        self.concurrency = max(1, concurrency)
        self.max_tokens = max_tokens
        self.progress = progress

    def score(self, requests: Sequence[MetricRequest]) -> list[float | None | BaseException]:
        if not requests:
            return []
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        return asyncio.run(self._score_all(list(requests)))

    def _build_metrics(self, llm: Any, embeddings: Any) -> dict[str, Any]:
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            Faithfulness,
        )

        factories = {
            "faithfulness": lambda: Faithfulness(llm=llm),
            "answer_relevancy": lambda: AnswerRelevancy(llm=llm, embeddings=embeddings),
            "context_precision": lambda: ContextPrecisionWithReference(llm=llm),
            "context_recall": lambda: ContextRecall(llm=llm),
        }
        return {metric: factories[metric]() for metric in self.metrics}

    async def _adapt(self, metrics: dict[str, Any], llm: Any, language: str) -> dict[str, Any]:
        from ragas.prompt.metrics.base_prompt import BasePrompt

        for metric in metrics.values():
            for name, value in list(vars(metric).items()):
                if isinstance(value, BasePrompt):
                    setattr(metric, name, await value.adapt(language, llm))
        return metrics

    async def _score_all(self, requests: list[MetricRequest]) -> list[float | None | BaseException]:
        from openai import AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.llms import llm_factory

        client = AsyncOpenAI(api_key=self.api_key)
        try:
            llm = llm_factory(
                self.judge_model,
                client=client,
                **judge_model_kwargs(self.judge_model, max_tokens=self.max_tokens),
            )
            embeddings = OpenAIEmbeddings(client=client, model=self.embedding_model)
            metric_sets = {"default": self._build_metrics(llm, embeddings)}
            if self.adapt_korean and any(request.lang == "ko" for request in requests):
                metric_sets["ko"] = await self._adapt(self._build_metrics(llm, embeddings), llm, "korean")

            semaphore = asyncio.Semaphore(self.concurrency)
            done = 0

            async def run(request: MetricRequest) -> float | None:
                nonlocal done
                metrics = metric_sets.get(request.lang, metric_sets["default"])
                async with semaphore:
                    try:
                        result = await metrics[request.metric].ascore(**_request_kwargs(request))
                    finally:
                        done += 1
                        if self.progress is not None:
                            self.progress(done, len(requests), request)
                value = getattr(result, "value", result)
                return None if value is None else float(value)

            return list(await asyncio.gather(*(run(request) for request in requests), return_exceptions=True))
        finally:
            await client.close()
