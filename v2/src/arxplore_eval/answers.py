import json
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Self

from arxplore.api.services.chat import ChatInput, ChatService, PreparedPaperChat
from arxplore.context import RetrievalObservation, observe_retrieval
from arxplore.domain.retrieval import RetrievalContext
from arxplore.llm.events import CollectedAnswer, collect_answer
from arxplore_eval.behavior import (
    DEFAULT_REFUSAL_PHRASES,
    OUTCOME_REJECTED,
    InputRejected,
    classify_outcome,
    prepare_chat_input,
)
from arxplore_eval.dataset import EvalQuery, dataset_files

AGENT_MODE = "agent"
PAPER_CHAT_MODE = "paper_chat"
MODES = (AGENT_MODE, PAPER_CHAT_MODE)
BOTH_MODES = "both"
REQUIRED_RECORD_KEYS = ("id", "query", "mode", "answer", "contexts")
LATENCY_DIGITS = 1

type AnswerRecord = dict[str, Any]
type ProgressFn = Callable[[int, int, str, EvalQuery, AnswerRecord], None]
type Clock = Callable[[], float]


class AnswersError(ValueError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    contexts: list[str]
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieval_mode: str | None = None
    arxiv_id: str | None = None
    hit_arxiv_ids: list[str] = field(default_factory=list)


type GenerateFn = Callable[[EvalQuery], Awaitable[GenerationResult]]


def resolve_modes(values: Iterable[str]) -> list[str]:
    modes: list[str] = []
    for value in values:
        name = value.strip()
        if not name:
            continue
        expanded = list(MODES) if name == BOTH_MODES else [name]
        for mode in expanded:
            if mode not in MODES:
                raise ValueError(f"알 수 없는 mode: {mode!r}. 사용 가능: {', '.join(MODES)}, {BOTH_MODES}")
            if mode not in modes:
                modes.append(mode)
    if not modes:
        raise ValueError("mode를 하나 이상 지정하세요.")
    return modes


def answer_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record["id"]), str(record["mode"])


def prepare_query(query: EvalQuery) -> EvalQuery:
    message, history = prepare_chat_input(query.query, query.history)
    return replace(query, query=message, history=tuple(history))


def build_answer_record(
    query: EvalQuery,
    mode: str,
    *,
    result: GenerationResult | None = None,
    error: str | None = None,
    rejection: str | None = None,
    latency_ms: float | None = None,
    reference_answer: str | None = None,
    model: str | None = None,
    history_turns: int | None = None,
    refusal_phrases: Sequence[str] = DEFAULT_REFUSAL_PHRASES,
) -> AnswerRecord:
    record: AnswerRecord = {
        "id": query.id,
        "query": query.query,
        "mode": mode,
        "lang": query.lang,
        "source": query.source,
        "relevant_arxiv_ids": list(query.relevant_arxiv_ids),
        "relevant_chunk_ids": list(query.relevant_chunk_ids),
        "reference_answer": reference_answer or None,
        "answer": result.answer if result else "",
        "contexts": list(result.contexts) if result else [],
        "citations": list(result.citations) if result else [],
        "retrieval_mode": result.retrieval_mode if result else None,
        "arxiv_id": result.arxiv_id if result else None,
        "model": model,
        "latency_ms": None if latency_ms is None else round(latency_ms, LATENCY_DIGITS),
        "error": error,
        "category": query.category,
        "expected_behavior": query.expected_behavior,
        "expect_rejected": query.expect_rejected,
        "must_not_contain": list(query.must_not_contain),
        "must_mention_arxiv_ids": list(query.must_mention_arxiv_ids),
        "history_turns": history_turns,
        "hit_arxiv_ids": list(result.hit_arxiv_ids) if result else [],
        "rejection": rejection,
    }
    record["outcome"] = OUTCOME_REJECTED if rejection else classify_outcome(record, refusal_phrases)
    return record


def load_reference_answers(path: str | Path) -> dict[str, str]:
    references: dict[str, str] = {}
    for file_path in dataset_files(path):
        with file_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                value = payload.get("reference_answer") if isinstance(payload, dict) else None
                if isinstance(value, str) and value.strip():
                    references[str(payload.get("id"))] = value.strip()
    return references


def _validated_record(payload: Any, location: str) -> AnswerRecord:
    if not isinstance(payload, dict):
        raise AnswersError(f"{location}: each line must be a JSON object")
    missing = [key for key in REQUIRED_RECORD_KEYS if key not in payload]
    if missing:
        raise AnswersError(f"{location}: missing keys {missing}")
    if payload["mode"] not in MODES:
        raise AnswersError(f"{location}: unknown mode {payload['mode']!r}")
    contexts = payload["contexts"]
    if not isinstance(contexts, list) or not all(isinstance(value, str) for value in contexts):
        raise AnswersError(f"{location}: 'contexts' must be a list of strings")
    return payload


def load_answer_records(path: str | Path) -> list[AnswerRecord]:
    source = Path(path)
    records: dict[tuple[str, str], AnswerRecord] = {}
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{source}:{line_number}"
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnswersError(f"{location}: invalid JSON ({exc.msg})") from exc
            record = _validated_record(payload, location)
            key = answer_key(record)
            records.pop(key, None)
            records[key] = record
    return list(records.values())


def append_answer_record(path: str | Path, record: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed_keys(records: Iterable[Mapping[str, Any]]) -> set[tuple[str, str]]:
    return {answer_key(record) for record in records if not record.get("error")}


async def _answer_record(
    query: EvalQuery,
    mode: str,
    generate: GenerateFn,
    *,
    keep_going: bool,
    common: Mapping[str, Any],
    clock: Clock,
) -> AnswerRecord:
    try:
        prepared = prepare_query(query)
    except InputRejected as exc:
        return build_answer_record(query, mode, rejection=str(exc), **common)
    started = clock()
    try:
        result = await generate(prepared)
    except Exception as exc:
        if not keep_going:
            raise
        return build_answer_record(
            query, mode, error=f"{type(exc).__name__}: {exc}", history_turns=len(prepared.history), **common
        )
    return build_answer_record(
        query,
        mode,
        result=result,
        latency_ms=(clock() - started) * 1000.0,
        history_turns=len(prepared.history),
        **common,
    )


async def collect_answers(
    queries: Sequence[EvalQuery],
    modes: Sequence[str],
    generators: Mapping[str, GenerateFn],
    *,
    done: Iterable[tuple[str, str]] = (),
    keep_going: bool = False,
    reference_answers: Mapping[str, str] | None = None,
    model: str | None = None,
    sink: Callable[[AnswerRecord], None] | None = None,
    progress: ProgressFn | None = None,
    clock: Clock = time.perf_counter,
    refusal_phrases: Sequence[str] = DEFAULT_REFUSAL_PHRASES,
) -> list[AnswerRecord]:
    skip = set(done)
    references = reference_answers or {}
    pending = [
        (query, mode)
        for query in queries
        for mode in modes
        if query.supports_mode(mode) and (query.id, mode) not in skip
    ]
    records: list[AnswerRecord] = []
    for index, (query, mode) in enumerate(pending, start=1):
        common = {
            "reference_answer": references.get(query.id),
            "model": model,
            "refusal_phrases": refusal_phrases,
        }
        record = await _answer_record(query, mode, generators[mode], keep_going=keep_going, common=common, clock=clock)
        if sink is not None:
            sink(record)
        records.append(record)
        if progress is not None:
            progress(index, len(pending), mode, query, record)
    return records


def _dedupe(texts: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(text for text in texts if text))


def context_text(context: RetrievalContext) -> str:
    return context.text or context.hit.chunk_text


@dataclass(frozen=True)
class CapturedContexts:
    texts: list[str]
    retrieval_modes: list[str]
    arxiv_ids: list[str]

    @classmethod
    def from_observations(cls, observations: Sequence[RetrievalObservation]) -> Self:
        contexts = [context for observation in observations for context in observation.result.contexts]
        return cls(
            texts=[context_text(context) for context in contexts],
            retrieval_modes=[observation.result.mode for observation in observations],
            arxiv_ids=[context.hit.arxiv_id for context in contexts],
        )


def _citation_payloads(answer: CollectedAnswer) -> list[dict[str, Any]]:
    return [citation.to_payload() for citation in answer.citations]


async def generate_agent_answer(service: ChatService, query: EvalQuery) -> GenerationResult:
    chat = ChatInput(message=query.query, history=list(query.history))
    with observe_retrieval() as observations:
        answer = await collect_answer(service.stream_agent_chat(chat))
    captured = CapturedContexts.from_observations(observations)
    modes = list(dict.fromkeys(captured.retrieval_modes))
    return GenerationResult(
        answer=answer.answer,
        contexts=_dedupe(captured.texts),
        citations=_citation_payloads(answer),
        retrieval_mode=",".join(modes) or None,
        hit_arxiv_ids=_dedupe(captured.arxiv_ids),
    )


async def generate_paper_chat_answer(service: ChatService, query: EvalQuery) -> GenerationResult:
    arxiv_id = query.paper_chat_arxiv_id
    if not arxiv_id:
        raise LookupError(f"{query.id}: 상세 챗 대상 논문(open_arxiv_id)이 없습니다.")
    paper = await service.store.get_paper(arxiv_id)
    if paper is None:
        raise LookupError(f"papers에 {arxiv_id}가 없습니다.")
    prepared = PreparedPaperChat(message=query.query, history=list(query.history), paper=paper)
    retrieval = await service.retrieve_paper_sources(prepared)
    answer = await collect_answer(service.answer_paper_sources(prepared, retrieval))
    return GenerationResult(
        answer=answer.answer,
        contexts=[source.text for source in retrieval.sources if source.text],
        citations=_citation_payloads(answer),
        retrieval_mode=retrieval.retrieval_mode,
        arxiv_id=arxiv_id,
        hit_arxiv_ids=[arxiv_id],
    )


def build_generators(service: ChatService, modes: Sequence[str]) -> dict[str, GenerateFn]:
    generators: dict[str, GenerateFn] = {}
    if AGENT_MODE in modes:
        generators[AGENT_MODE] = partial(generate_agent_answer, service)
    if PAPER_CHAT_MODE in modes:
        generators[PAPER_CHAT_MODE] = partial(generate_paper_chat_answer, service)
    return generators
