import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Self

from arxplore.api.services.chat import ChatInput, ChatService, PreparedPaperChat
from arxplore.context import RetrievalObservation, observe_retrieval
from arxplore.domain.papers import Paper
from arxplore.domain.retrieval import RetrievalContext
from arxplore.llm.agent import tools as agent_tools
from arxplore.llm.citations import paper_url
from arxplore.llm.events import CollectedAnswer, collect_answer
from arxplore.llm.paper_chat import PaperChatRetrieval, PaperChatSource, format_paper_chat_context
from arxplore.retrieval.service import Retriever
from arxplore_eval.agent_plan import (
    AgentPlan,
    PlanError,
    answer_from_plan,
    build_capture_graph,
    capture_plan,
    plan_from_payload,
    plan_payload,
    text_sha256,
)
from arxplore_eval.behavior import (
    DEFAULT_REFUSAL_PHRASES,
    OUTCOME_REJECTED,
    InputRejected,
    classify_outcome,
    prepare_chat_input,
)
from arxplore_eval.dataset import EvalQuery, dataset_files
from arxplore_eval.fusion_variants import check_fusion
from arxplore_eval.prompt_variants import PROMPT_CURRENT, paper_chat_prompt

AGENT_MODE = "agent"
PAPER_CHAT_MODE = "paper_chat"
MODES = (AGENT_MODE, PAPER_CHAT_MODE)
BOTH_MODES = "both"
REQUIRED_RECORD_KEYS = ("id", "query", "mode", "answer", "contexts")
LATENCY_DIGITS = 1
SEARCH_LIMIT_KEY = "search_limit"
SOURCES_FROM_RETRIEVAL = "retrieval"
SOURCES_FROM_REPLAY = "replay"
AGENT_ORIGIN_LIVE = "live"
AGENT_ORIGIN_PLAN_ONLY = "plan_only"
AGENT_ORIGIN_PLAN_REPLAY = "plan_replay"

type AnswerRecord = dict[str, Any]
type ProgressFn = Callable[[int, int, str, EvalQuery, AnswerRecord], None]
type Clock = Callable[[], float]


class AnswersError(ValueError):
    pass


@dataclass(frozen=True)
class PaperChatTrace:
    prompt_variant: str | None
    sources_origin: str
    sources_only: bool
    context_sha256: str
    sources: list[dict[str, Any]]

    def record_fields(self) -> dict[str, Any]:
        return {
            "prompt_variant": self.prompt_variant,
            "sources_origin": self.sources_origin,
            "sources_only": self.sources_only,
            "context_sha256": self.context_sha256,
            "sources": [dict(source) for source in self.sources],
        }


@dataclass(frozen=True)
class AgentTrace:
    fusion: str | None
    origin: str
    plan_sha256: str | None = None
    plan: dict[str, Any] | None = None
    searches: list[dict[str, Any]] = field(default_factory=list)
    tool_output_sha256: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    def record_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "fusion": self.fusion,
            "agent_origin": self.origin,
            "plan_sha256": self.plan_sha256,
            "searches": [dict(search) for search in self.searches],
            "usage": dict(self.usage),
        }
        if self.origin == AGENT_ORIGIN_PLAN_ONLY:
            fields["plan_only"] = True
            fields["plan"] = self.plan
        if self.origin == AGENT_ORIGIN_PLAN_REPLAY:
            fields["tool_output_sha256"] = list(self.tool_output_sha256)
        return fields


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    contexts: list[str]
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieval_mode: str | None = None
    arxiv_id: str | None = None
    hit_arxiv_ids: list[str] = field(default_factory=list)
    paper_chat: PaperChatTrace | None = None
    agent: AgentTrace | None = None


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


@contextmanager
def agent_search_limit(limit: int | None) -> Iterator[None]:
    if limit is None:
        yield
        return
    if limit < 1:
        raise ValueError(f"search_limit은 1 이상이어야 합니다: {limit}")
    original = agent_tools.SEARCH_RESULT_LIMIT
    agent_tools.SEARCH_RESULT_LIMIT = limit
    try:
        yield
    finally:
        agent_tools.SEARCH_RESULT_LIMIT = original


def recorded_search_limit(record: Mapping[str, Any]) -> int | None:
    value = record.get(SEARCH_LIMIT_KEY)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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
    search_limit: int | None = None,
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
    if search_limit is not None and mode == AGENT_MODE:
        record[SEARCH_LIMIT_KEY] = search_limit
    if result is not None and result.paper_chat is not None and mode == PAPER_CHAT_MODE:
        record.update(result.paper_chat.record_fields())
    if result is not None and result.agent is not None and mode == AGENT_MODE:
        record.update(result.agent.record_fields())
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
    limit = payload.get(SEARCH_LIMIT_KEY)
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise AnswersError(f"{location}: '{SEARCH_LIMIT_KEY}' must be a positive integer")
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
    search_limit: int | None = None,
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
    with agent_search_limit(search_limit):
        for index, (query, mode) in enumerate(pending, start=1):
            common = {
                "reference_answer": references.get(query.id),
                "model": model,
                "refusal_phrases": refusal_phrases,
                "search_limit": search_limit,
            }
            record = await _answer_record(
                query, mode, generators[mode], keep_going=keep_going, common=common, clock=clock
            )
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


@dataclass(frozen=True)
class AgentOptions:
    fusion: str | None = None
    plan_only: bool = False
    replay: Mapping[str, AgentPlan] | None = None
    replay_sha256: str | None = None
    retriever: Retriever | None = None

    def __post_init__(self) -> None:
        if self.fusion is not None:
            check_fusion(self.fusion)
        if self.plan_only and self.replay is not None:
            raise ValueError("계획 캡처와 계획 재생은 함께 쓸 수 없습니다.")
        if self.plan_only and self.fusion is not None:
            raise ValueError("계획 캡처는 융합 전 후보만 기록하므로 fusion을 받지 않습니다.")
        if self.replay is not None and self.fusion is None:
            raise ValueError("계획 재생에는 fusion이 필요합니다.")
        if (self.plan_only or self.replay is not None) and self.retriever is None:
            raise ValueError("계획 캡처·재생에는 Retriever가 필요합니다.")

    @property
    def origin(self) -> str:
        if self.plan_only:
            return AGENT_ORIGIN_PLAN_ONLY
        if self.replay is not None:
            return AGENT_ORIGIN_PLAN_REPLAY
        return AGENT_ORIGIN_LIVE

    def require_retriever(self) -> Retriever:
        if self.retriever is None:
            raise ValueError("Retriever가 없습니다.")
        return self.retriever


def search_trace(observations: Sequence[RetrievalObservation]) -> list[dict[str, Any]]:
    return [
        {
            "query": observation.query,
            "retrieval_mode": observation.result.mode,
            "chunk_ids": [context.hit.chunk_id for context in observation.result.contexts],
        }
        for observation in observations
    ]


def _agent_result(
    answer: str,
    citations: list[dict[str, Any]],
    observations: Sequence[RetrievalObservation],
    trace: AgentTrace | None,
) -> GenerationResult:
    captured = CapturedContexts.from_observations(observations)
    modes = list(dict.fromkeys(captured.retrieval_modes))
    return GenerationResult(
        answer=answer,
        contexts=_dedupe(captured.texts),
        citations=citations,
        retrieval_mode=",".join(modes) or None,
        hit_arxiv_ids=_dedupe(captured.arxiv_ids),
        agent=trace,
    )


async def generate_agent_answer(
    service: ChatService, query: EvalQuery, options: AgentOptions | None = None
) -> GenerationResult:
    settings = options or AgentOptions()
    if settings.plan_only:
        return await generate_agent_plan(service, query, settings)
    if settings.replay is not None:
        return await generate_agent_plan_answer(service, query, settings)
    chat = ChatInput(message=query.query, history=list(query.history))
    with observe_retrieval() as observations:
        answer = await collect_answer(service.stream_agent_chat(chat))
    trace = None
    if settings.fusion is not None:
        trace = AgentTrace(fusion=settings.fusion, origin=AGENT_ORIGIN_LIVE, searches=search_trace(observations))
    return _agent_result(answer.answer, _citation_payloads(answer), observations, trace)


async def generate_agent_plan(service: ChatService, query: EvalQuery, options: AgentOptions) -> GenerationResult:
    plan = await capture_plan(
        service,
        options.require_retriever(),
        build_capture_graph(service),
        query_id=query.id,
        question=query.query,
        history=list(query.history),
    )
    trace = AgentTrace(
        fusion=None,
        origin=AGENT_ORIGIN_PLAN_ONLY,
        plan=plan_payload(plan),
        searches=[{"query": search.query, "tool_call_id": search.tool_call_id} for search in plan.searches],
        usage=dict(plan.usage),
    )
    return GenerationResult(answer="", contexts=[], agent=trace)


async def generate_agent_plan_answer(service: ChatService, query: EvalQuery, options: AgentOptions) -> GenerationResult:
    plan = (options.replay or {}).get(query.id)
    if plan is None:
        raise LookupError(f"{query.id}: 재생할 에이전트 계획이 없습니다.")
    fusion = check_fusion(options.fusion or "")
    answer = await answer_from_plan(
        service,
        options.require_retriever(),
        plan,
        fusion,
        question=query.query,
        history=list(query.history),
    )
    trace = AgentTrace(
        fusion=fusion,
        origin=AGENT_ORIGIN_PLAN_REPLAY,
        plan_sha256=options.replay_sha256,
        searches=search_trace(answer.observations),
        tool_output_sha256=[text_sha256(output) for output in answer.tool_outputs],
        usage=dict(answer.usage),
    )
    return _agent_result(
        answer.answer, [citation.to_payload() for citation in answer.citations], answer.observations, trace
    )


def load_agent_plans(path: str | Path) -> tuple[dict[str, AgentPlan], dict[str, str]]:
    plans: dict[str, AgentPlan] = {}
    questions: dict[str, str] = {}
    for record in load_answer_records(path):
        if record["mode"] != AGENT_MODE or record.get("error") or record.get("rejection"):
            continue
        query_id = str(record["id"])
        if not record.get("plan_only"):
            raise AnswersError(f"{query_id}: 계획 캡처(--plan-only) 레코드가 아닙니다.")
        try:
            plans[query_id] = plan_from_payload(query_id, record.get("plan"))
        except PlanError as exc:
            raise AnswersError(str(exc)) from exc
        questions[query_id] = str(record["query"])
    return plans, questions


@dataclass(frozen=True)
class RecordedSource:
    chunk_id: int | None
    section_title: str | None
    text: str


@dataclass(frozen=True)
class RecordedSources:
    query_id: str
    arxiv_id: str | None
    retrieval_mode: str
    sources: tuple[RecordedSource, ...]
    context_sha256: str | None = None


@dataclass(frozen=True)
class PaperChatOptions:
    prompt_variant: str = PROMPT_CURRENT
    sources_only: bool = False
    replay: Mapping[str, RecordedSources] | None = None

    def __post_init__(self) -> None:
        paper_chat_prompt(self.prompt_variant)
        if self.sources_only and self.replay is not None:
            raise ValueError("검색 전용 수집과 소스 재생은 함께 쓸 수 없습니다.")


def source_payloads(sources: Sequence[PaperChatSource]) -> list[dict[str, Any]]:
    return [
        {"chunk_id": source.chunk_id, "section_title": source.section_title, "text": source.text} for source in sources
    ]


def context_digest(sources: Sequence[PaperChatSource]) -> str:
    return hashlib.sha256(format_paper_chat_context(sources).encode("utf-8")).hexdigest()


def _recorded_source(payload: Any, location: str) -> RecordedSource:
    if not isinstance(payload, Mapping):
        raise AnswersError(f"{location}: each source must be an object")
    chunk_id = payload.get("chunk_id")
    section_title = payload.get("section_title")
    text = payload.get("text")
    if chunk_id is not None and (not isinstance(chunk_id, int) or isinstance(chunk_id, bool)):
        raise AnswersError(f"{location}: 'chunk_id' must be an integer or null")
    if section_title is not None and not isinstance(section_title, str):
        raise AnswersError(f"{location}: 'section_title' must be a string or null")
    if not isinstance(text, str):
        raise AnswersError(f"{location}: 'text' must be a string")
    return RecordedSource(chunk_id=chunk_id, section_title=section_title, text=text)


def recorded_sources(record: Mapping[str, Any]) -> RecordedSources:
    location = f"{record.get('id')}"
    sources = record.get("sources")
    if not isinstance(sources, list):
        raise AnswersError(f"{location}: 'sources' must be a list (--sources-only로 수집한 파일인지 확인)")
    retrieval_mode = record.get("retrieval_mode")
    if not isinstance(retrieval_mode, str) or not retrieval_mode:
        raise AnswersError(f"{location}: 'retrieval_mode' is missing")
    digest = record.get("context_sha256")
    return RecordedSources(
        query_id=str(record["id"]),
        arxiv_id=str(record["arxiv_id"]) if record.get("arxiv_id") else None,
        retrieval_mode=retrieval_mode,
        sources=tuple(_recorded_source(item, f"{location} sources[{index}]") for index, item in enumerate(sources)),
        context_sha256=digest if isinstance(digest, str) and digest else None,
    )


def load_recorded_sources(path: str | Path) -> dict[str, RecordedSources]:
    recorded: dict[str, RecordedSources] = {}
    for record in load_answer_records(path):
        if record["mode"] != PAPER_CHAT_MODE or record.get("error") or record.get("rejection"):
            continue
        recorded[str(record["id"])] = recorded_sources(record)
    return recorded


def replayed_retrieval(paper: Paper, recorded: RecordedSources) -> PaperChatRetrieval:
    arxiv_id = str(paper.arxiv_id or "")
    title = str(paper.title or "")
    url = paper_url(arxiv_id, paper.pdf_url)
    sources = tuple(
        PaperChatSource(
            arxiv_id=arxiv_id,
            title=title,
            url=url,
            section_title=source.section_title,
            chunk_id=source.chunk_id,
            text=source.text,
        )
        for source in recorded.sources
    )
    if recorded.context_sha256 is not None and context_digest(sources) != recorded.context_sha256:
        raise ValueError(f"{recorded.query_id}: 재생한 발췌문 문맥이 기록(context_sha256)과 다릅니다.")
    return PaperChatRetrieval(sources=sources, retrieval_mode=recorded.retrieval_mode)


async def generate_paper_chat_answer(
    service: ChatService, query: EvalQuery, options: PaperChatOptions | None = None
) -> GenerationResult:
    settings = options or PaperChatOptions()
    arxiv_id = query.paper_chat_arxiv_id
    if not arxiv_id:
        raise LookupError(f"{query.id}: 상세 챗 대상 논문(open_arxiv_id)이 없습니다.")
    paper = await service.store.get_paper(arxiv_id)
    if paper is None:
        raise LookupError(f"papers에 {arxiv_id}가 없습니다.")
    prepared = PreparedPaperChat(message=query.query, history=list(query.history), paper=paper)
    if settings.replay is not None:
        recorded = settings.replay.get(query.id)
        if recorded is None:
            raise LookupError(f"{query.id}: 재생할 소스 기록이 없습니다.")
        if recorded.arxiv_id and recorded.arxiv_id != arxiv_id:
            raise LookupError(f"{query.id}: 기록된 논문 {recorded.arxiv_id}와 질의 논문 {arxiv_id}가 다릅니다.")
        retrieval = replayed_retrieval(paper, recorded)
    else:
        retrieval = await service.retrieve_paper_sources(prepared)
    trace = PaperChatTrace(
        prompt_variant=None if settings.sources_only else settings.prompt_variant,
        sources_origin=SOURCES_FROM_RETRIEVAL if settings.replay is None else SOURCES_FROM_REPLAY,
        sources_only=settings.sources_only,
        context_sha256=context_digest(retrieval.sources),
        sources=source_payloads(retrieval.sources),
    )
    contexts = [source.text for source in retrieval.sources if source.text]
    if settings.sources_only:
        return GenerationResult(
            answer="",
            contexts=contexts,
            retrieval_mode=retrieval.retrieval_mode,
            arxiv_id=arxiv_id,
            hit_arxiv_ids=[arxiv_id],
            paper_chat=trace,
        )
    answer = await collect_answer(
        service.answer_paper_sources(prepared, retrieval, prompt=paper_chat_prompt(settings.prompt_variant))
    )
    return GenerationResult(
        answer=answer.answer,
        contexts=contexts,
        citations=_citation_payloads(answer),
        retrieval_mode=retrieval.retrieval_mode,
        arxiv_id=arxiv_id,
        hit_arxiv_ids=[arxiv_id],
        paper_chat=trace,
    )


def build_generators(
    service: ChatService,
    modes: Sequence[str],
    *,
    paper_chat: PaperChatOptions | None = None,
    agent: AgentOptions | None = None,
) -> dict[str, GenerateFn]:
    generators: dict[str, GenerateFn] = {}
    if AGENT_MODE in modes:
        generators[AGENT_MODE] = partial(generate_agent_answer, service, options=agent)
    if PAPER_CHAT_MODE in modes:
        generators[PAPER_CHAT_MODE] = partial(generate_paper_chat_answer, service, options=paper_chat)
    return generators
