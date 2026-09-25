"""생성 품질 평가용 답변 수집.

에이전트(`agent`)와 상세 챗(`paper_chat`)을 Django 없이 제품 코드 그대로 호출하고, 답변과 함께
LLM 프롬프트에 들어간 발췌문 본문(`contexts`)을 JSONL 한 줄씩 기록한다. 호출 전에 제품 API의 입력 검증을
재현해(`eval.behavior.prepare_chat_input`) API가 거부할 입력은 생성하지 않고 `rejected` 결과로 남긴다.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from eval.behavior import (
    DEFAULT_REFUSAL_PHRASES,
    OUTCOME_REJECTED,
    InputRejected,
    classify_outcome,
    prepare_chat_input,
)
from eval.dataset import EvalQuery, dataset_files

MODES = ("agent", "paper_chat")
REQUIRED_RECORD_KEYS = ("id", "query", "mode", "answer", "contexts")


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


GenerateFn = Callable[[EvalQuery], GenerationResult]
ProgressFn = Callable[[int, int, str, EvalQuery, dict[str, Any]], None]


def resolve_modes(values: Iterable[str]) -> list[str]:
    """`both`는 두 모드로 펼치고, 모르는 이름이면 ValueError."""
    modes: list[str] = []
    for value in values:
        name = value.strip()
        if not name:
            continue
        expanded = list(MODES) if name == "both" else [name]
        for mode in expanded:
            if mode not in MODES:
                raise ValueError(f"알 수 없는 mode: {mode!r}. 사용 가능: {', '.join(MODES)}, both")
            if mode not in modes:
                modes.append(mode)
    if not modes:
        raise ValueError("mode를 하나 이상 지정하세요.")
    return modes


def answer_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record["id"]), str(record["mode"])


def prepare_query(query: EvalQuery) -> EvalQuery:
    """제품 API가 생성 함수에 넘기는 모양으로 바꾼 사본: 앞뒤 공백을 지운 질문과 정리된 이력.

    API가 거부할 입력이면 `InputRejected`.
    """
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
) -> dict[str, Any]:
    """질의 하나 × 모드 하나의 답변 레코드.

    실패하면 답변·발췌문은 비우고 `error`에 원인을, 입력 거부면 `rejection`에 API 안내 문구를 남긴다.
    `outcome`은 answered / refused / rejected / error 중 하나다.
    """
    record: dict[str, Any] = {
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
        "latency_ms": None if latency_ms is None else round(latency_ms, 1),
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
    """질의셋 JSONL에서 선택 필드 `reference_answer`(비어 있지 않은 문자열)만 id별로 읽는다."""
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


def load_answer_records(path: str | Path) -> list[dict[str, Any]]:
    """답변 JSONL을 읽는다. 같은 (id, mode)가 여러 번 있으면 마지막 줄이 이전 줄을 대체한다."""
    path = Path(path)
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AnswersError(f"{location}: invalid JSON ({exc.msg})") from exc
            if not isinstance(payload, dict):
                raise AnswersError(f"{location}: each line must be a JSON object")
            missing = [key for key in REQUIRED_RECORD_KEYS if key not in payload]
            if missing:
                raise AnswersError(f"{location}: missing keys {missing}")
            if payload["mode"] not in MODES:
                raise AnswersError(f"{location}: unknown mode {payload['mode']!r}")
            if not isinstance(payload["contexts"], list) or not all(
                isinstance(value, str) for value in payload["contexts"]
            ):
                raise AnswersError(f"{location}: 'contexts' must be a list of strings")
            key = answer_key(payload)
            records.pop(key, None)
            records[key] = payload
    return list(records.values())


def append_answer_record(path: str | Path, record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def completed_keys(records: Iterable[Mapping[str, Any]]) -> set[tuple[str, str]]:
    """오류 없이 끝난 (id, mode). 이어서 수집할 때 이 조합은 건너뛴다."""
    return {answer_key(record) for record in records if not record.get("error")}


def collect_answers(
    queries: Sequence[EvalQuery],
    modes: Sequence[str],
    generators: Mapping[str, GenerateFn],
    *,
    done: Iterable[tuple[str, str]] = (),
    keep_going: bool = False,
    reference_answers: Mapping[str, str] | None = None,
    model: str | None = None,
    sink: Callable[[dict[str, Any]], None] | None = None,
    progress: ProgressFn | None = None,
    clock: Callable[[], float] = time.perf_counter,
    refusal_phrases: Sequence[str] = DEFAULT_REFUSAL_PHRASES,
) -> list[dict[str, Any]]:
    """질의 × 모드마다 생성 함수를 호출해 레코드를 만들고 `sink`에 즉시 넘긴다.

    질의의 `mode` 힌트가 허용하지 않는 조합과 `done`에 있는 조합은 건너뛴다. 생성 함수에는 `prepare_query`로
    정리한 질의를 넘기고, 제품 API가 거부할 입력이면 생성 함수를 부르지 않고 `outcome="rejected"` 레코드를 남긴다.
    생성 함수가 예외를 던지면 `keep_going=False`일 때 그대로 다시 던지고(그 전까지의 레코드는 이미 `sink`로 나갔다),
    `keep_going=True`일 때 `error`를 채운 레코드를 남기고 계속한다.
    """
    skip = set(done)
    references = reference_answers or {}
    pending = [
        (query, mode)
        for query in queries
        for mode in modes
        if query.supports_mode(mode) and (query.id, mode) not in skip
    ]
    records: list[dict[str, Any]] = []
    for index, (query, mode) in enumerate(pending, start=1):
        common = {
            "reference_answer": references.get(query.id),
            "model": model,
            "refusal_phrases": refusal_phrases,
        }
        try:
            prepared = prepare_query(query)
        except InputRejected as exc:
            record = build_answer_record(query, mode, rejection=str(exc), **common)
        else:
            started = clock()
            try:
                result = generators[mode](prepared)
            except Exception as exc:
                if not keep_going:
                    raise
                record = build_answer_record(
                    query,
                    mode,
                    error=f"{type(exc).__name__}: {exc}",
                    history_turns=len(prepared.history),
                    **common,
                )
            else:
                record = build_answer_record(
                    query,
                    mode,
                    result=result,
                    latency_ms=(clock() - started) * 1000.0,
                    history_turns=len(prepared.history),
                    **common,
                )
        if sink is not None:
            sink(record)
        records.append(record)
        if progress is not None:
            progress(index, len(pending), mode, query, record)
    return records


def _dedupe(texts: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(text for text in texts if text))


@dataclass
class CapturedContexts:
    texts: list[str] = field(default_factory=list)
    retrieval_modes: list[str] = field(default_factory=list)
    arxiv_ids: list[str] = field(default_factory=list)


@contextmanager
def capture_agent_contexts() -> Iterator[CapturedContexts]:
    """블록 안에서 에이전트 검색 도구가 LLM에 넘기는 청크 본문과 검색 방식을 모은다.

    도구 모듈의 `retrieve_contexts`를 기록용 래퍼로 잠시 바꿨다가 블록이 끝나면 되돌린다.
    프로세스 전역 교체이므로 한 번에 질의 하나씩 순서대로 실행할 때만 쓴다.
    """
    from src.core.agent import tools
    from src.core.rag_types import context_text

    original = tools.retrieve_contexts
    captured = CapturedContexts()

    def recording(query: str, **kwargs: Any):
        contexts, mode = original(query, **kwargs)
        captured.texts.extend(context_text(context) for context in contexts)
        captured.arxiv_ids.extend(str(context.get("arxiv_id") or "") for context in contexts)
        captured.retrieval_modes.append(mode)
        return contexts, mode

    tools.retrieve_contexts = recording
    try:
        yield captured
    finally:
        tools.retrieve_contexts = original


def _collect_stream(events: Iterable[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    parts: list[str] = []
    citations: list[dict[str, Any]] = []
    for event in events:
        if "chunk" in event:
            parts.append(event["chunk"])
        elif "citations" in event:
            citations = list(event["citations"])
    return "".join(parts), citations


def generate_agent_answer(query: EvalQuery) -> GenerationResult:
    """LangGraph 에이전트로 답한다. 질의의 `history`를 대화 이력으로 넘긴다.

    contexts는 검색 도구가 반환해 LLM에 넘긴 청크 본문(중복 제거), hit_arxiv_ids는 그 청크들의 논문이다.
    """
    from src.core.agent.chatbot import agent_search

    with capture_agent_contexts() as captured:
        result = agent_search(query.query, chat_history=list(query.history))
    modes = list(dict.fromkeys(captured.retrieval_modes))
    return GenerationResult(
        answer=str(result.get("answer") or ""),
        contexts=_dedupe(captured.texts),
        citations=list(result.get("citations") or []),
        retrieval_mode=",".join(modes) or None,
        hit_arxiv_ids=_dedupe(captured.arxiv_ids),
    )


def make_paper_chat_generator(
    *,
    repository: Any | None = None,
    retriever_factory: Callable[[Any], Any] | None = None,
    llm: Any | None = None,
) -> GenerateFn:
    """질의의 `paper_chat_arxiv_id`(`open_arxiv_id`, 없으면 `relevant_arxiv_ids[0]`) 논문의 상세 챗으로 답하는
    생성 함수를 만든다.

    질의의 `history`를 대화 이력으로 넘긴다. contexts는 상세 챗 프롬프트의 번호 출처(초록 + 검색 청크) 본문
    그대로다. 대상 논문이 없거나 DB에 없으면 LookupError.
    """

    cache: dict[str, Any] = {}

    def resolve_repository() -> Any:
        if repository is not None:
            return repository
        if "repository" not in cache:
            from src.integrations.paper_repository import PaperRepository

            cache["repository"] = PaperRepository()
        return cache["repository"]

    def generate(query: EvalQuery) -> GenerationResult:
        from src.core.agent.paper_chat import retrieve_paper_chat_sources, stream_paper_chat_answer

        repo = resolve_repository()
        if retriever_factory is None:
            from src.integrations.paper_retriever import PaperRetriever

            retriever = PaperRetriever(repository=repo)
        else:
            retriever = retriever_factory(repo)

        arxiv_id = query.paper_chat_arxiv_id
        if not arxiv_id:
            raise LookupError(f"{query.id}: 상세 챗 대상 논문(open_arxiv_id)이 없습니다.")
        paper = repo.get_paper(arxiv_id)
        if not paper:
            raise LookupError(f"papers에 {arxiv_id}가 없습니다.")
        retrieval = retrieve_paper_chat_sources(paper, query.query, retriever=retriever)
        answer, citations = _collect_stream(
            stream_paper_chat_answer(
                query.query, paper=paper, retrieval=retrieval, chat_history=list(query.history), llm=llm
            )
        )
        return GenerationResult(
            answer=answer,
            contexts=[str(source.get("text") or "") for source in retrieval.sources if source.get("text")],
            citations=citations,
            retrieval_mode=retrieval.retrieval_mode,
            arxiv_id=arxiv_id,
            hit_arxiv_ids=[arxiv_id],
        )

    return generate


def build_product_generators(modes: Sequence[str]) -> dict[str, GenerateFn]:
    generators: dict[str, GenerateFn] = {}
    if "agent" in modes:
        generators["agent"] = generate_agent_answer
    if "paper_chat" in modes:
        generators["paper_chat"] = make_paper_chat_generator()
    return generators
