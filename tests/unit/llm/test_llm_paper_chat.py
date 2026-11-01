from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
from langchain_core.prompts import SystemMessagePromptTemplate

from arxplore.config import RetrievalMode
from arxplore.domain.chunks import StoredChunk
from arxplore.domain.papers import Paper
from arxplore.domain.retrieval import RetrievalContext, RetrievalHit, RetrievalResult
from arxplore.llm.events import AnswerChunk, AnswerCitations, collect_answer
from arxplore.llm.models import ChatModelFactory
from arxplore.llm.paper_chat import (
    ABSTRACT_SECTION_TITLE,
    FIRST_CHUNKS_MODE,
    PAPER_CHAT_CONTEXT_LIMIT,
    PaperChatRetrieval,
    format_paper_chat_context,
    retrieve_paper_chat_sources,
    stream_paper_chat_answer,
)
from arxplore.llm.prompts.paper_chat import PAPER_CHAT_PROMPT
from arxplore.llm.tracing import Tracer

PAPER = Paper(
    arxiv_id="2401.00001",
    title="Direct Preference Optimization Revisited",
    authors=["Ada"],
    abstract="We revisit DPO.",
    primary_category="cs.CL",
    categories=["cs.CL"],
    pdf_url=None,
    published_at=None,
    updated_at=None,
    upvotes=0,
    github_url=None,
    github_stars=None,
    citation_count=None,
)


def context(
    chunk_id: int, section: str = "3 Method", text: str | None = None, chunk_text: str = ""
) -> RetrievalContext:
    hit = RetrievalHit(
        chunk_id=chunk_id,
        arxiv_id=PAPER.arxiv_id,
        chunk_index=chunk_id,
        paper_title=PAPER.title,
        paper_abstract=PAPER.abstract,
        chunk_text=chunk_text or f"chunk {chunk_id}",
        section_title=section,
        content_role="body",
        method="hybrid",
        score=1.0,
        snippet="",
    )
    return RetrievalContext(hit=hit, chunks=(), text=f"context {chunk_id}" if text is None else text)


@dataclass
class FakeRetriever:
    contexts: Sequence[RetrievalContext] = ()
    mode: RetrievalMode = "hybrid"
    calls: list[tuple[str, int, str | None]] = field(default_factory=list)

    async def retrieve_contexts(self, query: str, *, limit: int = 5, arxiv_id: str | None = None) -> RetrievalResult:
        self.calls.append((query, limit, arxiv_id))
        return RetrievalResult(contexts=tuple(self.contexts), mode=self.mode)


@dataclass
class FakeChunks:
    chunks: Sequence[StoredChunk] = ()
    calls: list[tuple[str, int]] = field(default_factory=list)

    async def list_chunks(self, arxiv_id: str, *, limit: int) -> Sequence[StoredChunk]:
        self.calls.append((arxiv_id, limit))
        return self.chunks


def stored_chunk(chunk_id: int, text: str, section: str | None) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        arxiv_id=PAPER.arxiv_id,
        chunk_index=chunk_id,
        chunk_text=text,
        section_title=section,
        token_count=1,
        metadata={},
        updated_at=None,
    )


async def test_search_is_scoped_to_the_paper_and_the_abstract_comes_first() -> None:
    retriever = FakeRetriever(contexts=[context(10), context(11, "4 Experiments")])

    result = await retrieve_paper_chat_sources(PAPER, "what is the loss?", retriever=retriever, chunks=FakeChunks())

    assert retriever.calls == [("what is the loss?", PAPER_CHAT_CONTEXT_LIMIT, "2401.00001")]
    assert result.retrieval_mode == "hybrid"
    assert [source.section_title for source in result.sources] == [ABSTRACT_SECTION_TITLE, "3 Method", "4 Experiments"]
    assert (result.sources[0].text, result.sources[0].chunk_id) == ("We revisit DPO.", None)
    assert [source.text for source in result.sources[1:]] == ["context 10", "context 11"]
    assert {source.url for source in result.sources} == {"https://arxiv.org/abs/2401.00001"}


async def test_lexical_mode_is_reported() -> None:
    result = await retrieve_paper_chat_sources(
        PAPER, "q", retriever=FakeRetriever(contexts=[context(10)], mode="lexical"), chunks=FakeChunks()
    )

    assert result.retrieval_mode == "lexical"


async def test_empty_search_falls_back_to_the_first_chunks() -> None:
    chunks = FakeChunks(chunks=[stored_chunk(1, "intro text", "1 Introduction"), stored_chunk(2, "more", None)])

    result = await retrieve_paper_chat_sources(PAPER, "한국어 질문", retriever=FakeRetriever(), chunks=chunks)

    assert chunks.calls == [("2401.00001", PAPER_CHAT_CONTEXT_LIMIT)]
    assert result.retrieval_mode == FIRST_CHUNKS_MODE
    assert [(source.chunk_id, source.section_title, source.text) for source in result.sources[1:]] == [
        (1, "1 Introduction", "intro text"),
        (2, None, "more"),
    ]


async def test_duplicate_and_empty_contexts_are_dropped_and_chunk_text_is_the_fallback() -> None:
    contexts = [context(10), context(10), context(11, text="", chunk_text="  "), context(12, text="", chunk_text="raw")]

    result = await retrieve_paper_chat_sources(
        PAPER, "q", retriever=FakeRetriever(contexts=contexts), chunks=FakeChunks()
    )

    assert [(source.chunk_id, source.text) for source in result.sources] == [
        (None, "We revisit DPO."),
        (10, "context 10"),
        (12, "raw"),
    ]


async def test_paper_without_abstract_has_no_abstract_source() -> None:
    paper = replace(PAPER, abstract="")

    result = await retrieve_paper_chat_sources(
        paper, "q", retriever=FakeRetriever(contexts=[context(10)]), chunks=FakeChunks()
    )

    assert [source.chunk_id for source in result.sources] == [10]


async def test_context_format_numbers_sources_with_section_and_chunk_id() -> None:
    result = await retrieve_paper_chat_sources(
        PAPER, "q", retriever=FakeRetriever(contexts=[context(10)]), chunks=FakeChunks()
    )

    assert format_paper_chat_context(result.sources) == (
        "[1] 섹션: Abstract | chunk_id: -\nWe revisit DPO.\n\n[2] 섹션: 3 Method | chunk_id: 10\ncontext 10"
    )
    assert format_paper_chat_context(()) == "제공된 발췌문이 없습니다."


async def _retrieval() -> PaperChatRetrieval:
    return await retrieve_paper_chat_sources(
        PAPER, "q", retriever=FakeRetriever(contexts=[context(10), context(11)]), chunks=FakeChunks()
    )


async def test_answer_streams_chunks_then_citations_for_referenced_sources(
    fake_openai: Any, models: ChatModelFactory, tracer: Tracer
) -> None:
    fake_openai.chat_handler = lambda request: "손실 함수는 [2]에 나옵니다."

    events = [
        event
        async for event in stream_paper_chat_answer(
            "q", paper=PAPER, retrieval=await _retrieval(), history=[], model=models.create(), tracer=tracer
        )
    ]

    assert all(isinstance(event, AnswerChunk) for event in events[:-1])
    assert len(events) > 2
    assert "".join(event.text for event in events if isinstance(event, AnswerChunk)) == "손실 함수는 [2]에 나옵니다."
    last = events[-1]
    assert isinstance(last, AnswerCitations)
    assert [citation.to_payload() for citation in last.citations] == [
        {
            "arxiv_id": "2401.00001",
            "title": PAPER.title,
            "url": "https://arxiv.org/abs/2401.00001",
            "section_title": "3 Method",
            "chunk_id": 10,
            "in_answer": True,
        }
    ]


async def test_answer_request_carries_history_and_the_numbered_excerpts(
    fake_openai: Any, models: ChatModelFactory, tracer: Tracer
) -> None:
    await collect_answer(
        stream_paper_chat_answer(
            "질문입니다",
            paper=PAPER,
            retrieval=await _retrieval(),
            history=[("user", "이전 질문"), ("assistant", "이전 답변")],
            model=models.create(temperature=0.0),
            tracer=tracer,
        )
    )

    (request,) = fake_openai.requests_for("/v1/chat/completions")
    messages = request.body["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert messages[3]["content"].startswith(
        "## 논문\n제목: Direct Preference Optimization Revisited\narXiv ID: 2401.00001"
    )
    assert "[2] 섹션: 3 Method | chunk_id: 10\ncontext 10" in messages[3]["content"]
    assert messages[3]["content"].endswith("## 질문\n질문입니다")
    assert (request.body["model"], request.body["temperature"], request.body["stream"]) == ("gpt-4o", 0.0, True)


@pytest.mark.parametrize(
    "question",
    ["What loss does the paper optimize?", "이 논문은 어떤 손실 함수를 최적화하나요?"],
)
async def test_system_message_asks_for_the_question_language_for_any_question(
    question: str, fake_openai: Any, models: ChatModelFactory, tracer: Tracer
) -> None:
    await collect_answer(
        stream_paper_chat_answer(
            question, paper=PAPER, retrieval=await _retrieval(), history=[], model=models.create(), tracer=tracer
        )
    )

    template = PAPER_CHAT_PROMPT.messages[0]
    assert isinstance(template, SystemMessagePromptTemplate)
    (request,) = fake_openai.requests_for("/v1/chat/completions")
    system, user = request.body["messages"]
    assert system["role"] == "system"
    assert system["content"] == template.prompt.template
    assert "사용자의 마지막 질문과 같은 언어로" in system["content"]
    assert "영어 질문에는 영어로" in system["content"]
    assert "한국어로 답할 때는 정중한 존댓말" in system["content"]
    assert user["content"].endswith(f"## 질문\n{question}")


async def test_unreferenced_and_out_of_range_answers_return_every_source_unverified(
    fake_openai: Any, models: ChatModelFactory, tracer: Tracer
) -> None:
    for reply in ("번호 없는 답변입니다.", "근거 [9]"):
        fake_openai.chat_handler = lambda request, text=reply: text
        answer = await collect_answer(
            stream_paper_chat_answer(
                "q", paper=PAPER, retrieval=await _retrieval(), history=[], model=models.create(), tracer=tracer
            )
        )

        assert [(c.section_title, c.chunk_id, c.in_answer) for c in answer.citations] == [
            ("Abstract", None, False),
            ("3 Method", 10, False),
            ("3 Method", 11, False),
        ]
