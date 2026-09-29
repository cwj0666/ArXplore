from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from arxplore.domain.chunks import StoredChunk
from arxplore.domain.papers import Paper
from arxplore.domain.retrieval import RetrievalContext, RetrievalResult
from arxplore.llm.citations import CitationSource, optional_int, optional_str, paper_url, verify_numbered_citations
from arxplore.llm.events import AnswerChunk, AnswerCitations, AnswerEvent
from arxplore.llm.prompts.paper_chat import PAPER_CHAT_PROMPT
from arxplore.llm.tracing import Tracer

PAPER_CHAT_CONTEXT_LIMIT = 5
ABSTRACT_SECTION_TITLE = "Abstract"
FIRST_CHUNKS_MODE = "first_chunks"
NO_EXCERPTS_TEXT = "제공된 발췌문이 없습니다."
MISSING_FIELD = "-"

type ChatHistory = Sequence[tuple[str, str]]


class ContextRetriever(Protocol):
    async def retrieve_contexts(
        self, query: str, *, limit: int = ..., arxiv_id: str | None = None
    ) -> RetrievalResult: ...


class ChunkReader(Protocol):
    async def list_chunks(self, arxiv_id: str, *, limit: int) -> Sequence[StoredChunk]: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperChatSource:
    arxiv_id: str
    title: str
    url: str
    section_title: str | None
    chunk_id: int | None
    text: str

    def citation_source(self) -> CitationSource:
        return CitationSource(
            arxiv_id=self.arxiv_id,
            title=self.title,
            url=self.url,
            section_title=self.section_title,
            chunk_id=self.chunk_id,
        )


@dataclass(frozen=True, slots=True)
class PaperChatRetrieval:
    sources: tuple[PaperChatSource, ...]
    retrieval_mode: str


@dataclass(frozen=True, slots=True)
class _Excerpt:
    chunk_id: int | None
    section_title: str | None
    text: str


def _context_excerpt(context: RetrievalContext) -> _Excerpt:
    hit = context.hit
    return _Excerpt(chunk_id=hit.chunk_id, section_title=hit.section_title, text=context.text or hit.chunk_text)


def _chunk_excerpt(chunk: StoredChunk) -> _Excerpt:
    return _Excerpt(chunk_id=chunk.chunk_id, section_title=chunk.section_title, text=chunk.chunk_text)


def build_paper_chat_sources(paper: Paper, excerpts: Sequence[_Excerpt]) -> tuple[PaperChatSource, ...]:
    arxiv_id = str(paper.arxiv_id or "")
    title = str(paper.title or "")
    url = paper_url(arxiv_id, paper.pdf_url)
    sources: list[PaperChatSource] = []
    abstract = str(paper.abstract or "").strip()
    if abstract:
        sources.append(
            PaperChatSource(
                arxiv_id=arxiv_id,
                title=title,
                url=url,
                section_title=ABSTRACT_SECTION_TITLE,
                chunk_id=None,
                text=abstract,
            )
        )
    seen_chunk_ids: set[int] = set()
    for excerpt in excerpts:
        chunk_id = optional_int(excerpt.chunk_id)
        if chunk_id is not None:
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
        text = str(excerpt.text or "").strip()
        if not text:
            continue
        sources.append(
            PaperChatSource(
                arxiv_id=arxiv_id,
                title=title,
                url=url,
                section_title=optional_str(excerpt.section_title),
                chunk_id=chunk_id,
                text=text,
            )
        )
    return tuple(sources)


async def retrieve_paper_chat_sources(
    paper: Paper,
    question: str,
    *,
    retriever: ContextRetriever,
    chunks: ChunkReader,
    limit: int = PAPER_CHAT_CONTEXT_LIMIT,
) -> PaperChatRetrieval:
    arxiv_id = str(paper.arxiv_id or "")
    result = await retriever.retrieve_contexts(question, limit=limit, arxiv_id=arxiv_id)
    excerpts = [_context_excerpt(context) for context in result.contexts]
    retrieval_mode: str = result.mode
    if not excerpts:
        excerpts = [_chunk_excerpt(chunk) for chunk in await chunks.list_chunks(arxiv_id, limit=limit)]
        retrieval_mode = FIRST_CHUNKS_MODE
    return PaperChatRetrieval(sources=build_paper_chat_sources(paper, excerpts), retrieval_mode=retrieval_mode)


def format_paper_chat_context(sources: Sequence[PaperChatSource]) -> str:
    if not sources:
        return NO_EXCERPTS_TEXT
    blocks = []
    for index, source in enumerate(sources, 1):
        chunk_id = source.chunk_id if source.chunk_id is not None else MISSING_FIELD
        header = f"[{index}] 섹션: {source.section_title or MISSING_FIELD} | chunk_id: {chunk_id}"
        blocks.append(f"{header}\n{source.text or ''}")
    return "\n\n".join(blocks)


async def stream_paper_chat_answer(
    question: str,
    *,
    paper: Paper,
    retrieval: PaperChatRetrieval,
    history: ChatHistory,
    model: BaseChatModel,
    tracer: Tracer,
) -> AsyncGenerator[AnswerEvent]:
    chain = PAPER_CHAT_PROMPT | model | StrOutputParser()
    config = tracer.config(
        "paper_chat",
        metadata={
            "arxiv_id": paper.arxiv_id,
            "retrieval_mode": retrieval.retrieval_mode,
            "source_count": len(retrieval.sources),
        },
    )
    inputs = {
        "paper_title": paper.title or "",
        "arxiv_id": paper.arxiv_id or "",
        "context": format_paper_chat_context(retrieval.sources),
        "question": question,
        "chat_history": list(history),
    }
    parts: list[str] = []
    with tracer.scope():
        async for text in chain.astream(inputs, config=config):
            if not text:
                continue
            parts.append(text)
            yield AnswerChunk(text)
    sources = [source.citation_source() for source in retrieval.sources]
    yield AnswerCitations.of(verify_numbered_citations("".join(parts), sources))
