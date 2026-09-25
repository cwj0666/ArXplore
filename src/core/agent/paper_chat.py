"""논문 상세 페이지 챗: 해당 논문 안에서 질의 검색한 발췌문으로 답한다."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.core.prompts.paper_chat import PAPER_CHAT_PROMPT
from src.core.rag_types import context_text, optional_int, optional_str, paper_url
from src.core.tracing import build_paper_chat_trace_config

from .citations import verify_numbered_citations
from .llm import build_chat_llm
from .retrieval import retrieve_contexts

PAPER_CHAT_CONTEXT_LIMIT = 5
ABSTRACT_SECTION_TITLE = "Abstract"
FIRST_CHUNKS_MODE = "first_chunks"


@dataclass(frozen=True)
class PaperChatRetrieval:
    sources: list[dict[str, Any]]
    retrieval_mode: str


def retrieve_paper_chat_sources(
    paper: dict[str, Any],
    question: str,
    *,
    retriever: Any,
    limit: int = PAPER_CHAT_CONTEXT_LIMIT,
) -> PaperChatRetrieval:
    """초록을 1번 출처로 두고, 질의로 이 논문 안을 검색한 청크를 뒤에 붙인다.

    검색 결과가 비면 논문의 앞 `limit`개 청크로 대신하고 `retrieval_mode="first_chunks"`로 표시한다.
    """
    arxiv_id = str(paper.get("arxiv_id") or "")
    title = str(paper.get("title") or "")
    url = paper_url(arxiv_id, paper.get("pdf_url"))

    contexts, retrieval_mode = retrieve_contexts(question, retriever=retriever, arxiv_id=arxiv_id, limit=limit)
    if not contexts:
        contexts = retriever.repository.list_paper_chunks(arxiv_id, limit=limit)
        retrieval_mode = FIRST_CHUNKS_MODE

    sources: list[dict[str, Any]] = []
    abstract = str(paper.get("abstract") or "").strip()
    if abstract:
        sources.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "url": url,
                "section_title": ABSTRACT_SECTION_TITLE,
                "chunk_id": None,
                "text": abstract,
            }
        )

    seen_chunk_ids: set[int] = set()
    for context in contexts:
        chunk_id = optional_int(context.get("chunk_id"))
        if chunk_id is not None:
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
        text = context_text(context).strip()
        if not text:
            continue
        sources.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "url": url,
                "section_title": optional_str(context.get("section_title")),
                "chunk_id": chunk_id,
                "text": text,
            }
        )

    return PaperChatRetrieval(sources=sources, retrieval_mode=retrieval_mode)


def format_paper_chat_context(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "제공된 발췌문이 없습니다."
    blocks = []
    for index, source in enumerate(sources, 1):
        chunk_id = source.get("chunk_id")
        header = (
            f"[{index}] 섹션: {source.get('section_title') or '-'}"
            f" | chunk_id: {chunk_id if chunk_id is not None else '-'}"
        )
        blocks.append(f"{header}\n{source.get('text') or ''}")
    return "\n\n".join(blocks)


def stream_paper_chat_answer(
    question: str,
    *,
    paper: dict[str, Any],
    retrieval: PaperChatRetrieval,
    chat_history: list[tuple[str, str]] | None = None,
    llm: BaseChatModel | None = None,
    runtime: str | None = None,
    user: str | None = None,
) -> Iterator[dict[str, Any]]:
    """`{"chunk": str}`를 순서대로 내보내고 마지막에 `{"citations": [...]}`를 한 번 내보낸다."""
    chain = PAPER_CHAT_PROMPT | (llm or build_chat_llm()) | StrOutputParser()
    trace_config = build_paper_chat_trace_config(
        runtime=runtime,
        user=user,
        extra_metadata={
            "arxiv_id": paper.get("arxiv_id"),
            "retrieval_mode": retrieval.retrieval_mode,
            "source_count": len(retrieval.sources),
        },
    )
    inputs = {
        "paper_title": paper.get("title") or "",
        "arxiv_id": paper.get("arxiv_id") or "",
        "context": format_paper_chat_context(retrieval.sources),
        "question": question,
        "chat_history": list(chat_history or []),
    }

    parts: list[str] = []
    for text in chain.stream(inputs, config=trace_config):
        if not text:
            continue
        parts.append(text)
        yield {"chunk": text}

    yield {"citations": verify_numbered_citations("".join(parts), retrieval.sources)}
