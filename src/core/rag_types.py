"""RAG 응답 계층이 공유하는 citation shape와 변환 함수."""

from __future__ import annotations

from typing import Any, TypedDict


class Citation(TypedDict):
    arxiv_id: str
    title: str
    url: str
    section_title: str | None
    chunk_id: int | None
    in_answer: bool


UNTITLED = "제목 없음"


def paper_url(arxiv_id: str, pdf_url: str | None = None) -> str:
    normalized_pdf_url = str(pdf_url or "").strip()
    if normalized_pdf_url:
        return normalized_pdf_url
    return f"https://arxiv.org/abs/{arxiv_id}"


def context_title(context: dict[str, Any]) -> str:
    return str(context.get("paper_title") or context.get("title") or UNTITLED)


def context_text(context: dict[str, Any]) -> str:
    return str(
        context.get("context_text") or context.get("chunk_text") or context.get("text") or context.get("abstract") or ""
    )


def optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_str(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def source_from_context(context: dict[str, Any]) -> dict[str, Any]:
    """retrieval context 하나를 citation 후보(출처) 형태로 줄인다."""
    arxiv_id = str(context.get("arxiv_id") or "")
    return {
        "arxiv_id": arxiv_id,
        "title": context_title(context),
        "url": paper_url(arxiv_id, context.get("pdf_url")),
        "section_title": optional_str(context.get("section_title")),
        "chunk_id": optional_int(context.get("chunk_id")),
    }


def to_citation(source: dict[str, Any], *, in_answer: bool) -> Citation:
    return {
        "arxiv_id": str(source.get("arxiv_id") or ""),
        "title": str(source.get("title") or UNTITLED),
        "url": str(source.get("url") or ""),
        "section_title": optional_str(source.get("section_title")),
        "chunk_id": optional_int(source.get("chunk_id")),
        "in_answer": in_answer,
    }
