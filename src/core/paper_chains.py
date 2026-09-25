from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.shared import get_runtime_openai_api_key, get_runtime_openai_model

from .models import PaperDetailDocument
from .prompts import KEY_FINDINGS_PROMPT, OVERVIEW_PROMPT
from .tracing import build_paper_key_findings_trace_config, build_paper_overview_trace_config


def _build_llm() -> ChatOpenAI:
    api_key = get_runtime_openai_api_key()
    model = get_runtime_openai_model()
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않아 논문 상세 문서를 생성할 수 없습니다.")

    kwargs = {
        "model": model,
        "api_key": api_key,
    }
    if not model.startswith("gpt-5"):
        kwargs["temperature"] = 0.2

    return ChatOpenAI(**kwargs)


def _compact_text(value: Any, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _normalize_paper_detail_input(paper: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(paper)

    nested_fulltext = normalized.get("fulltext")
    if isinstance(nested_fulltext, dict):
        if not normalized.get("sections"):
            normalized["sections"] = nested_fulltext.get("sections") or []
        if not normalized.get("fulltext_text"):
            normalized["fulltext_text"] = nested_fulltext.get("text") or nested_fulltext.get("fulltext_text") or ""

    if not normalized.get("fulltext_text"):
        normalized["fulltext_text"] = normalized.get("text") or ""

    if not normalized.get("sections"):
        normalized["sections"] = []

    return normalized


def _extract_author_names(authors: Any) -> list[str]:
    if not authors:
        return []

    names: list[str] = []
    for author in authors:
        if isinstance(author, str):
            author_name = author.strip()
            if author_name:
                names.append(author_name)
            continue
        if isinstance(author, dict):
            author_name = str(author.get("name") or "").strip()
            if author_name:
                names.append(author_name)
    return names


def _format_paper_metadata(paper: dict[str, Any]) -> str:
    authors = ", ".join(_extract_author_names(paper.get("authors"))) or "저자 미상"
    categories = paper.get("categories") or []
    category_text = ", ".join(str(value) for value in categories if value) or str(
        paper.get("primary_category") or "카테고리 미상"
    )

    lines = [
        f"arXiv ID: {paper.get('arxiv_id') or '정보 없음'}",
        f"제목: {paper.get('title') or '제목 없음'}",
        f"저자: {authors}",
        f"발행일: {paper.get('published_at') or paper.get('publishedAt') or '정보 없음'}",
        f"카테고리: {category_text}",
        f"PDF: {paper.get('pdf_url') or '정보 없음'}",
        f"초록: {_compact_text(paper.get('abstract', paper.get('summary', '')), max_chars=1600) or '정보 없음'}",
    ]
    return "\n".join(lines)


def _select_sections(sections: list[dict[str, Any]], *, max_sections: int = 8) -> list[dict[str, Any]]:
    if not sections:
        return []

    priority_keywords = (
        "abstract",
        "introduction",
        "problem",
        "method",
        "approach",
        "model",
        "experiment",
        "result",
        "evaluation",
        "discussion",
        "limitation",
        "conclusion",
    )

    prioritized: list[dict[str, Any]] = []
    remainder: list[dict[str, Any]] = []
    for section in sections:
        title = str(section.get("title") or "").lower()
        if any(keyword in title for keyword in priority_keywords):
            prioritized.append(section)
        else:
            remainder.append(section)

    selected = prioritized[:max_sections]
    if len(selected) < max_sections:
        selected.extend(remainder[: max_sections - len(selected)])
    return selected


def _format_paper_sections(paper: dict[str, Any], *, max_chars_per_section: int = 2200) -> str:
    sections = paper.get("sections") or []
    if isinstance(sections, list) and sections:
        selected_sections = _select_sections([section for section in sections if isinstance(section, dict)])
        parts: list[str] = []
        for index, section in enumerate(selected_sections, 1):
            title = str(section.get("title") or f"Section {index}").strip() or f"Section {index}"
            text = _compact_text(section.get("text"), max_chars=max_chars_per_section)
            if not text:
                continue
            parts.append(f"[섹션 {index}] {title}\n{text}")
        if parts:
            return "\n\n---\n\n".join(parts)

    fulltext = _compact_text(paper.get("fulltext_text"), max_chars=7000)
    if fulltext:
        return f"[본문]\n{fulltext}"

    return "본문 섹션 정보가 없습니다."


def has_paper_detail_context(paper: dict[str, Any]) -> bool:
    normalized = _normalize_paper_detail_input(paper)
    sections = normalized.get("sections")
    if isinstance(sections, list) and any(isinstance(section, dict) and section.get("text") for section in sections):
        return True
    return bool(str(normalized.get("fulltext_text") or "").strip())


def _extract_key_findings(text: str, *, max_items: int = 6) -> list[str]:
    findings: list[str] = []
    for raw_line in text.splitlines():
        cleaned = raw_line.strip()
        cleaned = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", cleaned)
        cleaned = " ".join(cleaned.split())
        if not cleaned:
            continue
        if len(cleaned) < 10:
            continue
        if not cleaned.endswith((".", "!", "?")):
            cleaned += "."
        findings.append(cleaned)

    deduped: list[str] = []
    seen: set[str] = set()
    for finding in findings:
        key = re.sub(r"\W+", "", finding).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped[:max_items]


def build_paper_overview(
    paper: dict[str, Any],
    *,
    runtime: str = "dev",
    user: str | None = None,
    quality_score: float | None = None,
) -> str:
    normalized = _normalize_paper_detail_input(paper)
    if not has_paper_detail_context(normalized):
        raise ValueError("상세 논문 설명을 생성하려면 본문 섹션 또는 fulltext_text가 필요합니다.")

    chain = OVERVIEW_PROMPT | _build_llm() | StrOutputParser()
    config = build_paper_overview_trace_config(
        runtime=runtime,
        user=user,
        quality_score=quality_score,
    )
    return chain.invoke(
        {
            "paper_metadata": _format_paper_metadata(normalized),
            "paper_sections": _format_paper_sections(normalized),
        },
        config=config,
    )


def build_paper_key_findings(
    paper: dict[str, Any],
    *,
    runtime: str = "dev",
    user: str | None = None,
    quality_score: float | None = None,
) -> list[str]:
    normalized = _normalize_paper_detail_input(paper)
    if not has_paper_detail_context(normalized):
        raise ValueError("상세 논문 설명을 생성하려면 본문 섹션 또는 fulltext_text가 필요합니다.")

    chain = KEY_FINDINGS_PROMPT | _build_llm() | StrOutputParser()
    config = build_paper_key_findings_trace_config(
        runtime=runtime,
        user=user,
        quality_score=quality_score,
    )
    response = chain.invoke(
        {
            "paper_metadata": _format_paper_metadata(normalized),
            "paper_sections": _format_paper_sections(normalized),
        },
        config=config,
    )
    return _extract_key_findings(response)


def analyze_paper_detail(
    paper: dict[str, Any],
    *,
    generated_at: datetime | None = None,
    runtime: str = "dev",
    user: str | None = None,
) -> PaperDetailDocument:
    normalized = _normalize_paper_detail_input(paper)
    return PaperDetailDocument(
        arxiv_id=str(normalized.get("arxiv_id") or ""),
        title=str(normalized.get("title") or "제목 없음"),
        overview=build_paper_overview(normalized, runtime=runtime, user=user),
        key_findings=build_paper_key_findings(normalized, runtime=runtime, user=user),
        generated_at=generated_at or datetime.now(UTC),
    )
