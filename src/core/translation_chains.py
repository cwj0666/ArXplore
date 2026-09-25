"""한국어 번역 및 상세 요약 체인을 담당하는 모듈.

두 체인의 용도:
- translate_chunk(): RAG 근거 chunk 단위 번역. 짧은 구절 단위 입력.
- build_summary(): LangGraph 기반 논문 상세 요약. 섹션별 1차 해설 후 최종 본문을 생성.
"""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.shared import get_runtime_openai_api_key, get_runtime_openai_model

from .prompts import TRANSLATION_PROMPT
from .summary_graph import generate_summary_via_graph
from .tracing import build_translation_trace_config

_CHUNK_TEXT_MAX_CHARS = 2000


def _build_llm(temperature: float = 0.1) -> ChatOpenAI:
    api_key = get_runtime_openai_api_key()
    model = get_runtime_openai_model()
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않아 한국어 번역/요약 체인을 실행할 수 없습니다.")

    kwargs = {
        "model": model,
        "api_key": api_key,
    }
    if not model.startswith("gpt-5"):
        kwargs["temperature"] = temperature

    return ChatOpenAI(
        **kwargs,
    )


def _format_authors(authors: Any) -> str:
    if not authors:
        return "저자 미상"
    if isinstance(authors, str):
        return authors.strip() or "저자 미상"

    names: list[str] = []
    if isinstance(authors, list):
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
    return ", ".join(names) or "저자 미상"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[이하 생략]"


def translate_chunk(
    chunk_text: str,
    *,
    runtime: str = "dev",
    user: str | None = None,
    quality_score: float | None = None,
) -> str:
    """RAG 근거 chunk를 한국어로 번역한다."""
    text = _truncate(str(chunk_text or "").strip(), _CHUNK_TEXT_MAX_CHARS)
    if not text:
        return ""

    chain = TRANSLATION_PROMPT | _build_llm(temperature=0) | StrOutputParser()
    config = build_translation_trace_config(
        runtime=runtime,
        user=user,
        quality_score=quality_score,
    )
    return chain.invoke({"chunk_text": text}, config=config)


def build_summary(
    *,
    title: str,
    authors: Any,
    text: str,
    sections: list[dict[str, Any]] | None = None,
    runtime: str = "dev",
    user: str | None = None,
    quality_score: float | None = None,
) -> str:
    """논문 전체 본문을 입력받아 LangGraph 기반 한국어 구조화 요약을 생성한다."""
    return generate_summary_via_graph(
        title=title or "제목 없음",
        authors=_format_authors(authors),
        text=str(text or "").strip(),
        sections=sections,
        runtime=runtime,
        user=user,
        quality_score=quality_score,
    )
