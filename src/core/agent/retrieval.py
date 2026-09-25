"""제품 경로(에이전트 도구, 상세 챗)가 쓰는 검색 방식 선택."""

from __future__ import annotations

import logging
from typing import Any, Literal

from openai import OpenAIError

from src.shared import get_settings

logger = logging.getLogger(__name__)

RetrievalMode = Literal["hybrid", "lexical"]


def resolve_retrieval_mode(retriever: Any) -> RetrievalMode:
    """RETRIEVAL_MODE=hybrid이고 질의 임베딩에 쓸 키가 있을 때만 hybrid를 고른다."""
    if get_settings().retrieval_mode != "hybrid":
        return "lexical"
    embedding_client = getattr(retriever, "embedding_client", None)
    if embedding_client is None:
        return "lexical"
    is_available = getattr(embedding_client, "is_available", None)
    if callable(is_available) and not is_available():
        return "lexical"
    return "hybrid"


def retrieve_contexts(
    query: str,
    *,
    retriever: Any,
    limit: int,
    arxiv_id: str | None = None,
) -> tuple[list[dict], RetrievalMode]:
    """hybrid를 우선 시도하고, 키가 없거나 임베딩 호출이 실패하면 lexical로 내려간다."""
    if resolve_retrieval_mode(retriever) == "hybrid":
        try:
            contexts = retriever.search_paper_contexts_by_hybrid(query, arxiv_id=arxiv_id, limit=limit)
            return contexts, "hybrid"
        except OpenAIError as exc:
            logger.warning("질의 임베딩 실패로 lexical 검색으로 전환합니다: %s", type(exc).__name__)
    return retriever.search_paper_contexts(query, arxiv_id=arxiv_id, limit=limit), "lexical"
