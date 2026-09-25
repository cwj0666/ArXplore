"""도구 hit 기록과 답변 인용 사후 검증."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlsplit

from src.core.rag_types import Citation, to_citation

logger = logging.getLogger(__name__)

_tool_hits: ContextVar[list[dict[str, Any]] | None] = ContextVar("agent_tool_hits", default=None)

_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\[\]]+)\]\((https?://[^)\s]+)\)")
_ARXIV_PATH_PATTERN = re.compile(r"^/(?:abs|pdf)/([^\s?#)/]+(?:/[^\s?#)/]+)?)/?$", re.IGNORECASE)
_ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}
_VERSION_SUFFIX = re.compile(r"v\d+$")
_SOURCE_REF_PATTERN = re.compile(r"\[(\d+(?:\s*[,，]\s*\d+)*)\](?!\()")


@contextmanager
def collect_tool_hits() -> Iterator[list[dict[str, Any]]]:
    """이 블록 안에서 도구가 반환한 hit을 모은다.

    LangGraph는 도구를 컨텍스트를 복사한 스레드에서 실행하므로, ContextVar에는 같은 list 객체를
    넣어 두고 도구 쪽에서 append한다.
    """
    hits: list[dict[str, Any]] = []
    token = _tool_hits.set(hits)
    try:
        yield hits
    finally:
        _tool_hits.reset(token)


def record_tool_hits(sources: Iterable[dict[str, Any]]) -> None:
    hits = _tool_hits.get()
    if hits is None:
        return
    hits.extend(sources)


def extract_markdown_links(text: str) -> list[tuple[str, str]]:
    return [(match.group(1).strip(), match.group(2).strip()) for match in _MARKDOWN_LINK_PATTERN.finditer(text or "")]


def normalize_arxiv_id(value: str) -> str:
    normalized = str(value or "").strip()
    if normalized.lower().endswith(".pdf"):
        normalized = normalized[:-4]
    return _VERSION_SUFFIX.sub("", normalized)


def arxiv_id_from_url(url: str) -> str | None:
    try:
        parts = urlsplit(str(url or "").strip())
    except ValueError:
        return None
    if parts.scheme.lower() not in {"http", "https"} or (parts.hostname or "").lower() not in _ARXIV_HOSTS:
        return None
    match = _ARXIV_PATH_PATTERN.match(parts.path or "")
    if not match:
        return None
    return normalize_arxiv_id(match.group(1))


def _normalize_url(url: str) -> str:
    return str(url or "").strip().rstrip("/").lower()


def dedupe_sources(sources: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, int | None]] = set()
    unique: list[dict[str, Any]] = []
    for source in sources:
        key = (normalize_arxiv_id(source.get("arxiv_id") or ""), source.get("chunk_id"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def verify_agent_citations(answer: str, hits: Iterable[dict[str, Any]]) -> list[Citation]:
    """답변의 마크다운 링크를 도구 hit과 대조해 citation 목록을 만든다.

    - 답변에 링크가 있으면 링크 URL(또는 링크 속 arXiv ID)과 맞는 hit만 `in_answer=True`로 돌려준다.
    - 답변에 링크가 없으면 사용한 hit 전체를 `in_answer=False`로 돌려준다.
    - hit에 없는 링크는 경고로 남기고 답변은 고치지 않는다.
    """
    unique_hits = dedupe_sources(hits)
    links = extract_markdown_links(answer)
    if not links:
        return [to_citation(hit, in_answer=False) for hit in unique_hits]

    hit_urls = {_normalize_url(hit.get("url") or "") for hit in unique_hits}
    hit_ids = {normalize_arxiv_id(hit.get("arxiv_id") or "") for hit in unique_hits}
    linked_urls: set[str] = set()
    linked_ids: set[str] = set()
    for title, url in links:
        normalized_url = _normalize_url(url)
        linked_id = arxiv_id_from_url(url)
        linked_urls.add(normalized_url)
        if linked_id:
            linked_ids.add(linked_id)
        if normalized_url not in hit_urls and (not linked_id or linked_id not in hit_ids):
            logger.warning("도구 결과에 없는 링크가 답변에 있습니다: title=%r url=%s", title, url)

    citations: list[Citation] = []
    for hit in unique_hits:
        hit_id = normalize_arxiv_id(hit.get("arxiv_id") or "")
        if _normalize_url(hit.get("url") or "") in linked_urls or (hit_id and hit_id in linked_ids):
            citations.append(to_citation(hit, in_answer=True))
    return citations


def extract_source_refs(answer: str) -> list[int]:
    """`[1]`, `[2, 3]` 형태의 번호 인용을 등장 순서대로 꺼낸다. 마크다운 링크 텍스트는 제외한다."""
    refs: list[int] = []
    for match in _SOURCE_REF_PATTERN.finditer(answer or ""):
        for part in re.split(r"[,，]", match.group(1)):
            number = int(part.strip())
            if number not in refs:
                refs.append(number)
    return refs


def verify_numbered_citations(answer: str, sources: list[dict[str, Any]]) -> list[Citation]:
    """번호로 인용하는 상세 챗용. 번호가 있으면 그 출처만, 없으면 전체를 `in_answer=False`로 돌려준다."""
    refs = extract_source_refs(answer)
    referenced = {number for number in refs if 1 <= number <= len(sources)}
    unknown = [number for number in refs if number not in referenced]
    if unknown:
        logger.warning("컨텍스트에 없는 출처 번호가 답변에 있습니다: %s", unknown)
    if not referenced:
        return [to_citation(source, in_answer=False) for source in sources]
    return [to_citation(source, in_answer=True) for index, source in enumerate(sources, 1) if index in referenced]
