import logging
import re
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Self
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

UNTITLED = "제목 없음"
ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"
MARKDOWN_LINK = re.compile(r"\[([^\[\]]+)\]\((https?://[^)\s]+)\)")
ARXIV_PATH = re.compile(r"^/(?:abs|pdf)/([^\s?#)/]+(?:/[^\s?#)/]+)?)/?$", re.IGNORECASE)
ARXIV_HOSTS = frozenset({"arxiv.org", "www.arxiv.org", "export.arxiv.org"})
ARXIV_LINK_SCHEMES = frozenset({"http", "https"})
VERSION_SUFFIX = re.compile(r"v\d+$")
PDF_SUFFIX = ".pdf"
SOURCE_REFERENCE = re.compile(r"\[(\d+(?:\s*[,\uff0c]\s*\d+)*)\](?!\()")
REFERENCE_SEPARATOR = re.compile(r"[,\uff0c]")


def optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def optional_str(value: Any) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def paper_url(arxiv_id: str, pdf_url: str | None = None) -> str:
    normalized = str(pdf_url or "").strip()
    return normalized or ARXIV_ABS_URL.format(arxiv_id=arxiv_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class CitationSource:
    arxiv_id: str
    title: str
    url: str
    section_title: str | None = None
    chunk_id: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Citation:
    arxiv_id: str
    title: str
    url: str
    section_title: str | None
    chunk_id: int | None
    in_answer: bool

    @classmethod
    def from_source(cls, source: CitationSource, *, in_answer: bool) -> Self:
        return cls(
            arxiv_id=str(source.arxiv_id or ""),
            title=str(source.title or UNTITLED),
            url=str(source.url or ""),
            section_title=optional_str(source.section_title),
            chunk_id=optional_int(source.chunk_id),
            in_answer=in_answer,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "url": self.url,
            "section_title": self.section_title,
            "chunk_id": self.chunk_id,
            "in_answer": self.in_answer,
        }


_tool_hits: ContextVar[list[CitationSource] | None] = ContextVar("arxplore_agent_tool_hits", default=None)


@contextmanager
def collect_tool_hits() -> Iterator[list[CitationSource]]:
    hits: list[CitationSource] = []
    token = _tool_hits.set(hits)
    try:
        yield hits
    finally:
        _tool_hits.reset(token)


def record_tool_hits(sources: Iterable[CitationSource]) -> None:
    hits = _tool_hits.get()
    if hits is not None:
        hits.extend(sources)


def extract_markdown_links(text: str) -> list[tuple[str, str]]:
    return [(match.group(1).strip(), match.group(2).strip()) for match in MARKDOWN_LINK.finditer(text or "")]


def normalize_cited_arxiv_id(value: str) -> str:
    normalized = str(value or "").strip()
    if normalized.lower().endswith(PDF_SUFFIX):
        normalized = normalized[: -len(PDF_SUFFIX)]
    return VERSION_SUFFIX.sub("", normalized)


def arxiv_id_from_url(url: str) -> str | None:
    try:
        parts = urlsplit(str(url or "").strip())
    except ValueError:
        return None
    if parts.scheme.lower() not in ARXIV_LINK_SCHEMES or (parts.hostname or "").lower() not in ARXIV_HOSTS:
        return None
    match = ARXIV_PATH.match(parts.path or "")
    if not match:
        return None
    return normalize_cited_arxiv_id(match.group(1))


def _normalize_url(url: str) -> str:
    return str(url or "").strip().rstrip("/").lower()


def dedupe_sources(sources: Iterable[CitationSource]) -> list[CitationSource]:
    seen: set[tuple[str, int | None]] = set()
    unique: list[CitationSource] = []
    for source in sources:
        key = (normalize_cited_arxiv_id(source.arxiv_id or ""), source.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def verify_agent_citations(answer: str, hits: Iterable[CitationSource]) -> list[Citation]:
    unique_hits = dedupe_sources(hits)
    links = extract_markdown_links(answer)
    if not links:
        return [Citation.from_source(hit, in_answer=False) for hit in unique_hits]

    hit_urls = {_normalize_url(hit.url) for hit in unique_hits}
    hit_ids = {normalize_cited_arxiv_id(hit.arxiv_id) for hit in unique_hits}
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
        hit_id = normalize_cited_arxiv_id(hit.arxiv_id)
        if _normalize_url(hit.url) in linked_urls or (hit_id and hit_id in linked_ids):
            citations.append(Citation.from_source(hit, in_answer=True))
    return citations


def extract_source_refs(answer: str) -> list[int]:
    refs: list[int] = []
    for match in SOURCE_REFERENCE.finditer(answer or ""):
        for part in REFERENCE_SEPARATOR.split(match.group(1)):
            number = int(part.strip())
            if number not in refs:
                refs.append(number)
    return refs


def verify_numbered_citations(answer: str, sources: Sequence[CitationSource]) -> list[Citation]:
    refs = extract_source_refs(answer)
    referenced = {number for number in refs if 1 <= number <= len(sources)}
    unknown = [number for number in refs if number not in referenced]
    if unknown:
        logger.warning("컨텍스트에 없는 출처 번호가 답변에 있습니다: %s", unknown)
    if not referenced:
        return [Citation.from_source(source, in_answer=False) for source in sources]
    return [
        Citation.from_source(source, in_answer=True) for index, source in enumerate(sources, 1) if index in referenced
    ]
