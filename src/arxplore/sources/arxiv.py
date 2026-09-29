import asyncio
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import httpx

from arxplore.config import AppSettings

ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
ARXIV_URL_PREFIXES = (
    "http://arxiv.org/abs/",
    "https://arxiv.org/abs/",
    "http://arxiv.org/pdf/",
    "https://arxiv.org/pdf/",
)
PDF_SUFFIX = ".pdf"
VERSION_MARKER = "v"
PDF_LINK_TITLE = "pdf"
SEARCH_TERM_LIMIT = 8
SEARCH_TIMEOUT_CAP_SECONDS = 8
SEARCH_RETRY_COUNT = 1
SEARCH_MIN_RETRY_DELAY_SECONDS = 1.0
TIMEOUT_BACKOFF_MULTIPLIER = 2
RATE_LIMITED_BACKOFF_MULTIPLIER = 5
RATE_LIMITED_STATUS = 429
DEFAULT_SEARCH_RESULTS = 5


def normalize_arxiv_id(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        return normalized
    for prefix in ARXIV_URL_PREFIXES:
        normalized = normalized.removeprefix(prefix)
    if normalized.endswith(PDF_SUFFIX):
        normalized = normalized[: -len(PDF_SUFFIX)]
    if VERSION_MARKER in normalized:
        base, version = normalized.rsplit(VERSION_MARKER, 1)
        if version.isdigit():
            return base
    return normalized


@dataclass(frozen=True, slots=True, kw_only=True)
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str] = field(default_factory=list)
    published_at: str = ""
    updated_at: str = ""
    categories: list[str] = field(default_factory=list)
    primary_category: str | None = None
    pdf_url: str | None = None


def _element_text(node: ET.Element, path: str) -> str:
    element = node.find(path, ATOM_NAMESPACE)
    if element is None or element.text is None:
        return ""
    return " ".join(element.text.split())


def _element_attribute(node: ET.Element, path: str, key: str) -> str | None:
    element = node.find(path, ATOM_NAMESPACE)
    if element is None:
        return None
    return element.attrib.get(key)


def _pdf_link(entry: ET.Element) -> str | None:
    for link in entry.findall("atom:link", ATOM_NAMESPACE):
        if link.attrib.get("title") == PDF_LINK_TITLE:
            return link.attrib.get("href")
    return None


def parse_arxiv_feed(xml_text: str) -> dict[str, ArxivPaper]:
    root = ET.fromstring(xml_text)
    parsed: dict[str, ArxivPaper] = {}
    for entry in root.findall("atom:entry", ATOM_NAMESPACE):
        arxiv_id = normalize_arxiv_id(_element_text(entry, "atom:id"))
        parsed[arxiv_id] = ArxivPaper(
            arxiv_id=arxiv_id,
            title=_element_text(entry, "atom:title"),
            abstract=_element_text(entry, "atom:summary"),
            authors=[name.text.strip() for name in entry.findall("atom:author/atom:name", ATOM_NAMESPACE) if name.text],
            published_at=_element_text(entry, "atom:published"),
            updated_at=_element_text(entry, "atom:updated"),
            categories=[category.attrib["term"] for category in entry.findall("atom:category", ATOM_NAMESPACE)],
            primary_category=_element_attribute(entry, "arxiv:primary_category", "term"),
            pdf_url=_pdf_link(entry),
        )
    return parsed


def build_search_query(query: str) -> str:
    terms = [f'all:"{term}"' for term in " ".join(query.split()).split()[:SEARCH_TERM_LIMIT] if term.strip()]
    return " AND ".join(terms)


def retry_delay(error: httpx.HTTPError, *, attempt: int, base_seconds: float) -> float:
    if isinstance(error, httpx.TimeoutException):
        return base_seconds * attempt * TIMEOUT_BACKOFF_MULTIPLIER
    if isinstance(error, httpx.HTTPStatusError) and error.response.status_code == RATE_LIMITED_STATUS:
        return base_seconds * attempt * RATE_LIMITED_BACKOFF_MULTIPLIER
    return base_seconds * attempt


class ArxivClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        delay_seconds: float,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.delay_seconds = delay_seconds
        self._http = http or httpx.AsyncClient()
        self._owns_http = http is None

    @classmethod
    def from_settings(cls, settings: AppSettings, *, http: httpx.AsyncClient | None = None) -> Self:
        return cls(
            base_url=settings.arxiv_api_base_url,
            timeout_seconds=settings.arxiv_request_timeout_seconds,
            delay_seconds=settings.arxiv_request_delay_seconds,
            http=http,
        )

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def get_with_retries(
        self,
        params: Mapping[str, Any],
        *,
        timeout: float,
        retry_count: int,
        retry_delay_seconds: float,
    ) -> httpx.Response:
        last_error: httpx.HTTPError | None = None
        for attempt in range(1, retry_count + 1):
            try:
                response = await self._http.get(self.base_url, params=dict(params), timeout=timeout)
                response.raise_for_status()
                return response
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= retry_count:
                    break
                await asyncio.sleep(retry_delay(exc, attempt=attempt, base_seconds=retry_delay_seconds))
        if last_error is None:
            raise RuntimeError("HTTP 요청 재시도 중 알 수 없는 오류가 발생했습니다.")
        raise last_error

    async def search_papers(self, query: str, *, max_results: int = DEFAULT_SEARCH_RESULTS) -> list[ArxivPaper]:
        search_query = build_search_query(query)
        if not search_query:
            return []
        response = await self.get_with_retries(
            {
                "search_query": search_query,
                "start": 0,
                "max_results": max(1, int(max_results)),
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=min(self.timeout_seconds, SEARCH_TIMEOUT_CAP_SECONDS),
            retry_count=SEARCH_RETRY_COUNT,
            retry_delay_seconds=max(SEARCH_MIN_RETRY_DELAY_SECONDS, float(self.delay_seconds)),
        )
        return list(parse_arxiv_feed(response.text).values())
