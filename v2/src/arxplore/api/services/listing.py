import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from arxplore.api.services.store import PaperStore
from arxplore.domain.papers import Paper

MAX_RECENT_PAPERS = 1500
PAPERS_PER_PAGE = 21
SEARCH_MODE = "search"
AI_MODE = "ai"
VALID_SEARCH_MODES = frozenset({SEARCH_MODE, AI_MODE})
SORT_LATEST = "latest"
SORT_UPVOTES = "upvotes"


@dataclass(frozen=True, slots=True, kw_only=True)
class PageWindow:
    number: int
    num_pages: int
    start: int
    stop: int


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperPage:
    papers: list[Paper]
    page: int
    page_size: int
    total_items: int
    total_pages: int
    query: str
    sort: str
    mode: str


def normalize_mode(mode: str) -> str:
    return mode if mode in VALID_SEARCH_MODES else SEARCH_MODE


def normalize_sort(sort: str) -> str:
    return SORT_UPVOTES if sort == SORT_UPVOTES else SORT_LATEST


def parse_page_number(raw: Any) -> int:
    try:
        number = int(raw)
    except TypeError, ValueError:
        return 1
    return number if number > 0 else 1


def page_window(total_items: int, requested: int, per_page: int = PAPERS_PER_PAGE) -> PageWindow:
    num_pages = math.ceil(max(1, total_items) / per_page)
    number = requested if 1 <= requested <= num_pages else num_pages
    start = (number - 1) * per_page
    return PageWindow(number=number, num_pages=num_pages, start=start, stop=min(start + per_page, total_items))


def matches_query(paper: Paper, lowered_query: str) -> bool:
    return lowered_query in (paper.title or "").lower() or lowered_query in (paper.abstract or "").lower()


def sort_by_upvotes(papers: Sequence[Paper]) -> list[Paper]:
    return sorted(papers, key=lambda paper: paper.upvotes or 0, reverse=True)


async def list_papers(store: PaperStore, *, query: str, sort: str, mode: str, page: Any) -> PaperPage:
    papers = list(await store.list_recent_papers(limit=MAX_RECENT_PAPERS))
    normalized_query = query.strip()
    normalized_sort = normalize_sort(sort)
    if normalized_query:
        lowered = normalized_query.lower()
        papers = [paper for paper in papers if matches_query(paper, lowered)]
    if normalized_sort == SORT_UPVOTES:
        papers = sort_by_upvotes(papers)
    window = page_window(len(papers), parse_page_number(page))
    return PaperPage(
        papers=papers[window.start : window.stop],
        page=window.number,
        page_size=PAPERS_PER_PAGE,
        total_items=len(papers),
        total_pages=window.num_pages,
        query=normalized_query,
        sort=normalized_sort,
        mode=normalize_mode(mode),
    )
