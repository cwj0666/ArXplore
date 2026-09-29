import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from arxplore.api.services.store import PaperStore
from arxplore.db.repositories.listing import ListSort, search_tokens
from arxplore.domain.papers import Paper

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


def normalize_sort(sort: str) -> ListSort:
    return "upvotes" if sort == SORT_UPVOTES else "latest"


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


async def list_papers(store: PaperStore, *, query: str, sort: str, mode: str, page: Any) -> PaperPage:
    normalized_query = query.strip()
    normalized_sort = normalize_sort(sort)
    tokens = search_tokens(normalized_query)
    total = await store.count_papers(tokens)
    window = page_window(total, parse_page_number(page))
    papers: Sequence[Paper] = []
    if window.stop > window.start:
        papers = await store.list_papers_page(
            tokens, sort=normalized_sort, limit=window.stop - window.start, offset=window.start
        )
    return PaperPage(
        papers=list(papers),
        page=window.number,
        page_size=PAPERS_PER_PAGE,
        total_items=total,
        total_pages=window.num_pages,
        query=normalized_query,
        sort=normalized_sort,
        mode=normalize_mode(mode),
    )
