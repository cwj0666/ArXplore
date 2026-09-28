import asyncio
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from arxplore.api.services.detail import ArxivSearch
from arxplore.sources.arxiv import ArxivPaper

DEFAULT_TTL_SECONDS = 3600.0
DEFAULT_MAX_ENTRIES = 512

type SearchKey = tuple[str, int]


@dataclass(frozen=True, slots=True)
class CachedSearch:
    papers: tuple[ArxivPaper, ...]
    expires_at: float


class CachedArxivSearch:
    def __init__(
        self,
        search: ArxivSearch,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.search = search
        self.ttl_seconds = ttl_seconds
        self.max_entries = max(1, max_entries)
        self.clock = clock
        self._entries: OrderedDict[SearchKey, CachedSearch] = OrderedDict()
        self._in_flight: dict[SearchKey, asyncio.Task[tuple[ArxivPaper, ...]]] = {}

    async def search_papers(self, query: str, *, max_results: int = 5) -> list[ArxivPaper]:
        key = (query, max_results)
        cached = self._entries.get(key)
        if cached is not None and cached.expires_at > self.clock():
            self._entries.move_to_end(key)
            return list(cached.papers)
        task = self._in_flight.get(key)
        if task is None:
            task = asyncio.ensure_future(self._fetch(key))
            self._in_flight[key] = task
            task.add_done_callback(self._forget)
        return list(await asyncio.shield(task))

    def _forget(self, task: asyncio.Task[tuple[ArxivPaper, ...]]) -> None:
        for key, running in list(self._in_flight.items()):
            if running is task:
                del self._in_flight[key]
        if not task.cancelled():
            task.exception()

    async def _fetch(self, key: SearchKey) -> tuple[ArxivPaper, ...]:
        query, max_results = key
        papers = tuple(await self.search.search_papers(query, max_results=max_results))
        self._entries[key] = CachedSearch(papers=papers, expires_at=self.clock() + self.ttl_seconds)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return papers

    def clear(self) -> None:
        self._entries.clear()
