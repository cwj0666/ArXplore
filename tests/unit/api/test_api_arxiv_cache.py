import asyncio
from dataclasses import dataclass, field

import pytest

from arxplore.api.services.arxiv_cache import CachedArxivSearch
from arxplore.sources.arxiv import ArxivPaper


@dataclass
class Clock:
    now: float = 0.0

    def __call__(self) -> float:
        return self.now


@dataclass
class CountingSearch:
    calls: list[tuple[str, int]] = field(default_factory=list)
    failures: int = 0
    gate: asyncio.Event | None = None

    async def search_papers(self, query: str, *, max_results: int = 5) -> list[ArxivPaper]:
        self.calls.append((query, max_results))
        if self.gate is not None:
            await self.gate.wait()
        if self.failures:
            self.failures -= 1
            raise ConnectionError("arxiv down")
        return [ArxivPaper(arxiv_id=f"{query}-{len(self.calls)}", title=query, abstract="")]


async def test_repeated_queries_are_served_from_the_cache_until_the_ttl() -> None:
    clock = Clock()
    search = CountingSearch()
    cache = CachedArxivSearch(search, ttl_seconds=10, clock=clock)

    first = await cache.search_papers("dpo", max_results=10)
    again = await cache.search_papers("dpo", max_results=10)
    other_size = await cache.search_papers("dpo", max_results=5)
    clock.now = 10.5
    refreshed = await cache.search_papers("dpo", max_results=10)

    assert first == again
    assert first[0].arxiv_id == "dpo-1"
    assert other_size[0].arxiv_id == "dpo-2"
    assert refreshed[0].arxiv_id == "dpo-3"
    assert search.calls == [("dpo", 10), ("dpo", 5), ("dpo", 10)]


async def test_failures_are_not_cached() -> None:
    search = CountingSearch(failures=1)
    cache = CachedArxivSearch(search)

    with pytest.raises(ConnectionError):
        await cache.search_papers("dpo")
    assert [paper.arxiv_id for paper in await cache.search_papers("dpo")] == ["dpo-2"]


async def test_least_recently_used_entries_are_evicted() -> None:
    search = CountingSearch()
    cache = CachedArxivSearch(search, max_entries=2)

    await cache.search_papers("a")
    await cache.search_papers("b")
    await cache.search_papers("a")
    await cache.search_papers("c")
    await cache.search_papers("a")
    await cache.search_papers("b")

    assert [query for query, _ in search.calls] == ["a", "b", "c", "b"]


async def test_concurrent_misses_share_one_request_and_survive_a_cancelled_caller() -> None:
    gate = asyncio.Event()
    search = CountingSearch(gate=gate)
    cache = CachedArxivSearch(search)

    first = asyncio.create_task(cache.search_papers("dpo"))
    second = asyncio.create_task(cache.search_papers("dpo"))
    await asyncio.sleep(0)
    first.cancel()
    gate.set()

    assert [paper.arxiv_id for paper in await second] == ["dpo-1"]
    assert search.calls == [("dpo", 5)]
    with pytest.raises(asyncio.CancelledError):
        await first
    assert [paper.arxiv_id for paper in await cache.search_papers("dpo")] == ["dpo-1"]


async def test_clear_forgets_every_entry() -> None:
    search = CountingSearch()
    cache = CachedArxivSearch(search)
    await cache.search_papers("dpo")

    cache.clear()
    await cache.search_papers("dpo")

    assert len(search.calls) == 2
