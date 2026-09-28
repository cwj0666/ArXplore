from collections.abc import Sequence
from typing import Any, Protocol

from arxplore.db.pool import AnyPool
from arxplore.db.repositories import ai_cache, fulltexts, papers
from arxplore.domain.chunks import StoredChunk
from arxplore.domain.papers import AiOverview, DetailedSummary, Paper, PaperFulltext


class PaperStore(Protocol):
    async def get_paper(self, arxiv_id: str) -> Paper | None: ...

    async def list_recent_papers(self, *, limit: int) -> Sequence[Paper]: ...

    async def get_fulltext(self, arxiv_id: str) -> PaperFulltext | None: ...

    async def list_chunks(self, arxiv_id: str, *, limit: int) -> Sequence[StoredChunk]: ...

    async def get_overview(self, arxiv_id: str) -> AiOverview | None: ...

    async def save_overview(self, arxiv_id: str, *, overview: str, key_findings: list[Any], model: str) -> None: ...

    async def get_detailed_summary(self, arxiv_id: str, model: str) -> DetailedSummary | None: ...

    async def save_detailed_summary(self, arxiv_id: str, *, summary: str, model: str) -> None: ...


class PostgresPaperStore:
    def __init__(self, pool: AnyPool) -> None:
        self.pool = pool

    async def get_paper(self, arxiv_id: str) -> Paper | None:
        return await papers.get_paper(self.pool, arxiv_id)

    async def list_recent_papers(self, *, limit: int) -> Sequence[Paper]:
        return await papers.list_recent_papers(self.pool, limit=limit)

    async def get_fulltext(self, arxiv_id: str) -> PaperFulltext | None:
        return await fulltexts.get_fulltext(self.pool, arxiv_id)

    async def list_chunks(self, arxiv_id: str, *, limit: int) -> Sequence[StoredChunk]:
        return await fulltexts.list_chunks(self.pool, arxiv_id, limit=limit)

    async def get_overview(self, arxiv_id: str) -> AiOverview | None:
        return await ai_cache.get_overview(self.pool, arxiv_id)

    async def save_overview(self, arxiv_id: str, *, overview: str, key_findings: list[Any], model: str) -> None:
        await ai_cache.save_overview(self.pool, arxiv_id, overview=overview, key_findings=key_findings, model=model)

    async def get_detailed_summary(self, arxiv_id: str, model: str) -> DetailedSummary | None:
        return await ai_cache.get_detailed_summary(self.pool, arxiv_id, model)

    async def save_detailed_summary(self, arxiv_id: str, *, summary: str, model: str) -> None:
        await ai_cache.save_detailed_summary(self.pool, arxiv_id, summary=summary, model=model)
