from collections.abc import Sequence
from typing import Any, Protocol

from arxplore.db.pool import AnyPool
from arxplore.db.repositories import fulltexts, generation_cache, listing, papers
from arxplore.domain.chunks import StoredChunk
from arxplore.domain.papers import AiOverview, DetailedSummary, Paper, PaperFulltext


class PaperStore(Protocol):
    async def get_paper(self, arxiv_id: str) -> Paper | None: ...

    async def list_recent_papers(self, *, limit: int) -> Sequence[Paper]: ...

    async def get_fulltext(self, arxiv_id: str) -> PaperFulltext | None: ...

    async def list_chunks(self, arxiv_id: str, *, limit: int) -> Sequence[StoredChunk]: ...

    async def get_overview(self, arxiv_id: str, *, prompt_version: int) -> AiOverview | None: ...

    async def save_overview(
        self, arxiv_id: str, *, overview: str, key_findings: list[Any], model: str, prompt_version: int
    ) -> None: ...

    async def get_detailed_summary(
        self, arxiv_id: str, model: str, *, prompt_version: int
    ) -> DetailedSummary | None: ...

    async def save_detailed_summary(self, arxiv_id: str, *, summary: str, model: str, prompt_version: int) -> None: ...


class PostgresPaperStore:
    def __init__(self, pool: AnyPool) -> None:
        self.pool = pool

    async def get_paper(self, arxiv_id: str) -> Paper | None:
        return await papers.get_paper(self.pool, arxiv_id)

    async def list_recent_papers(self, *, limit: int) -> Sequence[Paper]:
        return await listing.list_recent_papers(self.pool, limit=limit)

    async def get_fulltext(self, arxiv_id: str) -> PaperFulltext | None:
        return await fulltexts.get_fulltext(self.pool, arxiv_id)

    async def list_chunks(self, arxiv_id: str, *, limit: int) -> Sequence[StoredChunk]:
        return await fulltexts.list_chunks(self.pool, arxiv_id, limit=limit)

    async def get_overview(self, arxiv_id: str, *, prompt_version: int) -> AiOverview | None:
        return await generation_cache.get_overview(self.pool, arxiv_id, prompt_version=prompt_version)

    async def save_overview(
        self, arxiv_id: str, *, overview: str, key_findings: list[Any], model: str, prompt_version: int
    ) -> None:
        await generation_cache.save_overview(
            self.pool,
            arxiv_id,
            overview=overview,
            key_findings=key_findings,
            model=model,
            prompt_version=prompt_version,
        )

    async def get_detailed_summary(self, arxiv_id: str, model: str, *, prompt_version: int) -> DetailedSummary | None:
        return await generation_cache.get_detailed_summary(self.pool, arxiv_id, model, prompt_version=prompt_version)

    async def save_detailed_summary(self, arxiv_id: str, *, summary: str, model: str, prompt_version: int) -> None:
        await generation_cache.save_detailed_summary(
            self.pool, arxiv_id, summary=summary, model=model, prompt_version=prompt_version
        )
