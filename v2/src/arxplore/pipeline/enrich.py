from dataclasses import dataclass, field
from typing import Any

import httpx

from arxplore.db.pool import AnyPool
from arxplore.db.repositories import papers
from arxplore.db.repositories.papers import ENRICHED_PAPER_SOURCE
from arxplore.domain.papers import EnrichmentCandidate, PaperUpsert
from arxplore.sources.arxiv import RATE_LIMITED_STATUS, ArxivClient, ArxivPaper
from arxplore.sources.arxiv_metadata import fetch_arxiv_metadata

DEFAULT_ENRICH_MAX_PAPERS = 30
SAMPLE_LIMIT = 5


def merge_enriched(existing: EnrichmentCandidate, enriched: ArxivPaper) -> PaperUpsert:
    return PaperUpsert.from_mapping(
        {
            "arxiv_id": existing.arxiv_id,
            "title": enriched.title or existing.title,
            "authors": enriched.authors or existing.authors,
            "abstract": enriched.abstract or existing.abstract,
            "primary_category": enriched.primary_category or existing.primary_category,
            "categories": enriched.categories or existing.categories,
            "pdf_url": enriched.pdf_url or existing.pdf_url,
            "published_at": enriched.published_at or existing.published_at,
            "updated_at": enriched.updated_at or existing.updated_at,
            "upvotes": existing.upvotes,
            "github_url": existing.github_url,
            "github_stars": existing.github_stars,
            "citation_count": existing.citation_count,
            "source": ENRICHED_PAPER_SOURCE,
        }
    )


def normalize_max_papers(max_papers: int | None) -> int:
    return DEFAULT_ENRICH_MAX_PAPERS if max_papers is None else max(1, int(max_papers))


@dataclass(slots=True, kw_only=True)
class EnrichResult:
    status: str
    candidate_count: int
    max_papers: int
    updated_count: int = 0
    skipped_count: int = 0
    sample_updated: list[dict[str, Any]] = field(default_factory=list)
    failed_arxiv_ids: list[str] | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "stage": "enrich_papers_metadata",
            "status": self.status,
            "candidate_count": self.candidate_count,
            "updated_count": self.updated_count,
            "skipped_count": self.skipped_count,
            "max_papers": self.max_papers,
            "sample_updated": list(self.sample_updated),
        }
        if self.failed_arxiv_ids is not None:
            result["failed_arxiv_ids"] = list(self.failed_arxiv_ids)
            result["error"] = self.error
        return result


def _soft_failure(status: str, candidates: list[EnrichmentCandidate], limit: int, exc: Exception) -> EnrichResult:
    return EnrichResult(
        status=status,
        candidate_count=len(candidates),
        max_papers=limit,
        failed_arxiv_ids=[paper.arxiv_id for paper in candidates[:SAMPLE_LIMIT] if paper.arxiv_id],
        error=str(exc),
    )


async def enrich_papers_metadata(
    pool: AnyPool,
    arxiv: ArxivClient,
    *,
    batch_size: int,
    max_papers: int | None = DEFAULT_ENRICH_MAX_PAPERS,
) -> EnrichResult:
    limit = normalize_max_papers(max_papers)
    candidates = await papers.list_papers_missing_arxiv_metadata(pool, limit=limit)
    arxiv_ids = [paper.arxiv_id for paper in candidates if paper.arxiv_id]
    try:
        metadata = await fetch_arxiv_metadata(arxiv, arxiv_ids, batch_size=batch_size)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != RATE_LIMITED_STATUS:
            raise
        return _soft_failure("rate_limited", candidates, limit, exc)
    except httpx.TimeoutException as exc:
        return _soft_failure("timed_out", candidates, limit, exc)

    result = EnrichResult(status="success", candidate_count=len(candidates), max_papers=limit)
    for paper in candidates:
        enriched = metadata.get(paper.arxiv_id)
        if enriched is None:
            result.skipped_count += 1
            continue
        merged = merge_enriched(paper, enriched)
        await papers.save_paper(pool, merged)
        result.updated_count += 1
        if len(result.sample_updated) < SAMPLE_LIMIT:
            result.sample_updated.append(
                {
                    "arxiv_id": paper.arxiv_id,
                    "primary_category": merged.primary_category,
                    "category_count": len(merged.categories or []),
                    "pdf_url": merged.pdf_url,
                }
            )
    return result
