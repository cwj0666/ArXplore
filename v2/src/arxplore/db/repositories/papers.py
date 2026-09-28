from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from arxplore.db.pool import ConnectionSource, use_connection
from arxplore.domain.papers import DEFAULT_PAPER_SOURCE, EnrichmentCandidate, Paper, PaperUpsert
from arxplore.text import strip_surrogates, strip_surrogates_deep

EMPTY_ARXIV_ID_MESSAGE = "paper['arxiv_id']는 비어 있을 수 없습니다."

HF_RAW_PAPER_SOURCE = "hf_daily_papers_raw"
ENRICHED_PAPER_SOURCE = "hf_daily_papers+arxiv"

_HF_ROW_OVER_ENRICHED_ROW = "(papers.source = %(enriched_source)s AND EXCLUDED.source = %(hf_raw_source)s)"

SAVE_PAPER_SQL = f"""
INSERT INTO papers (
    arxiv_id, title, authors, abstract, primary_category,
    categories, pdf_url, published_at, updated_at,
    upvotes, github_url, github_stars, citation_count, source, updated_at_utc
) VALUES (
    %(arxiv_id)s, %(title)s, %(authors)s, %(abstract)s, %(primary_category)s,
    %(categories)s, %(pdf_url)s, %(published_at)s, %(arxiv_updated_at)s,
    COALESCE(%(upvotes)s::integer, 0), %(github_url)s, %(github_stars)s, %(citation_count)s, %(source)s, NOW()
)
ON CONFLICT (arxiv_id)
DO UPDATE SET
    title = CASE WHEN {_HF_ROW_OVER_ENRICHED_ROW} THEN papers.title ELSE EXCLUDED.title END,
    authors = CASE WHEN {_HF_ROW_OVER_ENRICHED_ROW} THEN papers.authors ELSE EXCLUDED.authors END,
    abstract = CASE WHEN {_HF_ROW_OVER_ENRICHED_ROW} THEN papers.abstract ELSE EXCLUDED.abstract END,
    primary_category = COALESCE(EXCLUDED.primary_category, papers.primary_category),
    categories = CASE
        WHEN EXCLUDED.categories = '[]'::jsonb THEN papers.categories
        ELSE EXCLUDED.categories
    END,
    pdf_url = CASE WHEN {_HF_ROW_OVER_ENRICHED_ROW} THEN papers.pdf_url ELSE EXCLUDED.pdf_url END,
    published_at = CASE
        WHEN {_HF_ROW_OVER_ENRICHED_ROW} THEN papers.published_at
        ELSE COALESCE(EXCLUDED.published_at, papers.published_at)
    END,
    updated_at = CASE
        WHEN {_HF_ROW_OVER_ENRICHED_ROW} THEN papers.updated_at
        ELSE COALESCE(EXCLUDED.updated_at, papers.updated_at)
    END,
    upvotes = COALESCE(%(upvotes)s::integer, papers.upvotes),
    github_url = COALESCE(EXCLUDED.github_url, papers.github_url),
    github_stars = COALESCE(EXCLUDED.github_stars, papers.github_stars),
    citation_count = COALESCE(EXCLUDED.citation_count, papers.citation_count),
    source = CASE WHEN {_HF_ROW_OVER_ENRICHED_ROW} THEN papers.source ELSE EXCLUDED.source END,
    updated_at_utc = NOW()
"""

LIST_RECENT_PAPERS_SQL = """
SELECT arxiv_id, title, authors, abstract, primary_category, categories, pdf_url,
       published_at, updated_at, upvotes, github_url, github_stars, citation_count
FROM papers
ORDER BY COALESCE(published_at, updated_at_utc) DESC
LIMIT %s
"""

LIST_PAPERS_MISSING_ARXIV_METADATA_SQL = """
SELECT arxiv_id, title, authors, abstract, primary_category, categories, pdf_url,
       published_at, updated_at, upvotes, github_url, github_stars, citation_count, source
FROM papers
WHERE
    (
        primary_category IS NULL
        OR categories = '[]'::jsonb
        OR source = 'hf_daily_papers_raw'
    )
    AND (arxiv_lookup_retry_at IS NULL OR arxiv_lookup_retry_at <= NOW())
ORDER BY COALESCE(published_at, updated_at_utc) DESC
LIMIT %s
"""

RECORD_ARXIV_LOOKUP_HITS_SQL = """
UPDATE papers
SET arxiv_lookup_at = NOW(), arxiv_lookup_misses = 0, arxiv_lookup_retry_at = NULL
WHERE arxiv_id = ANY(%s)
"""

RECORD_ARXIV_LOOKUP_MISSES_SQL = """
UPDATE papers
SET
    arxiv_lookup_at = NOW(),
    arxiv_lookup_misses = arxiv_lookup_misses + 1,
    arxiv_lookup_retry_at = NOW() + LEAST(
        %(max_delay)s::interval,
        %(base_delay)s::interval * power(2, LEAST(arxiv_lookup_misses, %(max_exponent)s))
    )
WHERE arxiv_id = ANY(%(arxiv_ids)s)
"""

ARXIV_LOOKUP_MAX_EXPONENT = 16

LIST_PAPER_IDS_SQL = "SELECT arxiv_id FROM papers ORDER BY arxiv_id"

GET_PAPER_SQL = """
SELECT arxiv_id, title, authors, abstract, primary_category, categories, pdf_url,
       published_at, updated_at, upvotes, github_url, github_stars, citation_count
FROM papers
WHERE arxiv_id = %s
"""


def _clean_optional(value: str | None) -> str | None:
    return strip_surrogates(value) if isinstance(value, str) else value


def _save_params(paper: PaperUpsert, arxiv_id: str) -> dict[str, Any]:
    return {
        "arxiv_id": arxiv_id,
        "title": _clean_optional(paper.title),
        "authors": Jsonb(strip_surrogates_deep(paper.authors)),
        "abstract": _clean_optional(paper.abstract),
        "primary_category": _clean_optional(paper.primary_category),
        "categories": Jsonb(strip_surrogates_deep(paper.categories)),
        "pdf_url": _clean_optional(paper.pdf_url),
        "published_at": paper.published_at,
        "arxiv_updated_at": paper.updated_at,
        "upvotes": paper.upvotes,
        "github_url": _clean_optional(paper.github_url),
        "github_stars": paper.github_stars,
        "citation_count": paper.citation_count,
        "source": _clean_optional(paper.source),
        "enriched_source": ENRICHED_PAPER_SOURCE,
        "hf_raw_source": HF_RAW_PAPER_SOURCE,
    }


def _paper_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "arxiv_id": row["arxiv_id"],
        "title": row["title"],
        "authors": row["authors"] or [],
        "abstract": row["abstract"] or "",
        "primary_category": row["primary_category"],
        "categories": row["categories"] or [],
        "pdf_url": row["pdf_url"],
        "published_at": row["published_at"],
        "updated_at": row["updated_at"],
        "upvotes": row["upvotes"] or 0,
        "github_url": row["github_url"],
        "github_stars": row["github_stars"],
        "citation_count": row["citation_count"],
    }


async def save_paper(db: ConnectionSource, paper: PaperUpsert) -> str:
    arxiv_id = strip_surrogates(str(paper.arxiv_id)).strip()
    if not arxiv_id:
        raise ValueError(EMPTY_ARXIV_ID_MESSAGE)
    async with use_connection(db) as connection, connection.cursor() as cursor:
        await cursor.execute(SAVE_PAPER_SQL, _save_params(paper, arxiv_id))
    return arxiv_id


async def get_paper(db: ConnectionSource, arxiv_id: str) -> Paper | None:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(GET_PAPER_SQL, (arxiv_id,))
        row = await cursor.fetchone()
    return Paper(**_paper_fields(row)) if row is not None else None


async def list_recent_papers(db: ConnectionSource, *, limit: int = 200) -> list[Paper]:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(LIST_RECENT_PAPERS_SQL, (max(1, limit),))
        rows = await cursor.fetchall()
    return [Paper(**_paper_fields(row)) for row in rows]


async def list_papers_missing_arxiv_metadata(db: ConnectionSource, *, limit: int = 200) -> list[EnrichmentCandidate]:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(LIST_PAPERS_MISSING_ARXIV_METADATA_SQL, (max(1, limit),))
        rows = await cursor.fetchall()
    return [EnrichmentCandidate(**_paper_fields(row), source=row["source"] or DEFAULT_PAPER_SOURCE) for row in rows]


async def record_arxiv_lookups(
    db: ConnectionSource,
    *,
    found: Sequence[str],
    missing: Sequence[str],
    base_delay: timedelta,
    max_delay: timedelta,
) -> None:
    async with use_connection(db) as connection, connection.cursor() as cursor:
        if found:
            await cursor.execute(RECORD_ARXIV_LOOKUP_HITS_SQL, (list(found),))
        if missing:
            await cursor.execute(
                RECORD_ARXIV_LOOKUP_MISSES_SQL,
                {
                    "arxiv_ids": list(missing),
                    "base_delay": base_delay,
                    "max_delay": max_delay,
                    "max_exponent": ARXIV_LOOKUP_MAX_EXPONENT,
                },
            )


async def list_paper_ids(db: ConnectionSource) -> list[str]:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(LIST_PAPER_IDS_SQL)
        rows = await cursor.fetchall()
    return [str(row["arxiv_id"]) for row in rows]
