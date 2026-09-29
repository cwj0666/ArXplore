from typing import Any

from psycopg.rows import dict_row

from arxplore.db.pool import ConnectionSource, use_connection
from arxplore.domain.papers import Paper

LIST_RECENT_PAPERS_SQL = """
SELECT arxiv_id, title, authors, abstract, primary_category, categories, pdf_url,
       published_at, updated_at, upvotes, github_url, github_stars, citation_count
FROM papers
ORDER BY COALESCE(published_at, updated_at_utc) DESC, arxiv_id DESC
LIMIT %s
"""


def paper_from_row(row: dict[str, Any]) -> Paper:
    return Paper(
        arxiv_id=row["arxiv_id"],
        title=row["title"],
        authors=row["authors"] or [],
        abstract=row["abstract"] or "",
        primary_category=row["primary_category"],
        categories=row["categories"] or [],
        pdf_url=row["pdf_url"],
        published_at=row["published_at"],
        updated_at=row["updated_at"],
        upvotes=row["upvotes"] or 0,
        github_url=row["github_url"],
        github_stars=row["github_stars"],
        citation_count=row["citation_count"],
    )


async def list_recent_papers(db: ConnectionSource, *, limit: int) -> list[Paper]:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(LIST_RECENT_PAPERS_SQL, (max(1, limit),))
        rows = await cursor.fetchall()
    return [paper_from_row(row) for row in rows]
