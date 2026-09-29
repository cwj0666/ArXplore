from collections.abc import Sequence
from typing import Any, Literal

from psycopg import sql
from psycopg.rows import dict_row

from arxplore.db.pool import ConnectionSource, use_connection
from arxplore.domain.papers import Paper

type ListSort = Literal["latest", "upvotes"]

LIST_RECENT_PAPERS_SQL = """
SELECT arxiv_id, title, authors, abstract, primary_category, categories, pdf_url,
       published_at, updated_at, upvotes, github_url, github_stars, citation_count
FROM papers
ORDER BY COALESCE(published_at, updated_at_utc) DESC, arxiv_id DESC
LIMIT %s
"""
MAX_SEARCH_TOKENS = 20
LIKE_ESCAPE = "\\"
PAPER_COLUMNS = sql.SQL(
    "arxiv_id, title, authors, abstract, primary_category, categories, pdf_url, "
    "published_at, updated_at, upvotes, github_url, github_stars, citation_count"
)
ORDER_BY: dict[ListSort, sql.SQL] = {
    "latest": sql.SQL("COALESCE(published_at, updated_at_utc) DESC, arxiv_id DESC"),
    "upvotes": sql.SQL("upvotes DESC, COALESCE(published_at, updated_at_utc) DESC, arxiv_id DESC"),
}
TOKEN_MATCH = sql.SQL("(title ILIKE %s ESCAPE '\\' OR abstract ILIKE %s ESCAPE '\\')")
MATCH_ALL = sql.SQL("TRUE")
COUNT_SQL = sql.SQL("SELECT COUNT(*) AS total FROM papers WHERE {where}")
PAGE_SQL = sql.SQL("SELECT {columns} FROM papers WHERE {where} ORDER BY {order} LIMIT %s OFFSET %s")


def search_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in query.split():
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        tokens.append(token)
    return tokens[:MAX_SEARCH_TOKENS]


def like_pattern(token: str) -> str:
    escaped = (
        token.replace(LIKE_ESCAPE, LIKE_ESCAPE * 2).replace("%", f"{LIKE_ESCAPE}%").replace("_", f"{LIKE_ESCAPE}_")
    )
    return f"%{escaped}%"


def token_filter(tokens: Sequence[str]) -> tuple[sql.Composable, list[str]]:
    if not tokens:
        return MATCH_ALL, []
    params: list[str] = []
    for token in tokens:
        pattern = like_pattern(token)
        params.extend((pattern, pattern))
    return sql.SQL(" AND ").join([TOKEN_MATCH] * len(tokens)), params


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


async def count_papers(db: ConnectionSource, tokens: Sequence[str]) -> int:
    where, params = token_filter(tokens)
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(COUNT_SQL.format(where=where), params)
        row = await cursor.fetchone()
    return int(row["total"]) if row is not None else 0


async def list_papers_page(
    db: ConnectionSource, tokens: Sequence[str], *, sort: ListSort, limit: int, offset: int
) -> list[Paper]:
    where, params = token_filter(tokens)
    query = PAGE_SQL.format(columns=PAPER_COLUMNS, where=where, order=ORDER_BY[sort])
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(query, [*params, max(0, limit), max(0, offset)])
        rows = await cursor.fetchall()
    return [paper_from_row(row) for row in rows]
