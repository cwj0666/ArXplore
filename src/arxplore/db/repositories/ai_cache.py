from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from arxplore.db.pool import ConnectionSource, use_connection
from arxplore.domain.papers import AiOverview, DetailedSummary

GET_OVERVIEW_SQL = """
SELECT overview, key_findings, model, generated_at
FROM paper_ai_overviews
WHERE arxiv_id = %s
"""

GET_DETAILED_SUMMARY_SQL = """
SELECT summary, model, generated_at
FROM paper_ai_detailed_summaries
WHERE arxiv_id = %s AND model = %s
"""

SAVE_OVERVIEW_SQL = """
INSERT INTO paper_ai_overviews (arxiv_id, overview, key_findings, model, generated_at)
VALUES (%s, %s, %s, %s, NOW())
ON CONFLICT (arxiv_id) DO UPDATE SET
    overview = EXCLUDED.overview,
    key_findings = EXCLUDED.key_findings,
    model = EXCLUDED.model,
    generated_at = NOW()
"""

SAVE_DETAILED_SUMMARY_SQL = """
INSERT INTO paper_ai_detailed_summaries (arxiv_id, model, summary, generated_at)
VALUES (%s, %s, %s, NOW())
ON CONFLICT (arxiv_id, model) DO UPDATE SET
    summary = EXCLUDED.summary,
    generated_at = NOW()
"""


async def get_overview(db: ConnectionSource, arxiv_id: str) -> AiOverview | None:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(GET_OVERVIEW_SQL, (arxiv_id,))
        row = await cursor.fetchone()
    if row is None:
        return None
    return AiOverview(
        overview=row["overview"],
        key_findings=row["key_findings"] or [],
        model=row["model"],
        generated_at=row["generated_at"],
    )


async def get_detailed_summary(db: ConnectionSource, arxiv_id: str, model: str) -> DetailedSummary | None:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(GET_DETAILED_SUMMARY_SQL, (arxiv_id, model))
        row = await cursor.fetchone()
    if row is None:
        return None
    return DetailedSummary(summary=row["summary"], model=row["model"], generated_at=row["generated_at"])


async def save_overview(
    db: ConnectionSource,
    arxiv_id: str,
    *,
    overview: str,
    key_findings: list[Any],
    model: str,
) -> None:
    async with use_connection(db) as connection, connection.cursor() as cursor:
        await cursor.execute(SAVE_OVERVIEW_SQL, (arxiv_id, overview, Jsonb(key_findings), model))


async def save_detailed_summary(db: ConnectionSource, arxiv_id: str, *, summary: str, model: str) -> None:
    async with use_connection(db) as connection, connection.cursor() as cursor:
        await cursor.execute(SAVE_DETAILED_SUMMARY_SQL, (arxiv_id, model, summary))
