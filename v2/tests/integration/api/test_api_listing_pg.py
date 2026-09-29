from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from arxplore.db.migrate import migrate
from arxplore.db.repositories.listing import list_recent_papers

pytestmark = pytest.mark.integration

PAPERS = (
    ("p1", "Diffusion Models for Video", "latent video generation", 3, "2024-01-05"),
    ("p2", "Other Title", "We study diffusion models at scale", 9, "2024-01-04"),
    ("p3", "Models of Diffusion", "abstract", 3, "2024-01-03"),
    ("p4", "100% Accurate_Model", "percent sign and underscore", 1, "2024-01-02"),
    ("p5", "Unrelated", "nothing to see", 7, None),
)
TIED = (("t1", "2024-02-01"), ("t3", "2024-02-01"), ("t2", "2024-02-01"), ("t0", None))


async def seeded(url: str, papers: tuple[tuple[str, str, str, int, str | None], ...] = PAPERS) -> Any:
    connection = await psycopg.AsyncConnection.connect(url, autocommit=True, row_factory=dict_row)
    await migrate(connection)
    for arxiv_id, title, abstract, upvotes, published in papers:
        await connection.execute(
            """
            INSERT INTO papers (arxiv_id, title, abstract, upvotes, published_at, updated_at_utc)
            VALUES (%s, %s, %s, %s, %s, '2023-12-31T00:00:00+00')
            """,
            (arxiv_id, title, abstract, upvotes, published),
        )
    return connection


async def test_recent_papers_break_date_ties_by_arxiv_id_descending(empty_database: str) -> None:
    tied = tuple((arxiv_id, arxiv_id, "", 0, published) for arxiv_id, published in TIED)
    async with await seeded(empty_database, tied) as connection:
        papers = await list_recent_papers(connection, limit=10)
        first_two = await list_recent_papers(connection, limit=2)

    assert [paper.arxiv_id for paper in papers] == ["t3", "t2", "t1", "t0"]
    assert [paper.arxiv_id for paper in first_two] == ["t3", "t2"]


async def test_recent_papers_map_rows_to_papers(empty_database: str) -> None:
    async with await seeded(empty_database) as connection:
        papers = await list_recent_papers(connection, limit=0)

    assert [paper.arxiv_id for paper in papers] == ["p1"]
    assert (papers[0].title, papers[0].upvotes, papers[0].authors, papers[0].categories) == (
        "Diffusion Models for Video",
        3,
        [],
        [],
    )
