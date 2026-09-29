from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from arxplore.db.migrate import migrate
from arxplore.db.repositories.listing import (
    count_papers,
    like_pattern,
    list_papers_page,
    list_recent_papers,
    search_tokens,
)

pytestmark = pytest.mark.integration

PAPERS = (
    ("p1", "Diffusion Models for Video", "latent video generation", 3, "2024-01-05"),
    ("p2", "Other Title", "We study diffusion models at scale", 9, "2024-01-04"),
    ("p3", "Diffusion Policies", "we scale models", 3, "2024-01-03"),
    ("p4", "100% Accurate_Model", "percent sign and underscore", 1, "2024-01-02"),
    ("p5", "Unrelated", "diffusion alone", 7, None),
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


def test_tokens_are_deduplicated_case_insensitively_and_like_wildcards_are_escaped() -> None:
    assert search_tokens("  Diffusion models diffusion  MODELS ") == ["Diffusion", "models"]
    assert len(search_tokens(" ".join(f"t{index}" for index in range(30)))) == 20
    assert like_pattern("100%_a\\b") == "%100\\%\\_a\\\\b%"


async def test_all_tokens_match_in_any_order_across_title_and_abstract(empty_database: str) -> None:
    async with await seeded(empty_database) as connection:
        tokens = search_tokens("MODELS diffusion")

        assert await count_papers(connection, tokens) == 3
        latest = await list_papers_page(connection, tokens, sort="latest", limit=10, offset=0)
        by_votes = await list_papers_page(connection, tokens, sort="upvotes", limit=10, offset=0)

    assert [paper.arxiv_id for paper in latest] == ["p1", "p2", "p3"]
    assert [paper.arxiv_id for paper in by_votes] == ["p2", "p1", "p3"]


async def test_percent_and_underscore_match_literally(empty_database: str) -> None:
    async with await seeded(empty_database) as connection:
        literal = await list_papers_page(connection, ["100%"], sort="latest", limit=10, offset=0)
        underscore = await list_papers_page(connection, ["Accurate_Model"], sort="latest", limit=10, offset=0)
        wildcard_underscore = await count_papers(connection, ["Accurate_"])
        plain_percent = await count_papers(connection, ["%"])

    assert [paper.arxiv_id for paper in literal] == ["p4"]
    assert [paper.arxiv_id for paper in underscore] == ["p4"]
    assert (wildcard_underscore, plain_percent) == (1, 1)


async def test_no_tokens_pages_through_every_paper_newest_first(empty_database: str) -> None:
    async with await seeded(empty_database) as connection:
        total = await count_papers(connection, [])
        first = await list_papers_page(connection, [], sort="latest", limit=2, offset=0)
        rest = await list_papers_page(connection, [], sort="latest", limit=2, offset=2)
        tail = await list_papers_page(connection, [], sort="latest", limit=2, offset=4)

    assert total == 5
    assert [paper.arxiv_id for paper in [*first, *rest, *tail]] == ["p1", "p2", "p3", "p4", "p5"]
    assert first[0].published_at is not None


async def test_pages_over_tied_dates_neither_repeat_nor_skip_papers(empty_database: str) -> None:
    tied = tuple((f"t{index:02d}", f"tied {index}", "", index % 3, "2024-02-01") for index in range(30))
    async with await seeded(empty_database, tied) as connection:
        for sort in ("latest", "upvotes"):
            pages = [
                await list_papers_page(connection, [], sort=sort, limit=4, offset=offset) for offset in range(0, 30, 4)
            ]
            ids = [paper.arxiv_id for page in pages for paper in page]
            assert len(ids) == len(set(ids)) == 30
