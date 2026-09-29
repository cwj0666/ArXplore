from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from arxplore.db.repositories import ai_cache, papers
from arxplore.domain.papers import AiOverview, DetailedSummary, EnrichmentCandidate, Paper, PaperUpsert

pytestmark = pytest.mark.integration

LEGACY_RECENT_SQL = """
                SELECT arxiv_id, title, authors, abstract, primary_category, categories, pdf_url,
                       published_at, updated_at, upvotes, github_url, github_stars, citation_count
                FROM papers
                ORDER BY COALESCE(published_at, updated_at_utc) DESC
                LIMIT %s
                """

FULL_PAPER: dict[str, Any] = {
    "arxiv_id": "2601.00001",
    "title": "Diffusion Transformers",
    "authors": [{"name": "Ada"}],
    "abstract": "We scale diffusion.",
    "primary_category": "cs.CV",
    "categories": ["cs.CV", "cs.LG"],
    "pdf_url": "https://arxiv.org/pdf/2601.00001",
    "published_at": "2026-01-02T03:04:05Z",
    "updated_at": "2026-01-03T00:00:00+09:00",
    "upvotes": 12,
    "github_url": "https://github.com/example/dit",
    "github_stars": 40,
    "citation_count": 3,
    "source": "hf_daily_papers+arxiv",
}


async def _paper_row(probe, arxiv_id: str) -> dict[str, Any]:
    row = await probe.row("SELECT * FROM papers WHERE arxiv_id = %s", (arxiv_id,))
    assert row is not None
    return row


async def test_insert_uses_defaults_for_missing_values(pool, probe) -> None:
    assert await papers.save_paper(pool, PaperUpsert(arxiv_id="2601.00002")) == "2601.00002"

    row = await _paper_row(probe, "2601.00002")
    assert (row["title"], row["authors"], row["abstract"], row["categories"]) == ("", [], "", [])
    assert (row["upvotes"], row["source"], row["primary_category"], row["published_at"]) == (
        0,
        "hf_daily_papers",
        None,
        None,
    )
    assert row["updated_at_utc"] is not None


async def test_update_keeps_enriched_values_when_new_ones_are_empty(pool, probe) -> None:
    await papers.save_paper(pool, PaperUpsert.from_mapping(FULL_PAPER))
    enriched = await _paper_row(probe, "2601.00001")

    await papers.save_paper(
        pool,
        PaperUpsert.from_mapping({"arxiv_id": "2601.00001", "title": "HF title", "categories": []}),
    )

    row = await _paper_row(probe, "2601.00001")
    for kept in (
        "primary_category",
        "categories",
        "published_at",
        "updated_at",
        "upvotes",
        "github_url",
        "github_stars",
        "citation_count",
    ):
        assert row[kept] == enriched[kept], kept
    assert (row["title"], row["authors"], row["abstract"], row["pdf_url"], row["source"]) == (
        "HF title",
        [],
        "",
        None,
        "hf_daily_papers",
    )
    assert row["updated_at_utc"] > enriched["updated_at_utc"]


async def test_update_replaces_values_that_are_present(pool, probe) -> None:
    await papers.save_paper(pool, PaperUpsert.from_mapping(FULL_PAPER))

    await papers.save_paper(
        pool,
        PaperUpsert.from_mapping(
            {
                **FULL_PAPER,
                "primary_category": "cs.AI",
                "categories": ["cs.AI"],
                "published_at": "2026-02-01T00:00:00Z",
                "upvotes": "0",
                "github_stars": "41",
            }
        ),
    )

    row = await _paper_row(probe, "2601.00001")
    assert (row["primary_category"], row["categories"], row["upvotes"], row["github_stars"]) == (
        "cs.AI",
        ["cs.AI"],
        0,
        41,
    )
    assert row["published_at"] == datetime(2026, 2, 1, tzinfo=UTC)


async def test_arxiv_id_is_trimmed_and_must_not_be_empty(pool) -> None:
    assert await papers.save_paper(pool, PaperUpsert(arxiv_id=" 2601.00003 ", title="T")) == "2601.00003"
    assert await papers.get_paper(pool, "2601.00003") is not None

    with pytest.raises(ValueError, match="arxiv_id"):
        await papers.save_paper(pool, PaperUpsert(arxiv_id="  ", title="T"))


async def test_text_fields_are_cleaned_of_surrogates(pool) -> None:
    await papers.save_paper(
        pool,
        PaperUpsert(arxiv_id="2601.00004", title="T\ud800", authors=[{"na\udfffme": "A\ud800"}], abstract="x\udc00y"),
    )

    stored = await papers.get_paper(pool, "2601.00004")
    assert stored is not None
    assert (stored.title, stored.authors, stored.abstract) == ("T", [{"name": "A"}], "xy")


async def test_get_paper_maps_the_row(pool) -> None:
    await papers.save_paper(pool, PaperUpsert.from_mapping(FULL_PAPER))

    assert await papers.get_paper(pool, "missing") is None
    assert await papers.get_paper(pool, "2601.00001") == Paper(
        arxiv_id="2601.00001",
        title="Diffusion Transformers",
        authors=[{"name": "Ada"}],
        abstract="We scale diffusion.",
        primary_category="cs.CV",
        categories=["cs.CV", "cs.LG"],
        pdf_url="https://arxiv.org/pdf/2601.00001",
        published_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        updated_at=datetime(2026, 1, 2, 15, 0, tzinfo=UTC),
        upvotes=12,
        github_url="https://github.com/example/dit",
        github_stars=40,
        citation_count=3,
    )


async def test_recent_papers_order_by_published_at_then_save_time(pool) -> None:
    base = datetime(2026, 3, 1, tzinfo=UTC)
    await papers.save_paper(pool, PaperUpsert(arxiv_id="old", published_at=base))
    await papers.save_paper(pool, PaperUpsert(arxiv_id="new", published_at=base + timedelta(days=2)))
    await papers.save_paper(pool, PaperUpsert(arxiv_id="undated"))

    recent = await papers.list_recent_papers(pool, limit=10)

    assert [paper.arxiv_id for paper in recent] == ["undated", "new", "old"]
    assert [paper.arxiv_id for paper in await papers.list_recent_papers(pool, limit=0)] == ["undated"]


async def test_recent_paper_ties_follow_the_legacy_statement(pool, probe) -> None:
    same_day = datetime(2026, 9, 14, tzinfo=UTC)
    for index in range(40):
        await papers.save_paper(
            pool,
            PaperUpsert(arxiv_id=f"2609.{(index * 37) % 101:05d}", upvotes=index, published_at=same_day),
        )
    await probe.execute("UPDATE papers SET upvotes = upvotes + 1 WHERE upvotes %% 3 = 0")

    for limit in (5, 30, 1500):
        legacy = [row["arxiv_id"] for row in await probe.rows(LEGACY_RECENT_SQL, (limit,))]
        assert [paper.arxiv_id for paper in await papers.list_recent_papers(pool, limit=limit)] == legacy


async def test_enrichment_candidates_lack_arxiv_metadata(pool) -> None:
    await papers.save_paper(pool, PaperUpsert.from_mapping(FULL_PAPER))
    await papers.save_paper(pool, PaperUpsert(arxiv_id="no-category", categories=["cs.CL"]))
    await papers.save_paper(pool, PaperUpsert(arxiv_id="no-categories", primary_category="cs.CL"))
    await papers.save_paper(
        pool,
        PaperUpsert(arxiv_id="raw", primary_category="cs.CL", categories=["cs.CL"], source="hf_daily_papers_raw"),
    )

    candidates = await papers.list_papers_missing_arxiv_metadata(pool, limit=10)

    assert sorted(candidate.arxiv_id for candidate in candidates) == ["no-categories", "no-category", "raw"]
    assert all(isinstance(candidate, EnrichmentCandidate) for candidate in candidates)
    assert {candidate.arxiv_id: candidate.source for candidate in candidates}["raw"] == "hf_daily_papers_raw"
    assert len(await papers.list_papers_missing_arxiv_metadata(pool, limit=0)) == 1


async def test_overview_cache_round_trip(pool) -> None:
    await papers.save_paper(pool, PaperUpsert(arxiv_id="2601.00001", title="T"))
    assert await ai_cache.get_overview(pool, "2601.00001") is None

    await ai_cache.save_overview(pool, "2601.00001", overview="First", key_findings=["a", "b"], model="gpt-5-mini")
    await ai_cache.save_overview(pool, "2601.00001", overview="Second", key_findings=["c"], model="gpt-5")

    cached = await ai_cache.get_overview(pool, "2601.00001")
    assert cached is not None
    assert (cached.overview, cached.key_findings, cached.model) == ("Second", ["c"], "gpt-5")
    assert isinstance(cached, AiOverview)
    assert cached.generated_at is not None


async def test_detailed_summaries_are_cached_per_model(pool) -> None:
    await papers.save_paper(pool, PaperUpsert(arxiv_id="2601.00001", title="T"))

    await ai_cache.save_detailed_summary(pool, "2601.00001", summary="mini", model="gpt-5-mini")
    await ai_cache.save_detailed_summary(pool, "2601.00001", summary="full", model="gpt-5")
    await ai_cache.save_detailed_summary(pool, "2601.00001", summary="mini 2", model="gpt-5-mini")

    mini = await ai_cache.get_detailed_summary(pool, "2601.00001", "gpt-5-mini")
    assert mini is not None
    assert (mini.summary, mini.model) == ("mini 2", "gpt-5-mini")
    assert isinstance(mini, DetailedSummary)
    full = await ai_cache.get_detailed_summary(pool, "2601.00001", "gpt-5")
    assert full is not None
    assert full.summary == "full"
    assert await ai_cache.get_detailed_summary(pool, "2601.00001", "gpt-4o") is None
