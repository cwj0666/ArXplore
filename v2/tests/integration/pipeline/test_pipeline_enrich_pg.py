from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx

from arxplore.db.repositories import papers
from arxplore.domain.papers import PaperUpsert
from arxplore.pipeline.enrich import enrich_papers_metadata
from arxplore.sources import arxiv as arxiv_module
from arxplore.sources.arxiv import ArxivClient

pytestmark = pytest.mark.integration

ARXIV_URL = "https://export.arxiv.org/api/query"


def _entry(arxiv_id: str) -> str:
    return f"""
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}v2</id>
    <title>arXiv title {arxiv_id}</title>
    <summary>arXiv abstract {arxiv_id}</summary>
    <published>2026-08-01T00:00:00Z</published>
    <updated>2026-08-05T00:00:00Z</updated>
    <link href="https://arxiv.org/pdf/{arxiv_id}v2" rel="related" type="application/pdf" title="pdf"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:primary_category term="cs.LG"/>
    <author><name>Arxiv Author</name></author>
  </entry>"""


def _feed(request: httpx.Request, known: set[str]) -> httpx.Response:
    ids = parse_qs(urlsplit(str(request.url)).query)["id_list"][0].split(",")
    entries = "".join(_entry(arxiv_id) for arxiv_id in ids if arxiv_id in known)
    return httpx.Response(
        200,
        text=f'<feed xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">{entries}</feed>',
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(arxiv_module, "asyncio", SimpleNamespace(sleep=fake_sleep))


async def _hf_paper(pool: Any, arxiv_id: str, *, day: int, **fields: Any) -> None:
    await papers.save_paper(
        pool,
        PaperUpsert.from_mapping(
            {
                "arxiv_id": arxiv_id,
                "title": f"HF title {arxiv_id}",
                "authors": ["HF Author"],
                "abstract": f"HF abstract {arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "published_at": datetime(2026, 9, day, tzinfo=UTC),
                "upvotes": 17,
                "github_url": "https://github.com/hf/repo",
                "github_stars": 4,
                "source": "hf_daily_papers_raw",
                **fields,
            }
        ),
    )


def _client() -> ArxivClient:
    return ArxivClient(base_url=ARXIV_URL, timeout_seconds=45, delay_seconds=0)


@respx.mock
async def test_found_papers_take_arxiv_metadata_and_keep_hf_signals(pool, probe) -> None:
    await _hf_paper(pool, "2609.00001", day=10)
    await _hf_paper(pool, "2609.00002", day=11)
    respx.get(ARXIV_URL).mock(side_effect=lambda request: _feed(request, {"2609.00001"}))

    result = await enrich_papers_metadata(pool, _client(), batch_size=25)

    assert result.as_dict() == {
        "stage": "enrich_papers_metadata",
        "status": "success",
        "candidate_count": 2,
        "updated_count": 1,
        "skipped_count": 1,
        "max_papers": 30,
        "sample_updated": [
            {
                "arxiv_id": "2609.00001",
                "primary_category": "cs.LG",
                "category_count": 2,
                "pdf_url": "https://arxiv.org/pdf/2609.00001v2",
            }
        ],
    }
    row = await probe.row(
        """
        SELECT title, authors, abstract, primary_category, categories, pdf_url, published_at, updated_at,
               upvotes, github_url, github_stars, source
        FROM papers WHERE arxiv_id = '2609.00001'
        """
    )
    assert row == {
        "title": "arXiv title 2609.00001",
        "authors": ["Arxiv Author"],
        "abstract": "arXiv abstract 2609.00001",
        "primary_category": "cs.LG",
        "categories": ["cs.LG", "cs.AI"],
        "pdf_url": "https://arxiv.org/pdf/2609.00001v2",
        "published_at": datetime(2026, 8, 1, tzinfo=UTC),
        "updated_at": datetime(2026, 8, 5, tzinfo=UTC),
        "upvotes": 17,
        "github_url": "https://github.com/hf/repo",
        "github_stars": 4,
        "source": "hf_daily_papers+arxiv",
    }
    assert await probe.scalar("SELECT source FROM papers WHERE arxiv_id = '2609.00002'") == "hf_daily_papers_raw"


@respx.mock
async def test_candidates_are_the_newest_papers_missing_metadata(pool) -> None:
    for day in range(1, 5):
        await _hf_paper(pool, f"2609.0000{day}", day=day)
    await _hf_paper(
        pool, "2609.00009", day=9, source="hf_daily_papers+arxiv", primary_category="cs.CL", categories=["cs.CL"]
    )
    route = respx.get(ARXIV_URL).mock(side_effect=lambda request: _feed(request, set()))

    result = await enrich_papers_metadata(pool, _client(), batch_size=2, max_papers=3)

    queried = [parse_qs(urlsplit(str(call.request.url)).query)["id_list"][0] for call in route.calls]
    assert queried == ["2609.00004,2609.00003", "2609.00002"]
    assert (result.candidate_count, result.skipped_count, result.max_papers) == (3, 3, 3)


@pytest.mark.parametrize(
    "response,status",
    [(httpx.Response(429), "rate_limited"), (httpx.ReadTimeout("slow"), "timed_out")],
)
@respx.mock
async def test_rate_limits_and_timeouts_are_soft_failures(pool, response, status) -> None:
    for day in range(1, 8):
        await _hf_paper(pool, f"2609.0000{day}", day=day)
    respx.get(ARXIV_URL).mock(side_effect=[response] * 4)

    result = (await enrich_papers_metadata(pool, _client(), batch_size=25)).as_dict()

    assert result["status"] == status
    assert (result["candidate_count"], result["updated_count"], result["skipped_count"]) == (7, 0, 0)
    assert result["failed_arxiv_ids"] == ["2609.00007", "2609.00006", "2609.00005", "2609.00004", "2609.00003"]
    assert result["error"]
    assert result["sample_updated"] == []


@respx.mock
async def test_other_http_errors_are_raised(pool) -> None:
    await _hf_paper(pool, "2609.00001", day=1)
    respx.get(ARXIV_URL).respond(500)

    with pytest.raises(httpx.HTTPStatusError):
        await enrich_papers_metadata(pool, _client(), batch_size=25)
