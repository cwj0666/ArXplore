import html
import json
from datetime import date
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx

from arxplore.config import AppSettings
from arxplore.sources import arxiv as arxiv_module
from arxplore.sources import arxiv_metadata as arxiv_metadata_module
from arxplore.sources.arxiv import ArxivClient
from arxplore.sources.arxiv_metadata import fetch_arxiv_metadata, metadata_batches, unique_arxiv_ids
from arxplore.sources.hf_daily import (
    DailyPapersFormatError,
    HfDailyPapersClient,
    daily_papers_from_page,
    extract_daily_papers_props,
)

ARXIV_URL = "https://export.arxiv.org/api/query"
PAPERS = [{"paper": {"id": "2609.00001", "title": "A & B"}}, {"paper": {"id": "2609.00002"}}]


def _page(props: object) -> str:
    encoded = html.escape(json.dumps(props), quote=True)
    return f'<html><div data-target="DailyPapers" data-props="{encoded}"></div></html>'


def _feed(*arxiv_ids: str) -> str:
    entries = "".join(
        f"""
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}v1</id>
    <title>Title {arxiv_id}</title>
    <summary>Summary {arxiv_id}</summary>
    <published>2026-09-01T00:00:00Z</published>
    <updated>2026-09-02T00:00:00Z</updated>
    <link href="https://arxiv.org/pdf/{arxiv_id}v1" rel="related" type="application/pdf" title="pdf"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:primary_category term="cs.CL"/>
    <author><name>Ada Fake</name></author>
  </entry>"""
        for arxiv_id in arxiv_ids
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">'
        f"{entries}</feed>"
    )


def test_daily_papers_are_read_from_the_unescaped_data_props() -> None:
    assert daily_papers_from_page(_page({"dailyPapers": PAPERS, "other": 1})) == PAPERS
    assert extract_daily_papers_props(_page({"dailyPapers": []})) == {"dailyPapers": []}


def test_missing_daily_papers_key_means_an_empty_list() -> None:
    assert daily_papers_from_page(_page({"other": 1})) == []


@pytest.mark.parametrize(
    "page",
    [
        "<html>no props here</html>",
        '<div data-target="DailyPapers" data-props="{&quot;dailyPapers&quot;: []}',
        _page(["not", "a", "dict"]),
        _page({"dailyPapers": {"not": "a list"}}),
    ],
)
def test_malformed_pages_are_rejected(page: str) -> None:
    with pytest.raises(DailyPapersFormatError):
        daily_papers_from_page(page)


@respx.mock
async def test_fetch_reads_the_dated_page_with_the_configured_timeout() -> None:
    route = respx.get("https://huggingface.co/papers/date/2026-09-15").respond(200, text=_page({"dailyPapers": PAPERS}))
    client = HfDailyPapersClient.from_settings(AppSettings.model_validate({"HF_REQUEST_TIMEOUT_SECONDS": "7"}))
    try:
        assert await client.fetch_daily_papers(date(2026, 9, 15)) == PAPERS
    finally:
        await client.aclose()
    assert route.call_count == 1
    assert route.calls.last.request.extensions["timeout"]["read"] == 7


@respx.mock
async def test_fetch_raises_http_errors() -> None:
    respx.get("https://huggingface.co/papers/date/2026-09-15").respond(429)
    client = HfDailyPapersClient(base_url="https://huggingface.co/papers", timeout_seconds=5)
    try:
        with pytest.raises(httpx.HTTPStatusError, match="429"):
            await client.fetch_daily_papers(date(2026, 9, 15))
    finally:
        await client.aclose()


def test_metadata_ids_are_normalized_deduplicated_and_batched_in_order() -> None:
    ids = ["2609.00003v2", "https://arxiv.org/abs/2609.00001", "2609.00003", "", "2609.00002.pdf"]
    assert unique_arxiv_ids(ids) == ["2609.00003", "2609.00001", "2609.00002"]
    assert metadata_batches(ids, 2) == [["2609.00003", "2609.00001"], ["2609.00002"]]
    assert metadata_batches(ids, 0) == [["2609.00003"], ["2609.00001"], ["2609.00002"]]
    assert metadata_batches([], 25) == []


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    fake_asyncio = SimpleNamespace(sleep=fake_sleep)
    monkeypatch.setattr(arxiv_module, "asyncio", fake_asyncio)
    monkeypatch.setattr(arxiv_metadata_module, "asyncio", fake_asyncio)
    return recorded


@respx.mock
async def test_metadata_is_fetched_in_id_list_batches_with_a_pause_between(sleeps: list[float]) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        ids = parse_qs(urlsplit(str(request.url)).query)["id_list"][0].split(",")
        return httpx.Response(200, text=_feed(*ids))

    route = respx.get(ARXIV_URL).mock(side_effect=respond)
    client = ArxivClient(base_url=ARXIV_URL, timeout_seconds=45, delay_seconds=3.0)
    ids = [f"2609.{number:05d}" for number in range(1, 6)]

    papers = await fetch_arxiv_metadata(client, ids, batch_size=2)

    assert list(papers) == ids
    assert papers["2609.00001"].primary_category == "cs.CL"
    queries = [parse_qs(urlsplit(str(call.request.url)).query)["id_list"][0] for call in route.calls]
    assert queries == ["2609.00001,2609.00002", "2609.00003,2609.00004", "2609.00005"]
    assert [call.request.extensions["timeout"]["read"] for call in route.calls] == [45, 45, 45]
    assert sleeps == [3.0, 3.0]


@respx.mock
async def test_metadata_requests_retry_four_times_with_the_legacy_backoff(sleeps: list[float]) -> None:
    route = respx.get(ARXIV_URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.ReadTimeout("slow"),
            httpx.Response(500),
            httpx.Response(200, text=_feed("2609.00001")),
        ]
    )
    client = ArxivClient(base_url=ARXIV_URL, timeout_seconds=45, delay_seconds=1.0)

    papers = await fetch_arxiv_metadata(client, ["2609.00001"], batch_size=25)

    assert list(papers) == ["2609.00001"]
    assert route.call_count == 4
    assert sleeps == [2.0 * 1 * 5, 2.0 * 2 * 2, 2.0 * 3 * 1]


@respx.mock
async def test_metadata_raises_the_last_error_after_four_attempts(sleeps: list[float]) -> None:
    route = respx.get(ARXIV_URL).respond(429)
    client = ArxivClient(base_url=ARXIV_URL, timeout_seconds=45, delay_seconds=3.0)

    with pytest.raises(httpx.HTTPStatusError) as caught:
        await fetch_arxiv_metadata(client, ["2609.00001"], batch_size=25)

    assert caught.value.response.status_code == 429
    assert route.call_count == 4
    assert sleeps == [3.0 * 1 * 5, 3.0 * 2 * 5, 3.0 * 3 * 5]


async def test_no_ids_means_no_request() -> None:
    client = ArxivClient(base_url="http://127.0.0.1:9/never", timeout_seconds=1, delay_seconds=0)
    assert await fetch_arxiv_metadata(client, [], batch_size=25) == {}
