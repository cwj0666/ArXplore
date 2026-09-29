from datetime import date
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx

from arxplore.config import AppSettings
from arxplore.sources import arxiv as arxiv_module
from arxplore.sources import arxiv_metadata as arxiv_metadata_module
from arxplore.sources.arxiv import ArxivClient
from arxplore.sources.arxiv_metadata import fetch_arxiv_metadata, metadata_batches, unique_arxiv_ids
from arxplore.sources.hf_daily import DailyPapersFormatError, HfDailyPapersClient, daily_papers_api_url

ARXIV_URL = "https://export.arxiv.org/api/query"
PAPERS = [{"paper": {"id": "2609.00001", "title": "A & B"}}, {"paper": {"id": "2609.00002"}}]


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


API_URL = "https://huggingface.co/api/daily_papers"


@pytest.fixture
def hf_sleeps() -> list[float]:
    return []


def _hf_client(hf_sleeps: list[float], **overrides: Any) -> HfDailyPapersClient:
    async def fake_sleep(seconds: float) -> None:
        hf_sleeps.append(seconds)

    values: dict[str, Any] = {"base_url": "https://huggingface.co/papers", "timeout_seconds": 20, "sleep": fake_sleep}
    values.update(overrides)
    return HfDailyPapersClient(**values)


def test_the_api_lives_on_the_origin_of_the_configured_base_url() -> None:
    assert daily_papers_api_url("https://huggingface.co/papers") == API_URL
    assert daily_papers_api_url("http://127.0.0.1:18302/mirror/papers/") == "http://127.0.0.1:18302/api/daily_papers"
    client = HfDailyPapersClient.from_settings(AppSettings.model_validate({}))
    assert (client.api_url, client.timeout_seconds) == (API_URL, 20)


@respx.mock
async def test_fetch_asks_the_json_api_for_the_date_with_a_user_agent(hf_sleeps: list[float]) -> None:
    route = respx.get(API_URL).respond(200, json=PAPERS)
    client = _hf_client(hf_sleeps, timeout_seconds=7)

    assert await client.fetch_daily_papers(date(2026, 9, 15)) == PAPERS

    request = route.calls.last.request
    assert request.url.params["date"] == "2026-09-15"
    assert request.headers["User-Agent"] == "arxplore/2.0"
    assert request.headers["Accept"] == "application/json"
    assert request.extensions["timeout"]["read"] == 7
    assert hf_sleeps == []
    await client.aclose()


@respx.mock
async def test_a_weekend_day_is_an_empty_list(hf_sleeps: list[float]) -> None:
    respx.get(API_URL).respond(200, json=[])
    assert await _hf_client(hf_sleeps).fetch_daily_papers(date(2026, 9, 12)) == []


@respx.mock
async def test_server_errors_timeouts_and_rate_limits_are_retried_with_backoff(hf_sleeps: list[float]) -> None:
    route = respx.get(API_URL).mock(
        side_effect=[httpx.Response(503), httpx.ReadTimeout("slow"), httpx.Response(200, json=PAPERS)]
    )

    assert await _hf_client(hf_sleeps).fetch_daily_papers(date(2026, 9, 15)) == PAPERS
    assert route.call_count == 3
    assert hf_sleeps == [2.0, 4.0]


@respx.mock
async def test_retry_after_is_honoured_up_to_a_minute(hf_sleeps: list[float]) -> None:
    respx.get(API_URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(429, headers={"Retry-After": "900"}),
            httpx.Response(200, json=PAPERS),
        ]
    )

    assert await _hf_client(hf_sleeps).fetch_daily_papers(date(2026, 9, 15)) == PAPERS
    assert hf_sleeps == [7.0, 60.0]


@respx.mock
async def test_the_last_error_is_raised_after_the_attempts(hf_sleeps: list[float]) -> None:
    route = respx.get(API_URL).respond(429)

    with pytest.raises(httpx.HTTPStatusError, match="429"):
        await _hf_client(hf_sleeps).fetch_daily_papers(date(2026, 9, 15))
    assert route.call_count == 3


@respx.mock
async def test_client_errors_are_not_retried(hf_sleeps: list[float]) -> None:
    route = respx.get(API_URL).respond(404)

    with pytest.raises(httpx.HTTPStatusError):
        await _hf_client(hf_sleeps).fetch_daily_papers(date(2026, 9, 15))
    assert route.call_count == 1
    assert hf_sleeps == []


@respx.mock
async def test_a_response_that_is_not_a_list_is_rejected(hf_sleeps: list[float]) -> None:
    respx.get(API_URL).respond(200, json={"dailyPapers": PAPERS})
    with pytest.raises(DailyPapersFormatError):
        await _hf_client(hf_sleeps).fetch_daily_papers(date(2026, 9, 15))


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
