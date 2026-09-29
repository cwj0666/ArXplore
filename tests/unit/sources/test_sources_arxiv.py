from collections.abc import Callable
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from arxplore.config import AppSettings
from arxplore.sources.arxiv import (
    ArxivClient,
    ArxivPaper,
    build_search_query,
    normalize_arxiv_id,
    parse_arxiv_feed,
    retry_delay,
)

FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2609.90001v2</id>
    <title>Fake  External
      Study</title>
    <updated>2026-09-03T00:00:00Z</updated>
    <link href="https://arxiv.org/abs/2609.90001v2" rel="alternate" type="text/html"/>
    <link href="https://arxiv.org/pdf/2609.90001v2" rel="related" type="application/pdf" title="pdf"/>
    <summary>  We describe
      a fake paper. </summary>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.IR" scheme="http://arxiv.org/schemas/atom"/>
    <published>2026-09-01T00:00:00Z</published>
    <arxiv:primary_category term="cs.CL"/>
    <author><name> Ada Fake </name></author>
    <author><name></name></author>
    <author><name>Ben Placeholder</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/hep-th/9901001v1</id>
    <title>Old Style</title>
  </entry>
</feed>
"""


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2401.00001", "2401.00001"),
        ("  2401.00001v3  ", "2401.00001"),
        ("2401.00001.pdf", "2401.00001"),
        ("https://arxiv.org/abs/2401.00001v2", "2401.00001"),
        ("http://arxiv.org/pdf/2401.00001v2.pdf", "2401.00001"),
        ("https://arxiv.org/pdf/2401.00001", "2401.00001"),
        ("hep-th/9901001v2", "hep-th/9901001"),
        ("2401.00001v", "2401.00001v"),
        ("2401.00001vX", "2401.00001vX"),
        ("https://export.arxiv.org/abs/2401.00001v2", "https://export.arxiv.org/abs/2401.00001"),
        ("   ", ""),
    ],
)
def test_normalize_arxiv_id_keeps_the_legacy_rules(raw: str, expected: str) -> None:
    assert normalize_arxiv_id(raw) == expected


def test_feed_entries_are_parsed_by_normalized_id() -> None:
    papers = parse_arxiv_feed(FEED)

    assert list(papers) == ["2609.90001", "hep-th/9901001"]
    assert papers["2609.90001"] == ArxivPaper(
        arxiv_id="2609.90001",
        title="Fake External Study",
        abstract="We describe a fake paper.",
        authors=["Ada Fake", "Ben Placeholder"],
        published_at="2026-09-01T00:00:00Z",
        updated_at="2026-09-03T00:00:00Z",
        categories=["cs.CL", "cs.IR"],
        primary_category="cs.CL",
        pdf_url="https://arxiv.org/pdf/2609.90001v2",
    )
    assert papers["hep-th/9901001"] == ArxivPaper(arxiv_id="hep-th/9901001", title="Old Style", abstract="")


def test_search_query_quotes_at_most_eight_terms() -> None:
    assert build_search_query("  retrieval   augmented\ngeneration ") == (
        'all:"retrieval" AND all:"augmented" AND all:"generation"'
    )
    assert build_search_query(" ".join(f"t{index}" for index in range(10))).count("all:") == 8
    assert build_search_query("   ") == ""


def _client(handler: Callable[[httpx.Request], httpx.Response], **options: float) -> ArxivClient:
    return ArxivClient(
        base_url="http://arxiv.test/api/query",
        timeout_seconds=options.get("timeout_seconds", 45),
        delay_seconds=options.get("delay_seconds", 0),
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


async def test_search_sends_the_legacy_parameters_and_parses_the_feed() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=FEED)

    papers = await _client(handler).search_papers("fake  study", max_results=10)

    assert [paper.arxiv_id for paper in papers] == ["2609.90001", "hep-th/9901001"]
    (request,) = requests
    assert f"{request.url.scheme}://{request.url.host}{request.url.path}" == "http://arxiv.test/api/query"
    assert parse_qs(urlsplit(str(request.url)).query) == {
        "search_query": ['all:"fake" AND all:"study"'],
        "start": ["0"],
        "max_results": ["10"],
        "sortBy": ["relevance"],
        "sortOrder": ["descending"],
    }
    assert request.extensions["timeout"] == {"connect": 8, "read": 8, "write": 8, "pool": 8}


async def test_search_without_terms_sends_nothing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    assert await _client(handler).search_papers("   ") == []


async def test_search_tries_once_and_raises_the_http_error() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503, text="down")

    with pytest.raises(httpx.HTTPStatusError):
        await _client(handler).search_papers("fake study")
    assert len(calls) == 1


async def test_short_timeouts_are_kept() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=FEED)

    await _client(handler, timeout_seconds=3).search_papers("fake")

    assert requests[0].extensions["timeout"]["read"] == 3


async def test_get_with_retries_retries_until_success() -> None:
    statuses = iter([503, 429, 200])
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        seen.append(status)
        return httpx.Response(status, text=FEED)

    response = await _client(handler).get_with_retries(
        {"id_list": "2609.90001"}, timeout=5, retry_count=4, retry_delay_seconds=0
    )

    assert response.status_code == 200
    assert seen == [503, 429, 200]


def test_retry_delay_backs_off_more_for_timeouts_and_rate_limits() -> None:
    request = httpx.Request("GET", "http://arxiv.test/api/query")
    limited = httpx.HTTPStatusError("429", request=request, response=httpx.Response(429, request=request))
    failed = httpx.HTTPStatusError("503", request=request, response=httpx.Response(503, request=request))

    assert retry_delay(httpx.ReadTimeout("slow", request=request), attempt=2, base_seconds=3) == 12
    assert retry_delay(limited, attempt=2, base_seconds=3) == 30
    assert retry_delay(failed, attempt=2, base_seconds=3) == 6


def test_client_reads_the_arxiv_settings() -> None:
    settings = AppSettings.model_validate(
        {"ARXIV_API_BASE_URL": "http://arxiv.test/q", "ARXIV_REQUEST_TIMEOUT_SECONDS": "5"}
    )

    client = ArxivClient.from_settings(settings)

    assert (client.base_url, client.timeout_seconds, client.delay_seconds) == ("http://arxiv.test/q", 5, 3.0)
