import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from arxplore.config import AppSettings
from arxplore.parsing.layout_client import LayoutParserClient, normalize_segment, normalize_segments

BASE_URL = "http://layout-parser:5060"
RAW_SEGMENT: dict[str, Any] = {
    "left": 50,
    "top": "120.5",
    "width": 500.0,
    "height": 12,
    "page_number": "2",
    "page_width": 612,
    "page_height": 792,
    "text": None,
    "type": "Text",
    "extra": "ignored",
}

type Handler = Callable[[httpx.Request], httpx.Response]


def mocked_client(handler: Handler, requests: list[httpx.Request]) -> httpx.AsyncClient:
    def record(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    return httpx.AsyncClient(transport=httpx.MockTransport(record))


def respond_json(payload: Any, status_code: int = 200) -> Handler:
    return lambda request: httpx.Response(status_code, content=json.dumps(payload).encode())


def test_normalize_segment_coerces_the_required_fields():
    assert normalize_segment(RAW_SEGMENT) == {
        "left": 50.0,
        "top": 120.5,
        "width": 500.0,
        "height": 12.0,
        "page_number": 2,
        "page_width": 612.0,
        "page_height": 792.0,
        "text": "",
        "type": "Text",
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"segments": []}, r"Layout parser response must be a list of segments\."),
        (["not an object"], r"Layout parser segment must be a JSON object\."),
        (
            [{"text": "x"}],
            "missing required fields: height, left, page_height, page_number, page_width, top, type, width",
        ),
    ],
)
def test_normalize_segments_rejects_invalid_payloads(payload, message):
    with pytest.raises(ValueError, match=message):
        normalize_segments(payload)


async def test_posts_the_pdf_as_a_multipart_form():
    requests: list[httpx.Request] = []
    async with mocked_client(respond_json([RAW_SEGMENT]), requests) as http_client:
        client = LayoutParserClient(f" {BASE_URL}/ ", http_client=http_client)

        segments = await client.analyze_pdf_bytes(b"%PDF-1.4 fake")

    assert [segment["page_number"] for segment in segments] == [2]
    assert len(requests) == 1
    request = requests[0]
    body = request.read()
    assert request.method == "POST"
    assert str(request.url) == f"{BASE_URL}/"
    assert b'name="fast"\r\n\r\nfalse\r\n' in body
    assert b'name="parse_tables_and_math"\r\n\r\ntrue\r\n' in body
    assert b'name="file"; filename="document.pdf"\r\nContent-Type: application/pdf\r\n\r\n%PDF-1.4 fake' in body


async def test_form_flags_follow_the_client_and_call_options():
    requests: list[httpx.Request] = []
    async with mocked_client(respond_json([]), requests) as http_client:
        client = LayoutParserClient(BASE_URL, fast=True, parse_tables_and_math=False, http_client=http_client)

        await client.analyze_pdf_bytes(b"%PDF")
        await client.analyze_pdf_bytes(b"%PDF", fast=False, parse_tables_and_math=True, filename="x.pdf")

    first, second = (request.read() for request in requests)
    assert b'name="fast"\r\n\r\ntrue\r\n' in first
    assert b'name="parse_tables_and_math"\r\n\r\nfalse\r\n' in first
    assert b'name="fast"\r\n\r\nfalse\r\n' in second
    assert b'name="parse_tables_and_math"\r\n\r\ntrue\r\n' in second
    assert b'filename="x.pdf"' in second


async def test_http_errors_are_raised_without_retrying():
    requests: list[httpx.Request] = []
    async with mocked_client(respond_json({"detail": "busy"}, 503), requests) as http_client:
        client = LayoutParserClient(BASE_URL, http_client=http_client)

        with pytest.raises(httpx.HTTPStatusError):
            await client.analyze_pdf_bytes(b"%PDF")

    assert len(requests) == 1


async def test_invalid_json_is_a_value_error():
    requests: list[httpx.Request] = []
    async with mocked_client(lambda request: httpx.Response(200, content=b"<html>"), requests) as http_client:
        client = LayoutParserClient(BASE_URL, http_client=http_client)

        with pytest.raises(ValueError):
            await client.analyze_pdf_bytes(b"%PDF")


async def test_empty_content_and_missing_base_url_are_rejected_before_any_request():
    with pytest.raises(ValueError, match=r"PDF content is empty\."):
        await LayoutParserClient(BASE_URL).analyze_pdf_bytes(b"")
    for base_url in ("  ", " / "):
        with pytest.raises(ValueError, match=r"LAYOUT_PARSER_BASE_URL is not configured\."):
            await LayoutParserClient(base_url).analyze_pdf_bytes(b"%PDF")


@respx.mock
async def test_without_an_injected_client_a_request_scoped_client_is_used():
    route = respx.post(f"{BASE_URL}/").mock(return_value=httpx.Response(200, json=[RAW_SEGMENT]))
    client = LayoutParserClient(BASE_URL, timeout_seconds=5)

    segments = await client.analyze_pdf_bytes(b"%PDF")

    assert route.call_count == 1
    assert len(segments) == 1


@respx.mock
async def test_transport_errors_are_raised_without_retrying():
    route = respx.post(f"{BASE_URL}/").mock(side_effect=httpx.ConnectError("down"))

    with pytest.raises(httpx.ConnectError):
        await LayoutParserClient(BASE_URL).analyze_pdf_bytes(b"%PDF")

    assert route.call_count == 1


def test_is_configured():
    assert LayoutParserClient(BASE_URL).is_configured()
    assert not LayoutParserClient(None).is_configured()
    assert not LayoutParserClient("  ").is_configured()
    assert LayoutParserClient(" / ").is_configured()


def test_from_settings(clean_env: Path):
    settings = AppSettings.model_validate(
        {
            "LAYOUT_PARSER_BASE_URL": f"{BASE_URL}/",
            "LAYOUT_PARSER_TIMEOUT_SECONDS": "120",
            "LAYOUT_PARSER_FAST": "true",
            "LAYOUT_PARSER_PARSE_TABLES_AND_MATH": "false",
        }
    )

    client = LayoutParserClient.from_settings(settings)

    assert (client.base_url, client.timeout_seconds, client.fast, client.parse_tables_and_math) == (
        BASE_URL,
        120,
        True,
        False,
    )
    assert not LayoutParserClient.from_settings(AppSettings.model_validate({})).is_configured()
