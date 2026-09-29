import json
import logging
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("left", None),
        ("top", "not a number"),
        ("width", "nan"),
        ("height", float("inf")),
        ("page_width", True),
        ("page_height", [612]),
        ("page_number", None),
        ("page_number", 2.5),
        ("page_number", "2.0"),
        ("page_number", False),
    ],
)
def test_invalid_numeric_fields_fail_validation(field, value):
    with pytest.raises(ValueError, match=f"field {field} is not a valid number"):
        normalize_segment({**RAW_SEGMENT, field: value})


def test_integral_float_and_numeric_string_page_numbers_are_accepted():
    assert normalize_segment({**RAW_SEGMENT, "page_number": 3.0})["page_number"] == 3
    assert normalize_segment({**RAW_SEGMENT, "page_number": "4"})["page_number"] == 4


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


class RecordedSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def respond_in_turn(*responses: httpx.Response | Exception) -> Handler:
    queue = list(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return handle


@pytest.mark.parametrize("status_code", [422, 500, 502, 503, 504])
async def test_retryable_statuses_are_retried_with_exponential_backoff(caplog, status_code):
    requests: list[httpx.Request] = []
    sleep = RecordedSleep()
    async with mocked_client(respond_json({"detail": "busy"}, status_code), requests) as http_client:
        client = LayoutParserClient(BASE_URL, http_client=http_client, sleep=sleep)

        with caplog.at_level(logging.WARNING), pytest.raises(httpx.HTTPStatusError):
            await client.analyze_pdf_bytes(b"%PDF")

    assert len(requests) == 4
    assert sleep.delays == [1.0, 2.0, 4.0]
    assert "layout parser attempt 1 of 4 failed" in caplog.text
    assert "layout parser attempt 4 of 4" not in caplog.text


async def test_a_retry_that_succeeds_returns_the_segments():
    requests: list[httpx.Request] = []
    sleep = RecordedSleep()
    handler = respond_in_turn(
        httpx.Response(503),
        httpx.Response(422),
        httpx.Response(200, content=json.dumps([RAW_SEGMENT]).encode()),
    )
    async with mocked_client(handler, requests) as http_client:
        client = LayoutParserClient(BASE_URL, http_client=http_client, sleep=sleep)

        segments = await client.analyze_pdf_bytes(b"%PDF")

    assert len(segments) == 1
    assert len(requests) == 3
    assert sleep.delays == [1.0, 2.0]
    assert all(b"%PDF" in request.read() for request in requests)


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("refused"),
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadError("connection reset"),
        httpx.RemoteProtocolError("server disconnected"),
    ],
)
async def test_timeouts_and_connection_errors_are_retried(error):
    requests: list[httpx.Request] = []
    sleep = RecordedSleep()
    handler = respond_in_turn(error, httpx.Response(200, content=json.dumps([RAW_SEGMENT]).encode()))
    async with mocked_client(handler, requests) as http_client:
        client = LayoutParserClient(BASE_URL, http_client=http_client, sleep=sleep)

        segments = await client.analyze_pdf_bytes(b"%PDF")

    assert len(segments) == 1
    assert len(requests) == 2
    assert sleep.delays == [1.0]


@pytest.mark.parametrize("status_code", [400, 404, 413, 415])
async def test_other_client_errors_are_not_retried(status_code):
    requests: list[httpx.Request] = []
    sleep = RecordedSleep()
    async with mocked_client(respond_json({"detail": "bad"}, status_code), requests) as http_client:
        client = LayoutParserClient(BASE_URL, http_client=http_client, sleep=sleep)

        with pytest.raises(httpx.HTTPStatusError):
            await client.analyze_pdf_bytes(b"%PDF")

    assert len(requests) == 1
    assert sleep.delays == []


async def test_retry_count_and_backoff_come_from_the_client():
    requests: list[httpx.Request] = []
    sleep = RecordedSleep()
    async with mocked_client(respond_json({}, 503), requests) as http_client:
        once = LayoutParserClient(BASE_URL, http_client=http_client, sleep=sleep, retries=0)
        with pytest.raises(httpx.HTTPStatusError):
            await once.analyze_pdf_bytes(b"%PDF")
        twice = LayoutParserClient(BASE_URL, http_client=http_client, sleep=sleep, retries=2, backoff_seconds=0.5)
        with pytest.raises(httpx.HTTPStatusError):
            await twice.analyze_pdf_bytes(b"%PDF")

    assert len(requests) == 1 + 3
    assert sleep.delays == [0.5, 1.0]


def test_negative_retry_settings_are_rejected():
    with pytest.raises(ValueError, match="retries"):
        LayoutParserClient(BASE_URL, retries=-1)
    with pytest.raises(ValueError, match="backoff_seconds"):
        LayoutParserClient(BASE_URL, backoff_seconds=-0.1)


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
async def test_invalid_json_and_segments_are_not_retried():
    route = respx.post(f"{BASE_URL}/").mock(return_value=httpx.Response(200, json={"not": "a list"}))
    sleep = RecordedSleep()

    with pytest.raises(ValueError):
        await LayoutParserClient(BASE_URL, sleep=sleep).analyze_pdf_bytes(b"%PDF")

    assert route.call_count == 1
    assert sleep.delays == []


@respx.mock
async def test_a_request_scoped_client_also_retries():
    route = respx.post(f"{BASE_URL}/").mock(side_effect=httpx.ConnectError("down"))
    sleep = RecordedSleep()

    with pytest.raises(httpx.ConnectError):
        await LayoutParserClient(BASE_URL, sleep=sleep).analyze_pdf_bytes(b"%PDF")

    assert route.call_count == 4
    assert sleep.delays == [1.0, 2.0, 4.0]


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
