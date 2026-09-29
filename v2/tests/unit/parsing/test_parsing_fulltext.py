import logging

import httpx
import pytest

from arxplore.parsing.chunker import build_chunks
from arxplore.parsing.fulltext import fallback_result, parse_fulltext
from arxplore.parsing.layout_client import LayoutParserClient
from arxplore.parsing.models import LayoutSegment

BODY = "We propose a method for the alignment of large language models."


def layout_segment(text: str, segment_type: str, top: float) -> LayoutSegment:
    return {
        "left": 50.0,
        "top": top,
        "width": 500.0,
        "height": 12.0,
        "page_number": 1,
        "page_width": 612.0,
        "page_height": 792.0,
        "text": text,
        "type": segment_type,
    }


LAYOUT_SEGMENTS = [
    layout_segment("1 Introduction", "Section header", 10.0),
    layout_segment(BODY, "Text", 20.0),
]


class FakeLayoutClient:
    def __init__(
        self,
        *,
        configured: bool = True,
        segments: list[LayoutSegment] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.configured = configured
        self.segments = segments or []
        self.error = error
        self.calls: list[bytes] = []

    def is_configured(self) -> bool:
        return self.configured

    async def analyze_pdf_bytes(self, content: bytes) -> list[LayoutSegment]:
        self.calls.append(content)
        if self.error is not None:
            raise self.error
        return self.segments


def test_fallback_result_normalizes_the_abstract():
    result = fallback_result("  We align   models.  ")

    assert result.source == "fallback_abstract"
    assert result.text == "We align models."
    assert result.sections == [{"title": "Abstract", "text": "We align models."}]
    assert result.quality_metrics == {
        "parse_source": "fallback_abstract",
        "fallback_used": True,
        "text_length": 16,
        "section_count": 1,
        "avg_section_chars": 16.0,
        "max_section_chars": 16,
    }
    assert result.artifacts == {}
    assert result.parser_metadata == {}


def test_empty_fallback_has_no_sections():
    result = fallback_result("")

    assert result.sections == []
    assert result.quality_metrics["avg_section_chars"] == 0
    assert type(result.quality_metrics["avg_section_chars"]) is int


async def test_missing_pdf_falls_back_without_calling_the_layout_parser():
    client = FakeLayoutClient(segments=LAYOUT_SEGMENTS)

    result = await parse_fulltext(None, fallback_text="We align models.", layout_client=client)

    assert result.source == "fallback_abstract"
    assert client.calls == []


async def test_layout_parser_result_wins(text_pdf):
    client = FakeLayoutClient(segments=LAYOUT_SEGMENTS)
    pdf = text_pdf(["1 Introduction", BODY])

    result = await parse_fulltext(pdf, fallback_text="abstract", layout_client=client)

    assert result.source == "layout_pdf"
    assert client.calls == [pdf]
    assert [section["title"] for section in result.sections] == ["1 Introduction"]


async def test_unconfigured_layout_parser_uses_pypdf(text_pdf):
    client = FakeLayoutClient(configured=False, segments=LAYOUT_SEGMENTS)

    result = await parse_fulltext(text_pdf(["1 Introduction", BODY]), layout_client=client)

    assert result.source == "pdf"
    assert client.calls == []


async def test_no_layout_client_uses_pypdf(text_pdf):
    result = await parse_fulltext(text_pdf(["1 Introduction", BODY]))

    assert result.source == "pdf"


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("layout parser down"),
        httpx.ReadTimeout("read timed out"),
        httpx.HTTPStatusError(
            "503 Service Unavailable",
            request=httpx.Request("POST", "http://parser/"),
            response=httpx.Response(503),
        ),
        ValueError("Layout parser response must be a list of segments."),
    ],
)
async def test_layout_parser_errors_degrade_to_pypdf_and_are_logged(caplog, text_pdf, error):
    client = FakeLayoutClient(error=error)

    with caplog.at_level(logging.WARNING):
        result = await parse_fulltext(text_pdf(["1 Introduction", BODY]), layout_client=client)

    assert result.source == "pdf"
    assert f"layout parser unavailable: {error}" in caplog.text


async def test_layout_parser_that_keeps_failing_falls_back_to_pypdf_after_retries(caplog, text_pdf):
    calls: list[httpx.Request] = []
    delays: list[float] = []

    def unavailable(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503)

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(unavailable)) as http_client:
        client = LayoutParserClient("http://layout-parser:5060", http_client=http_client, sleep=record_sleep)
        with caplog.at_level(logging.WARNING):
            result = await parse_fulltext(text_pdf(["1 Introduction", BODY]), layout_client=client)

    assert result.source == "pdf"
    assert len(calls) == 4
    assert delays == [1.0, 2.0, 4.0]
    assert "layout parser unavailable" in caplog.text


async def test_unexpected_layout_errors_propagate(text_pdf):
    client = FakeLayoutClient(error=RuntimeError("bug"))

    with pytest.raises(RuntimeError):
        await parse_fulltext(text_pdf(["1 Introduction", BODY]), layout_client=client)


async def test_layout_segments_without_text_fall_through_to_pypdf(text_pdf):
    client = FakeLayoutClient(segments=[layout_segment("", "Picture", 1.0)])

    result = await parse_fulltext(text_pdf(["1 Introduction", BODY]), layout_client=client)

    assert result.source == "pdf"


@pytest.mark.parametrize("content", [b"", b"not a pdf"])
async def test_unreadable_pdf_falls_back_to_the_abstract(caplog, content):
    with caplog.at_level(logging.WARNING):
        result = await parse_fulltext(content, fallback_text="We align models.")

    assert result.source == "fallback_abstract"
    assert result.text == "We align models."
    assert "pypdf text extraction failed" in caplog.text


async def test_encrypted_pdf_falls_back_to_the_abstract(caplog, encrypted_pdf):
    with caplog.at_level(logging.WARNING):
        result = await parse_fulltext(
            encrypted_pdf(["1 Introduction", "Secret body text."]), fallback_text="We align models."
        )

    assert result.source == "fallback_abstract"
    assert "pypdf text extraction failed" in caplog.text


async def test_fallback_abstract_can_be_chunked():
    result = await parse_fulltext(None, fallback_text=BODY * 40)
    chunks = build_chunks(result.text, result.sections)

    assert chunks
    assert {chunk["section_title"] for chunk in chunks} == {"Abstract"}
