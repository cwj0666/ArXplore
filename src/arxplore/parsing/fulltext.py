import asyncio
import logging
from typing import Protocol

from arxplore.parsing.cleaner import normalize_text
from arxplore.parsing.layout import parse_layout_segments
from arxplore.parsing.layout_client import LAYOUT_CLIENT_ERRORS
from arxplore.parsing.models import FALLBACK_SOURCE, FulltextParseResult, LayoutSegment, Section
from arxplore.parsing.pdf_text import parse_pdf_bytes
from arxplore.parsing.quality import fulltext_quality_metrics
from arxplore.parsing.sections import ABSTRACT_TITLE

logger = logging.getLogger(__name__)


class LayoutAnalyzer(Protocol):
    def is_configured(self) -> bool: ...

    async def analyze_pdf_bytes(self, content: bytes) -> list[LayoutSegment]: ...


def fallback_result(fallback_text: str) -> FulltextParseResult:
    cleaned = normalize_text(fallback_text)
    sections: list[Section] = [{"title": ABSTRACT_TITLE, "text": cleaned}] if cleaned else []
    return FulltextParseResult(
        text=cleaned,
        sections=sections,
        source=FALLBACK_SOURCE,
        quality_metrics=fulltext_quality_metrics(text=cleaned, sections=sections, source=FALLBACK_SOURCE),
    )


async def _parse_with_layout(content: bytes, layout_client: LayoutAnalyzer | None) -> FulltextParseResult | None:
    if layout_client is None or not layout_client.is_configured():
        return None
    try:
        segments = await layout_client.analyze_pdf_bytes(content)
    except LAYOUT_CLIENT_ERRORS as exc:
        logger.warning("layout parser unavailable: %s", exc)
        return None
    return await asyncio.to_thread(parse_layout_segments, segments)


async def parse_fulltext(
    pdf_content: bytes | None,
    *,
    fallback_text: str = "",
    layout_client: LayoutAnalyzer | None = None,
) -> FulltextParseResult:
    if pdf_content is None:
        return fallback_result(fallback_text)
    layout_result = await _parse_with_layout(pdf_content, layout_client)
    if layout_result is not None:
        return layout_result
    pdf_result = await asyncio.to_thread(parse_pdf_bytes, pdf_content)
    if pdf_result is not None:
        return pdf_result
    return fallback_result(fallback_text)
