from typing import Any, Self

import httpx

from arxplore.config import AppSettings
from arxplore.parsing.models import LayoutSegment

DEFAULT_FILENAME = "document.pdf"
DEFAULT_TIMEOUT_SECONDS = 600.0
REQUIRED_SEGMENT_FIELDS = frozenset(
    {"left", "top", "width", "height", "page_number", "page_width", "page_height", "text", "type"}
)
LAYOUT_CLIENT_ERRORS: tuple[type[Exception], ...] = (httpx.HTTPError, httpx.InvalidURL, ValueError)


def _form_flag(value: bool) -> str:
    return str(value).lower()


def normalize_segment(segment: Any) -> LayoutSegment:
    if not isinstance(segment, dict):
        raise ValueError("Layout parser segment must be a JSON object.")
    missing = REQUIRED_SEGMENT_FIELDS.difference(segment)
    if missing:
        raise ValueError(f"Layout parser segment is missing required fields: {', '.join(sorted(missing))}")
    return {
        "left": float(segment["left"]),
        "top": float(segment["top"]),
        "width": float(segment["width"]),
        "height": float(segment["height"]),
        "page_number": int(segment["page_number"]),
        "page_width": float(segment["page_width"]),
        "page_height": float(segment["page_height"]),
        "text": str(segment.get("text") or ""),
        "type": str(segment.get("type") or ""),
    }


def normalize_segments(payload: Any) -> list[LayoutSegment]:
    if not isinstance(payload, list):
        raise ValueError("Layout parser response must be a list of segments.")
    return [normalize_segment(segment) for segment in payload]


class LayoutParserClient:
    def __init__(
        self,
        base_url: str | None,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        fast: bool = False,
        parse_tables_and_math: bool = True,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._configured_url = (base_url or "").strip()
        self.base_url = self._configured_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.fast = fast
        self.parse_tables_and_math = parse_tables_and_math
        self._http_client = http_client

    @classmethod
    def from_settings(cls, settings: AppSettings, *, http_client: httpx.AsyncClient | None = None) -> Self:
        return cls(
            settings.layout_parser_base_url,
            timeout_seconds=settings.layout_parser_timeout_seconds,
            fast=settings.layout_parser_fast,
            parse_tables_and_math=settings.layout_parser_parse_tables_and_math,
            http_client=http_client,
        )

    def is_configured(self) -> bool:
        return bool(self._configured_url)

    async def analyze_pdf_bytes(
        self,
        content: bytes,
        *,
        filename: str = DEFAULT_FILENAME,
        fast: bool | None = None,
        parse_tables_and_math: bool | None = None,
    ) -> list[LayoutSegment]:
        if not content:
            raise ValueError("PDF content is empty.")
        if not self.base_url:
            raise ValueError("LAYOUT_PARSER_BASE_URL is not configured.")
        form = {
            "fast": _form_flag(self.fast if fast is None else fast),
            "parse_tables_and_math": _form_flag(
                self.parse_tables_and_math if parse_tables_and_math is None else parse_tables_and_math
            ),
        }
        files = {"file": (filename, content, "application/pdf")}
        url = f"{self.base_url}/"
        if self._http_client is not None:
            response = await self._http_client.post(url, files=files, data=form, timeout=self.timeout_seconds)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, files=files, data=form)
        response.raise_for_status()
        return normalize_segments(response.json())
