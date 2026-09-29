import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from typing import Any, Self

import httpx

from arxplore.config import AppSettings
from arxplore.parsing.models import LayoutSegment

DEFAULT_FILENAME = "document.pdf"
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1.0
RETRYABLE_STATUS_CODES = frozenset({422, *range(500, 600)})
RETRYABLE_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (httpx.NetworkError, httpx.RemoteProtocolError)
REQUIRED_SEGMENT_FIELDS = frozenset(
    {"left", "top", "width", "height", "page_number", "page_width", "page_height", "text", "type"}
)
LAYOUT_CLIENT_ERRORS: tuple[type[Exception], ...] = (httpx.HTTPError, httpx.InvalidURL, ValueError)

type Sleep = Callable[[float], Awaitable[None]]

logger = logging.getLogger(__name__)


def _form_flag(value: bool) -> str:
    return str(value).lower()


def _invalid_field(field: str, value: Any) -> ValueError:
    return ValueError(f"Layout parser segment field {field} is not a valid number: {value!r}")


def _coordinate(segment: dict[str, Any], field: str) -> float:
    value = segment[field]
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise _invalid_field(field, value)
    try:
        number = float(value)
    except ValueError as exc:
        raise _invalid_field(field, value) from exc
    if not math.isfinite(number):
        raise _invalid_field(field, value)
    return number


def _page_number(segment: dict[str, Any]) -> int:
    value = segment["page_number"]
    if isinstance(value, bool):
        raise _invalid_field("page_number", value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise _invalid_field("page_number", value) from exc
    raise _invalid_field("page_number", value)


def normalize_segment(segment: Any) -> LayoutSegment:
    if not isinstance(segment, dict):
        raise ValueError("Layout parser segment must be a JSON object.")
    missing = REQUIRED_SEGMENT_FIELDS.difference(segment)
    if missing:
        raise ValueError(f"Layout parser segment is missing required fields: {', '.join(sorted(missing))}")
    return {
        "left": _coordinate(segment, "left"),
        "top": _coordinate(segment, "top"),
        "width": _coordinate(segment, "width"),
        "height": _coordinate(segment, "height"),
        "page_number": _page_number(segment),
        "page_width": _coordinate(segment, "page_width"),
        "page_height": _coordinate(segment, "page_height"),
        "text": str(segment.get("text") or ""),
        "type": str(segment.get("type") or ""),
    }


def is_retryable_error(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_STATUS_CODES
    return isinstance(error, RETRYABLE_TRANSPORT_ERRORS)


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
        retries: int = DEFAULT_RETRIES,
        backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
        http_client: httpx.AsyncClient | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if retries < 0:
            raise ValueError("retries must not be negative.")
        if backoff_seconds < 0:
            raise ValueError("backoff_seconds must not be negative.")
        self._configured_url = (base_url or "").strip()
        self.base_url = self._configured_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.fast = fast
        self.parse_tables_and_math = parse_tables_and_math
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self._http_client = http_client
        self._sleep = sleep

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

    def retry_delay(self, retry: int) -> float:
        return float(self.backoff_seconds * 2**retry)

    async def _post(self, url: str, files: dict[str, Any], form: dict[str, str]) -> httpx.Response:
        if self._http_client is not None:
            response = await self._http_client.post(url, files=files, data=form, timeout=self.timeout_seconds)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, files=files, data=form)
        response.raise_for_status()
        return response

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
        files: dict[str, Any] = {"file": (filename, content, "application/pdf")}
        url = f"{self.base_url}/"
        retry = 0
        while True:
            try:
                response = await self._post(url, files, form)
            except LAYOUT_CLIENT_ERRORS as exc:
                if retry >= self.retries or not is_retryable_error(exc):
                    raise
                delay = self.retry_delay(retry)
                logger.warning(
                    "layout parser attempt %d of %d failed: %s; retrying in %.1fs",
                    retry + 1,
                    self.retries + 1,
                    exc,
                    delay,
                )
                await self._sleep(delay)
                retry += 1
            else:
                return normalize_segments(response.json())
