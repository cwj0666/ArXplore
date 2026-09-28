import asyncio
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any, Self
from urllib.parse import urlsplit, urlunsplit

import httpx

from arxplore.config import AppSettings

USER_AGENT = "arxplore/2.0"
DAILY_PAPERS_API_PATH = "/api/daily_papers"
REQUEST_ATTEMPTS = 3
RETRY_BASE_SECONDS = 2.0
RETRY_AFTER_CAP_SECONDS = 60.0
RATE_LIMITED_STATUS = 429
SERVER_ERROR_STATUS = 500

type Sleep = Callable[[float], Awaitable[None]]


class DailyPapersFormatError(ValueError):
    pass


def daily_papers_api_url(base_url: str) -> str:
    parts = urlsplit(base_url.strip())
    return urlunsplit((parts.scheme, parts.netloc, DAILY_PAPERS_API_PATH, "", ""))


def daily_papers_from_response(payload: Any) -> list[Any]:
    if not isinstance(payload, list):
        raise DailyPapersFormatError("HF Daily Papers API 응답을 list로 해석하지 못했습니다.")
    return payload


def is_retryable(error: httpx.HTTPError) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status == RATE_LIMITED_STATUS or status >= SERVER_ERROR_STATUS
    return isinstance(error, httpx.TransportError)


def retry_after_seconds(error: httpx.HTTPError) -> float | None:
    if not isinstance(error, httpx.HTTPStatusError):
        return None
    value = error.response.headers.get("Retry-After", "").strip()
    try:
        seconds = float(value)
    except ValueError:
        return None
    return min(max(0.0, seconds), RETRY_AFTER_CAP_SECONDS)


def retry_delay_seconds(error: httpx.HTTPError, attempt: int) -> float:
    announced = retry_after_seconds(error)
    return announced if announced is not None else RETRY_BASE_SECONDS * 2 ** (attempt - 1)


class HfDailyPapersClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        http: httpx.AsyncClient | None = None,
        attempts: int = REQUEST_ATTEMPTS,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.api_url = daily_papers_api_url(base_url)
        self.timeout_seconds = timeout_seconds
        self.attempts = max(1, attempts)
        self._sleep = sleep
        self._http = http or httpx.AsyncClient(follow_redirects=True)
        self._owns_http = http is None

    @classmethod
    def from_settings(cls, settings: AppSettings, *, http: httpx.AsyncClient | None = None) -> Self:
        return cls(
            base_url=settings.hf_daily_papers_base_url,
            timeout_seconds=settings.hf_request_timeout_seconds,
            http=http,
        )

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def _get(self, target_date: date) -> httpx.Response:
        response = await self._http.get(
            self.api_url,
            params={"date": target_date.isoformat()},
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return response

    async def fetch_daily_papers(self, target_date: date) -> list[Any]:
        attempt = 1
        while True:
            try:
                response = await self._get(target_date)
            except httpx.HTTPError as exc:
                if attempt >= self.attempts or not is_retryable(exc):
                    raise
                await self._sleep(retry_delay_seconds(exc, attempt))
                attempt += 1
                continue
            return daily_papers_from_response(response.json())
