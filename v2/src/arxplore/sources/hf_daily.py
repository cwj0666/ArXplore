import html
import json
from datetime import date
from typing import Any, Self

import httpx

from arxplore.config import AppSettings

DAILY_PAPERS_PROPS_MARKER = 'data-target="DailyPapers" data-props="'
DAILY_PAPERS_KEY = "dailyPapers"


class DailyPapersFormatError(ValueError):
    pass


def extract_daily_papers_props(page_html: str) -> dict[str, Any]:
    start = page_html.find(DAILY_PAPERS_PROPS_MARKER)
    if start == -1:
        raise DailyPapersFormatError("HF Daily Papers 페이지에서 DailyPapers data-props를 찾지 못했습니다.")
    start += len(DAILY_PAPERS_PROPS_MARKER)
    end = page_html.find('"', start)
    if end == -1:
        raise DailyPapersFormatError("HF Daily Papers data-props 속성이 비정상적으로 종료되었습니다.")
    props = json.loads(html.unescape(page_html[start:end]))
    if not isinstance(props, dict):
        raise DailyPapersFormatError("HF Daily Papers data-props를 dict로 해석하지 못했습니다.")
    return props


def daily_papers_from_page(page_html: str) -> list[Any]:
    daily_papers = extract_daily_papers_props(page_html).get(DAILY_PAPERS_KEY, [])
    if not isinstance(daily_papers, list):
        raise DailyPapersFormatError("HF Daily Papers 응답에서 dailyPapers 필드를 list로 해석하지 못했습니다.")
    return daily_papers


class HfDailyPapersClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
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

    def page_url(self, target_date: date) -> str:
        return f"{self.base_url}/date/{target_date.isoformat()}"

    async def fetch_daily_papers(self, target_date: date) -> list[Any]:
        response = await self._http.get(self.page_url(target_date), timeout=self.timeout_seconds)
        response.raise_for_status()
        return daily_papers_from_page(response.text)
