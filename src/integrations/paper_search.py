from __future__ import annotations

import html
import json
import time
import xml.etree.ElementTree as ET
from datetime import date as date_cls
from typing import Any

import requests

from src.shared import AppSettings, get_settings

ATOM_NAMESPACE = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class PaperSearchClient:
    """AI 논문 수집용 API 클라이언트의 공통 진입점."""

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.session = session or requests.Session()

    def fetch_daily_papers(self, date: str) -> list[dict[str, Any]]:
        """HF Daily Papers 페이지에서 날짜별 큐레이션 논문 목록을 추출한다."""
        target_date = self._validate_date(date)
        response = self.session.get(
            f"{self.settings.hf_daily_papers_base_url}/date/{target_date}",
            timeout=self.settings.hf_request_timeout_seconds,
        )
        response.raise_for_status()

        props = self._extract_daily_papers_props(response.text)
        daily_papers = props.get("dailyPapers", [])
        if not isinstance(daily_papers, list):
            raise ValueError("HF Daily Papers 응답에서 dailyPapers 필드를 list로 해석하지 못했습니다.")
        return daily_papers

    @staticmethod
    def normalize_arxiv_id(value: str) -> str:
        return PaperSearchClient._normalize_arxiv_id(value)

    def fetch_arxiv_metadata(self, arxiv_ids: list[str]) -> dict[str, dict[str, Any]]:
        """arXiv API에서 카테고리, PDF 링크, 발행일 등 메타데이터를 보강한다."""
        normalized_ids = [self._normalize_arxiv_id(arxiv_id) for arxiv_id in arxiv_ids if arxiv_id]
        unique_ids = list(dict.fromkeys(normalized_ids))
        if not unique_ids:
            return {}

        batch_size = max(1, self.settings.arxiv_request_batch_size)
        results: dict[str, dict[str, Any]] = {}

        for index in range(0, len(unique_ids), batch_size):
            batch = unique_ids[index : index + batch_size]
            params = {"id_list": ",".join(batch)}
            response = self._get_with_retries(
                self.settings.arxiv_api_base_url,
                params=params,
                timeout=self.settings.arxiv_request_timeout_seconds,
                retry_count=4,
                retry_delay_seconds=max(2.0, float(self.settings.arxiv_request_delay_seconds)),
            )
            results.update(self._parse_arxiv_feed(response.text))

            has_more = index + batch_size < len(unique_ids)
            if has_more and self.settings.arxiv_request_delay_seconds > 0:
                time.sleep(self.settings.arxiv_request_delay_seconds)

        return results

    def search_arxiv_papers(self, query: str, *, max_results: int = 5) -> list[dict[str, Any]]:
        """arXiv API에서 관련 논문 후보를 검색한다."""
        normalized_query = " ".join(query.split())
        if not normalized_query:
            return []

        search_terms = [f'all:"{term}"' for term in normalized_query.split()[:8] if term.strip()]
        if not search_terms:
            return []

        response = self._get_with_retries(
            self.settings.arxiv_api_base_url,
            params={
                "search_query": " AND ".join(search_terms),
                "start": 0,
                "max_results": max(1, int(max_results)),
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            timeout=min(self.settings.arxiv_request_timeout_seconds, 8),
            retry_count=1,
            retry_delay_seconds=max(1.0, float(self.settings.arxiv_request_delay_seconds)),
        )
        return list(self._parse_arxiv_feed(response.text).values())

    def _get_with_retries(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        timeout: int,
        retry_count: int,
        retry_delay_seconds: float,
    ) -> requests.Response:
        last_error: requests.RequestException | None = None
        for attempt in range(1, retry_count + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=timeout,
                )
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= retry_count:
                    break
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if isinstance(exc, requests.Timeout):
                    multiplier = 2
                else:
                    multiplier = 5 if status_code == 429 else 1
                time.sleep(retry_delay_seconds * attempt * multiplier)
        if last_error is None:
            raise RuntimeError("HTTP 요청 재시도 중 알 수 없는 오류가 발생했습니다.")
        raise last_error

    @staticmethod
    def _validate_date(value: str) -> str:
        return date_cls.fromisoformat(value).isoformat()

    @staticmethod
    def _extract_daily_papers_props(page_html: str) -> dict[str, Any]:
        marker = 'data-target="DailyPapers" data-props="'
        start = page_html.find(marker)
        if start == -1:
            raise ValueError("HF Daily Papers 페이지에서 DailyPapers data-props를 찾지 못했습니다.")

        start += len(marker)
        end = page_html.find('"', start)
        if end == -1:
            raise ValueError("HF Daily Papers data-props 속성이 비정상적으로 종료되었습니다.")

        encoded_props = page_html[start:end]
        decoded_props = html.unescape(encoded_props)
        props = json.loads(decoded_props)
        if not isinstance(props, dict):
            raise ValueError("HF Daily Papers data-props를 dict로 해석하지 못했습니다.")
        return props

    def _parse_arxiv_feed(self, xml_text: str) -> dict[str, dict[str, Any]]:
        root = ET.fromstring(xml_text)
        parsed: dict[str, dict[str, Any]] = {}

        for entry in root.findall("atom:entry", ATOM_NAMESPACE):
            entry_id = self._normalize_arxiv_id(self._get_text(entry, "atom:id"))
            authors = [
                author_name.text.strip()
                for author_name in entry.findall("atom:author/atom:name", ATOM_NAMESPACE)
                if author_name.text
            ]
            categories = [category.attrib["term"] for category in entry.findall("atom:category", ATOM_NAMESPACE)]
            pdf_url = None
            for link in entry.findall("atom:link", ATOM_NAMESPACE):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href")
                    break

            parsed[entry_id] = {
                "arxiv_id": entry_id,
                "title": self._get_text(entry, "atom:title"),
                "abstract": self._get_text(entry, "atom:summary"),
                "authors": authors,
                "published_at": self._get_text(entry, "atom:published"),
                "updated_at": self._get_text(entry, "atom:updated"),
                "categories": categories,
                "primary_category": self._get_attr(entry, "arxiv:primary_category", "term"),
                "pdf_url": pdf_url,
            }

        return parsed

    @staticmethod
    def _get_text(node: ET.Element, path: str) -> str:
        element = node.find(path, ATOM_NAMESPACE)
        if element is None or element.text is None:
            return ""
        return " ".join(element.text.split())

    @staticmethod
    def _get_attr(node: ET.Element, path: str, key: str) -> str | None:
        element = node.find(path, ATOM_NAMESPACE)
        if element is None:
            return None
        return element.attrib.get(key)

    @staticmethod
    def _normalize_arxiv_id(value: str) -> str:
        normalized = value.strip()
        if not normalized:
            return normalized
        if normalized.startswith("http://arxiv.org/abs/"):
            normalized = normalized.removeprefix("http://arxiv.org/abs/")
        if normalized.startswith("https://arxiv.org/abs/"):
            normalized = normalized.removeprefix("https://arxiv.org/abs/")
        if normalized.startswith("http://arxiv.org/pdf/"):
            normalized = normalized.removeprefix("http://arxiv.org/pdf/")
        if normalized.startswith("https://arxiv.org/pdf/"):
            normalized = normalized.removeprefix("https://arxiv.org/pdf/")
        if normalized.endswith(".pdf"):
            normalized = normalized[:-4]
        if "v" in normalized:
            base, version = normalized.rsplit("v", 1)
            if version.isdigit():
                return base
        return normalized
