from __future__ import annotations

import io
from typing import Any

import requests

from .types import FulltextParseResult

try:
    from pypdf import PdfReader
except ModuleNotFoundError:
    PdfReader = None


class PdfExtractorMixin:
    """extractor mixin logic."""

    def parse_from_pdf_url(self, pdf_url: str, *, fallback_text: str='') -> FulltextParseResult:
        """PDF를 다운로드해 텍스트를 추출한다. 실패 시 fallback 텍스트를 사용한다."""
        normalized_url = (pdf_url or '').strip()
        if not normalized_url:
            return self._build_fallback_result(fallback_text)
        try:
            response = requests.get(normalized_url, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException:
            return self._build_fallback_result(fallback_text)
        layout_result = self._parse_with_layout_parser(response.content)
        if layout_result is not None:
            return layout_result
        parsed_text = self._extract_pdf_text(response.content)
        if parsed_text:
            sections = self._extract_sections(parsed_text)
            return FulltextParseResult(text=parsed_text, sections=sections or [{'title': 'Full Text', 'text': parsed_text}], source='pdf', quality_metrics=self._build_fulltext_quality_metrics(text=parsed_text, sections=sections or [{'title': 'Full Text', 'text': parsed_text}], source='pdf'))
        return self._build_fallback_result(fallback_text)

    def _build_fallback_result(self, fallback_text: str) -> FulltextParseResult:
        cleaned = self._normalize_text(fallback_text)
        sections = [{'title': 'Abstract', 'text': cleaned}] if cleaned else []
        return FulltextParseResult(text=cleaned, sections=sections, source='fallback_abstract', quality_metrics=self._build_fulltext_quality_metrics(text=cleaned, sections=sections, source='fallback_abstract'))

    @classmethod
    def _extract_pdf_text(cls, content: bytes) -> str:
        if PdfReader is None:
            return ''
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception:
            return ''
        pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ''
            cleaned_page = cls._normalize_extracted_page_text(page_text)
            if cleaned_page:
                pages.append(cleaned_page)
        return cls._normalize_text('\n\n'.join(pages))

    @staticmethod
    def _build_fulltext_quality_metrics(*, text: str, sections: list[dict[str, Any]], source: str) -> dict[str, Any]:
        section_lengths = [len(str(section.get('text') or '')) for section in sections]
        return {'parse_source': source, 'fallback_used': source == 'fallback_abstract', 'text_length': len(text), 'section_count': len(sections), 'avg_section_chars': round(sum(section_lengths) / len(section_lengths), 2) if section_lengths else 0, 'max_section_chars': max(section_lengths) if section_lengths else 0}

