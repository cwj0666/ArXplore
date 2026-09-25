from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import requests

from src.integrations.layout_parser_client import LayoutParserClient

from .types import FulltextParseResult

logger = logging.getLogger(__name__)


class LayoutIntegrationMixin:
    """layout_parser mixin logic."""

    def _parse_with_layout_parser(self, content: bytes) -> FulltextParseResult | None:
        client = self.layout_parser_client or LayoutParserClient()
        if not client.is_configured():
            return None
        try:
            segments = client.analyze_pdf_bytes(content)
        except (requests.RequestException, ValueError) as exc:
            logger.warning("layout parser unavailable: %s", exc)
            return None
        layout_text = self._build_layout_text(segments)
        if not layout_text:
            return None
        sections = self._extract_sections(layout_text)
        artifacts = self._extract_layout_artifacts(segments)
        quality_metrics = {**self._build_fulltext_quality_metrics(text=layout_text, sections=sections or [{'title': 'Full Text', 'text': layout_text}], source='layout_pdf'), 'layout_provider': 'huridocs', 'layout_parse_success': True, 'artifact_count': sum(len(values) for values in artifacts.values())}
        parser_metadata = {'provider': 'huridocs', 'segment_count': len(segments), 'segment_type_counts': dict(Counter(str(segment.get('type') or '') for segment in segments))}
        return FulltextParseResult(text=layout_text, sections=sections or [{'title': 'Full Text', 'text': layout_text}], source='layout_pdf', quality_metrics=quality_metrics, artifacts=artifacts, parser_metadata=parser_metadata)

    @classmethod
    def _build_layout_text(cls, segments: list[dict[str, Any]]) -> str:
        ordered_segments = sorted(segments, key=lambda segment: (int(segment.get('page_number', 0) or 0), float(segment.get('top', 0.0) or 0.0), float(segment.get('left', 0.0) or 0.0)))
        parts: list[str] = []
        for segment in ordered_segments:
            segment_type = str(segment.get('type') or '')
            text = cls._normalize_text(str(segment.get('text') or ''))
            if not text or segment_type in cls._LAYOUT_IGNORED_TYPES:
                continue
            if segment_type not in cls._LAYOUT_TEXT_TYPES:
                continue
            if segment_type in {'Title', 'Section header', 'Caption'}:
                text = cls._normalize_layout_heading_like_text(text)
            parts.append(text)
            if segment_type in {'Title', 'Section header', 'Caption', 'Table'}:
                parts.append('')
            elif not text.endswith(('.', '?', '!', ':')):
                parts.append('')
        return cls._normalize_text('\n'.join(parts))

    @classmethod
    def _extract_layout_artifacts(cls, segments: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        ordered_segments = sorted(segments, key=lambda segment: (int(segment.get('page_number', 0) or 0), float(segment.get('top', 0.0) or 0.0), float(segment.get('left', 0.0) or 0.0)))
        captions = [segment for segment in ordered_segments if str(segment.get('type') or '') == 'Caption']
        tables: list[dict[str, Any]] = []
        figures: list[dict[str, Any]] = []
        for segment in ordered_segments:
            segment_type = str(segment.get('type') or '')
            if segment_type not in cls._LAYOUT_ARTIFACT_TYPES:
                continue
            caption = cls._find_nearest_caption(segment, captions)
            if segment_type == 'Table':
                tables.append({'page': int(segment.get('page_number', 0) or 0), 'caption': caption, 'raw_text': str(segment.get('text') or ''), 'confidence': 1.0})
            elif segment_type == 'Picture':
                figures.append({'page': int(segment.get('page_number', 0) or 0), 'caption': caption, 'confidence': 1.0 if caption else 0.5})
        return {'tables': tables, 'figures': figures}

    @classmethod
    def _find_nearest_caption(cls, segment: dict[str, Any], captions: list[dict[str, Any]]) -> str | None:
        page_number = int(segment.get('page_number', 0) or 0)
        top = float(segment.get('top', 0.0) or 0.0)
        same_page_captions = [caption for caption in captions if int(caption.get('page_number', 0) or 0) == page_number]
        if not same_page_captions:
            return None
        nearest = min(same_page_captions, key=lambda caption: abs(float(caption.get('top', 0.0) or 0.0) - top))
        text = str(nearest.get('text') or '').strip()
        if not text:
            return None
        return cls._normalize_layout_heading_like_text(text)

