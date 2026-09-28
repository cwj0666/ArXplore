from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from arxplore.parsing.cleaner import normalize_heading_like_text, normalize_text
from arxplore.parsing.models import LAYOUT_SOURCE, FulltextParseResult, JsonObject, Section
from arxplore.parsing.quality import LAYOUT_PROVIDER, layout_quality_metrics
from arxplore.parsing.sections import FULL_TEXT_TITLE, extract_sections

type SegmentLike = Mapping[str, Any]

LAYOUT_TEXT_TYPES = frozenset(
    {"Title", "Section header", "Text", "Table", "Caption", "Formula", "List item", "Footnote"}
)
LAYOUT_IGNORED_TYPES = frozenset({"Page header", "Page footer"})
LAYOUT_ARTIFACT_TYPES = frozenset({"Table", "Picture", "Caption"})
HEADING_LIKE_TYPES = frozenset({"Title", "Section header", "Caption"})
BLOCK_TYPES = frozenset({"Title", "Section header", "Caption", "Table"})
PARAGRAPH_ENDINGS = (".", "?", "!", ":")


def _segment_type(segment: SegmentLike) -> str:
    return str(segment.get("type") or "")


def _segment_page(segment: SegmentLike) -> int:
    return int(segment.get("page_number", 0) or 0)


def _segment_top(segment: SegmentLike) -> float:
    return float(segment.get("top", 0.0) or 0.0)


def _reading_order_key(segment: SegmentLike) -> tuple[int, float, float]:
    return (_segment_page(segment), _segment_top(segment), float(segment.get("left", 0.0) or 0.0))


def order_segments(segments: Sequence[SegmentLike]) -> list[SegmentLike]:
    return sorted(segments, key=_reading_order_key)


def build_layout_text(segments: Sequence[SegmentLike]) -> str:
    parts: list[str] = []
    for segment in order_segments(segments):
        segment_type = _segment_type(segment)
        text = normalize_text(str(segment.get("text") or ""))
        if not text or segment_type in LAYOUT_IGNORED_TYPES:
            continue
        if segment_type not in LAYOUT_TEXT_TYPES:
            continue
        if segment_type in HEADING_LIKE_TYPES:
            text = normalize_heading_like_text(text)
        parts.append(text)
        if segment_type in BLOCK_TYPES or not text.endswith(PARAGRAPH_ENDINGS):
            parts.append("")
    return normalize_text("\n".join(parts))


def _nearest_caption(segment: SegmentLike, captions: Sequence[SegmentLike]) -> str | None:
    page_number = _segment_page(segment)
    top = _segment_top(segment)
    same_page_captions = [caption for caption in captions if _segment_page(caption) == page_number]
    if not same_page_captions:
        return None
    nearest = min(same_page_captions, key=lambda caption: abs(_segment_top(caption) - top))
    text = str(nearest.get("text") or "").strip()
    if not text:
        return None
    return normalize_heading_like_text(text)


def extract_layout_artifacts(segments: Sequence[SegmentLike]) -> dict[str, list[JsonObject]]:
    ordered_segments = order_segments(segments)
    captions = [segment for segment in ordered_segments if _segment_type(segment) == "Caption"]
    tables: list[JsonObject] = []
    figures: list[JsonObject] = []
    for segment in ordered_segments:
        segment_type = _segment_type(segment)
        if segment_type == "Table":
            tables.append(
                {
                    "page": _segment_page(segment),
                    "caption": _nearest_caption(segment, captions),
                    "raw_text": str(segment.get("text") or ""),
                    "confidence": 1.0,
                }
            )
        elif segment_type == "Picture":
            caption = _nearest_caption(segment, captions)
            figures.append(
                {
                    "page": _segment_page(segment),
                    "caption": caption,
                    "confidence": 1.0 if caption else 0.5,
                }
            )
    return {"tables": tables, "figures": figures}


def layout_parser_metadata(segments: Sequence[SegmentLike]) -> JsonObject:
    return {
        "provider": LAYOUT_PROVIDER,
        "segment_count": len(segments),
        "segment_type_counts": dict(Counter(_segment_type(segment) for segment in segments)),
    }


def parse_layout_segments(segments: Sequence[SegmentLike]) -> FulltextParseResult | None:
    text = build_layout_text(segments)
    if not text:
        return None
    sections: list[Section] = extract_sections(text) or [{"title": FULL_TEXT_TITLE, "text": text}]
    artifacts = extract_layout_artifacts(segments)
    return FulltextParseResult(
        text=text,
        sections=sections,
        source=LAYOUT_SOURCE,
        quality_metrics=layout_quality_metrics(
            text=text,
            sections=sections,
            source=LAYOUT_SOURCE,
            artifact_count=sum(len(values) for values in artifacts.values()),
        ),
        artifacts=dict(artifacts),
        parser_metadata=layout_parser_metadata(segments),
    )
