from collections.abc import Mapping, Sequence
from typing import Any

from arxplore.parsing.models import FALLBACK_SOURCE, JsonObject

LAYOUT_PROVIDER = "huridocs"


def section_length_metrics(*, text: str, sections: Sequence[Mapping[str, Any]]) -> JsonObject:
    section_lengths = [len(str(section.get("text") or "")) for section in sections]
    return {
        "text_length": len(text),
        "section_count": len(sections),
        "avg_section_chars": round(sum(section_lengths) / len(section_lengths), 2) if section_lengths else 0,
        "max_section_chars": max(section_lengths) if section_lengths else 0,
    }


def fulltext_quality_metrics(*, text: str, sections: Sequence[Mapping[str, Any]], source: str) -> JsonObject:
    return {
        "parse_source": source,
        "fallback_used": source == FALLBACK_SOURCE,
        **section_length_metrics(text=text, sections=sections),
    }


def layout_quality_metrics(
    *, text: str, sections: Sequence[Mapping[str, Any]], source: str, artifact_count: int
) -> JsonObject:
    return {
        **fulltext_quality_metrics(text=text, sections=sections, source=source),
        "layout_provider": LAYOUT_PROVIDER,
        "layout_parse_success": True,
        "artifact_count": artifact_count,
    }
