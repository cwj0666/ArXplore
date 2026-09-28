import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, cast

from arxplore.domain.papers import FulltextState
from arxplore.parsing.models import FALLBACK_SOURCE, LAYOUT_SOURCE, PYPDF_SOURCE, FulltextParseResult, Section
from arxplore.parsing.quality import section_length_metrics
from arxplore.parsing.sections import FRONT_MATTER_TITLE, FULL_TEXT_TITLE
from arxplore.text import collapse_whitespace, strip_surrogates

FULLTEXT_SOURCE_RANK: Mapping[str, int] = {LAYOUT_SOURCE: 3, PYPDF_SOURCE: 2, FALLBACK_SOURCE: 1}
UNRANKED_SOURCE = 0
TOP_SOURCE_RANK = max(FULLTEXT_SOURCE_RANK.values())
ABSTRACT_SECTION_KEY = "abstract"
UNTITLED_SECTION_TITLES = frozenset({"", FRONT_MATTER_TITLE, FULL_TEXT_TITLE})
SECTION_TEXT_KEYS = ("title", "text")


class SaveDecision(StrEnum):
    SAVE = "save"
    SKIP_LOWER_RANK = "skip_lower_rank"
    SKIP_UNCHANGED = "skip_unchanged"


def fulltext_source_rank(source: str | None) -> int:
    return FULLTEXT_SOURCE_RANK.get(str(source or ""), UNRANKED_SOURCE)


def compute_content_hash(text: str, sections: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "text": collapse_whitespace(str(text or "")),
        "section_titles": [collapse_whitespace(str(section.get("title") or "")) for section in sections or []],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def rebuild_text_from_sections(sections: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for section in sections:
        title = str(section.get("title") or "").strip()
        text = str(section.get("text") or "").strip()
        if not text:
            continue
        parts.append(text if title in UNTITLED_SECTION_TITLES else f"{title}\n{text}")
    return "\n\n".join(parts).strip()


def _needs_abstract_repair(section: Mapping[str, Any]) -> bool:
    title = str(section.get("title") or "").strip().lower()
    text = collapse_whitespace(str(section.get("text") or ""))
    return title == ABSTRACT_SECTION_KEY and bool(text) and text[0].islower()


def repair_abstract_with_metadata(parsed: FulltextParseResult, abstract: str) -> FulltextParseResult:
    metadata_abstract = collapse_whitespace(str(abstract or ""))
    if not metadata_abstract or not parsed.sections:
        return parsed
    if not any(_needs_abstract_repair(section) for section in parsed.sections):
        return parsed
    sections: list[Section] = [
        {**section, "text": metadata_abstract} if _needs_abstract_repair(section) else {**section}
        for section in parsed.sections
    ]
    text = rebuild_text_from_sections(sections) or parsed.text
    metrics = {**parsed.quality_metrics, **section_length_metrics(text=text, sections=sections)}
    return dataclasses.replace(parsed, text=text, sections=sections, quality_metrics=metrics)


def _strip_section_surrogates(section: Section) -> Section:
    cleaned: dict[str, Any] = dict(section)
    for key in SECTION_TEXT_KEYS:
        value = cleaned.get(key)
        if isinstance(value, str):
            cleaned[key] = strip_surrogates(value)
    return cast(Section, cleaned)


def strip_result_surrogates(parsed: FulltextParseResult) -> FulltextParseResult:
    return dataclasses.replace(
        parsed,
        text=strip_surrogates(parsed.text),
        sections=[_strip_section_surrogates(section) for section in parsed.sections],
    )


def decide_fulltext_save(
    existing: FulltextState | None,
    *,
    source: str,
    content_hash: str,
    force: bool,
) -> SaveDecision:
    if existing is None or force:
        return SaveDecision.SAVE
    has_chunks = existing.chunk_count > 0
    if fulltext_source_rank(source) < fulltext_source_rank(existing.source) and has_chunks:
        return SaveDecision.SKIP_LOWER_RANK
    if existing.source == source and existing.content_hash == content_hash and has_chunks:
        return SaveDecision.SKIP_UNCHANGED
    return SaveDecision.SAVE


def fulltext_is_final(existing: FulltextState | None, *, force: bool) -> bool:
    if force or existing is None:
        return False
    return (
        fulltext_source_rank(existing.source) == TOP_SOURCE_RANK
        and existing.chunk_count > 0
        and existing.content_hash is not None
    )
