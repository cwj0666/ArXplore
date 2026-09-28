import hashlib

import pytest

from arxplore.domain.papers import FulltextState
from arxplore.parsing.models import FulltextParseResult, Section
from arxplore.parsing.quality import fulltext_quality_metrics
from arxplore.pipeline.prepare.policy import (
    SaveDecision,
    compute_content_hash,
    decide_fulltext_save,
    fulltext_is_final,
    fulltext_source_rank,
    rebuild_text_from_sections,
    repair_abstract_with_metadata,
    strip_result_surrogates,
)

HASH = compute_content_hash("parsed text", [{"title": "Introduction"}])


def _parsed(sections: list[Section], *, text: str = "original text", source: str = "layout_pdf") -> FulltextParseResult:
    return FulltextParseResult(
        text=text,
        sections=sections,
        source=source,
        quality_metrics={**fulltext_quality_metrics(text=text, sections=sections, source=source), "extra": 1},
    )


def test_source_rank_orders_layout_over_pdf_over_abstract_over_unknown() -> None:
    ranks = [fulltext_source_rank(source) for source in ("layout_pdf", "pdf", "fallback_abstract", "legacy", None)]
    assert ranks == [3, 2, 1, 0, 0]


def test_content_hash_is_sha256_of_the_legacy_json_payload() -> None:
    expected_payload = '{"text": "a b c ✓", "section_titles": ["Intro", "2 Method"]}'
    expected = hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
    assert compute_content_hash(" a  b\nc ✓ ", [{"title": " Intro "}, {"title": "2\nMethod", "text": "x"}]) == expected


def test_content_hash_normalizes_whitespace_and_tracks_section_titles() -> None:
    base = compute_content_hash("a  b\nc", [{"title": "Intro"}])
    assert base == compute_content_hash(" a b c ", [{"title": " Intro "}])
    assert base != compute_content_hash("a b c", [{"title": "Method"}])
    assert base != compute_content_hash("a b d", [{"title": "Intro"}])
    assert compute_content_hash("", []) == compute_content_hash("", [])


def test_rebuilt_text_joins_titled_sections_and_skips_empty_ones() -> None:
    sections = [
        {"title": "Front Matter", "text": " Title page "},
        {"title": "Abstract", "text": "Summary."},
        {"title": "Empty", "text": "  "},
        {"title": "Full Text", "text": "Body."},
        {"title": "", "text": "Loose."},
    ]
    assert rebuild_text_from_sections(sections) == "Title page\n\nAbstract\nSummary.\n\nBody.\n\nLoose."


def test_lowercase_abstract_is_replaced_by_the_metadata_abstract() -> None:
    sections: list[Section] = [
        {"title": "Front Matter", "text": "Paper title"},
        {"title": "Abstract", "text": "  continued from a broken line."},
        {"title": "1 Introduction", "text": "Intro text."},
    ]
    parsed = _parsed(sections)

    repaired = repair_abstract_with_metadata(parsed, "  The  real abstract.  ")

    assert repaired.sections[1] == {"title": "Abstract", "text": "The real abstract."}
    assert repaired.text == "Paper title\n\nAbstract\nThe real abstract.\n\n1 Introduction\nIntro text."
    assert repaired.quality_metrics["text_length"] == len(repaired.text)
    assert repaired.quality_metrics["section_count"] == 3
    assert repaired.quality_metrics["max_section_chars"] == len("The real abstract.")
    assert repaired.quality_metrics["avg_section_chars"] == round((11 + 18 + 11) / 3, 2)
    assert repaired.quality_metrics["extra"] == 1
    assert parsed.sections[1]["text"] == "  continued from a broken line."


@pytest.mark.parametrize(
    "sections,abstract",
    [
        ([{"title": "Abstract", "text": "Starts uppercase."}], "Metadata."),
        ([{"title": "Summary", "text": "lowercase but not abstract."}], "Metadata."),
        ([{"title": "Abstract", "text": "   "}], "Metadata."),
        ([{"title": "Abstract", "text": "lowercase."}], "   "),
        ([], "Metadata."),
    ],
)
def test_abstract_repair_leaves_other_results_alone(sections: list[Section], abstract: str) -> None:
    parsed = _parsed(sections)
    assert repair_abstract_with_metadata(parsed, abstract) is parsed


def test_surrogates_are_removed_from_text_titles_and_section_text() -> None:
    parsed = _parsed([{"title": "Intro \ud835", "text": "Body \udc00 text"}], text="Intro \ud835 body")
    cleaned = strip_result_surrogates(parsed)
    assert cleaned.text == "Intro  body"
    assert cleaned.sections == [{"title": "Intro ", "text": "Body  text"}]
    assert parsed.text == "Intro \ud835 body"


def _state(source: str | None, content_hash: str | None = "old", chunk_count: int = 1) -> FulltextState | None:
    if source is None:
        return None
    return FulltextState(source=source, content_hash=content_hash, chunk_count=chunk_count)


@pytest.mark.parametrize(
    "existing,new_source",
    [("layout_pdf", "fallback_abstract"), ("pdf", "fallback_abstract"), ("layout_pdf", "pdf")],
)
def test_lower_ranked_source_does_not_overwrite_stored_chunks(existing: str, new_source: str) -> None:
    decision = decide_fulltext_save(_state(existing), source=new_source, content_hash=HASH, force=False)
    assert decision is SaveDecision.SKIP_LOWER_RANK
    assert decide_fulltext_save(_state(existing), source=new_source, content_hash=HASH, force=True) is SaveDecision.SAVE


def test_lower_ranked_source_is_saved_when_no_chunks_are_stored() -> None:
    state = _state("layout_pdf", chunk_count=0)
    assert decide_fulltext_save(state, source="pdf", content_hash=HASH, force=False) is SaveDecision.SAVE


@pytest.mark.parametrize(
    "existing,new_source",
    [
        (None, "fallback_abstract"),
        ("fallback_abstract", "fallback_abstract"),
        ("fallback_abstract", "pdf"),
        ("pdf", "layout_pdf"),
        ("layout_pdf", "layout_pdf"),
        ("legacy_source", "fallback_abstract"),
    ],
)
def test_equal_or_higher_ranked_changed_content_is_saved(existing: str | None, new_source: str) -> None:
    assert (
        decide_fulltext_save(_state(existing), source=new_source, content_hash=HASH, force=False) is SaveDecision.SAVE
    )


def test_unchanged_content_needs_the_same_source_hash_and_stored_chunks() -> None:
    assert (
        decide_fulltext_save(_state("layout_pdf", HASH), source="layout_pdf", content_hash=HASH, force=False)
        is SaveDecision.SKIP_UNCHANGED
    )
    assert (
        decide_fulltext_save(_state("pdf", HASH), source="layout_pdf", content_hash=HASH, force=False)
        is SaveDecision.SAVE
    )
    assert (
        decide_fulltext_save(_state("layout_pdf", HASH, 0), source="layout_pdf", content_hash=HASH, force=False)
        is SaveDecision.SAVE
    )
    assert (
        decide_fulltext_save(_state("layout_pdf", None), source="layout_pdf", content_hash=HASH, force=False)
        is SaveDecision.SAVE
    )
    assert (
        decide_fulltext_save(_state("layout_pdf", HASH), source="layout_pdf", content_hash=HASH, force=True)
        is SaveDecision.SAVE
    )


@pytest.mark.parametrize(
    "state,force,final",
    [
        (FulltextState(source="layout_pdf", content_hash="h", chunk_count=3), False, True),
        (FulltextState(source="layout_pdf", content_hash="h", chunk_count=3), True, False),
        (FulltextState(source="layout_pdf", content_hash=None, chunk_count=3), False, False),
        (FulltextState(source="layout_pdf", content_hash="h", chunk_count=0), False, False),
        (FulltextState(source="pdf", content_hash="h", chunk_count=3), False, False),
        (FulltextState(source="fallback_abstract", content_hash="h", chunk_count=1), False, False),
        (None, False, False),
    ],
)
def test_only_a_complete_layout_fulltext_is_final(state: FulltextState | None, force: bool, final: bool) -> None:
    assert fulltext_is_final(state, force=force) is final
