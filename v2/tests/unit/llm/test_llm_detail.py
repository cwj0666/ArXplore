from typing import Any

import pytest

from arxplore.llm.detail import (
    NO_SECTIONS_TEXT,
    MissingPaperContextError,
    PaperDetailInput,
    PaperDetailWriter,
    compact_text,
    extract_key_findings,
    format_paper_metadata,
    format_paper_sections,
    has_paper_detail_context,
    select_detail_sections,
)
from arxplore.llm.models import ChatModelFactory
from arxplore.llm.tracing import Tracer


def _paper(**overrides: Any) -> PaperDetailInput:
    values: dict[str, Any] = {
        "arxiv_id": "2401.00001",
        "title": "Direct Preference Optimization Revisited",
        "authors": ["Ada", {"name": " Ben "}, {"name": ""}, "  ", 3],
        "abstract": "We   revisit\nDPO.",
        "published_at": "2024-01-01T09:00:00+09:00",
        "categories": ["cs.CL", "", "cs.LG"],
        "primary_category": "cs.CL",
        "pdf_url": "https://arxiv.org/pdf/2401.00001",
        "sections": [{"title": "1 Introduction", "text": "Intro body."}],
        "fulltext": "",
    }
    values.update(overrides)
    return PaperDetailInput(**values)


def test_compact_text_collapses_whitespace_and_marks_cuts() -> None:
    assert compact_text("  a \n b  ", max_chars=10) == "a b"
    assert compact_text("abcdef ghij", max_chars=8) == "abcdef…"
    assert compact_text(None, max_chars=5) == ""


def test_metadata_block_follows_the_legacy_format() -> None:
    assert format_paper_metadata(_paper()) == (
        "arXiv ID: 2401.00001\n"
        "제목: Direct Preference Optimization Revisited\n"
        "저자: Ada, Ben\n"
        "발행일: 2024-01-01T09:00:00+09:00\n"
        "카테고리: cs.CL, cs.LG\n"
        "PDF: https://arxiv.org/pdf/2401.00001\n"
        "초록: We revisit DPO."
    )


def test_metadata_placeholders_for_missing_values() -> None:
    block = format_paper_metadata(
        _paper(authors=[], published_at=None, categories=[], primary_category=None, pdf_url=None, abstract="")
    )

    assert "저자: 저자 미상" in block
    assert "발행일: 정보 없음" in block
    assert "카테고리: 카테고리 미상" in block
    assert "PDF: 정보 없음" in block
    assert block.endswith("초록: 정보 없음")
    assert "카테고리: cs.AI" in format_paper_metadata(_paper(categories=[], primary_category="cs.AI"))


def test_metadata_abstract_is_capped_at_1600_characters() -> None:
    block = format_paper_metadata(_paper(abstract="word " * 1000))

    abstract = block.split("초록: ", 1)[1]
    assert len(abstract) == 1600
    assert abstract.endswith("…")


def test_section_selection_prefers_keyword_titles_then_fills_to_eight() -> None:
    sections = [{"title": f"Appendix {index}", "text": "x"} for index in range(5)] + [
        {"title": title, "text": "x"} for title in ("Method", "Results", "Related Work", "Conclusion")
    ]

    titles = [section["title"] for section in select_detail_sections(sections)]

    assert titles == [
        "Method",
        "Results",
        "Conclusion",
        "Appendix 0",
        "Appendix 1",
        "Appendix 2",
        "Appendix 3",
        "Appendix 4",
    ]


def test_sections_are_numbered_compacted_and_separated() -> None:
    paper = _paper(
        sections=[
            {"title": "Abstract", "text": "a " * 2000},
            {"title": "  ", "text": "untitled body"},
            {"title": "Method", "text": "   "},
            "not a section",
        ]
    )

    formatted = format_paper_sections(paper)
    blocks = formatted.split("\n\n---\n\n")

    assert len(blocks) == 2
    assert blocks[0].startswith("[섹션 1] Abstract\n")
    assert len(blocks[0].split("\n", 1)[1]) == 2200
    assert blocks[1] == "[섹션 3] Section 3\nuntitled body"


def test_fulltext_and_placeholder_fallbacks() -> None:
    assert format_paper_sections(_paper(sections=[], fulltext="x " * 5000)).startswith("[본문]\n")
    assert len(format_paper_sections(_paper(sections=[], fulltext="x " * 5000))) == len("[본문]\n") + 7000
    assert format_paper_sections(_paper(sections=[{"title": "T", "text": " "}], fulltext="")) == NO_SECTIONS_TEXT


def test_context_needs_section_text_or_fulltext() -> None:
    assert has_paper_detail_context(_paper())
    assert has_paper_detail_context(_paper(sections=[], fulltext="body"))
    assert not has_paper_detail_context(_paper(sections=[{"title": "T", "text": ""}], fulltext="   "))


class TestExtractKeyFindings:
    def test_strips_bullets_and_numbering_and_adds_period(self) -> None:
        text = (
            "- First finding about scaling\n* Second finding here!\n• Third finding bullet\n"
            "1. Fourth numbered item\n2) Fifth numbered item?"
        )

        assert extract_key_findings(text) == [
            "First finding about scaling.",
            "Second finding here!",
            "Third finding bullet.",
            "Fourth numbered item.",
            "Fifth numbered item?",
        ]

    def test_drops_short_lines_and_dedupes_ignoring_punctuation_and_case(self) -> None:
        text = "short\n\nThe model improves accuracy.\nthe model improves accuracy\nTHE MODEL, IMPROVES ACCURACY!"

        assert extract_key_findings(text) == ["The model improves accuracy."]

    def test_keeps_at_most_six(self) -> None:
        text = "\n".join(f"Finding number {index} is here" for index in range(10))

        assert len(extract_key_findings(text)) == 6
        assert len(extract_key_findings(text, max_items=2)) == 2

    def test_collapses_internal_whitespace(self) -> None:
        assert extract_key_findings("-   spaced    out   finding  ") == ["spaced out finding."]

    def test_leading_decimal_number_is_mistaken_for_list_numbering(self) -> None:
        assert extract_key_findings("3.5% gain on GSM8K over the baseline") == ["5% gain on GSM8K over the baseline."]


async def test_writer_calls_overview_then_key_findings_with_the_chosen_model(
    fake_openai: Any, models: ChatModelFactory, tracer: Tracer
) -> None:
    fake_openai.chat_handler = lambda request: (
        "- 첫 번째 핵심 포인트입니다\n- 두 번째 핵심 포인트입니다."
        if "핵심 포인트" in request["messages"][0]["content"]
        else "개요 문단입니다."
    )

    analysis = await PaperDetailWriter(models, tracer).analyze(_paper(), model="gpt-5-mini")

    assert analysis.overview == "개요 문단입니다."
    assert analysis.key_findings == ["첫 번째 핵심 포인트입니다.", "두 번째 핵심 포인트입니다."]
    bodies = [request.body for request in fake_openai.requests_for("/v1/chat/completions")]
    assert [body["model"] for body in bodies] == ["gpt-5-mini", "gpt-5-mini"]
    assert all("temperature" not in body and body["stream"] is False for body in bodies)
    assert "깊이 있게 설명하는" in bodies[0]["messages"][0]["content"]
    assert "핵심 포인트" in bodies[1]["messages"][0]["content"]
    assert "## 본문 섹션 발췌\n[섹션 1] 1 Introduction\nIntro body." in bodies[0]["messages"][1]["content"]


async def test_writer_sends_temperature_for_other_models(
    fake_openai: Any, models: ChatModelFactory, tracer: Tracer
) -> None:
    await PaperDetailWriter(models, tracer).overview(_paper(), model="gpt-4o")

    (request,) = fake_openai.requests_for("/v1/chat/completions")
    assert request.body["temperature"] == 0.2


async def test_writer_refuses_papers_without_context_before_calling_the_model(
    fake_openai: Any, models: ChatModelFactory, tracer: Tracer
) -> None:
    with pytest.raises(MissingPaperContextError):
        await PaperDetailWriter(models, tracer).analyze(_paper(sections=[], fulltext=""), model="gpt-5-mini")

    assert fake_openai.requests == []
