from typing import Any

from arxplore.parsing.layout import (
    build_layout_text,
    extract_layout_artifacts,
    layout_parser_metadata,
    parse_layout_segments,
)

BODY_PARAGRAPH = (
    "We study how preference data shapes the alignment of large language models. "
    "Our approach builds on prior work and extends it with a simple objective that is easy to optimize. "
) * 12


def segment(text: str, segment_type: str, *, top: float, page: int = 1, left: float = 50.0) -> dict[str, Any]:
    return {
        "left": left,
        "top": top,
        "width": 500.0,
        "height": 12.0,
        "page_number": page,
        "page_width": 612.0,
        "page_height": 792.0,
        "text": text,
        "type": segment_type,
    }


SMALL_DOCUMENT = [
    segment("A 1\nB 2", "Table", top=50.0, page=2),
    segment("Table 1: Scores.", "Caption", top=30.0, page=2),
    segment("Aligning Models", "Title", top=40.0),
    segment("Running header", "Page header", top=10.0),
    segment("1 Introduction", "Section header", top=100.0),
    segment("It works well", "Text", top=140.0),
    segment("We propose a method.", "Text", top=120.0),
    segment("", "Picture", top=200.0),
    segment("Figure 1 : Overview.", "Caption", top=260.0),
    segment("x = y", "Formula", top=300.0),
    segment("Unknown kind", "Mystery", top=310.0),
    segment("3", "Page footer", top=780.0),
]

PAPER_DOCUMENT = [
    segment("Aligning Language Models with Preferences", "Title", top=40.0),
    segment("Page 1 running header", "Page header", top=10.0),
    segment("1 Introduction", "Section header", top=100.0),
    segment(BODY_PARAGRAPH, "Text", top=120.0),
    segment("2 Direct Preference Optimization", "Section header", top=300.0),
    segment(BODY_PARAGRAPH, "Text", top=320.0),
    segment("Table 1: Win rates against the reference policy.", "Caption", top=500.0),
    segment("Model Win rate\nDPO 61.2\nPPO 57.4", "Table", top=520.0),
    segment("7 References", "Section header", top=60.0, page=2),
    segment("[1] A. Smith. Learning to align. In Proceedings of NeurIPS, 2023.", "Text", top=80.0, page=2),
]


def test_build_layout_text_orders_filters_and_separates_segments():
    assert build_layout_text(SMALL_DOCUMENT) == (
        "Aligning Models\n\n1 Introduction\n\nWe propose a method.\nIt works well\n\n"
        "Figure 1: Overview.\n\nx = y\n\nTable 1: Scores.\n\nA 1\nB 2"
    )


def test_build_layout_text_sorts_by_page_then_top_then_left():
    segments = [
        segment("right column.", "Text", top=100.0, left=300.0),
        segment("left column.", "Text", top=100.0, left=50.0),
        segment("next page.", "Text", top=10.0, page=2),
    ]

    assert build_layout_text(segments) == "left column.\nright column.\nnext page."


def test_build_layout_text_keeps_camel_case_names():
    segments = [
        segment("ChatGPT and PyTorch on arXiv.", "Text", top=1.0),
        segment("RelatedWork", "Section header", top=2.0),
    ]

    assert build_layout_text(segments) == "ChatGPT and PyTorch on arXiv.\nRelatedWork"


def test_extract_layout_artifacts_pairs_tables_and_figures_with_the_nearest_caption_on_the_page():
    segments = [
        *SMALL_DOCUMENT,
        segment("", "Picture", top=100.0, page=3),
        segment("Figure 2: Far.", "Caption", top=700.0, page=1),
    ]

    assert extract_layout_artifacts(segments) == {
        "tables": [{"page": 2, "caption": "Table 1: Scores.", "raw_text": "A 1\nB 2", "confidence": 1.0}],
        "figures": [
            {"page": 1, "caption": "Figure 1: Overview.", "confidence": 1.0},
            {"page": 3, "caption": None, "confidence": 0.5},
        ],
    }


def test_layout_parser_metadata_counts_segment_types():
    metadata = layout_parser_metadata(PAPER_DOCUMENT)

    assert metadata["provider"] == "huridocs"
    assert metadata["segment_count"] == len(PAPER_DOCUMENT)
    assert metadata["segment_type_counts"]["Section header"] == 3


def test_parse_layout_segments_builds_a_layout_result():
    result = parse_layout_segments(PAPER_DOCUMENT)

    assert result is not None
    assert result.source == "layout_pdf"
    assert "Page 1 running header" not in result.text
    assert [section["title"] for section in result.sections] == [
        "Front Matter",
        "1 Introduction",
        "2 Direct Preference Optimization",
        "7 References",
    ]
    assert result.artifacts["tables"][0]["caption"] == "Table 1: Win rates against the reference policy."
    assert result.quality_metrics["parse_source"] == "layout_pdf"
    assert result.quality_metrics["layout_provider"] == "huridocs"
    assert result.quality_metrics["layout_parse_success"] is True
    assert result.quality_metrics["artifact_count"] == 1
    assert result.parser_metadata["segment_count"] == len(PAPER_DOCUMENT)


def test_section_titles_come_from_line_rules_not_layout_labels():
    segments = [
        segment("Our Findings", "Section header", top=10.0),
        segment("We found things.", "Text", top=20.0),
        segment("3 Results", "Text", top=30.0),
        segment("They are good.", "Text", top=40.0),
    ]

    result = parse_layout_segments(segments)

    assert result is not None
    assert [section["title"] for section in result.sections] == ["Front Matter", "3 Results"]


def test_parse_layout_segments_without_text_returns_none():
    assert parse_layout_segments([segment("", "Picture", top=1.0), segment("header", "Page header", top=2.0)]) is None
    assert parse_layout_segments([]) is None


def test_layout_functions_tolerate_segments_with_missing_fields():
    segments = [{"type": "Text", "text": "Only text."}, {"text": "No type."}, {"type": "Picture"}]

    assert build_layout_text(segments) == "Only text."
    assert extract_layout_artifacts(segments) == {
        "tables": [],
        "figures": [{"page": 0, "caption": None, "confidence": 0.5}],
    }
