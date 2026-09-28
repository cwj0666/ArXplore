import pytest

from arxplore.parsing.sections import (
    extract_sections,
    looks_like_numbered_heading,
    normalize_section_heading,
    normalize_section_text,
    numeric_section_sort_key,
    prettify_section_title,
    reorder_sections,
    should_drop_section,
    strip_trailing_reference_like_tail,
)

BODY_PARAGRAPH = (
    "We study how preference data shapes the alignment of large language models. "
    "Our approach builds on prior work and extends it with a simple objective that is easy to optimize. "
) * 12

PAPER_TEXT = "\n".join(
    [
        "Aligning Language Models with Preferences",
        "",
        "Abstract",
        BODY_PARAGRAPH,
        "",
        "1 Introduction",
        BODY_PARAGRAPH,
        "",
        "2 Direct Preference Optimization",
        BODY_PARAGRAPH,
        "",
        "3.2 Reference Model",
        BODY_PARAGRAPH,
        "",
        "4 Experiments",
        BODY_PARAGRAPH * 2,
        "",
        "7 References",
        "[1] A. Smith, B. Jones. Learning to align. In Proceedings of NeurIPS, 2023.",
        "[2] C. Lee, D. Kim. Preference models. arXiv preprint arXiv:2301.00001, 2023.",
    ]
)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Abstract", "Abstract"),
        ("INTRODUCTION", "Introduction"),
        ("related work", "related work"),
        ("3.2 Reference Model", "3.2 Reference Model"),
        ("2) Direct Preference Optimization", "2 Direct Preference Optimization"),
        ("4. Experiments", "4 Experiments"),
        ("B.1 Additional Results", "B.1 Additional Results"),
        ("Appendix Implementation Details", "Appendix Implementation Details"),
        ("A.2 Proofs", "A.2 Proofs"),
        ("Appendix Table of Contents", "Appendix"),
        ("1 INTRODUCTION", "1 Introduction"),
        ("5 LLM AND GPU RESULTS", "5 LLM And GPU Results"),
        ("2 Method \u2013 Overview", "2 Method - Overview"),
        ("I NTRODUCTION", "Introduction"),
    ],
)
def test_normalize_section_heading_accepts_headings(line, expected):
    assert normalize_section_heading(line) == expected


@pytest.mark.parametrize(
    "line",
    [
        "Figure",
        "Figure 3: Accuracy over time",
        "Table 2 Results",
        "arXiv:2609.12345",
        "42",
        "20 Section Twenty",
        "3.1 Results . . . 12",
        "3 Experimental Results 12",
        "4 Table of numbers",
        "5 This Heading Has Far Too Many Words To Be A Real Heading",
        "6 Averyveryverylongwordthatexceedslimits Title",
        "7 One, Two, Three, Four, Five, Six, Seven",
        "8 We Propose Things",
        "9 results are shown in the table below",
        "II Method",
        "Methods and results are described below.",
    ],
)
def test_normalize_section_heading_rejects_non_headings(line):
    assert normalize_section_heading(line) is None


def test_single_letter_prefixes_are_merged_into_the_next_capitalized_word():
    assert normalize_section_heading("A Proofs of the Main Theorem") is None
    assert normalize_section_heading("Appendix A Implementation Details") == "Appendix Aimplementation Details"
    assert normalize_section_heading("APPENDIX B PROOFS") == "Appendix Bproofs"


def test_looks_like_numbered_heading_uses_the_capitalized_word_ratio():
    assert looks_like_numbered_heading("Direct Preference Optimization")
    assert looks_like_numbered_heading("Training of Models")
    assert not looks_like_numbered_heading("Training of models")
    assert not looks_like_numbered_heading("results are shown in the table below for all")
    assert not looks_like_numbered_heading("...")


def test_prettify_section_title_capitalizes_all_caps_words_except_known_acronyms():
    assert prettify_section_title("RAG WITH LLM   AGENTS") == "RAG With LLM Agents"
    assert prettify_section_title("Mixed Case Title") == "Mixed Case Title"


def test_extract_sections_on_a_paper():
    titles = [section["title"] for section in extract_sections(PAPER_TEXT)]

    assert titles == [
        "Front Matter",
        "Abstract",
        "1 Introduction",
        "2 Direct Preference Optimization",
        "3.2 Reference Model",
        "4 Experiments",
        "7 References",
    ]


def test_extract_sections_renames_a_single_front_matter_section():
    assert extract_sections("Just some text without headings.") == [
        {"title": "Full Text", "text": "Just some text without headings."}
    ]


def test_extract_sections_drops_table_of_contents_sections():
    text = "Contents\n1 Introduction 3\n2 Method 5\n1 Introduction\nWe start here."

    assert extract_sections(text) == [{"title": "1 Introduction", "text": "We start here."}]


def test_extract_sections_reorders_numbered_sections_that_arrive_out_of_order():
    text = "\n".join(
        [
            "Title of the paper",
            "Abstract",
            "We summarize the work.",
            "3 Results",
            "The results are good.",
            "1 Introduction",
            "We introduce the topic.",
            "Conclusion",
            "We conclude the paper.",
        ]
    )

    assert [section["title"] for section in extract_sections(text)] == [
        "Front Matter",
        "Abstract",
        "1 Introduction",
        "3 Results",
        "Conclusion",
    ]


def test_reorder_sections_keeps_sorted_or_short_lists():
    ordered = [{"title": "1 A", "text": "a"}, {"title": "Z", "text": "z"}, {"title": "2 B", "text": "b"}]

    assert reorder_sections(ordered) == ordered
    assert reorder_sections(ordered[:2]) == ordered[:2]


def test_numeric_section_sort_key():
    assert numeric_section_sort_key(" 3.10.2 Title") == (3, 10, 2)
    assert numeric_section_sort_key("12abc") is None
    assert numeric_section_sort_key("Appendix") is None


def test_strip_trailing_reference_like_tail_only_for_conclusion_like_sections():
    text = "We conclude.\n\n[1] A. Smith. Paper. 2020. [2] B. Jones. Paper. 2021."

    assert strip_trailing_reference_like_tail("5 Conclusion", text) == "We conclude."
    assert strip_trailing_reference_like_tail("4 Experiments", text) == text


def test_strip_trailing_reference_like_tail_keeps_text_when_everything_looks_like_references():
    text = "[1] A. Smith. Paper. 2020. [2] B. Jones. Paper. 2021."

    assert strip_trailing_reference_like_tail("Appendix", text) == text


def test_normalize_section_text_drops_a_lowercase_lead_fragment_in_body_sections():
    text = "tion of the previous page. In this section we describe the model."

    assert normalize_section_text("3 Method", text) == "In this section we describe the model."
    assert normalize_section_text("Appendix", text) == text


def test_should_drop_section():
    assert should_drop_section("1 Introduction", "   ")
    assert should_drop_section("Table of Contents", "anything")
    assert should_drop_section("Front Matter", "1 Introduction 3\n2 Method 5")
    assert not should_drop_section("1 Introduction", "We start here.")
