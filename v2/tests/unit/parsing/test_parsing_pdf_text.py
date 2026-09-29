import logging

import pytest

from arxplore.parsing.pdf_text import (
    extract_page_texts,
    extract_pdf_text,
    find_running_lines,
    merge_lines,
    normalize_page_text,
    parse_pdf_bytes,
    parse_pdf_text,
    should_merge_lines,
    split_camel_case,
    split_inline_heading_line,
    text_from_pages,
)


@pytest.mark.parametrize(
    "line",
    [
        "12",
        "2026-09-29",
        "arXiv:2609.12345v2 [cs.CL] 29 Sep 2026",
        "*Equal contribution",
        "/uni00000031/uni00000032",
        "a /uni00000031 b /uni00000032 c /uni00000033",
        "Contents",
        "2.1 Setup . . . . . . 7",
    ],
)
def test_normalize_page_text_drops_page_furniture(line):
    assert normalize_page_text(f"Body sentence one.\n{line}\nBody sentence two.") == (
        "Body sentence one.\nBody sentence two."
    )


def test_normalize_page_text_merges_wrapped_lines_and_keeps_paragraph_breaks():
    page = "We propose a new method that\nimproves accuracy.\n\n\nThe second paragraph starts here."

    assert normalize_page_text(page) == (
        "We propose a new method that improves accuracy.\n\nThe second paragraph starts here."
    )


def test_normalize_page_text_splits_an_inline_heading_line():
    assert normalize_page_text("Training Details. We train for ten epochs.") == (
        "Training Details.\nWe train for ten epochs."
    )


def test_normalize_page_text_does_not_merge_into_a_section_heading():
    assert normalize_page_text("we describe the setup and\n2 Method\nthe model") == (
        "we describe the setup and\n2 Method the model"
    )


def test_should_merge_lines_rules():
    assert should_merge_lines("state-", "of-the-art")
    assert not should_merge_lines("A full sentence.", "next line")
    assert not should_merge_lines("text without end", "3 Experiments")
    assert should_merge_lines("text without end", "continues here")
    assert should_merge_lines("a list,", "Next Item")
    assert should_merge_lines("x" * 41, "Next Line Has Words")
    assert not should_merge_lines("short", "Next Line Has Words")


def test_merge_lines_joins_hyphenated_words_without_a_space():
    assert merge_lines("state-", "of-the-art") == "stateof-the-art"
    assert merge_lines("wrapped ", " line") == "wrapped line"


def test_split_inline_heading_line():
    assert split_inline_heading_line("   ") == [""]
    assert split_inline_heading_line("Plain body text") == ["Plain body text"]


def body_lines(page: int) -> list[str]:
    letters = "abcdefghijklmnopqrstuvwxyz"
    return [f"The {letters[page % 26]}{letters[offset]} paragraph continues on this page." for offset in range(6)]


def page_with(page: int, *, top: list[str], bottom: list[str]) -> str:
    return "\n".join([*top, *body_lines(page), *bottom])


def test_find_running_lines_needs_the_same_edge_line_on_enough_pages():
    pages = [
        page_with(page, top=["Published as a conference paper at ICLR 2026"], bottom=[str(page + 1)])
        for page in range(10)
    ]

    assert find_running_lines(pages) == frozenset({"Published as a conference paper at ICLR #"})
    assert find_running_lines(pages[:2]) == frozenset()


def test_find_running_lines_ignores_lines_repeated_only_on_a_few_pages():
    pages = [page_with(page, top=["Frontier proprietary models"] if page < 3 else [], bottom=[]) for page in range(10)]

    assert find_running_lines(pages) == frozenset()


def test_find_running_lines_ignores_repeated_lines_in_the_middle_of_pages():
    pages = ["\n".join([*body_lines(page)[:3], "Repeated middle line", *body_lines(page)[3:]]) for page in range(5)]

    assert find_running_lines(pages) == frozenset()


def test_find_running_lines_matches_page_numbers_inside_the_line():
    pages = [page_with(page, top=[f"The Authors {page * 2 + 1}"], bottom=[]) for page in range(6)]

    assert find_running_lines(pages) == frozenset({"The Authors #"})


def test_text_from_pages_drops_running_headers_and_keeps_body_lines_ending_in_numbers():
    unique_top = ["Table 3 shows accuracy of 95", "Example 1", "Rasmus Aagaard 1, 2, Nicki Skafte Detlefsen 1"]
    pages = [
        page_with(
            page,
            top=["Journal Title 12(3)", *unique_top[page : page + 1]],
            bottom=["Prepared using sagej.cls"],
        )
        for page in range(5)
    ]

    text = text_from_pages(pages)

    assert "Journal Title" not in text
    assert "sagej" not in text
    for line in unique_top:
        assert line in text


def test_a_running_line_is_only_dropped_at_the_page_edges():
    pages = ["\n".join(["Preprint", *body_lines(page)[:3], "Preprint", *body_lines(page)[3:]]) for page in range(4)]

    assert text_from_pages(pages).count("Preprint") == 4


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ChatGPT with PyTorch", "Chat GPT with Py Torch"),
        ("RelatedWork and AblationStudy", "Related Work and Ablation Study"),
        ("arXiv GitHub LiDAR ModelScope DeepSeek HuggingFace", "arXiv GitHub LiDAR ModelScope DeepSeek HuggingFace"),
        ("myGitHub and DeepSeekGitHub", "my GitHub and DeepSeek GitHub"),
        ("Hugging Face and ar Xiv keep their spaces", "Hugging Face and ar Xiv keep their spaces"),
        ("arXivs and iPhone", "ar Xivs and i Phone"),
        ("lower case and UPPER CASE", "lower case and UPPER CASE"),
    ],
)
def test_split_camel_case_separates_glued_words_except_known_names(text, expected):
    assert split_camel_case(text) == expected


def test_text_from_pages_splits_words_that_pypdf_glued_together():
    text = text_from_pages(["2 RelatedWork\nWe review priorWork on arXiv and GitHub."])

    assert text == "2 Related Work\nWe review prior Work on arXiv and GitHub."
    assert text_from_pages([text]) == text


def test_glued_pypdf_headings_still_become_sections():
    result = parse_pdf_text(text_from_pages(["2 RelatedWork\nWe review prior work.\n3 MainResults\nWe win."]))

    assert result is not None
    assert [section["title"] for section in result.sections] == ["2 Related Work", "3 Main Results"]


def test_text_from_pages_joins_non_empty_pages_with_blank_lines():
    assert text_from_pages(["First page text.", "12", "Second page text."]) == ("First page text.\n\nSecond page text.")


def test_extract_page_texts_reads_each_page(text_pdf):
    pages = extract_page_texts(text_pdf(["1 Introduction", "We propose a method."]))

    assert len(pages) == 1
    assert "We propose a method." in pages[0]


def test_extract_pdf_text_logs_and_returns_empty_on_unreadable_pdf(caplog, encrypted_pdf):
    with caplog.at_level(logging.WARNING):
        assert extract_pdf_text(encrypted_pdf(["1 Introduction", "Secret body."])) == ""
        assert extract_pdf_text(b"not a pdf") == ""

    assert caplog.text.count("pypdf text extraction failed") == 2


def test_parse_pdf_bytes_builds_sections(text_pdf):
    pdf = text_pdf(
        [
            "1 Introduction",
            "We propose a method for the alignment of large language models.",
            "2 Direct Preference Optimization",
            "We optimize the policy directly from pairwise preference data.",
        ]
    )

    result = parse_pdf_bytes(pdf)

    assert result is not None
    assert result.source == "pdf"
    assert result.quality_metrics["fallback_used"] is False
    assert [section["title"] for section in result.sections] == ["1 Introduction", "2 Direct Preference Optimization"]
    assert result.artifacts == {}
    assert result.parser_metadata == {}


def test_parse_pdf_text_uses_full_text_when_no_section_survives():
    result = parse_pdf_text("Contents")

    assert result is not None
    assert result.sections == [{"title": "Full Text", "text": "Contents"}]
    assert parse_pdf_text("") is None
