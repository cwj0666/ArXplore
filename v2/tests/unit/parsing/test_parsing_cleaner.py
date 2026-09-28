import pytest

from arxplore.parsing.cleaner import (
    collapse_whitespace,
    find_inline_heading_split,
    looks_like_toc_line,
    normalize_chunk_opening,
    normalize_heading_like_text,
    normalize_text,
    strip_inline_heading_prefix,
)


def test_normalize_text_replaces_nul_and_joins_hyphenated_line_breaks():
    assert normalize_text("exam-\nple\x00text and state-\n of") == "example text and state-\n of"


def test_normalize_text_drops_bullet_page_number_headers_and_footers():
    text = "Body one\n12•Some Conference Header\nBody two\nShort Title•13\nBody three"

    assert normalize_text(text) == "Body one\nBody two\nBody three"


def test_normalize_text_drops_a_running_header_that_ends_with_a_page_number():
    assert normalize_text("Body one\nLarge Language Models 12\nBody two") == "Body one\nBody two"


def test_normalize_text_drops_a_title_line_followed_by_a_page_number_line():
    assert normalize_text("Body one\nLarge Language Models\n12\nBody two") == "Body one\nBody two"


def test_normalize_text_compresses_horizontal_whitespace_and_blank_lines():
    assert normalize_text("  a\t\tb   c\n\n\n\nd  ") == "a b c\n\nd"


def test_normalize_text_splits_camel_case_words():
    assert normalize_text("ChatGPT, LangChain and PyTorch") == "Chat GPT, Lang Chain and Py Torch"


def test_normalize_text_restores_brand_names_after_splitting():
    text = "arXiv GitHub LiDAR ModelScope DeepSeek HuggingFace Hugging Face"

    assert normalize_text(text) == "arXiv GitHub LiDAR ModelScope DeepSeek HuggingFace HuggingFace"


def test_normalize_text_removes_only_every_other_consecutive_running_header_per_pass():
    text = "Body one\nLarge Language Models 12\nLarge Language Models 13\nBody two"

    once = normalize_text(text)

    assert once == "Body one\nLarge Language Models 13\nBody two"
    assert normalize_text(once) == "Body one\nBody two"


def test_collapse_whitespace():
    assert collapse_whitespace(" a\n b\t\tc ") == "a b c"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I NTRODUCTION", "INTRODUCTION"),
        ("R ELATED W ORK", "RELATED WORK"),
        ("A.1 E XPERIMENTAL S ETUP", "A.1 EXPERIMENTAL SETUP"),
        ("A Blation Study", "Ablation Study"),
        ("Table 1 :  Results", "Table 1: Results"),
        ("  Plain   heading ", "Plain heading"),
    ],
)
def test_normalize_heading_like_text_joins_spaced_small_caps(text, expected):
    assert normalize_heading_like_text(text) == expected


def test_find_inline_heading_split_detects_a_short_title_sentence():
    assert find_inline_heading_split("Training Details. We train for ten epochs.") == (
        "Training Details.",
        "We train for ten epochs.",
    )


def test_find_inline_heading_split_detects_a_numbered_heading_before_a_body_starter():
    assert find_inline_heading_split("3.2 Training Details We train the model for ten epochs.") == (
        "3.2 Training Details",
        "We train the model for ten epochs.",
    )


@pytest.mark.parametrize(
    "text",
    [
        "We train. The model converges.",
        "An extremely long first sentence that has more than six words in it. Then more.",
        "results are shown. The model converges.",
        "3 Method We",
    ],
)
def test_find_inline_heading_split_ignores_body_sentences(text):
    assert find_inline_heading_split(text) is None


def test_strip_inline_heading_prefix_drops_a_short_first_sentence_and_collapses_whitespace():
    assert strip_inline_heading_prefix("Results are strong. The model\nimproves.") == "The model improves."


def test_normalize_chunk_opening_drops_text_before_an_early_bullet():
    assert normalize_chunk_opening("Our contributions are: • A new benchmark") == "A new benchmark"


def test_normalize_chunk_opening_drops_leading_closing_punctuation():
    assert normalize_chunk_opening(" ), and the rest") == "and the rest"


def test_normalize_chunk_opening_keeps_a_bullet_after_a_sentence():
    text = "We list the results below, which matter. • First item"

    assert normalize_chunk_opening(text) == "First item"
    assert normalize_chunk_opening("Results (%) • First") == "Results (%) • First"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Contents", True),
        ("Table of Contents", True),
        ("Introduction . . . . . . . 3", True),
        ("2.1 Setup . . . 7", True),
        ("3 Experimental Results 12", True),
        ("A Proofs 25", True),
        ("1 Intro to v1.2.3.4.5 7", True),
        ("3 Experimental Results", False),
        ("We evaluate on 12 datasets.", False),
    ],
)
def test_looks_like_toc_line(line, expected):
    assert looks_like_toc_line(line) is expected
