import random

import pytest

from arxplore.parsing.cleaner import (
    collapse_whitespace,
    find_inline_heading_split,
    looks_like_toc_line,
    normalize_heading_like_text,
    normalize_text,
)


def test_normalize_text_replaces_nul_and_joins_hyphenated_line_breaks():
    assert normalize_text("exam-\nple\x00text and state-\n of") == "example text and state-\n of"


@pytest.mark.parametrize(
    "line",
    [
        "Table 3 shows accuracy of 95",
        "Example 1",
        "Rasmus Aagaard 1, 2, Nicki Skafte Detlefsen 1",
        "Large Language Models 12",
        "12•Some Conference Header",
        "Short Title•13",
    ],
)
def test_normalize_text_keeps_lines_that_merely_end_with_a_number(line):
    assert normalize_text(f"Body one\n{line}\nBody two") == f"Body one\n{line}\nBody two"


def test_normalize_text_compresses_horizontal_whitespace_and_blank_lines():
    assert normalize_text("  a\t\tb   c\n\n\n\nd  ") == "a b c\n\nd"


def test_normalize_text_keeps_camel_case_words():
    assert normalize_text("ChatGPT, LangChain, PyTorch and iPhone") == "ChatGPT, LangChain, PyTorch and iPhone"


def test_normalize_text_keeps_brand_names_as_written():
    text = "arXiv GitHub LiDAR ModelScope DeepSeek HuggingFace Hugging Face"

    assert normalize_text(text) == text


def random_page_like_text(rng: random.Random) -> str:
    pieces = [
        "\n",
        "\n\n",
        "\n\n\n",
        " ",
        "\t",
        "\x00",
        "exam-\nple ",
        "\n12•Conference Header\n",
        "\nShort Title•13\n",
        "\nLarge Language Models 14\n",
        "\nA Study Of Things 7\n",
        "Body sentence here. ",
        "word ",
        "3",
        "Header",
    ]
    return "".join(rng.choice(pieces) for _ in range(rng.randint(0, 40)))


def test_normalize_text_is_idempotent():
    rng = random.Random(4)
    for _ in range(5000):
        text = random_page_like_text(rng)
        once = normalize_text(text)

        assert normalize_text(once) == once


def test_collapse_whitespace():
    assert collapse_whitespace(" a\n b\t\tc ") == "a b c"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I NTRODUCTION", "INTRODUCTION"),
        ("R ELATED W ORK", "RELATED WORK"),
        ("A.1 E XPERIMENTAL S ETUP", "A.1 EXPERIMENTAL SETUP"),
        ("A BSTRACT", "ABSTRACT"),
        ("Idea A MBIG : Benchmarking", "Idea AMBIG: Benchmarking"),
        ("Table 1 :  Results", "Table 1: Results"),
        ("  Plain   heading ", "Plain heading"),
    ],
)
def test_normalize_heading_like_text_joins_spaced_small_caps(text, expected):
    assert normalize_heading_like_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "A Training Details",
        "E Limitations",
        "Appendix A Implementation Details",
        "APPENDIX B PROOFS",
        "5 A Symmetry Audit of Common Representation Quantities",
        "Robot Learning: A Survey",
        "the following two lines: A or B Reason:",
    ],
)
def test_normalize_heading_like_text_keeps_a_single_letter_apart_from_the_next_word(text):
    assert normalize_heading_like_text(text) == text


def test_normalize_heading_like_text_keeps_the_appendix_label_while_joining_small_caps():
    assert normalize_heading_like_text("A PPENDIX B P ROOFS") == "APPENDIX B PROOFS"
    assert normalize_heading_like_text("Appendix A P ROOFS") == "Appendix A PROOFS"


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
