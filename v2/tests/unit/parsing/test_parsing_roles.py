import pytest

from arxplore.parsing.roles import (
    chunk_role,
    is_references_section_title,
    looks_like_body_chunk,
    looks_like_reference_chunk,
    looks_like_table_like_chunk,
    refined_chunk_role,
    section_role,
)

BODY_TEXT = (
    "We evaluate the model on three benchmarks and report the average accuracy. "
    "The results show that the proposed objective improves alignment with human preferences."
)
REFERENCE_TEXT = (
    "[1] A. Smith, B. Jones. Learning to align. In Proceedings of NeurIPS, 2023. "
    "[2] C. Lee, D. Kim. Preference models. arXiv preprint arXiv:2301.00001, 2023."
)
TABLE_TEXT = "\n".join(
    ["Model Acc F1", "DPO 61.2 55.0", "PPO 57.4 52.1", "SFT 50.3 49.9", "Base 45.0 40.2", "RM 1 2", "X 3 4"]
)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("7 References", "references"),
        ("7. References", "references"),
        ("References", "references"),
        ("REFERENCES", "references"),
        ("VII. References", "references"),
        ("Bibliography", "references"),
        ("Works Cited", "references"),
        ("Literature Cited and Notes", "references"),
        ("Direct Preference Optimization", "body"),
        ("2 Direct Preference Optimization", "body"),
        ("3.2 Reference Model", "body"),
        ("Reference Model", "body"),
        ("Cross-Reference Analysis", "body"),
        ("Appendix A References", "appendix"),
        ("Front Matter", "front_matter"),
        ("Table of Contents", "toc"),
    ],
)
def test_section_role(title, expected):
    assert section_role(title) == expected


def test_is_references_section_title_ignores_letter_prefixes():
    assert is_references_section_title("7 References")
    assert is_references_section_title(" references. ")
    assert not is_references_section_title("A References")
    assert not is_references_section_title("Mix References")


def test_chunk_role_follows_non_body_section_roles():
    assert chunk_role("7 References", BODY_TEXT) == "references"
    assert chunk_role("Appendix B", BODY_TEXT) == "appendix"
    assert chunk_role("Front Matter", BODY_TEXT) == "front_matter"
    assert chunk_role("Front Matter", TABLE_TEXT) == "table_like"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (BODY_TEXT, "body"),
        ("1 Introduction . . . . . . . 3", "toc"),
        ("Figure 3: Accuracy of each model over training steps.", "figure_caption"),
        ("Figure 3 shows the accuracy of each model over training steps.", "body"),
        (REFERENCE_TEXT, "references"),
        (TABLE_TEXT, "table_like"),
        ("Glyphs /uni00000031 in the text", "table_like"),
        ("<td>1</td><td>2</td>", "table_like"),
    ],
)
def test_chunk_role_in_body_sections(text, expected):
    assert chunk_role("4 Experiments", text) == expected


def test_refined_chunk_role():
    assert refined_chunk_role("References", BODY_TEXT, "body") == "references"
    assert refined_chunk_role("4 Experiments", BODY_TEXT, "references") == "body"
    assert refined_chunk_role("4 Experiments", REFERENCE_TEXT, "body") == "references"
    assert refined_chunk_role("Appendix B", REFERENCE_TEXT, "appendix") == "references"
    assert refined_chunk_role("4 Experiments", TABLE_TEXT, "body") == "table_like"
    assert refined_chunk_role("4 Experiments", BODY_TEXT, "figure_caption") == "figure_caption"


def test_body_and_reference_heuristics():
    assert looks_like_body_chunk("1 Introduction", BODY_TEXT)
    assert not looks_like_body_chunk("1 Introduction", "")
    assert looks_like_reference_chunk(" ".join(REFERENCE_TEXT.split()))
    assert not looks_like_reference_chunk(BODY_TEXT)
    assert looks_like_table_like_chunk(TABLE_TEXT, " ".join(TABLE_TEXT.split()))
    assert not looks_like_table_like_chunk(BODY_TEXT, BODY_TEXT)
