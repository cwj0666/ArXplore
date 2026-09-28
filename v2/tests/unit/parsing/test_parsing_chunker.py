import random
import statistics
from itertools import pairwise

from arxplore.parsing.chunker import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    build_chunks,
    ends_mid_sentence,
    rough_token_count,
    starts_mid_sentence,
    summarize_chunks,
)
from arxplore.parsing.cleaner import normalize_text

BODY_PARAGRAPH = (
    "We study how preference data shapes the alignment of large language models. "
    "Our approach builds on prior work and extends it with a simple objective that is easy to optimize. "
) * 12
LONG_BODY = BODY_PARAGRAPH * 3
METADATA_KEYS = {
    "section_index",
    "section_chunk_index",
    "section_char_length",
    "char_start",
    "char_end",
    "char_length",
    "starts_mid_sentence",
    "ends_mid_sentence",
    "content_role",
    "prev_chunk_index",
    "next_chunk_index",
    "continues_previous",
    "continues_next",
}


def section(title: str, text: str) -> dict[str, str]:
    return {"title": title, "text": text}


def random_prose(seed: int, sentences: int) -> str:
    rng = random.Random(seed)
    words = ["model", "data", "training", "results", "we", "the", "loss", "layer", "attention", "improves", "shows"]
    parts = []
    for _ in range(sentences):
        body = " ".join(rng.choice(words) for _ in range(rng.randint(4, 30)))
        parts.append(body[0].upper() + body[1:] + rng.choice([".", ".", ".", "?", "!"]))
        if rng.random() < 0.1:
            parts.append("\n\n")
    return " ".join(parts).replace(" \n\n ", "\n\n")


def overlaps(chunks: list) -> list[int]:
    return [
        previous["metadata"]["char_end"] - current["metadata"]["char_start"]
        for previous, current in pairwise(chunks)
        if previous["metadata"]["section_index"] == current["metadata"]["section_index"]
    ]


def test_empty_text_has_no_chunks():
    assert build_chunks("") == []
    assert build_chunks(" \n ", [section("1 Introduction", "text")]) == []


def test_text_without_sections_becomes_a_full_text_section():
    chunks = build_chunks("A short paper body.")

    assert [(chunk["section_title"], chunk["chunk_text"]) for chunk in chunks] == [("Full Text", "A short paper body.")]


def test_chunks_are_built_from_sections_only():
    chunks = build_chunks("ignored body", [section("", "Section text."), section("Empty", "  ")])

    assert [(chunk["section_title"], chunk["chunk_text"]) for chunk in chunks] == [("Full Text", "Section text.")]


def test_chunk_indexes_links_and_metadata():
    chunks = build_chunks(LONG_BODY, [section("1 Introduction", LONG_BODY), section("2 Method", BODY_PARAGRAPH)])

    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(set(chunk["metadata"]) == METADATA_KEYS for chunk in chunks)
    assert chunks[0]["metadata"]["prev_chunk_index"] is None
    assert chunks[-1]["metadata"]["next_chunk_index"] is None
    assert [chunk["metadata"]["next_chunk_index"] for chunk in chunks[:-1]] == list(range(1, len(chunks)))
    method_chunks = [chunk for chunk in chunks if chunk["section_title"] == "2 Method"]
    assert [chunk["metadata"]["section_chunk_index"] for chunk in method_chunks] == list(range(len(method_chunks)))
    assert {chunk["metadata"]["section_index"] for chunk in method_chunks} == {1}
    for chunk in chunks:
        metadata = chunk["metadata"]
        assert metadata["char_length"] == len(chunk["chunk_text"])
        assert chunk["token_count"] == max(1, len(chunk["chunk_text"]) // 4)
        assert metadata["continues_previous"] == metadata["starts_mid_sentence"]
        assert metadata["continues_next"] == metadata["ends_mid_sentence"]


def test_chunk_ends_prefer_the_last_sentence_break_within_max_chars():
    chunks = build_chunks(LONG_BODY, [section("2 Method", LONG_BODY)])

    assert [(chunk["metadata"]["char_start"], chunk["metadata"]["char_end"]) for chunk in chunks] == [
        (0, 1750),
        (1575, 3325),
        (3150, 4900),
        (4725, 6299),
    ]
    assert max(len(chunk["chunk_text"]) for chunk in chunks) == 1749
    assert all(chunk["chunk_text"].endswith(".") for chunk in chunks)


def test_consecutive_chunks_overlap_and_restart_at_a_sentence():
    chunks = build_chunks(LONG_BODY, [section("2 Method", LONG_BODY)])

    assert overlaps(chunks) == [175, 175, 175]
    for chunk in chunks[1:]:
        start = chunk["metadata"]["char_start"]
        assert LONG_BODY[start - 2 : start] == ". "
        assert LONG_BODY[start:].startswith(chunk["chunk_text"][:60])


def test_overlap_length_stays_near_the_target():
    measured = []
    for seed in range(20):
        text = random_prose(seed, 400)
        measured.extend(overlaps(build_chunks(text, [section("2 Method", text)])))

    assert len(measured) > 100
    assert min(measured) >= DEFAULT_OVERLAP_CHARS // 2
    assert max(measured) <= DEFAULT_OVERLAP_CHARS + DEFAULT_OVERLAP_CHARS // 2
    assert abs(statistics.mean(measured) - DEFAULT_OVERLAP_CHARS) < 25


def test_overlap_falls_back_to_word_boundaries_without_sentence_breaks():
    text = " ".join(f"token{index}" for index in range(1500))

    chunks = build_chunks(text, [section("2 Method", text)])

    assert len(chunks) > 3
    assert all(100 <= overlap <= 300 for overlap in overlaps(chunks))
    for chunk in chunks[1:]:
        start = chunk["metadata"]["char_start"]
        assert text[start - 1] == " "
        assert chunk["chunk_text"].startswith("token")


def test_zero_overlap_restarts_at_the_previous_end():
    chunks = build_chunks(LONG_BODY, [section("2 Method", LONG_BODY)], overlap_chars=0)

    assert overlaps(chunks) == [0, 0, 0]


def test_small_chunk_windows_always_make_progress():
    text = "Sentence one ends here. " + "word " * 50 + "y" * 400

    chunks = build_chunks(text, [section("2 Method", text)], max_chars=300, overlap_chars=200)

    assert chunks[-1]["metadata"]["char_end"] == len(text)


def test_chunks_never_exceed_max_chars():
    texts = [
        LONG_BODY,
        random_prose(7, 600),
        lossy_prone_section(3),
        "x" * 5000,
        "\n".join(f"row {index} 1.0 2.0 3.0" for index in range(500)),
        ("A" * 1500 + " ") * 6,
    ]
    for max_chars, overlap_chars in [(DEFAULT_MAX_CHARS, DEFAULT_OVERLAP_CHARS), (600, 100), (300, 200), (1000, 0)]:
        chunks = build_chunks(
            "body", [section("2 Method", text) for text in texts], max_chars=max_chars, overlap_chars=overlap_chars
        )

        assert max(len(chunk["chunk_text"]) for chunk in chunks) <= max_chars


def test_a_text_without_boundaries_is_cut_at_max_chars():
    chunks = build_chunks("x" * 5000, [section("2 Method", "x" * 5000)])

    assert [(chunk["metadata"]["char_start"], chunk["metadata"]["char_end"]) for chunk in chunks] == [
        (0, 1800),
        (1600, 3400),
        (3200, 5000),
    ]


def test_a_short_tail_stays_a_separate_chunk_when_absorbing_would_exceed_max_chars():
    text = BODY_PARAGRAPH[:2001] + "and a short tail"

    chunks = build_chunks(text, [section("2 Method", text)])

    assert [(chunk["metadata"]["char_start"], chunk["metadata"]["char_end"]) for chunk in chunks] == [
        (0, 1750),
        (1575, len(text)),
    ]
    assert chunks[1]["chunk_text"].endswith("large language models. and a short tail")


def test_a_short_section_is_absorbed_into_the_previous_section_with_the_same_title():
    chunks = build_chunks("body", [section("2 Method", BODY_PARAGRAPH[:500]), section("2 Method", "and a short tail")])

    assert len(chunks) == 1
    assert chunks[0]["chunk_text"] == BODY_PARAGRAPH[:500].strip() + "\nand a short tail"


def test_a_short_first_sentence_is_kept():
    text = "Results are strong. The model improves accuracy on every benchmark.\nWe also report more."

    chunks = build_chunks(text, [section("4 Experiments", text)])

    assert chunks[0]["chunk_text"] == text


def test_text_before_an_early_bullet_is_kept():
    text = "Our contributions are: • A new benchmark. • A new model."

    chunks = build_chunks(text, [section("1 Introduction", text)])

    assert chunks[0]["chunk_text"] == text


def test_leading_punctuation_is_kept():
    text = "), and the rest of the sentence continues here."

    assert build_chunks(text, [section("1 Introduction", text)])[0]["chunk_text"] == text


def lossy_prone_section(seed: int) -> str:
    rng = random.Random(seed)
    openings = [
        "Results are strong. ",
        "Our contributions are: • A new benchmark. • A new model. ",
        "), and a dangling clause. ",
        "3.2 Training Details We train the model for ten epochs. ",
        "Setup. ",
    ]
    parts = []
    for index in range(60):
        parts.append(rng.choice(openings) if index % 3 == 0 else "")
        parts.append(random_prose(seed * 100 + index, 3))
        parts.append(rng.choice([" ", "\n", "\n\n"]))
    return "".join(parts).strip()


def test_chunks_are_exact_source_spans_that_rebuild_each_section():
    sections = [section(f"{index} Section", lossy_prone_section(index)) for index in range(1, 6)]

    chunks = build_chunks("body", sections)

    for section_index, source in enumerate(sections):
        section_text = normalize_text(source["text"])
        section_chunks = [chunk for chunk in chunks if chunk["metadata"]["section_index"] == section_index]
        rebuilt = ""
        covered = 0
        for chunk in section_chunks:
            start, end = chunk["metadata"]["char_start"], chunk["metadata"]["char_end"]
            assert chunk["chunk_text"] == section_text[start:end].strip()
            assert start <= covered
            rebuilt += section_text[covered:end]
            covered = end
        assert rebuilt == section_text


def test_small_max_chars_is_raised_to_the_minimum():
    assert build_chunks(LONG_BODY, max_chars=10) == build_chunks(LONG_BODY, max_chars=300)


def test_references_sections_are_marked_as_references():
    references = "[1] A. Smith, B. Jones. Learning to align. In Proceedings of NeurIPS, 2023."

    chunks = build_chunks(references, [section("7 References", references)])

    assert chunks[0]["metadata"]["content_role"] == "references"


def test_rough_token_count():
    assert rough_token_count("") == 1
    assert rough_token_count("a" * 9) == 2


def test_mid_sentence_detection():
    text = "The model works, and it scales. Next sentence"

    assert not starts_mid_sentence(text, 0)
    assert starts_mid_sentence(text, text.index("and"))
    assert not starts_mid_sentence(text, text.index("Next"))
    assert ends_mid_sentence(text, text.index("works") + 3)
    assert not ends_mid_sentence(text, text.index("Next"))
    assert not ends_mid_sentence(text, len(text))


def test_summarize_chunks():
    chunks = build_chunks(LONG_BODY, [section("Front Matter", "a, tiny front part"), section("2 Method", LONG_BODY)])

    summary = summarize_chunks(chunks)

    assert summary["chunk_count"] == len(chunks)
    assert summary["front_matter_chunk_count"] == 1
    assert summary["suspicious_start_count"] == 1
    assert summary["tiny_chunk_count"] == 1
    assert summary["max_chunk_chars"] == max(len(chunk["chunk_text"]) for chunk in chunks)
    assert summary["avg_chunk_chars"] == round(sum(len(chunk["chunk_text"]) for chunk in chunks) / len(chunks), 2)


def test_summarize_no_chunks_uses_integer_zeros():
    summary = summarize_chunks([])

    assert len(summary) == 15
    assert all(value == 0 and type(value) is int for value in summary.values())
