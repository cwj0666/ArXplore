from arxplore.parsing.chunker import (
    DEFAULT_MAX_CHARS,
    build_chunks,
    ends_mid_sentence,
    rough_token_count,
    starts_mid_sentence,
    summarize_chunks,
)

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


def test_chunk_ends_prefer_the_last_sentence_break_inside_the_lookahead_window():
    chunks = build_chunks(LONG_BODY, [section("2 Method", LONG_BODY)])

    assert [(chunk["metadata"]["char_start"], chunk["metadata"]["char_end"]) for chunk in chunks] == [
        (0, 2001),
        (2001, 3926),
        (3926, 5851),
        (5851, 6299),
    ]
    assert max(len(chunk["chunk_text"]) for chunk in chunks) == 2000
    assert all(chunk["chunk_text"].endswith(".") for chunk in chunks)


def test_the_next_chunk_starts_where_the_previous_one_ended():
    chunks = build_chunks(LONG_BODY, [section("2 Method", LONG_BODY)])

    starts = [chunk["metadata"]["char_start"] for chunk in chunks[1:]]
    ends = [chunk["metadata"]["char_end"] for chunk in chunks[:-1]]
    assert starts == ends


def test_chunk_length_can_exceed_max_chars_by_the_lookahead():
    chunks = build_chunks(LONG_BODY, [section("2 Method", LONG_BODY)])

    assert max(len(chunk["chunk_text"]) for chunk in chunks) > DEFAULT_MAX_CHARS


def test_a_short_tail_is_absorbed_into_the_previous_chunk_with_a_newline():
    text = BODY_PARAGRAPH[:2001] + "and a short tail"

    chunks = build_chunks(text, [section("2 Method", text)])

    assert len(chunks) == 1
    assert chunks[0]["chunk_text"].endswith("large language models.\nand a short tail")
    assert chunks[0]["metadata"]["char_end"] == len(text)


def test_a_short_first_sentence_is_dropped_and_whitespace_collapsed():
    text = "Results are strong. The model improves accuracy on every benchmark.\nWe also report more."

    chunks = build_chunks(text, [section("4 Experiments", text)])

    assert chunks[0]["chunk_text"] == "The model improves accuracy on every benchmark. We also report more."


def test_text_before_an_early_bullet_is_dropped():
    text = "Our contributions are: • A new benchmark. • A new model."

    chunks = build_chunks(text, [section("1 Introduction", text)])

    assert chunks[0]["chunk_text"] == "A new benchmark. • A new model."


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
