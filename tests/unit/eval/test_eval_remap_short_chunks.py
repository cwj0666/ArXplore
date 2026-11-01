import pytest

from arxplore_eval.remap import (
    COMPLETE_COVERAGE,
    MIN_GAIN,
    MIN_GAIN_CHARS,
    SKIPPED_OVERLAP,
    QueryRemap,
    StoredText,
    map_chunk,
    normalize,
    report_lines,
)

PAPER = "2401.00002"
SENTENCES = [
    f"Sentence number {index} explains how the method handles case {index} in the experiments." for index in range(24)
]
FILLER = " ".join(
    f"Appendix note {letter} lists hardware budgets, licensing terms and reviewer remarks for run {letter}."
    for letter in "abcdefghijklmnop"
)
OLD = " ".join(SENTENCES)


def _text(start: int, stop: int) -> str:
    return " ".join(SENTENCES[start:stop])


def _chunk(chunk_id: int, index: int, text: str) -> StoredText:
    return StoredText(chunk_id=chunk_id, arxiv_id=PAPER, chunk_index=index, section_title="D.2 Decoding", text=text)


def _old() -> StoredText:
    return _chunk(1, 0, OLD)


def _head() -> StoredText:
    return _chunk(101, 75, _text(0, 20))


def _tail() -> StoredText:
    return _chunk(103, 77, f"{_text(21, 24)} {FILLER}")


def test_the_missing_sentence_is_above_the_char_floor_and_below_the_ratio() -> None:
    missing = len(SENTENCES[20]) + 1
    assert MIN_GAIN_CHARS <= missing < MIN_GAIN * len(normalize(OLD))


def test_a_short_chunk_inside_the_old_chunk_is_kept_even_with_a_small_gain() -> None:
    short = _chunk(102, 76, _text(19, 21))

    mapping = map_chunk(_old(), [_head(), short, _tail()])

    assert mapping.new_chunk_ids == (101, 102, 103)
    assert mapping.overlap >= COMPLETE_COVERAGE
    assert mapping.problem is None
    by_id = {share.chunk_id: share for share in mapping.new_chunks}
    assert by_id[102].inside_share == pytest.approx(1.0)


def test_without_the_short_chunk_the_gap_stays_uncovered() -> None:
    mapping = map_chunk(_old(), [_head(), _tail()])

    assert mapping.new_chunk_ids == (101, 103)
    assert mapping.overlap < 0.99
    assert mapping.problem is None


def test_a_repeated_sentence_inside_a_far_chunk_is_not_added() -> None:
    far = _chunk(140, 140, f"{FILLER} {SENTENCES[20]} {FILLER}")

    mapping = map_chunk(_old(), [_head(), far, _tail()])

    assert mapping.new_chunk_ids == (101, 103)
    assert mapping.problem is None
    assert mapping.skipped == ()


def test_a_large_share_dropped_by_the_gain_rule_needs_review() -> None:
    mixed = _chunk(120, 90, f"{FILLER} {_text(17, 21)} {FILLER}")

    mapping = map_chunk(_old(), [_head(), mixed, _tail()])
    remap = QueryRemap("cs-1", (1,), (mapping,), min_overlap=0.9)

    assert mapping.new_chunk_ids == (101, 103)
    assert [share.chunk_id for share in mapping.skipped] == [120]
    assert mapping.skipped[0].share >= 0.1
    assert mapping.problem == SKIPPED_OVERLAP
    assert remap.overlap >= 0.9 and remap.needs_review
    line = report_lines([remap])[0]
    assert "[제외 후보 120(#90 " in line
    assert line.endswith(f"검토 필요 ({SKIPPED_OVERLAP})")


def test_a_fully_redundant_candidate_is_not_flagged() -> None:
    redundant = _chunk(110, 80, f"{_text(10, 14)} {FILLER}")

    mapping = map_chunk(_old(), [_head(), redundant, _tail()])

    assert mapping.new_chunk_ids == (101, 103)
    assert mapping.skipped == ()
    assert mapping.problem is None
