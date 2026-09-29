import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from arxplore_eval.cli import app
from arxplore_eval.dataset import EvalQuery, dump_queries
from arxplore_eval.remap import (
    LOW_OVERLAP,
    MISSING_NEW_PAPER,
    MISSING_OLD_CHUNK,
    QueryRemap,
    StoredText,
    covered_length,
    map_chunk,
    matched_spans,
    remap_queries,
    report_lines,
    rewrite_chunk_ids,
)

PAPER = "2401.00001"
SENTENCES = [
    f"Sentence number {index} explains how the method handles case {index} in the experiments." for index in range(40)
]


def _text(start: int, stop: int) -> str:
    return " ".join(SENTENCES[start:stop])


def _chunk(chunk_id: int, index: int, text: str, arxiv_id: str = PAPER) -> StoredText:
    return StoredText(chunk_id=chunk_id, arxiv_id=arxiv_id, chunk_index=index, section_title="3 Method", text=text)


def _query(query_id: str, chunk_ids: tuple[int, ...]) -> EvalQuery:
    return EvalQuery(
        id=query_id,
        query="q",
        lang="en",
        relevant_arxiv_ids=(PAPER,),
        relevant_chunk_ids=chunk_ids,
        source="llm_synth",
    )


class TestSpans:
    def test_containment_is_one_span(self) -> None:
        assert matched_spans("abc", "xxabcxx") == [(0, 3)]

    def test_blocks_shorter_than_the_minimum_are_ignored(self) -> None:
        old = "the policy loss compares chosen and rejected responses"
        spans = matched_spans(old, "prefix the policy loss compares chosen answers", min_block=10)
        assert spans and all(stop - start >= 10 for start, stop in spans)
        assert matched_spans("abcdef", "xyz", min_block=2) == []

    def test_covered_length_merges_overlaps(self) -> None:
        assert covered_length([(0, 10), (5, 15), (20, 25)]) == 20
        assert covered_length([(3, 4), (0, 10)]) == 10
        assert covered_length([]) == 0


class TestMapChunk:
    def test_a_contained_chunk_maps_to_that_chunk_alone(self) -> None:
        old = _chunk(1, 5, _text(10, 14))
        new = [_chunk(101, 0, _text(0, 9)), _chunk(102, 1, _text(8, 16)), _chunk(103, 2, _text(15, 25))]

        mapping = map_chunk(old, new)

        assert mapping.new_chunk_ids == (102,)
        assert mapping.new_chunks[0].contained and mapping.overlap == 1.0
        assert mapping.problem is None

    def test_a_split_chunk_maps_to_the_neighbours_that_cover_it(self) -> None:
        old = _chunk(1, 5, _text(10, 20))
        new = [_chunk(101, 0, _text(0, 13)), _chunk(102, 1, _text(12, 24)), _chunk(103, 2, _text(30, 40))]

        mapping = map_chunk(old, new)

        assert mapping.new_chunk_ids == (101, 102)
        assert mapping.overlap == pytest.approx(1.0)
        assert not any(share.contained for share in mapping.new_chunks)

    def test_a_small_extra_match_is_not_added_once_the_text_is_covered(self) -> None:
        old = _chunk(1, 5, _text(10, 14))
        repeated = _chunk(109, 9, f"{SENTENCES[11]} and then something unrelated follows here.")
        mapping = map_chunk(old, [_chunk(102, 1, _text(9, 16)), repeated])
        assert mapping.new_chunk_ids == (102,)

    def test_rewritten_text_is_flagged_by_low_overlap(self) -> None:
        old = _chunk(1, 5, _text(10, 14))
        new = [_chunk(101, 0, _text(10, 11) + " Completely different wording replaces the rest of the passage.")]

        remap = QueryRemap("q", (1,), (map_chunk(old, new),), min_overlap=0.9)

        assert remap.overlap < 0.9
        assert remap.needs_review

    def test_missing_papers_and_empty_text(self) -> None:
        assert map_chunk(_chunk(1, 0, _text(0, 2)), []).problem == MISSING_NEW_PAPER
        assert map_chunk(_chunk(1, 0, "   "), [_chunk(2, 0, "x")]).problem == LOW_OVERLAP


class TestRemapQueries:
    def test_queries_without_chunk_answers_are_skipped_and_missing_old_chunks_need_review(self) -> None:
        old = {1: _chunk(1, 5, _text(10, 14)), 2: _chunk(2, 6, _text(20, 24))}
        new = {PAPER: [_chunk(101, 0, _text(8, 16)), _chunk(102, 1, _text(18, 26))]}
        queries = [
            _query("both", (1, 2)),
            _query("missing", (7,)),
            EvalQuery(id="plain", query="q", lang="en", relevant_arxiv_ids=(PAPER,), source="manual"),
        ]

        remaps = {remap.query_id: remap for remap in remap_queries(queries, old, new)}

        assert set(remaps) == {"both", "missing"}
        assert remaps["both"].new_chunk_ids == (101, 102)
        assert remaps["both"].changed and not remaps["both"].needs_review
        assert remaps["missing"].mappings[0].problem == MISSING_OLD_CHUNK
        assert remaps["missing"].needs_review

    def test_unchanged_ids_are_not_reported_as_changes(self) -> None:
        old = {1: _chunk(1, 5, _text(10, 14))}
        (remap,) = remap_queries([_query("same", (1,))], old, {PAPER: [_chunk(1, 5, _text(10, 14))]})
        assert not remap.changed

    def test_report_lines(self) -> None:
        old = {1: _chunk(1, 5, _text(10, 14))}
        remaps = remap_queries([_query("cs-1", (1,)), _query("cs-7", (7,))], old, {PAPER: [_chunk(9, 2, _text(8, 16))]})

        lines = report_lines(remaps)

        assert lines[0] == "cs-1\t1 → 9(#2 1.000 포함)\toverlap 1.000\tok"
        assert lines[1] == f"cs-7\t7 → -\toverlap 0.000\t검토 필요 ({MISSING_OLD_CHUNK})"
        assert lines[-1] == "청크 정답 질의 2건: 검토 필요 1건, 바뀌는 질의 2건"


def test_rewrite_changes_only_the_remapped_lines(tmp_path: Path) -> None:
    queries = [
        _query("cs-1", (1,)),
        EvalQuery(id="ki-1", query="q", lang="ko", relevant_arxiv_ids=(PAPER,), source="known_item"),
    ]
    path = tmp_path / "queries.generated.jsonl"
    path.write_text(dump_queries(queries), encoding="utf-8")
    before = path.read_text(encoding="utf-8").splitlines()
    old = {1: _chunk(1, 5, _text(10, 20))}
    remaps = remap_queries(queries, old, {PAPER: [_chunk(11, 0, _text(0, 13)), _chunk(12, 1, _text(12, 24))]})

    assert rewrite_chunk_ids(path, remaps) == 1

    after = path.read_text(encoding="utf-8").splitlines()
    assert after[1] == before[1]
    assert json.loads(after[0])["relevant_chunk_ids"] == [11, 12]
    assert {key: value for key, value in json.loads(after[0]).items() if key != "relevant_chunk_ids"} == {
        key: value for key, value in json.loads(before[0]).items() if key != "relevant_chunk_ids"
    }


def test_missing_query_file_exits_before_touching_a_database(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["remap-chunks", "--old-db", "postgresql://127.0.0.1:9/x", "--queries", str(tmp_path / "none.jsonl")]
    )
    assert result.exit_code == 2
    assert "질의셋을 읽을 수 없습니다" in result.stderr
