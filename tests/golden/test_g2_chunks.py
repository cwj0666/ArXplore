from typing import Any

import pytest

from tests.golden import g2_chunks
from tests.golden.store import diff, format_problems, read_golden, resolve_golden

ORIGINAL_FILE = g2_chunks.GOLDEN_PATH
ORIGINAL = read_golden(ORIGINAL_FILE) if ORIGINAL_FILE.is_file() else {"papers": {}}
GOLDEN_FILE = resolve_golden(ORIGINAL_FILE)
GOLDEN = ORIGINAL if GOLDEN_FILE == ORIGINAL_FILE else read_golden(GOLDEN_FILE)
PAPER_IDS = sorted(GOLDEN["papers"])
CORPUS_PAPERS = 275
CORPUS_CHUNKS = 16010


@pytest.fixture(scope="module")
def database_inputs(golden_database_url: str) -> dict[str, Any]:
    return g2_chunks.load_inputs(golden_database_url)


@pytest.fixture(scope="module")
def database_chunks(golden_database_url: str) -> dict[str, Any]:
    return g2_chunks.load_database_chunks(golden_database_url)


def _first_difference(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> int | None:
    for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
        if left != right:
            return index
    return None if len(expected) == len(actual) else min(len(expected), len(actual))


def test_original_golden_covers_the_corpus():
    assert ORIGINAL.get("format") == g2_chunks.FORMAT
    assert len(ORIGINAL["papers"]) == CORPUS_PAPERS
    assert sum(len(entry["chunks"]) for entry in ORIGINAL["papers"].values()) == CORPUS_CHUNKS


def test_original_golden_matches_the_database_chunks(database_chunks):
    original_ids = sorted(ORIGINAL["papers"])
    assert sorted(database_chunks) == original_ids
    problems = [
        arxiv_id for arxiv_id in original_ids if ORIGINAL["papers"][arxiv_id]["chunks"] != database_chunks[arxiv_id]
    ]
    assert not problems, f"original golden differs from paper_chunks for {len(problems)} papers: {problems[:10]}"


def test_golden_in_use_covers_every_paper_with_the_same_inputs():
    assert GOLDEN.get("format") == g2_chunks.FORMAT
    assert sorted(ORIGINAL["papers"]) == PAPER_IDS
    assert [arxiv_id for arxiv_id in PAPER_IDS if not GOLDEN["papers"][arxiv_id]["chunks"]] == []
    assert {arxiv_id: GOLDEN["papers"][arxiv_id]["input_sha256"] for arxiv_id in PAPER_IDS} == {
        arxiv_id: ORIGINAL["papers"][arxiv_id]["input_sha256"] for arxiv_id in PAPER_IDS
    }


@pytest.mark.parametrize("arxiv_id", PAPER_IDS)
def test_chunks_match_golden(parser_adapter, database_inputs, golden_database_url, arxiv_id):
    entry = GOLDEN["papers"][arxiv_id]
    text, sections = database_inputs[arxiv_id]
    assert g2_chunks.input_digest(text, sections) == entry["input_sha256"], "database input changed since capture"

    actual = g2_chunks.chunk_records(parser_adapter, text, sections)
    problems = diff(entry["chunks"], actual)
    if problems:
        index = _first_difference(entry["chunks"], actual)
        chunk_text = None if index is None else g2_chunks.database_chunk_text(golden_database_url, arxiv_id, index)
        problems.append(f"paper_chunks[{index}].chunk_text starts with {str(chunk_text)[:160]!r}")
    assert not problems, format_problems(arxiv_id, problems)
