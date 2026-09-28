from typing import Any

import pytest

from tests.golden import g2_chunks
from tests.golden.store import diff, format_problems, read_golden, resolve_golden

GOLDEN_FILE = resolve_golden(g2_chunks.GOLDEN_PATH)
GOLDEN = read_golden(GOLDEN_FILE) if GOLDEN_FILE.is_file() else {"papers": {}}
PAPER_IDS = sorted(GOLDEN["papers"])


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


def test_golden_covers_the_corpus():
    assert GOLDEN.get("format") == g2_chunks.FORMAT
    assert len(PAPER_IDS) == 275
    assert sum(len(entry["chunks"]) for entry in GOLDEN["papers"].values()) == 16010


def test_golden_matches_the_database_chunks(database_chunks):
    assert sorted(database_chunks) == PAPER_IDS
    problems = [arxiv_id for arxiv_id in PAPER_IDS if GOLDEN["papers"][arxiv_id]["chunks"] != database_chunks[arxiv_id]]
    assert not problems, f"golden differs from paper_chunks for {len(problems)} papers: {problems[:10]}"


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
