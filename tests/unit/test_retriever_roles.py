from __future__ import annotations

import pytest

from src.integrations.paper_retriever import PaperRetriever

CLEAN_BODY = (
    "The policy is trained to prefer chosen responses over rejected ones using a simple "
    "classification loss on pairwise comparisons. This removes the need for a separate reward model."
)


class FakeRepository:
    def __init__(self, candidates: list[dict]) -> None:
        self.candidates = candidates

    def list_chunk_candidates_by_query(self, query: str, *, limit: int, arxiv_id: str | None = None) -> list[dict]:
        return list(self.candidates)


class FakeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


class FakeVectorRepository:
    def __init__(self, candidates: list[dict]) -> None:
        self.candidates = candidates

    def search_paper_chunks(self, embedding, *, limit: int, arxiv_id: str | None = None) -> list[dict]:
        return list(self.candidates)


def _candidate(chunk_id: int, section_title: str, *, content_role: str = "body", score: float = 0.5) -> dict:
    return {
        "chunk_id": chunk_id,
        "arxiv_id": f"2401.0000{chunk_id}",
        "chunk_index": 0,
        "chunk_text": CLEAN_BODY,
        "section_title": section_title,
        "content_role": content_role,
        "paper_title": "Some Paper",
        "paper_abstract": "",
        "score": score,
    }


def _retriever(candidates: list[dict]) -> PaperRetriever:
    return PaperRetriever(
        repository=FakeRepository(candidates),
        embedding_client=FakeEmbeddingClient(),
        vector_repository=FakeVectorRepository(candidates),
    )


CANDIDATES = [
    _candidate(1, "References"),
    _candidate(2, "Preference Optimization"),
    _candidate(3, "3.2 Reference Model"),
    _candidate(4, "7 References"),
]


@pytest.mark.parametrize("query", ["how does the policy loss work", "direct preference optimization loss"])
def test_lexical_filter_drops_reference_titles_only(query: str):
    results = _retriever(CANDIDATES).search_paper_chunks(query, limit=10)
    titles = {result["section_title"] for result in results}

    assert titles == {"Preference Optimization", "3.2 Reference Model"}


def test_lexical_filter_keeps_references_when_user_asks_for_them():
    results = _retriever(CANDIDATES).search_paper_chunks("list the references of this paper", limit=10)

    assert {result["section_title"] for result in results} == {c["section_title"] for c in CANDIDATES}


def test_lexical_filter_still_uses_content_role():
    candidates = [_candidate(1, "Preference Optimization", content_role="references")]
    assert _retriever(candidates).search_paper_chunks("policy loss", limit=5) == []


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("show me the references", True),
        ("which bibliography entries", True),
        ("direct preference optimization", False),
        ("print the reference list", True),
        ("preferences of annotators", False),
    ],
)
def test_reference_intent_uses_word_boundaries(query: str, expected: bool):
    assert PaperRetriever._reference_intent_requested(query) is expected


def test_vector_rerank_penalizes_reference_title_only():
    results = _retriever(CANDIDATES).search_paper_chunks_by_vector("policy loss", limit=10)
    adjustment = {result["section_title"]: result["rerank_adjustment"] for result in results}

    assert adjustment["References"] == pytest.approx(adjustment["Preference Optimization"] - 0.18)
    assert adjustment["7 References"] == pytest.approx(adjustment["Preference Optimization"] - 0.18)
    assert adjustment["3.2 Reference Model"] == pytest.approx(adjustment["Preference Optimization"])
    assert {result["section_title"] for result in results[-2:]} == {"References", "7 References"}


def test_vector_rerank_skips_penalty_when_query_asks_for_citations():
    results = _retriever(CANDIDATES).search_paper_chunks_by_vector("citation list", limit=10)
    adjustment = {result["section_title"]: result["rerank_adjustment"] for result in results}

    assert adjustment["References"] == pytest.approx(adjustment["Preference Optimization"])
