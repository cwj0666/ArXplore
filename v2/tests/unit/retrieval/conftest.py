from typing import Protocol

import pytest

from arxplore.domain.retrieval import LexicalScores, RetrievalHit, RetrievalMethod, VectorScores

BODY = (
    "The policy is trained to prefer chosen responses over rejected ones using a simple "
    "classification loss on pairwise comparisons. This removes the need for a separate reward model."
)


class HitFactory(Protocol):
    def __call__(
        self,
        chunk_id: int,
        *,
        method: RetrievalMethod = ...,
        score: float = ...,
        arxiv_id: str | None = ...,
        section_title: str = ...,
        content_role: str = ...,
        text: str = ...,
        strict: bool = ...,
        snippet: str = ...,
        abstract: str = ...,
        chunk_index: int = ...,
    ) -> RetrievalHit: ...


def build_hit(
    chunk_id: int,
    *,
    method: RetrievalMethod = "lexical",
    score: float = 0.5,
    arxiv_id: str | None = None,
    section_title: str = "Method",
    content_role: str = "body",
    text: str = BODY,
    strict: bool = True,
    snippet: str = "",
    abstract: str = "",
    chunk_index: int = 0,
) -> RetrievalHit:
    lexical = None
    vector = None
    if method == "lexical":
        lexical = LexicalScores(
            fts_score=score,
            ilike_bonus=0.0,
            content_role_adjustment=0.0,
            section_boost=0.0,
            structural_adjustment=0.0,
            coverage=1.0,
            strict_match=strict,
        )
    elif method == "vector":
        vector = VectorScores(raw_similarity_score=score, content_role_adjustment=0.0, section_boost=0.0)
    return RetrievalHit(
        chunk_id=chunk_id,
        arxiv_id=arxiv_id if arxiv_id is not None else f"2401.{chunk_id:05d}",
        chunk_index=chunk_index,
        paper_title=f"Paper {chunk_id}",
        paper_abstract=abstract,
        chunk_text=text,
        section_title=section_title,
        content_role=content_role,
        method=method,
        score=score,
        snippet=snippet,
        lexical=lexical,
        vector=vector,
    )


@pytest.fixture
def make_hit() -> HitFactory:
    return build_hit
