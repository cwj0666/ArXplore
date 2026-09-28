from dataclasses import dataclass
from typing import Literal

from arxplore.config import RetrievalMode

type RetrievalMethod = Literal["lexical", "vector", "hybrid"]
type RetrievalChannel = Literal["lexical", "vector"]


@dataclass(frozen=True, slots=True, kw_only=True)
class LexicalScores:
    fts_score: float
    ilike_bonus: float
    content_role_adjustment: float
    section_boost: float
    structural_adjustment: float
    coverage: float
    strict_match: bool
    section_intent_bonus: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class VectorScores:
    raw_similarity_score: float
    content_role_adjustment: float
    section_boost: float
    rerank_adjustment: float = 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class ChannelFusion:
    rank: int
    score: float
    normalized_score: float
    weight: float


@dataclass(frozen=True, slots=True, kw_only=True)
class FusionScores:
    lexical: ChannelFusion | None = None
    vector: ChannelFusion | None = None

    @property
    def matched_methods(self) -> tuple[RetrievalChannel, ...]:
        matched: list[RetrievalChannel] = []
        if self.lexical is not None:
            matched.append("lexical")
        if self.vector is not None:
            matched.append("vector")
        return tuple(matched)


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalHit:
    chunk_id: int
    arxiv_id: str
    chunk_index: int
    paper_title: str
    paper_abstract: str
    chunk_text: str
    section_title: str
    content_role: str
    method: RetrievalMethod
    score: float
    snippet: str
    lexical: LexicalScores | None = None
    vector: VectorScores | None = None
    fusion: FusionScores | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ContextChunk:
    chunk_id: int
    arxiv_id: str
    chunk_index: int
    chunk_text: str
    section_title: str
    content_role: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalContext:
    hit: RetrievalHit
    chunks: tuple[ContextChunk, ...]
    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class HybridInputs:
    query: str
    lexical: tuple[RetrievalHit, ...]
    vector: tuple[RetrievalHit, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class RetrievalResult:
    contexts: tuple[RetrievalContext, ...]
    mode: RetrievalMode
