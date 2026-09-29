import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Self

from arxplore.text import strip_surrogates_deep


def chunk_text_hash(chunk_text: str) -> str:
    return hashlib.md5(str(chunk_text or "").encode("utf-8"), usedforsecurity=False).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkInput:
    chunk_index: int
    chunk_text: str
    section_title: str | None = None
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, chunk: Mapping[str, Any]) -> Self:
        cleaned: dict[str, Any] = strip_surrogates_deep(dict(chunk))
        return cls(
            chunk_index=int(cleaned.get("chunk_index", 0)),
            chunk_text=cleaned.get("chunk_text", ""),
            section_title=cleaned.get("section_title"),
            token_count=int(cleaned.get("token_count", 0)),
            metadata=cleaned.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StoredChunk:
    chunk_id: int
    arxiv_id: str
    chunk_index: int
    chunk_text: str
    section_title: str | None
    token_count: int
    metadata: dict[str, Any]
    updated_at: datetime | None


@dataclass(frozen=True, slots=True, kw_only=True)
class EmbeddingCandidate:
    chunk_id: int
    arxiv_id: str
    chunk_index: int
    chunk_text: str
    section_title: str | None
    metadata: dict[str, Any]
    paper_title: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkEmbedding:
    chunk_id: int
    embedding: Sequence[float]
    model_name: str
