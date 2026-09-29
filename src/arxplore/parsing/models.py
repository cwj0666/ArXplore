from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

type FulltextSource = Literal["layout_pdf", "pdf", "fallback_abstract"]
type JsonObject = dict[str, Any]

LAYOUT_SOURCE: FulltextSource = "layout_pdf"
PYPDF_SOURCE: FulltextSource = "pdf"
FALLBACK_SOURCE: FulltextSource = "fallback_abstract"


class Section(TypedDict):
    title: str
    text: str


class LayoutSegment(TypedDict):
    left: float
    top: float
    width: float
    height: float
    page_number: int
    page_width: float
    page_height: float
    text: str
    type: str


class ChunkMetadata(TypedDict):
    section_index: int
    section_chunk_index: int
    section_char_length: int
    char_start: int
    char_end: int
    char_length: int
    starts_mid_sentence: bool
    ends_mid_sentence: bool
    content_role: str
    prev_chunk_index: int | None
    next_chunk_index: int | None
    continues_previous: bool
    continues_next: bool


class Chunk(TypedDict):
    chunk_index: int
    chunk_text: str
    section_title: str
    token_count: int
    metadata: ChunkMetadata


class ChunkSummary(TypedDict):
    chunk_count: int
    body_chunk_count: int
    avg_chunk_chars: float
    max_chunk_chars: int
    avg_chunk_tokens: float
    max_chunk_tokens: int
    suspicious_start_count: int
    body_suspicious_start_count: int
    mid_sentence_start_count: int
    mid_sentence_end_count: int
    front_matter_chunk_count: int
    non_body_chunk_count: int
    reference_chunk_count: int
    table_like_chunk_count: int
    tiny_chunk_count: int


@dataclass(frozen=True, slots=True)
class FulltextParseResult:
    text: str
    sections: list[Section]
    source: FulltextSource
    quality_metrics: JsonObject
    artifacts: JsonObject = field(default_factory=dict)
    parser_metadata: JsonObject = field(default_factory=dict)
