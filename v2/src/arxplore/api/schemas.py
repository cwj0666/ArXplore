from typing import Literal

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str


class RateLimitedResponse(BaseModel):
    error: str
    retry_after: int = Field(ge=1, le=60)


class BootstrapResponse(BaseModel):
    default_summary_model: str
    available_summary_models: list[str]


class PaperListItem(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published_at: str | None
    upvotes: int
    pdf_url: str | None


class PaperListResponse(BaseModel):
    items: list[PaperListItem]
    page: int
    page_size: int
    total_items: int
    total_pages: int
    query: str
    sort: Literal["latest", "upvotes"]
    mode: Literal["search", "ai"]


class RelatedPaperItem(PaperListItem):
    source: Literal["local", "arxiv"]
    relation_score: float


class PaperDetail(PaperListItem):
    related_papers: list[RelatedPaperItem]


class PaperDetailResponse(BaseModel):
    paper: PaperDetail


class AnalysisResponse(BaseModel):
    overview: str | None
    key_findings: list[str]
    cached: bool


class SummaryRequest(BaseModel):
    model: str


class SummaryResponse(BaseModel):
    summary: str | None
    cached: bool
    model: str


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatHistoryMessage] = Field(default_factory=list)


class CitationItem(BaseModel):
    arxiv_id: str
    title: str
    url: str
    section_title: str | None
    chunk_id: int | None
    in_answer: bool


class PaperChatResponse(BaseModel):
    answer: str
    citations: list[CitationItem]
    retrieval_mode: Literal["hybrid", "lexical", "first_chunks"]


class AgentChatResponse(BaseModel):
    answer: str
    citations: list[CitationItem]


class StreamChunkEvent(BaseModel):
    chunk: str


class StreamCitationsEvent(BaseModel):
    citations: list[CitationItem]


class StreamErrorEvent(BaseModel):
    error: str


REQUEST_MODELS: tuple[type[BaseModel], ...] = (SummaryRequest, ChatRequest)
EVENT_MODELS: tuple[type[BaseModel], ...] = (StreamChunkEvent, StreamCitationsEvent, StreamErrorEvent)
