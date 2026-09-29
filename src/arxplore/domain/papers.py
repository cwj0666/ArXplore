from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Self

from arxplore.text import strip_surrogates_deep

DEFAULT_PAPER_SOURCE = "hf_daily_papers"
DEFAULT_FULLTEXT_SOURCE = "pdf"


def parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        return datetime.fromisoformat(normalized)
    return None


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperUpsert:
    arxiv_id: str
    title: str = ""
    authors: list[Any] = field(default_factory=list)
    abstract: str = ""
    primary_category: str | None = None
    categories: list[Any] = field(default_factory=list)
    pdf_url: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    upvotes: int | None = None
    github_url: str | None = None
    github_stars: int | None = None
    citation_count: int | None = None
    source: str = DEFAULT_PAPER_SOURCE

    @classmethod
    def from_mapping(cls, paper: Mapping[str, Any]) -> Self:
        cleaned: dict[str, Any] = strip_surrogates_deep(dict(paper))
        return cls(
            arxiv_id=str(cleaned["arxiv_id"]).strip(),
            title=cleaned.get("title", ""),
            authors=cleaned.get("authors", []),
            abstract=cleaned.get("abstract", ""),
            primary_category=cleaned.get("primary_category"),
            categories=cleaned.get("categories", []),
            pdf_url=cleaned.get("pdf_url"),
            published_at=parse_timestamp(cleaned.get("published_at")),
            updated_at=parse_timestamp(cleaned.get("updated_at")),
            upvotes=int_or_none(cleaned.get("upvotes")),
            github_url=cleaned.get("github_url"),
            github_stars=int_or_none(cleaned.get("github_stars")),
            citation_count=int_or_none(cleaned.get("citation_count")),
            source=cleaned.get("source", DEFAULT_PAPER_SOURCE),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Paper:
    arxiv_id: str
    title: str
    authors: list[Any]
    abstract: str
    primary_category: str | None
    categories: list[Any]
    pdf_url: str | None
    published_at: datetime | None
    updated_at: datetime | None
    upvotes: int
    github_url: str | None
    github_stars: int | None
    citation_count: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class EnrichmentCandidate(Paper):
    source: str


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperFulltext:
    arxiv_id: str
    text: str
    sections: list[Any]
    source: str
    quality_metrics: dict[str, Any]
    artifacts: dict[str, Any]
    parser_metadata: dict[str, Any]
    updated_at: datetime | None


@dataclass(frozen=True, slots=True, kw_only=True)
class FulltextState:
    source: str
    content_hash: str | None
    chunk_count: int


@dataclass(frozen=True, slots=True, kw_only=True)
class AiOverview:
    overview: str | None
    key_findings: list[Any]
    model: str
    generated_at: datetime | None


@dataclass(frozen=True, slots=True, kw_only=True)
class DetailedSummary:
    summary: str
    model: str
    generated_at: datetime | None
