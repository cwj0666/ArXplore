import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from arxplore.api.errors import PaperNotFoundError
from arxplore.api.services.store import PaperStore
from arxplore.domain.papers import Paper
from arxplore.sources.arxiv import ArxivPaper, normalize_arxiv_id

logger = logging.getLogger(__name__)

RELATED_PAPER_LIMIT = 5
RELATED_PAPER_CANDIDATE_LIMIT = 300
EXTERNAL_QUERY_TOKENS = 8
EXTERNAL_MIN_RESULTS = 5
CATEGORY_WEIGHT = 2.0
SAME_PRIMARY_CATEGORY_BONUS = 1.5
TITLE_TOKEN_WEIGHT = 0.8
ABSTRACT_TOKEN_WEIGHT = 0.12
ABSTRACT_OVERLAP_CAP = 12
LOCAL_SOURCE = "local"
ARXIV_SOURCE = "arxiv"
KEYWORD_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9-]{2,}")
KEYWORD_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "based",
        "between",
        "from",
        "into",
        "model",
        "models",
        "paper",
        "that",
        "the",
        "their",
        "this",
        "through",
        "using",
        "with",
    }
)


class ArxivSearch(Protocol):
    async def search_papers(self, query: str, *, max_results: int = ...) -> list[ArxivPaper]: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class RelatedPaper:
    arxiv_id: str
    title: str
    authors: list[Any] = field(default_factory=list)
    abstract: str = ""
    published_at: str | None = None
    upvotes: int = 0
    pdf_url: str | None = None
    categories: list[Any] = field(default_factory=list)
    primary_category: str | None = None
    source: str = LOCAL_SOURCE
    relation_score: float = 0.0


def keyword_tokens_in_order(text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in KEYWORD_TOKEN.findall(text.lower()):
        if token in KEYWORD_STOPWORDS or token in seen:
            continue
        tokens.append(token)
        seen.add(token)
    return tokens


def keyword_tokens(text: str) -> set[str]:
    return set(keyword_tokens_in_order(text))


def published_at_text(paper: Paper) -> str | None:
    return paper.published_at.isoformat() if paper.published_at else None


class Scorable(Protocol):
    @property
    def title(self) -> str: ...

    @property
    def abstract(self) -> str: ...

    @property
    def categories(self) -> Sequence[Any]: ...

    @property
    def primary_category(self) -> str | None: ...


def score_related_paper(source: Scorable, candidate: Scorable) -> float:
    category_overlap = len(set(source.categories or []) & set(candidate.categories or []))
    same_primary = bool(source.primary_category) and source.primary_category == candidate.primary_category
    title_overlap = len(keyword_tokens(source.title or "") & keyword_tokens(candidate.title or ""))
    abstract_overlap = len(keyword_tokens(source.abstract or "") & keyword_tokens(candidate.abstract or ""))
    return (
        category_overlap * CATEGORY_WEIGHT
        + (SAME_PRIMARY_CATEGORY_BONUS if same_primary else 0.0)
        + title_overlap * TITLE_TOKEN_WEIGHT
        + min(abstract_overlap, ABSTRACT_OVERLAP_CAP) * ABSTRACT_TOKEN_WEIGHT
    )


def local_related_paper(paper: Paper, score: float) -> RelatedPaper:
    return RelatedPaper(
        arxiv_id=paper.arxiv_id,
        title=paper.title or "",
        authors=list(paper.authors or []),
        abstract=paper.abstract or "",
        published_at=published_at_text(paper),
        upvotes=paper.upvotes or 0,
        pdf_url=paper.pdf_url,
        categories=list(paper.categories or []),
        primary_category=paper.primary_category,
        source=LOCAL_SOURCE,
        relation_score=score,
    )


def external_related_paper(paper: ArxivPaper, score: float) -> RelatedPaper:
    return RelatedPaper(
        arxiv_id=paper.arxiv_id,
        title=paper.title or "",
        authors=list(paper.authors),
        abstract=paper.abstract or "",
        published_at=paper.published_at,
        upvotes=0,
        pdf_url=paper.pdf_url,
        categories=list(paper.categories),
        primary_category=paper.primary_category,
        source=ARXIV_SOURCE,
        relation_score=score,
    )


def rank_local_related(paper: Paper, candidates: Sequence[Paper], seen_ids: set[str]) -> list[RelatedPaper]:
    related: list[RelatedPaper] = []
    for candidate in candidates:
        candidate_id = str(candidate.arxiv_id or "")
        if not candidate_id or candidate_id in seen_ids:
            continue
        score = score_related_paper(paper, candidate)
        if score <= 0:
            continue
        related.append(local_related_paper(candidate, score))
        seen_ids.add(candidate_id)
    return sorted(
        related,
        key=lambda item: (item.relation_score, item.upvotes, item.published_at or "", item.arxiv_id),
        reverse=True,
    )


async def search_external_related(
    paper: Paper, arxiv: ArxivSearch, *, seen_ids: set[str], limit: int
) -> list[RelatedPaper]:
    if limit <= 0:
        return []
    try:
        query = " ".join(keyword_tokens_in_order(paper.title or "")[:EXTERNAL_QUERY_TOKENS])
        if not query:
            return []
        found = await arxiv.search_papers(query, max_results=max(limit * 2, EXTERNAL_MIN_RESULTS))
    except Exception:
        logger.warning("arXiv 관련 논문 검색 실패: arxiv_id=%s", paper.arxiv_id, exc_info=True)
        return []
    related: list[RelatedPaper] = []
    for external in found:
        arxiv_id = str(external.arxiv_id or "")
        if not arxiv_id or arxiv_id in seen_ids:
            continue
        related.append(external_related_paper(external, score_related_paper(paper, external)))
        seen_ids.add(arxiv_id)
        if len(related) >= limit:
            break
    return related


async def find_related_papers(
    store: PaperStore, arxiv: ArxivSearch, paper: Paper, *, limit: int = RELATED_PAPER_LIMIT
) -> list[RelatedPaper]:
    seen_ids = {str(paper.arxiv_id or "")}
    candidates = await store.list_recent_papers(limit=RELATED_PAPER_CANDIDATE_LIMIT)
    related = rank_local_related(paper, candidates, seen_ids)
    if len(related) < limit:
        related.extend(await search_external_related(paper, arxiv, seen_ids=seen_ids, limit=limit - len(related)))
    return related[:limit]


async def find_paper(store: PaperStore, arxiv_id: str) -> Paper:
    paper = await store.get_paper(arxiv_id)
    if paper is None:
        normalized = normalize_arxiv_id(arxiv_id)
        if normalized != arxiv_id:
            paper = await store.get_paper(normalized)
    if paper is None:
        raise PaperNotFoundError()
    return paper
