from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from arxplore.domain.papers import PaperUpsert
from arxplore.sources.arxiv import normalize_arxiv_id

HF_RAW_PAPER_SOURCE = "hf_daily_papers_raw"
ARXIV_PDF_URL_TEMPLATE = "https://arxiv.org/pdf/{arxiv_id}.pdf"
CANDIDATE_ID_KEYS = ("id", "arxivId", "arxiv_id")
GITHUB_URL_KEYS = ("url", "repoUrl", "repository")


@dataclass(frozen=True, slots=True, kw_only=True)
class HfSignals:
    upvotes: int
    github_url: Any
    github_stars: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class PrepareCandidate:
    arxiv_id: str
    fields: Mapping[str, Any]

    @property
    def title(self) -> Any:
        return self.fields.get("title", "")

    @property
    def primary_category(self) -> Any:
        return self.fields.get("primary_category")

    @property
    def abstract(self) -> str:
        return str(self.fields.get("abstract") or "")

    @property
    def pdf_url(self) -> str:
        return str(self.fields.get("pdf_url") or "")

    def to_upsert(self) -> PaperUpsert:
        return PaperUpsert.from_mapping(self.fields)


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateSelection:
    target_date: date
    raw_count: int
    deduplicated_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    candidates: tuple[PrepareCandidate, ...]


def _paper_object(item: Mapping[str, Any]) -> Mapping[str, Any]:
    paper = item.get("paper")
    return paper if isinstance(paper, dict) else {}


def _first_text(source: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_candidate_id(item: Mapping[str, Any]) -> str | None:
    paper = item.get("paper")
    if isinstance(paper, dict):
        found = _first_text(paper, CANDIDATE_ID_KEYS)
        if found:
            return found
    return _first_text(item, CANDIDATE_ID_KEYS)


def extract_hf_signals(item: Mapping[str, Any]) -> HfSignals:
    paper = _paper_object(item)
    upvotes = item.get("upvotes", paper.get("upvotes", 0))
    github = item.get("github", paper.get("github", {}))
    github_url = paper.get("githubRepo")
    github_stars = paper.get("githubStars")
    if isinstance(github, dict):
        github_url = next((github.get(key) for key in GITHUB_URL_KEYS if github.get(key)), None) or github_url
        github_stars = github.get("stars") if github.get("stars") is not None else github_stars
    return HfSignals(
        upvotes=int(upvotes or 0),
        github_url=github_url,
        github_stars=int(github_stars) if github_stars is not None else None,
    )


def extract_hf_authors(item: Mapping[str, Any]) -> list[str]:
    authors = _paper_object(item).get("authors")
    if not isinstance(authors, list):
        return []
    names = (str(author.get("name") or "").strip() for author in authors if isinstance(author, dict))
    return [name for name in names if name]


def build_pdf_url(arxiv_id: str) -> str:
    normalized = arxiv_id.strip()
    return ARXIV_PDF_URL_TEMPLATE.format(arxiv_id=normalized) if normalized else ""


def paper_fields_from_hf_item(arxiv_id: str, item: Mapping[str, Any]) -> dict[str, Any]:
    paper = _paper_object(item)
    signals = extract_hf_signals(item)
    return {
        "arxiv_id": arxiv_id,
        "title": str(paper.get("title") or "").strip(),
        "authors": extract_hf_authors(item),
        "abstract": str(paper.get("summary") or "").strip(),
        "primary_category": None,
        "categories": [],
        "pdf_url": build_pdf_url(arxiv_id),
        "published_at": paper.get("publishedAt"),
        "updated_at": paper.get("publishedAt"),
        "upvotes": signals.upvotes,
        "github_url": signals.github_url,
        "github_stars": signals.github_stars,
        "source": HF_RAW_PAPER_SOURCE,
    }


def normalize_max_papers(max_papers: int | None) -> int | None:
    if max_papers is None or max_papers <= 0:
        return None
    return max_papers


def select_candidates(
    target_date: date, payload: Sequence[Any], *, max_papers: int | None = None
) -> CandidateSelection:
    items = [item for item in payload if isinstance(item, dict)]
    keyed = [(normalize_arxiv_id(candidate_id), item) for item in items if (candidate_id := extract_candidate_id(item))]
    deduplicated = list(dict.fromkeys(arxiv_id for arxiv_id, _ in keyed))
    selected = deduplicated[: normalize_max_papers(max_papers)]
    items_by_id = dict(keyed)
    candidates = tuple(
        PrepareCandidate(arxiv_id=arxiv_id, fields=paper_fields_from_hf_item(arxiv_id, items_by_id[arxiv_id]))
        for arxiv_id in selected
        if items_by_id.get(arxiv_id)
    )
    return CandidateSelection(
        target_date=target_date,
        raw_count=len(payload),
        deduplicated_ids=tuple(deduplicated),
        selected_ids=tuple(selected),
        candidates=candidates,
    )
