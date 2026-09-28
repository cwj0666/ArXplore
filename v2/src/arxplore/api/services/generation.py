from dataclasses import dataclass
from typing import Any

from arxplore.api.errors import UNSUPPORTED_MODEL_MESSAGE, AiUnavailableError, InvalidRequestError
from arxplore.api.services.detail import find_paper, published_at_text
from arxplore.api.services.store import PaperStore
from arxplore.domain.papers import Paper, PaperFulltext
from arxplore.llm.detail import PaperDetailInput, PaperDetailWriter
from arxplore.llm.models import ChatModelFactory
from arxplore.llm.summary_graph import SummaryWriter

OVERVIEW_MODEL = "gpt-5-mini"
DEFAULT_SUMMARY_MODEL = "gpt-5-mini"
AVAILABLE_SUMMARY_MODELS = ("gpt-5-mini", "gpt-5")


@dataclass(frozen=True, slots=True)
class AnalysisPayload:
    overview: str | None
    key_findings: list[Any]
    cached: bool


@dataclass(frozen=True, slots=True)
class SummaryPayload:
    summary: str | None
    cached: bool
    model: str


def detail_input(paper: Paper, fulltext: PaperFulltext | None) -> PaperDetailInput:
    return PaperDetailInput(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        authors=paper.authors,
        abstract=paper.abstract,
        published_at=published_at_text(paper),
        categories=paper.categories,
        primary_category=paper.primary_category,
        pdf_url=paper.pdf_url,
        sections=list(fulltext.sections) if fulltext else [],
        fulltext=fulltext.text if fulltext else "",
    )


def summary_source_text(fulltext: PaperFulltext | None) -> str:
    if fulltext is None:
        return ""
    if fulltext.text:
        return fulltext.text
    return "\n\n".join(f"[{section.get('title', '')}]\n{section.get('text', '')}" for section in fulltext.sections)


def validate_summary_model(model: str) -> str:
    normalized = model.strip()
    if normalized not in AVAILABLE_SUMMARY_MODELS:
        raise InvalidRequestError(UNSUPPORTED_MODEL_MESSAGE)
    return normalized


class GenerationService:
    def __init__(
        self,
        store: PaperStore,
        models: ChatModelFactory,
        detail_writer: PaperDetailWriter,
        summary_writer: SummaryWriter,
    ) -> None:
        self.store = store
        self.models = models
        self.detail_writer = detail_writer
        self.summary_writer = summary_writer

    def require_api_key(self) -> None:
        if not self.models.is_available():
            raise AiUnavailableError()

    async def analysis(self, arxiv_id: str) -> AnalysisPayload:
        paper = await find_paper(self.store, arxiv_id)
        cached = await self.store.get_overview(arxiv_id)
        if cached is not None and cached.overview:
            return AnalysisPayload(overview=cached.overview, key_findings=cached.key_findings or [], cached=True)
        self.require_api_key()
        fulltext = await self.store.get_fulltext(arxiv_id)
        analysis = await self.detail_writer.analyze(detail_input(paper, fulltext), model=OVERVIEW_MODEL)
        if analysis.overview:
            await self.store.save_overview(
                arxiv_id, overview=analysis.overview, key_findings=analysis.key_findings, model=OVERVIEW_MODEL
            )
        return AnalysisPayload(overview=analysis.overview, key_findings=analysis.key_findings, cached=False)

    async def summary(self, arxiv_id: str, *, model: str) -> SummaryPayload:
        normalized_model = validate_summary_model(model)
        paper = await find_paper(self.store, arxiv_id)
        cached = await self.store.get_detailed_summary(arxiv_id, normalized_model)
        if cached is not None and cached.summary:
            return SummaryPayload(summary=cached.summary, cached=True, model=normalized_model)
        self.require_api_key()
        fulltext = await self.store.get_fulltext(arxiv_id)
        summary = await self.summary_writer.summarize(
            title=paper.title,
            authors=paper.authors,
            text=summary_source_text(fulltext),
            sections=list(fulltext.sections) if fulltext else [],
            model=normalized_model,
        )
        if summary:
            await self.store.save_detailed_summary(arxiv_id, summary=summary, model=normalized_model)
        return SummaryPayload(summary=summary, cached=False, model=normalized_model)
