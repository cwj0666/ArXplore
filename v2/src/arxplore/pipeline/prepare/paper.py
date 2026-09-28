import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from arxplore.db.pool import AnyPool
from arxplore.db.repositories import fulltexts, papers
from arxplore.domain.chunks import ChunkInput
from arxplore.parsing.chunker import build_chunks, summarize_chunks
from arxplore.parsing.fulltext import LayoutAnalyzer, parse_fulltext
from arxplore.parsing.models import Chunk, FulltextParseResult, JsonObject, Section
from arxplore.pipeline.prepare.mapping import PrepareCandidate
from arxplore.pipeline.prepare.policy import (
    SaveDecision,
    compute_content_hash,
    decide_fulltext_save,
    fulltext_is_final,
    repair_abstract_with_metadata,
    strip_result_surrogates,
)

logger = logging.getLogger(__name__)

PDF_DOWNLOAD_TIMEOUT_SECONDS = 30.0
PDF_DOWNLOAD_ERRORS: tuple[type[Exception], ...] = (httpx.HTTPError, httpx.InvalidURL)
EMPTY_CANDIDATE_MESSAGE = "candidate.arxiv_id는 비어 있을 수 없습니다."


class PdfSource(Protocol):
    async def fetch(self, url: str) -> bytes | None: ...


class FulltextParser(Protocol):
    async def parse(self, pdf_content: bytes | None, *, fallback_text: str) -> FulltextParseResult: ...

    def chunk(self, text: str, sections: Sequence[Section]) -> list[Chunk]: ...


class HttpPdfSource:
    def __init__(self, http: httpx.AsyncClient, *, timeout_seconds: float = PDF_DOWNLOAD_TIMEOUT_SECONDS) -> None:
        self._http = http
        self.timeout_seconds = timeout_seconds

    async def fetch(self, url: str) -> bytes | None:
        normalized = (url or "").strip()
        if not normalized:
            return None
        try:
            response = await self._http.get(normalized, timeout=self.timeout_seconds)
            response.raise_for_status()
        except PDF_DOWNLOAD_ERRORS as exc:
            logger.warning("pdf download failed (%s): %s", normalized, exc)
            return None
        return response.content


@dataclass(frozen=True, slots=True)
class PdfFulltextParser:
    layout_client: LayoutAnalyzer | None = None

    async def parse(self, pdf_content: bytes | None, *, fallback_text: str) -> FulltextParseResult:
        return await parse_fulltext(pdf_content, fallback_text=fallback_text, layout_client=self.layout_client)

    def chunk(self, text: str, sections: Sequence[Section]) -> list[Chunk]:
        return build_chunks(text, sections)


@dataclass(frozen=True, slots=True)
class PrepareContext:
    db: AnyPool
    pdfs: PdfSource
    parser: FulltextParser


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperOutcome:
    arxiv_id: str
    title: Any
    primary_category: Any
    chunk_count: int
    fulltext_source: str
    fallback_used: Any
    section_count: Any
    text_length: Any
    saved_fulltext: int
    saved_chunks: int
    content_hash: str | None
    artifacts: JsonObject = field(default_factory=dict)
    parser_metadata: JsonObject = field(default_factory=dict)
    quality_metrics: JsonObject = field(default_factory=dict)
    saved_paper: int = 1
    skipped_lower_rank_overwrite: bool = False
    existing_fulltext_source: str | None = None
    skipped_unchanged: bool = False
    chunks_unchanged: bool = False
    parse_skipped: bool = False

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "primary_category": self.primary_category,
            "chunk_count": self.chunk_count,
            "fulltext_source": self.fulltext_source,
            "fallback_used": self.fallback_used,
            "section_count": self.section_count,
            "text_length": self.text_length,
            "saved_paper": self.saved_paper,
            "saved_fulltext": self.saved_fulltext,
            "saved_chunks": self.saved_chunks,
            "content_hash": self.content_hash,
            "artifacts": self.artifacts,
            "parser_metadata": self.parser_metadata,
            "quality_metrics": self.quality_metrics,
        }
        if self.skipped_lower_rank_overwrite:
            result["skipped_lower_rank_overwrite"] = True
            result["existing_fulltext_source"] = self.existing_fulltext_source
        if self.skipped_unchanged:
            result["skipped_unchanged"] = True
        if self.chunks_unchanged:
            result["chunks_unchanged"] = True
        if self.parse_skipped:
            result["parse_skipped"] = True
        return result


def _chunk_inputs(chunks: Sequence[Chunk]) -> list[ChunkInput]:
    return [ChunkInput.from_mapping(chunk) for chunk in chunks]


async def prepare_paper(candidate: PrepareCandidate, context: PrepareContext, *, force: bool = False) -> PaperOutcome:
    arxiv_id = candidate.arxiv_id.strip()
    if not arxiv_id:
        raise ValueError(EMPTY_CANDIDATE_MESSAGE)
    db = context.db
    paper = candidate.to_upsert()

    stored = await fulltexts.get_fulltext_state(db, arxiv_id)
    if stored is not None and fulltext_is_final(stored, force=force):
        await papers.save_paper(db, paper)
        return PaperOutcome(
            arxiv_id=arxiv_id,
            title=candidate.title,
            primary_category=candidate.primary_category,
            chunk_count=stored.chunk_count,
            fulltext_source=stored.source,
            fallback_used=False,
            section_count=None,
            text_length=None,
            saved_fulltext=0,
            saved_chunks=0,
            content_hash=stored.content_hash,
            skipped_unchanged=True,
            parse_skipped=True,
        )

    pdf_content = await context.pdfs.fetch(candidate.pdf_url)
    parsed = await context.parser.parse(pdf_content, fallback_text=candidate.abstract)
    parsed = strip_result_surrogates(repair_abstract_with_metadata(parsed, candidate.abstract))
    chunks = await asyncio.to_thread(
        context.parser.chunk, parsed.text or str(candidate.fields.get("abstract", "")), parsed.sections
    )
    metrics: JsonObject = {**parsed.quality_metrics, **summarize_chunks(chunks)}
    content_hash = compute_content_hash(parsed.text, parsed.sections)

    await papers.save_paper(db, paper)
    existing = await fulltexts.get_fulltext_state(db, arxiv_id)
    existing_chunk_count = existing.chunk_count if existing else 0
    decision = decide_fulltext_save(existing, source=parsed.source, content_hash=content_hash, force=force)

    saved_fulltext = 0
    saved_chunks = 0
    chunks_unchanged = False
    if decision is SaveDecision.SAVE:
        if parsed.text:
            await fulltexts.save_fulltext(
                db,
                arxiv_id,
                text=parsed.text,
                sections=list(parsed.sections),
                source=parsed.source,
                quality_metrics=metrics,
                artifacts=parsed.artifacts,
                parser_metadata=parsed.parser_metadata,
                content_hash=None,
            )
            saved_fulltext = 1
        if chunks:
            if await fulltexts.save_chunks(db, arxiv_id, _chunk_inputs(chunks)):
                saved_chunks = len(chunks)
            else:
                chunks_unchanged = True
        if saved_fulltext:
            await fulltexts.update_content_hash(db, arxiv_id, content_hash)

    skipped_lower_rank = decision is SaveDecision.SKIP_LOWER_RANK
    skipped_unchanged = decision is SaveDecision.SKIP_UNCHANGED
    if saved_chunks or skipped_unchanged:
        chunk_count = len(chunks)
    elif skipped_lower_rank:
        chunk_count = existing_chunk_count
    else:
        chunk_count = 0
    return PaperOutcome(
        arxiv_id=arxiv_id,
        title=candidate.title,
        primary_category=candidate.primary_category,
        chunk_count=chunk_count,
        fulltext_source=parsed.source,
        fallback_used=metrics.get("fallback_used"),
        section_count=metrics.get("section_count", 0),
        text_length=metrics.get("text_length", 0),
        saved_fulltext=saved_fulltext,
        saved_chunks=saved_chunks,
        content_hash=content_hash,
        artifacts=parsed.artifacts,
        parser_metadata=parsed.parser_metadata,
        quality_metrics=metrics,
        skipped_lower_rank_overwrite=skipped_lower_rank,
        existing_fulltext_source=existing.source if skipped_lower_rank and existing else None,
        skipped_unchanged=skipped_unchanged,
        chunks_unchanged=chunks_unchanged,
    )
