import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from arxplore.llm.models import ChatModelFactory
from arxplore.llm.prompts.key_findings import KEY_FINDINGS_PROMPT
from arxplore.llm.prompts.overview import OVERVIEW_PROMPT
from arxplore.llm.tracing import DEFAULT_TRACE_RUNTIME, Tracer, TraceStage
from arxplore.text import collapse_whitespace

ANALYSIS_PROMPT_VERSION = 1
DETAIL_TEMPERATURE = 0.2
METADATA_ABSTRACT_MAX_CHARS = 1600
SECTION_TEXT_MAX_CHARS = 2200
FULLTEXT_MAX_CHARS = 7000
MAX_SELECTED_SECTIONS = 8
MAX_KEY_FINDINGS = 6
MIN_KEY_FINDING_CHARS = 10
ELLIPSIS = "…"
SECTION_SEPARATOR = "\n\n---\n\n"
NO_SECTIONS_TEXT = "본문 섹션 정보가 없습니다."
UNKNOWN_AUTHORS = "저자 미상"
UNKNOWN_CATEGORY = "카테고리 미상"
UNKNOWN_VALUE = "정보 없음"
UNTITLED = "제목 없음"
MISSING_CONTEXT_MESSAGE = "상세 논문 설명을 생성하려면 본문 섹션 또는 fulltext_text가 필요합니다."
PRIORITY_SECTION_KEYWORDS = (
    "abstract",
    "introduction",
    "problem",
    "method",
    "approach",
    "model",
    "experiment",
    "result",
    "evaluation",
    "discussion",
    "limitation",
    "conclusion",
)
FINDING_MARKER = re.compile(r"^(?:[-*•]|\d+[.)])\s*")
FINDING_KEY_NOISE = re.compile(r"\W+")
SENTENCE_ENDINGS = (".", "!", "?")


class MissingPaperContextError(ValueError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperDetailInput:
    arxiv_id: str
    title: str
    authors: Sequence[Any] = ()
    abstract: str = ""
    published_at: str | None = None
    categories: Sequence[Any] = ()
    primary_category: str | None = None
    pdf_url: str | None = None
    sections: Sequence[Any] = field(default_factory=tuple)
    fulltext: str = ""


@dataclass(frozen=True, slots=True)
class PaperAnalysis:
    overview: str
    key_findings: list[str]


def compact_text(value: Any, *, max_chars: int) -> str:
    text = collapse_whitespace(str(value or ""))
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + ELLIPSIS


def author_names(authors: Any) -> list[str]:
    if not authors:
        return []
    names: list[str] = []
    for author in authors:
        if isinstance(author, str):
            name = author.strip()
        elif isinstance(author, Mapping):
            name = str(author.get("name") or "").strip()
        else:
            continue
        if name:
            names.append(name)
    return names


def format_paper_metadata(paper: PaperDetailInput) -> str:
    authors = ", ".join(author_names(paper.authors)) or UNKNOWN_AUTHORS
    categories = ", ".join(str(value) for value in paper.categories if value) or str(
        paper.primary_category or UNKNOWN_CATEGORY
    )
    lines = [
        f"arXiv ID: {paper.arxiv_id or UNKNOWN_VALUE}",
        f"제목: {paper.title or UNTITLED}",
        f"저자: {authors}",
        f"발행일: {paper.published_at or UNKNOWN_VALUE}",
        f"카테고리: {categories}",
        f"PDF: {paper.pdf_url or UNKNOWN_VALUE}",
        f"초록: {compact_text(paper.abstract, max_chars=METADATA_ABSTRACT_MAX_CHARS) or UNKNOWN_VALUE}",
    ]
    return "\n".join(lines)


def _section_title(section: Mapping[str, Any]) -> str:
    return str(section.get("title") or "").lower()


def select_detail_sections(sections: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    prioritized: list[Mapping[str, Any]] = []
    remainder: list[Mapping[str, Any]] = []
    for section in sections:
        title = _section_title(section)
        if any(keyword in title for keyword in PRIORITY_SECTION_KEYWORDS):
            prioritized.append(section)
        else:
            remainder.append(section)
    selected = prioritized[:MAX_SELECTED_SECTIONS]
    if len(selected) < MAX_SELECTED_SECTIONS:
        selected.extend(remainder[: MAX_SELECTED_SECTIONS - len(selected)])
    return selected


def _dict_sections(paper: PaperDetailInput) -> list[Mapping[str, Any]]:
    return [section for section in paper.sections if isinstance(section, Mapping)]


def format_paper_sections(paper: PaperDetailInput) -> str:
    sections = _dict_sections(paper)
    if sections:
        parts: list[str] = []
        for index, section in enumerate(select_detail_sections(sections), 1):
            fallback_title = f"Section {index}"
            title = str(section.get("title") or fallback_title).strip() or fallback_title
            text = compact_text(section.get("text"), max_chars=SECTION_TEXT_MAX_CHARS)
            if text:
                parts.append(f"[섹션 {index}] {title}\n{text}")
        if parts:
            return SECTION_SEPARATOR.join(parts)
    fulltext = compact_text(paper.fulltext, max_chars=FULLTEXT_MAX_CHARS)
    if fulltext:
        return f"[본문]\n{fulltext}"
    return NO_SECTIONS_TEXT


def has_paper_detail_context(paper: PaperDetailInput) -> bool:
    if any(section.get("text") for section in _dict_sections(paper)):
        return True
    return bool(paper.fulltext.strip())


def extract_key_findings(text: str, *, max_items: int = MAX_KEY_FINDINGS) -> list[str]:
    findings: list[str] = []
    for raw_line in text.splitlines():
        cleaned = collapse_whitespace(FINDING_MARKER.sub("", raw_line.strip()))
        if len(cleaned) < MIN_KEY_FINDING_CHARS:
            continue
        if not cleaned.endswith(SENTENCE_ENDINGS):
            cleaned += "."
        findings.append(cleaned)
    unique: list[str] = []
    seen: set[str] = set()
    for finding in findings:
        key = FINDING_KEY_NOISE.sub("", finding).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return unique[:max_items]


class PaperDetailWriter:
    def __init__(
        self,
        models: ChatModelFactory,
        tracer: Tracer,
        *,
        trace_runtime: str | None = DEFAULT_TRACE_RUNTIME,
    ) -> None:
        self.models = models
        self.tracer = tracer
        self.trace_runtime = trace_runtime

    async def _run(self, prompt: ChatPromptTemplate, stage: TraceStage, paper: PaperDetailInput, model: str) -> str:
        if not has_paper_detail_context(paper):
            raise MissingPaperContextError(MISSING_CONTEXT_MESSAGE)
        chain = prompt | self.models.create(model=model, temperature=DETAIL_TEMPERATURE) | StrOutputParser()
        inputs = {"paper_metadata": format_paper_metadata(paper), "paper_sections": format_paper_sections(paper)}
        with self.tracer.scope():
            return await chain.ainvoke(inputs, config=self.tracer.config(stage, runtime=self.trace_runtime))

    async def overview(self, paper: PaperDetailInput, *, model: str) -> str:
        return await self._run(OVERVIEW_PROMPT, "paper_overview", paper, model)

    async def key_findings(self, paper: PaperDetailInput, *, model: str) -> list[str]:
        return extract_key_findings(await self._run(KEY_FINDINGS_PROMPT, "paper_key_findings", paper, model))

    async def analyze(self, paper: PaperDetailInput, *, model: str) -> PaperAnalysis:
        overview = await self.overview(paper, model=model)
        key_findings = await self.key_findings(paper, model=model)
        return PaperAnalysis(overview=overview, key_findings=key_findings)
