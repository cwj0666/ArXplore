from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from arxplore.llm.detail import author_names, compact_text
from arxplore.llm.models import ChatModelFactory
from arxplore.llm.prompts.summary import SUMMARY_BUCKET_PROMPT, SUMMARY_PROMPT, SUMMARY_SECTION_PROMPT
from arxplore.llm.tracing import Tracer
from arxplore.parsing.roles import FRONT_MATTER_ROLE, REFERENCES_ROLE, TOC_ROLE, section_role
from arxplore.text import collapse_whitespace

SUMMARY_PROMPT_VERSION = 2
FALLBACK_TEXT_MAX_CHARS = 12000
SECTION_TEXT_MAX_CHARS = 3200
MERGED_SUMMARY_MAX_CHARS = 22000
EVIDENCE_SECTIONS_PER_BUCKET = 2
EVIDENCE_TEXT_MAX_CHARS = 900
MAX_SELECTED_SECTIONS = 12
TRUNCATION_SUFFIX = "\n\n[이하 생략]"
SECTION_TEMPERATURE = 0.1
BUCKET_TEMPERATURE = 0.15
FINAL_TEMPERATURE = 0.2
OTHER_BUCKET = "other"
UNTITLED = "제목 없음"
UNKNOWN_AUTHORS = "저자 미상"
NON_BODY_ROLES = frozenset({FRONT_MATTER_ROLE, REFERENCES_ROLE, TOC_ROLE})
ACKNOWLEDGMENT_TITLE_MARKER = "acknowledg"


@dataclass(frozen=True, slots=True)
class SummaryBucket:
    name: str
    label: str
    keywords: tuple[str, ...]
    selection_limit: int


CLASSIFIED_BUCKETS = (
    SummaryBucket(
        "background",
        "배경/문제",
        ("abstract", "introduction", "motivation", "background", "related work", "preliminar"),
        2,
    ),
    SummaryBucket(
        "method",
        "방법",
        ("method", "approach", "model", "architecture", "algorithm", "framework", "training"),
        3,
    ),
    SummaryBucket(
        "experiments",
        "실험/결과",
        ("experiment", "evaluation", "result", "benchmark", "analysis", "ablation"),
        3,
    ),
    SummaryBucket(
        "limitations",
        "한계/결론",
        ("limitation", "discussion", "failure", "conclusion", "future work"),
        3,
    ),
)
OTHER_SUMMARY_BUCKET = SummaryBucket(OTHER_BUCKET, "기타", (), 0)
SUMMARIZED_BUCKETS = (*CLASSIFIED_BUCKETS, OTHER_SUMMARY_BUCKET)

type Section = Mapping[str, Any]


def truncate_with_marker(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + TRUNCATION_SUFFIX


def format_summary_authors(authors: Any) -> str:
    if not authors:
        return UNKNOWN_AUTHORS
    if isinstance(authors, str):
        return authors.strip() or UNKNOWN_AUTHORS
    if not isinstance(authors, list):
        return UNKNOWN_AUTHORS
    return ", ".join(author_names(authors)) or UNKNOWN_AUTHORS


def classify_section_bucket(title: str) -> str:
    normalized = title.strip().lower()
    if not normalized:
        return OTHER_BUCKET
    for bucket in CLASSIFIED_BUCKETS:
        if any(keyword in normalized for keyword in bucket.keywords):
            return bucket.name
    return OTHER_BUCKET


def _bucket_of(section: Section) -> str:
    return classify_section_bucket(str(section.get("title") or ""))


def is_summarizable_other_title(title: str) -> bool:
    return section_role(title) not in NON_BODY_ROLES and ACKNOWLEDGMENT_TITLE_MARKER not in title.lower()


def select_summary_sections(sections: Sequence[Any]) -> list[Section]:
    by_bucket: dict[str, list[Section]] = {bucket.name: [] for bucket in CLASSIFIED_BUCKETS}
    by_bucket[OTHER_BUCKET] = []
    for section in sections:
        if not isinstance(section, Mapping) or not str(section.get("text") or "").strip():
            continue
        bucket_name = _bucket_of(section)
        if bucket_name == OTHER_BUCKET and not is_summarizable_other_title(str(section.get("title") or "")):
            continue
        by_bucket[bucket_name].append(section)
    selected: list[Section] = []
    for bucket in CLASSIFIED_BUCKETS:
        selected.extend(by_bucket[bucket.name][: bucket.selection_limit])
    if len(selected) < MAX_SELECTED_SECTIONS:
        selected.extend(by_bucket[OTHER_BUCKET][: MAX_SELECTED_SECTIONS - len(selected)])
    return selected[:MAX_SELECTED_SECTIONS]


def group_summary_sections(sections: Sequence[Section]) -> dict[str, list[Section]]:
    grouped: dict[str, list[Section]] = {bucket.name: [] for bucket in SUMMARIZED_BUCKETS}
    for section in sections:
        bucket = _bucket_of(section)
        if bucket in grouped:
            grouped[bucket].append(section)
    return grouped


def _section_label(section: Section, index: int) -> str:
    fallback = f"Section {index}"
    return str(section.get("title") or fallback).strip() or fallback


def build_bucket_evidence(sections: Sequence[Section]) -> str:
    parts: list[str] = []
    for index, section in enumerate(sections[:EVIDENCE_SECTIONS_PER_BUCKET], 1):
        body = compact_text(section.get("text"), max_chars=EVIDENCE_TEXT_MAX_CHARS)
        if body:
            parts.append(f"[근거 {index}] {_section_label(section, index)}\n{body}")
    return "\n\n".join(parts)


def merge_bucket_summaries(
    summaries: Mapping[str, str], grouped: Mapping[str, Sequence[Section]], fallback_text: str
) -> str:
    parts: list[str] = []
    for bucket in SUMMARIZED_BUCKETS:
        text = collapse_whitespace(str(summaries.get(bucket.name) or ""))
        if not text:
            continue
        evidence = build_bucket_evidence(grouped.get(bucket.name, []))
        if evidence:
            parts.append(f"[{bucket.label} 해설]\n{text}\n\n[{bucket.label} 원문 근거]\n{evidence}")
        else:
            parts.append(f"[{bucket.label} 해설]\n{text}")
    merged = "\n\n".join(parts) or fallback_text
    return truncate_with_marker(merged, MERGED_SUMMARY_MAX_CHARS)


def _merge_summaries(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    return {**left, **right}


class SummaryState(TypedDict, total=False):
    title: str
    authors: str
    sections: list[Section]
    fallback_text: str
    grouped_sections: dict[str, list[Section]]
    bucket_summaries: Annotated[dict[str, str], _merge_summaries]
    merged_section_summaries: str
    final_summary: str


@dataclass(frozen=True, slots=True)
class SummaryContext:
    models: ChatModelFactory
    model: str
    trace: RunnableConfig


async def _summarize_bucket(context: SummaryContext, *, title: str, bucket: str, sections: Sequence[Section]) -> str:
    if not sections:
        return ""
    section_chain = (
        SUMMARY_SECTION_PROMPT
        | context.models.create(model=context.model, temperature=SECTION_TEMPERATURE)
        | StrOutputParser()
    )
    per_section: list[str] = []
    for index, section in enumerate(sections, 1):
        section_title = _section_label(section, index)
        section_text = compact_text(section.get("text"), max_chars=SECTION_TEXT_MAX_CHARS)
        if not section_text:
            continue
        section_summary = await section_chain.ainvoke(
            {
                "title": title or UNTITLED,
                "section_title": section_title,
                "section_bucket": bucket,
                "section_text": section_text,
            },
            config=context.trace,
        )
        section_summary = collapse_whitespace(str(section_summary or ""))
        if section_summary:
            per_section.append(f"[{section_title}]\n{section_summary}")
    if not per_section:
        return ""
    bucket_chain = (
        SUMMARY_BUCKET_PROMPT
        | context.models.create(model=context.model, temperature=BUCKET_TEMPERATURE)
        | StrOutputParser()
    )
    return await bucket_chain.ainvoke(
        {"title": title or UNTITLED, "bucket": bucket, "section_summaries": "\n\n".join(per_section)},
        config=context.trace,
    )


async def _normalize_input(state: SummaryState) -> dict[str, Any]:
    return {
        "sections": [section for section in state.get("sections", []) if isinstance(section, Mapping)],
        "fallback_text": truncate_with_marker(state.get("fallback_text", "").strip(), FALLBACK_TEXT_MAX_CHARS),
        "grouped_sections": {},
        "bucket_summaries": {},
    }


async def _select_sections(state: SummaryState) -> dict[str, Any]:
    return {"grouped_sections": group_summary_sections(select_summary_sections(state.get("sections", [])))}


class BucketNode(Protocol):
    def __call__(self, state: SummaryState, *, runtime: Runtime[SummaryContext]) -> Awaitable[dict[str, Any]]: ...


def _bucket_node(bucket: str) -> BucketNode:
    async def summarize(state: SummaryState, *, runtime: Runtime[SummaryContext]) -> dict[str, Any]:
        text = await _summarize_bucket(
            runtime.context,
            title=state["title"],
            bucket=bucket,
            sections=state.get("grouped_sections", {}).get(bucket, []),
        )
        return {"bucket_summaries": {bucket: text}}

    return summarize


async def _merge_section_summaries(state: SummaryState) -> dict[str, Any]:
    merged = merge_bucket_summaries(
        state.get("bucket_summaries", {}),
        state.get("grouped_sections", {}),
        state.get("fallback_text", ""),
    )
    return {"merged_section_summaries": merged}


async def _generate_summary(state: SummaryState, runtime: Runtime[SummaryContext]) -> dict[str, Any]:
    source_text = state.get("merged_section_summaries") or state.get("fallback_text") or ""
    if not source_text:
        return {"final_summary": ""}
    context = runtime.context
    chain = (
        SUMMARY_PROMPT | context.models.create(model=context.model, temperature=FINAL_TEMPERATURE) | StrOutputParser()
    )
    summary = await chain.ainvoke(
        {"title": state["title"] or UNTITLED, "authors": state["authors"], "text": source_text},
        config=context.trace,
    )
    return {"final_summary": summary}


def build_summary_graph() -> CompiledStateGraph[SummaryState, SummaryContext, SummaryState, SummaryState]:
    graph = StateGraph(SummaryState, context_schema=SummaryContext)
    graph.add_node("normalize_input", _normalize_input)
    graph.add_node("select_sections", _select_sections)
    graph.add_node("merge_section_summaries", _merge_section_summaries)
    graph.add_node("generate_summary", _generate_summary)
    graph.add_edge(START, "normalize_input")
    graph.add_edge("normalize_input", "select_sections")
    for bucket in SUMMARIZED_BUCKETS:
        node = f"summarize_{bucket.name}"
        graph.add_node(node, _bucket_node(bucket.name))
        graph.add_edge("select_sections", node)
        graph.add_edge(node, "merge_section_summaries")
    graph.add_edge("merge_section_summaries", "generate_summary")
    graph.add_edge("generate_summary", END)
    return graph.compile()


class SummaryWriter:
    def __init__(self, models: ChatModelFactory, tracer: Tracer) -> None:
        self.models = models
        self.tracer = tracer
        self.graph = build_summary_graph()

    async def summarize(
        self,
        *,
        title: str,
        authors: Any,
        text: str,
        sections: Sequence[Any] | None,
        model: str,
    ) -> str:
        context = SummaryContext(
            models=self.models,
            model=model,
            trace=self.tracer.config("summary"),
        )
        state: SummaryState = {
            "title": title or UNTITLED,
            "authors": format_summary_authors(authors),
            "sections": list(sections or []),
            "fallback_text": str(text or "").strip(),
        }
        with self.tracer.scope():
            result = await self.graph.ainvoke(state, context=context)
        return str(result.get("final_summary") or "")
