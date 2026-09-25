from __future__ import annotations

from functools import lru_cache
from typing import Any, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.shared import get_runtime_openai_api_key, get_runtime_openai_model

from .prompts import SUMMARY_BUCKET_PROMPT, SUMMARY_PROMPT, SUMMARY_SECTION_PROMPT
from .tracing import build_summary_trace_config

_FALLBACK_TEXT_MAX_CHARS = 12000
_SECTION_TEXT_MAX_CHARS = 3200
_MERGED_SUMMARY_MAX_CHARS = 22000


class SummaryGraphState(TypedDict, total=False):
    title: str
    authors: str
    sections: list[dict[str, Any]]
    fallback_text: str
    selected_sections: list[dict[str, Any]]
    grouped_sections: dict[str, list[dict[str, Any]]]
    runtime: str
    user: str | None
    quality_score: float | None
    background_summary: str
    method_summary: str
    experiments_summary: str
    limitations_summary: str
    merged_section_summaries: str
    final_summary: str


def _build_llm(temperature: float = 0.1) -> ChatOpenAI:
    api_key = get_runtime_openai_api_key()
    model = get_runtime_openai_model()
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않아 한국어 번역/요약 체인을 실행할 수 없습니다.")

    kwargs = {
        "model": model,
        "api_key": api_key,
    }
    if not model.startswith("gpt-5"):
        kwargs["temperature"] = temperature

    return ChatOpenAI(**kwargs)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[이하 생략]"


def _compact_text(value: Any, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _classify_section_bucket(title: str) -> str:
    normalized = title.strip().lower()
    if not normalized:
        return "other"

    rules = (
        ("background", ("abstract", "introduction", "motivation", "background", "related work", "preliminar")),
        ("method", ("method", "approach", "model", "architecture", "algorithm", "framework", "training")),
        ("experiments", ("experiment", "evaluation", "result", "benchmark", "analysis", "ablation")),
        ("limitations", ("limitation", "discussion", "failure", "conclusion", "future work")),
    )
    for bucket, keywords in rules:
        if any(keyword in normalized for keyword in keywords):
            return bucket
    return "other"


def _select_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "background": [],
        "method": [],
        "experiments": [],
        "limitations": [],
        "other": [],
    }

    for section in sections:
        if not isinstance(section, dict):
            continue
        if not str(section.get("text") or "").strip():
            continue
        bucket = _classify_section_bucket(str(section.get("title") or ""))
        buckets[bucket].append(section)

    selected: list[dict[str, Any]] = []
    limits = {
        "background": 2,
        "method": 3,
        "experiments": 3,
        "limitations": 3,
    }
    for bucket in ("background", "method", "experiments", "limitations"):
        selected.extend(buckets[bucket][: limits[bucket]])

    if len(selected) < 12:
        selected.extend(buckets["other"][: 12 - len(selected)])
    return selected[:12]


def _group_sections(sections: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "background": [],
        "method": [],
        "experiments": [],
        "limitations": [],
    }
    for section in sections:
        bucket = _classify_section_bucket(str(section.get("title") or ""))
        if bucket in grouped:
            grouped[bucket].append(section)
    return grouped


def _build_bucket_text(sections: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for index, section in enumerate(sections, 1):
        title = str(section.get("title") or f"Section {index}").strip() or f"Section {index}"
        body = _compact_text(section.get("text"), max_chars=_SECTION_TEXT_MAX_CHARS)
        if not body:
            continue
        parts.append(f"[섹션 {index}] {title}\n{body}")
    return "\n\n".join(parts)


def _build_bucket_evidence(sections: list[dict[str, Any]], *, max_items: int = 2, max_chars: int = 900) -> str:
    parts: list[str] = []
    for index, section in enumerate(sections[:max_items], 1):
        title = str(section.get("title") or f"Section {index}").strip() or f"Section {index}"
        body = _compact_text(section.get("text"), max_chars=max_chars)
        if not body:
            continue
        parts.append(f"[근거 {index}] {title}\n{body}")
    return "\n\n".join(parts)


def _summarize_bucket(
    *,
    title: str,
    bucket: str,
    sections: list[dict[str, Any]],
    runtime: str,
    user: str | None,
    quality_score: float | None,
) -> str:
    if not sections:
        return ""

    config = build_summary_trace_config(
        runtime=runtime,
        user=user,
        quality_score=quality_score,
    )
    section_chain = SUMMARY_SECTION_PROMPT | _build_llm(temperature=0.1) | StrOutputParser()
    per_section_summaries: list[str] = []
    for index, section in enumerate(sections, 1):
        section_title = str(section.get("title") or f"Section {index}").strip() or f"Section {index}"
        section_text = _compact_text(section.get("text"), max_chars=_SECTION_TEXT_MAX_CHARS)
        if not section_text:
            continue
        section_summary = section_chain.invoke(
            {
                "title": title or "제목 없음",
                "section_title": section_title,
                "section_bucket": bucket,
                "section_text": section_text,
            },
            config=config,
        )
        section_summary = " ".join(str(section_summary or "").split())
        if not section_summary:
            continue
        per_section_summaries.append(f"[{section_title}]\n{section_summary}")

    if not per_section_summaries:
        return ""

    bucket_chain = SUMMARY_BUCKET_PROMPT | _build_llm(temperature=0.15) | StrOutputParser()
    return bucket_chain.invoke(
        {
            "title": title or "제목 없음",
            "bucket": bucket,
            "section_summaries": "\n\n".join(per_section_summaries),
        },
        config=config,
    )


def _normalize_input(state: SummaryGraphState) -> SummaryGraphState:
    normalized_sections = [section for section in state.get("sections", []) if isinstance(section, dict)]
    return {
        "title": state["title"],
        "authors": state["authors"],
        "sections": normalized_sections,
        "fallback_text": _truncate(state.get("fallback_text", "").strip(), _FALLBACK_TEXT_MAX_CHARS),
        "selected_sections": [],
        "grouped_sections": {},
        "runtime": state["runtime"],
        "user": state.get("user"),
        "quality_score": state.get("quality_score"),
    }


def _select_sections_node(state: SummaryGraphState) -> dict[str, Any]:
    selected_sections = _select_sections(state.get("sections", []))
    return {
        "selected_sections": selected_sections,
        "grouped_sections": _group_sections(selected_sections),
    }


def _summarize_background_node(state: SummaryGraphState) -> dict[str, str]:
    return {
        "background_summary": _summarize_bucket(
            title=state["title"],
            bucket="background",
            sections=state.get("grouped_sections", {}).get("background", []),
            runtime=state["runtime"],
            user=state.get("user"),
            quality_score=state.get("quality_score"),
        )
    }


def _summarize_method_node(state: SummaryGraphState) -> dict[str, str]:
    return {
        "method_summary": _summarize_bucket(
            title=state["title"],
            bucket="method",
            sections=state.get("grouped_sections", {}).get("method", []),
            runtime=state["runtime"],
            user=state.get("user"),
            quality_score=state.get("quality_score"),
        )
    }


def _summarize_experiments_node(state: SummaryGraphState) -> dict[str, str]:
    return {
        "experiments_summary": _summarize_bucket(
            title=state["title"],
            bucket="experiments",
            sections=state.get("grouped_sections", {}).get("experiments", []),
            runtime=state["runtime"],
            user=state.get("user"),
            quality_score=state.get("quality_score"),
        )
    }


def _summarize_limitations_node(state: SummaryGraphState) -> dict[str, str]:
    return {
        "limitations_summary": _summarize_bucket(
            title=state["title"],
            bucket="limitations",
            sections=state.get("grouped_sections", {}).get("limitations", []),
            runtime=state["runtime"],
            user=state.get("user"),
            quality_score=state.get("quality_score"),
        )
    }


def _merge_section_summaries_node(state: SummaryGraphState) -> dict[str, str]:
    parts: list[str] = []
    mapping = (
        ("background_summary", "배경/문제", "background"),
        ("method_summary", "방법", "method"),
        ("experiments_summary", "실험/결과", "experiments"),
        ("limitations_summary", "한계/결론", "limitations"),
    )
    grouped_sections = state.get("grouped_sections", {})
    for key, label, bucket in mapping:
        text = " ".join(str(state.get(key) or "").split())
        if not text:
            continue
        evidence = _build_bucket_evidence(grouped_sections.get(bucket, []))
        if evidence:
            parts.append(f"[{label} 해설]\n{text}\n\n[{label} 원문 근거]\n{evidence}")
        else:
            parts.append(f"[{label} 해설]\n{text}")

    merged = "\n\n".join(parts)
    if not merged:
        merged = state.get("fallback_text", "")
    return {"merged_section_summaries": _truncate(merged, _MERGED_SUMMARY_MAX_CHARS)}


def _generate_summary_node(state: SummaryGraphState) -> dict[str, str]:
    source_text = state.get("merged_section_summaries") or state.get("fallback_text") or ""
    if not source_text:
        return {"final_summary": ""}

    chain = SUMMARY_PROMPT | _build_llm(temperature=0.2) | StrOutputParser()
    config = build_summary_trace_config(
        runtime=state["runtime"],
        user=state.get("user"),
        quality_score=state.get("quality_score"),
    )
    summary = chain.invoke(
        {
            "title": state["title"] or "제목 없음",
            "authors": state["authors"],
            "text": source_text,
        },
        config=config,
    )
    return {"final_summary": summary}


@lru_cache(maxsize=1)
def _build_graph():
    graph = StateGraph(SummaryGraphState)
    graph.add_node("normalize_input", _normalize_input)
    graph.add_node("select_sections", _select_sections_node)
    graph.add_node("summarize_background", _summarize_background_node)
    graph.add_node("summarize_method", _summarize_method_node)
    graph.add_node("summarize_experiments", _summarize_experiments_node)
    graph.add_node("summarize_limitations", _summarize_limitations_node)
    graph.add_node("merge_section_summaries", _merge_section_summaries_node)
    graph.add_node("generate_summary", _generate_summary_node)

    graph.add_edge(START, "normalize_input")
    graph.add_edge("normalize_input", "select_sections")
    graph.add_edge("select_sections", "summarize_background")
    graph.add_edge("select_sections", "summarize_method")
    graph.add_edge("select_sections", "summarize_experiments")
    graph.add_edge("select_sections", "summarize_limitations")
    graph.add_edge("summarize_background", "merge_section_summaries")
    graph.add_edge("summarize_method", "merge_section_summaries")
    graph.add_edge("summarize_experiments", "merge_section_summaries")
    graph.add_edge("summarize_limitations", "merge_section_summaries")
    graph.add_edge("merge_section_summaries", "generate_summary")
    graph.add_edge("generate_summary", END)
    return graph.compile()


def generate_summary_via_graph(
    *,
    title: str,
    authors: str,
    text: str,
    sections: list[dict[str, Any]] | None,
    runtime: str,
    user: str | None,
    quality_score: float | None,
) -> str:
    graph = _build_graph()
    result = graph.invoke(
        {
            "title": title or "제목 없음",
            "authors": authors,
            "sections": sections or [],
            "fallback_text": text or "",
            "runtime": runtime,
            "user": user,
            "quality_score": quality_score,
            "selected_sections": [],
            "grouped_sections": {},
        }
    )
    return str(result.get("final_summary") or "")
