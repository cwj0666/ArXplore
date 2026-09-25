"""평가 질의 데이터셋(JSONL) 스키마, 로더, 질의 생성 보조 함수."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_LANGS = frozenset({"ko", "en"})
VALID_SOURCES = frozenset({"known_item", "llm_synth", "manual"})


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class EvalQuery:
    id: str
    query: str
    lang: str
    relevant_arxiv_ids: tuple[str, ...]
    source: str
    relevant_chunk_ids: tuple[int, ...] = field(default_factory=tuple)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "query": self.query,
            "lang": self.lang,
            "relevant_arxiv_ids": list(self.relevant_arxiv_ids),
        }
        if self.relevant_chunk_ids:
            payload["relevant_chunk_ids"] = list(self.relevant_chunk_ids)
        payload["source"] = self.source
        payload["notes"] = self.notes
        return payload


def parse_query(payload: dict[str, Any], *, location: str = "") -> EvalQuery:
    prefix = f"{location}: " if location else ""
    if not isinstance(payload, dict):
        raise DatasetError(f"{prefix}each line must be a JSON object")

    def require_str(key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise DatasetError(f"{prefix}'{key}' must be a non-empty string")
        return value.strip()

    query_id = require_str("id")
    query = require_str("query")
    lang = require_str("lang")
    if lang not in VALID_LANGS:
        raise DatasetError(f"{prefix}'lang' must be one of {sorted(VALID_LANGS)}, got {lang!r}")
    source = require_str("source")
    if source not in VALID_SOURCES:
        raise DatasetError(f"{prefix}'source' must be one of {sorted(VALID_SOURCES)}, got {source!r}")

    arxiv_ids = payload.get("relevant_arxiv_ids")
    if not isinstance(arxiv_ids, list) or not arxiv_ids:
        raise DatasetError(f"{prefix}'relevant_arxiv_ids' must be a non-empty list")
    if not all(isinstance(value, str) and value.strip() for value in arxiv_ids):
        raise DatasetError(f"{prefix}'relevant_arxiv_ids' must contain non-empty strings")

    chunk_ids = payload.get("relevant_chunk_ids") or []
    if not isinstance(chunk_ids, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in chunk_ids
    ):
        raise DatasetError(f"{prefix}'relevant_chunk_ids' must be a list of integers")

    notes = payload.get("notes", "")
    if not isinstance(notes, str):
        raise DatasetError(f"{prefix}'notes' must be a string")

    return EvalQuery(
        id=query_id,
        query=query,
        lang=lang,
        relevant_arxiv_ids=tuple(dict.fromkeys(value.strip() for value in arxiv_ids)),
        source=source,
        relevant_chunk_ids=tuple(dict.fromkeys(chunk_ids)),
        notes=notes,
    )


def parse_lines(lines: Iterable[str], *, origin: str = "<input>") -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        location = f"{origin}:{line_number}"
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{location}: invalid JSON ({exc.msg})") from exc
        query = parse_query(payload, location=location)
        if query.id in seen_ids:
            raise DatasetError(f"{location}: duplicate id {query.id!r}")
        seen_ids.add(query.id)
        queries.append(query)
    return queries


def load_queries(path: str | Path) -> list[EvalQuery]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        return parse_lines(handle, origin=str(path))


def dump_queries(queries: Iterable[EvalQuery]) -> str:
    return "".join(json.dumps(query.to_dict(), ensure_ascii=False) + "\n" for query in queries)


_WORD_PATTERN = re.compile(r"[0-9a-z]+|[가-힣]+")


def longest_shared_word_run(candidate: str, source: str) -> int:
    """두 텍스트가 연속으로 공유하는 가장 긴 단어 수(대소문자·구두점 무시)."""
    candidate_words = _WORD_PATTERN.findall(candidate.lower())
    source_words = _WORD_PATTERN.findall(source.lower())
    if not candidate_words or not source_words:
        return 0
    best = 0
    previous = [0] * (len(source_words) + 1)
    for candidate_word in candidate_words:
        current = [0] * (len(source_words) + 1)
        for index, source_word in enumerate(source_words, start=1):
            if candidate_word == source_word:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best


KNOWN_ITEM_SYSTEM_PROMPT = """You write evaluation queries for a paper search engine.
Given a paper's abstract and key findings, write the kind of query a researcher would type
when they remember what the paper does but not its title.

Rules:
- Paraphrase. Never copy a phrase of 5 or more consecutive words from the input.
- Do not include the paper title or the proposed method/model/dataset name.
- Each query is a single sentence or phrase, 8-25 words (Korean: 15-60 characters).
- Produce exactly one Korean query ("ko") and one English query ("en") about the same content.
- The Korean query should read naturally in Korean; keep common technical terms in English if Korean researchers would."""

CHUNK_SYNTH_SYSTEM_PROMPT = """You write evaluation questions for a paper search engine.
Given one passage from a paper, write one question that can be answered only from this passage
(a specific number, design choice, finding, or setting it states), not from general knowledge.

Rules:
- Paraphrase. Never copy a phrase of 5 or more consecutive words from the passage.
- Do not include the paper title.
- Write the question in the requested language only.
- If the passage has no specific answerable content (e.g. it is boilerplate or a list of citations), answer with an empty question."""


def build_known_item_prompt(*, title: str, abstract: str, key_findings: Iterable[str] = ()) -> str:
    findings = [str(item).strip() for item in key_findings if str(item).strip()]
    parts = [f"Title (do not reuse): {title.strip()}", f"Abstract:\n{abstract.strip()}"]
    if findings:
        parts.append("Key findings:\n" + "\n".join(f"- {item}" for item in findings))
    return "\n\n".join(parts)


def build_chunk_synth_prompt(*, title: str, section_title: str, chunk_text: str, lang: str) -> str:
    language = "Korean" if lang == "ko" else "English"
    return (
        f"Paper title (do not reuse): {title.strip()}\n"
        f"Section: {section_title.strip() or '(unknown)'}\n"
        f"Question language: {language}\n\n"
        f"Passage:\n{chunk_text.strip()}"
    )


def known_item_queries(
    *,
    arxiv_id: str,
    title: str,
    queries_by_lang: dict[str, str],
    basis: str,
) -> list[EvalQuery]:
    records: list[EvalQuery] = []
    for lang in ("ko", "en"):
        text = str(queries_by_lang.get(lang) or "").strip()
        if not text:
            continue
        records.append(
            EvalQuery(
                id=f"ki-{arxiv_id}-{lang}",
                query=text,
                lang=lang,
                relevant_arxiv_ids=(arxiv_id,),
                source="known_item",
                notes=f"basis={basis}; title={title.strip()}",
            )
        )
    return records


def chunk_synth_query(
    *,
    chunk_id: int,
    arxiv_id: str,
    chunk_index: int,
    section_title: str,
    question: str,
    lang: str,
) -> EvalQuery | None:
    text = question.strip()
    if not text:
        return None
    return EvalQuery(
        id=f"cs-{chunk_id}",
        query=text,
        lang=lang,
        relevant_arxiv_ids=(arxiv_id,),
        relevant_chunk_ids=(int(chunk_id),),
        source="llm_synth",
        notes=f"chunk_index={chunk_index}; section={section_title.strip()}",
    )
