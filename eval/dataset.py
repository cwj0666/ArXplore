"""평가 질의 데이터셋(JSONL) 스키마, 로더, 질의 생성 보조 함수."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_LANGS = frozenset({"ko", "en"})
VALID_SOURCES = frozenset({"known_item", "llm_synth", "manual"})
VALID_BEHAVIORS = ("answer", "refuse", "clarify")
VALID_MODE_HINTS = ("agent", "paper_chat", "both")
HISTORY_ROLES = ("user", "assistant")
DEFAULT_BEHAVIOR = "answer"
DEFAULT_MODE_HINT = "both"

PLACEHOLDER_PATTERN = re.compile(r"0000\.0000[a-z]")
TITLE_TEMPLATE_PATTERN = re.compile(r"\{\{(title|title_head):(0000\.0000[a-z])(?::(\d+))?\}\}")


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
    category: str = ""
    expected_behavior: str = DEFAULT_BEHAVIOR
    mode: str = DEFAULT_MODE_HINT
    history: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    must_not_contain: tuple[str, ...] = field(default_factory=tuple)
    must_mention_arxiv_ids: tuple[str, ...] = field(default_factory=tuple)
    expect_rejected: bool = False
    open_arxiv_id: str = ""

    @property
    def paper_chat_arxiv_id(self) -> str | None:
        """상세 챗에서 열어 둔 논문. `open_arxiv_id`가 없으면 `relevant_arxiv_ids[0]`, 둘 다 없으면 None."""
        if self.open_arxiv_id:
            return self.open_arxiv_id
        return self.relevant_arxiv_ids[0] if self.relevant_arxiv_ids else None

    def supports_mode(self, mode: str) -> bool:
        """`mode` 힌트가 `both`이거나 주어진 답변 경로와 같으면 True."""
        return self.mode == DEFAULT_MODE_HINT or self.mode == mode

    @property
    def retrieval_eligible(self) -> bool:
        """검색 평가 대상 여부: 거절·입력 거부 기대 케이스, 정답 논문이 없는 케이스, 상세 챗 전용·대화 이력 케이스는 제외."""
        return (
            self.expected_behavior != "refuse"
            and not self.expect_rejected
            and bool(self.relevant_arxiv_ids)
            and self.mode != "paper_chat"
            and not self.history
        )

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
        if self.category:
            payload["category"] = self.category
        if self.expected_behavior != DEFAULT_BEHAVIOR:
            payload["expected_behavior"] = self.expected_behavior
        if self.mode != DEFAULT_MODE_HINT:
            payload["mode"] = self.mode
        if self.open_arxiv_id:
            payload["open_arxiv_id"] = self.open_arxiv_id
        if self.history:
            payload["history"] = [{"role": role, "content": content} for role, content in self.history]
        if self.must_not_contain:
            payload["must_not_contain"] = list(self.must_not_contain)
        if self.must_mention_arxiv_ids:
            payload["must_mention_arxiv_ids"] = list(self.must_mention_arxiv_ids)
        if self.expect_rejected:
            payload["expect_rejected"] = True
        return payload


def _string_list(payload: dict[str, Any], key: str, prefix: str) -> tuple[str, ...]:
    values = payload.get(key)
    if values is None:
        return ()
    if not isinstance(values, list) or not all(isinstance(value, str) and value.strip() for value in values):
        raise DatasetError(f"{prefix}'{key}' must be a list of non-empty strings")
    return tuple(dict.fromkeys(value.strip() for value in values))


def _parse_history(payload: dict[str, Any], prefix: str) -> tuple[tuple[str, str], ...]:
    history = payload.get("history")
    if history is None:
        return ()
    if not isinstance(history, list):
        raise DatasetError(f"{prefix}'history' must be a list of {{role, content}} objects")
    turns: list[tuple[str, str]] = []
    for index, turn in enumerate(history):
        if not isinstance(turn, dict):
            raise DatasetError(f"{prefix}'history[{index}]' must be an object with 'role' and 'content'")
        role = turn.get("role")
        if role not in HISTORY_ROLES:
            raise DatasetError(f"{prefix}'history[{index}].role' must be one of {list(HISTORY_ROLES)}, got {role!r}")
        content = turn.get("content")
        if not isinstance(content, str):
            raise DatasetError(f"{prefix}'history[{index}].content' must be a string")
        turns.append((role, content))
    return tuple(turns)


def _choice(payload: dict[str, Any], key: str, choices: Sequence[str], default: str, prefix: str) -> str:
    value = payload.get(key, default)
    if value not in choices:
        raise DatasetError(f"{prefix}'{key}' must be one of {list(choices)}, got {value!r}")
    return value


def parse_query(payload: dict[str, Any], *, location: str = "") -> EvalQuery:
    """JSON 객체 하나를 검증해 `EvalQuery`로 만든다. 새 필드가 없는 예전 레코드는 기본값으로 읽는다.

    `query`는 앞뒤 공백을 포함해 그대로 보존한다. `expect_rejected`가 참이면 빈 문자열·공백만 있는 질의도 허용하고,
    `expected_behavior`가 `answer`인 케이스(입력 거부 기대 제외)는 `relevant_arxiv_ids`나
    `must_mention_arxiv_ids` 중 하나가 비어 있지 않아야 한다. 상세 챗 경로를 쓰는 케이스는 열어 둘 논문
    (`open_arxiv_id` 또는 `relevant_arxiv_ids[0]`)이 있어야 한다.
    """
    prefix = f"{location}: " if location else ""
    if not isinstance(payload, dict):
        raise DatasetError(f"{prefix}each line must be a JSON object")

    def require_str(key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise DatasetError(f"{prefix}'{key}' must be a non-empty string")
        return value.strip()

    query_id = require_str("id")
    expect_rejected = payload.get("expect_rejected", False)
    if not isinstance(expect_rejected, bool):
        raise DatasetError(f"{prefix}'expect_rejected' must be a boolean")
    query = payload.get("query")
    if not isinstance(query, str) or (not expect_rejected and not query.strip()):
        raise DatasetError(f"{prefix}'query' must be a non-empty string")
    lang = require_str("lang")
    if lang not in VALID_LANGS:
        raise DatasetError(f"{prefix}'lang' must be one of {sorted(VALID_LANGS)}, got {lang!r}")
    source = require_str("source")
    if source not in VALID_SOURCES:
        raise DatasetError(f"{prefix}'source' must be one of {sorted(VALID_SOURCES)}, got {source!r}")

    expected_behavior = _choice(payload, "expected_behavior", VALID_BEHAVIORS, DEFAULT_BEHAVIOR, prefix)
    mode = _choice(payload, "mode", VALID_MODE_HINTS, DEFAULT_MODE_HINT, prefix)
    category = payload.get("category", "")
    if not isinstance(category, str):
        raise DatasetError(f"{prefix}'category' must be a string")
    open_arxiv_id = payload.get("open_arxiv_id", "")
    if not isinstance(open_arxiv_id, str):
        raise DatasetError(f"{prefix}'open_arxiv_id' must be a string")
    must_not_contain = _string_list(payload, "must_not_contain", prefix)
    must_mention = _string_list(payload, "must_mention_arxiv_ids", prefix)
    history = _parse_history(payload, prefix)

    arxiv_ids = payload.get("relevant_arxiv_ids", [])
    if not isinstance(arxiv_ids, list):
        raise DatasetError(f"{prefix}'relevant_arxiv_ids' must be a list")
    if not all(isinstance(value, str) and value.strip() for value in arxiv_ids):
        raise DatasetError(f"{prefix}'relevant_arxiv_ids' must contain non-empty strings")
    if expected_behavior == DEFAULT_BEHAVIOR and not expect_rejected and not arxiv_ids and not must_mention:
        raise DatasetError(
            f"{prefix}'relevant_arxiv_ids' must be a non-empty list for answer cases (or give 'must_mention_arxiv_ids')"
        )

    chunk_ids = payload.get("relevant_chunk_ids") or []
    if not isinstance(chunk_ids, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in chunk_ids
    ):
        raise DatasetError(f"{prefix}'relevant_chunk_ids' must be a list of integers")

    notes = payload.get("notes", "")
    if not isinstance(notes, str):
        raise DatasetError(f"{prefix}'notes' must be a string")

    record = EvalQuery(
        id=query_id,
        query=query,
        lang=lang,
        relevant_arxiv_ids=tuple(dict.fromkeys(value.strip() for value in arxiv_ids)),
        source=source,
        relevant_chunk_ids=tuple(dict.fromkeys(chunk_ids)),
        notes=notes,
        category=category.strip(),
        expected_behavior=expected_behavior,
        mode=mode,
        history=history,
        must_not_contain=must_not_contain,
        must_mention_arxiv_ids=must_mention,
        expect_rejected=expect_rejected,
        open_arxiv_id=open_arxiv_id.strip(),
    )
    if mode != "agent" and not expect_rejected and record.paper_chat_arxiv_id is None:
        raise DatasetError(
            f"{prefix}mode {mode!r} runs paper_chat and needs 'open_arxiv_id' or 'relevant_arxiv_ids'; "
            "use mode 'agent' for cases without a paper"
        )
    return record


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


def dataset_files(path: str | Path) -> list[Path]:
    """파일이면 그 파일 하나, 디렉터리면 그 안의 `*.jsonl`을 이름순으로 돌려준다."""
    path = Path(path)
    if path.is_dir():
        return sorted(child for child in path.glob("*.jsonl") if child.is_file())
    if not path.exists():
        raise FileNotFoundError(path)
    return [path]


def load_queries(path: str | Path) -> list[EvalQuery]:
    """JSONL 파일 하나 또는 `*.jsonl` 파일이 모인 디렉터리를 읽는다. id는 파일을 가로질러 유일해야 한다."""
    queries: list[EvalQuery] = []
    origins: dict[str, str] = {}
    for file_path in dataset_files(path):
        with file_path.open(encoding="utf-8") as handle:
            loaded = parse_lines(handle, origin=str(file_path))
        for query in loaded:
            if query.id in origins:
                raise DatasetError(f"{file_path}: duplicate id {query.id!r} (first seen in {origins[query.id]})")
            origins[query.id] = str(file_path)
        queries.extend(loaded)
    return queries


def dataset_digest(path: str | Path) -> str:
    """질의셋 파일(디렉터리면 `*.jsonl` 전체를 이름순으로 이어 붙인 바이트)의 sha256 앞 12자리."""
    digest = hashlib.sha256()
    for file_path in dataset_files(path):
        digest.update(file_path.read_bytes())
    return digest.hexdigest()[:12]


def dump_queries(queries: Iterable[EvalQuery]) -> str:
    return "".join(json.dumps(query.to_dict(), ensure_ascii=False) + "\n" for query in queries)


def find_placeholders(value: Any) -> set[str]:
    """문자열·리스트·딕셔너리 안에 남아 있는 자리표시자 arxiv_id(`0000.0000a` 형식)를 모은다."""
    if isinstance(value, str):
        return set(PLACEHOLDER_PATTERN.findall(value))
    if isinstance(value, Mapping):
        return set().union(*(find_placeholders(item) for item in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(find_placeholders(item) for item in value)) if value else set()
    return set()


def _fill_text(text: str, papers: Mapping[str, tuple[str, str]]) -> str:
    def title(match: re.Match[str]) -> str:
        kind, placeholder, words = match.group(1), match.group(2), match.group(3)
        if placeholder not in papers:
            return match.group(0)
        paper_title = " ".join(papers[placeholder][1].split())
        if kind == "title_head":
            return " ".join(paper_title.split()[: max(1, int(words or 4))])
        return paper_title

    filled = TITLE_TEMPLATE_PATTERN.sub(title, text)
    return PLACEHOLDER_PATTERN.sub(
        lambda match: papers[match.group(0)][0] if match.group(0) in papers else match.group(0), filled
    )


def fill_placeholders(value: Any, papers: Mapping[str, tuple[str, str]]) -> Any:
    """자리표시자를 실제 논문으로 바꾼 사본을 돌려준다.

    `papers`는 자리표시자 → (arxiv_id, 제목). 문자열 안의 `{{title:ID}}`는 제목 전체로,
    `{{title_head:ID:N}}`은 제목 앞 N단어로, 나머지 `ID`는 arxiv_id로 바꾼다. 매핑에 없는 자리표시자는 그대로 둔다.
    """
    if isinstance(value, str):
        return _fill_text(value, papers)
    if isinstance(value, Mapping):
        return {key: fill_placeholders(item, papers) for key, item in value.items()}
    if isinstance(value, list):
        return [fill_placeholders(item, papers) for item in value]
    return value


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
