import json
import random
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from arxplore_eval.dataset import (
    CHUNK_SYNTH_SYSTEM_PROMPT,
    KNOWN_ITEM_SYSTEM_PROMPT,
    PLACEHOLDER_PATTERN,
    DatasetError,
    EvalQuery,
    PaperFill,
    build_chunk_synth_prompt,
    build_known_item_prompt,
    chunk_synth_query,
    dataset_files,
    fill_placeholders,
    find_placeholders,
    known_item_queries,
    load_queries,
    longest_shared_word_run,
)

KNOWN_ITEM = "known_item"
CHUNK_SYNTH = "chunk_synth"
BOTH = "both"
BUILD_MODES = (KNOWN_ITEM, CHUNK_SYNTH, BOTH)
CHUNK_LANG_POLICIES = ("alternate", "ko", "en")
CHUNK_PROMPT_MAX_CHARS = 3000
QUERY_TEMPERATURE = 0.2
DEFAULT_PAPERS = 20
DEFAULT_CHUNKS = 10
DEFAULT_SEED = 42
DEFAULT_MIN_ABSTRACT_CHARS = 300
DEFAULT_MIN_CHUNK_CHARS = 600
DEFAULT_MAX_SHARED_WORDS = 5
DEFAULT_CANDIDATES = 5
PLACEHOLDER_REGISTRY_NAME = "placeholders.json"
GENERATED_QUERIES_FILE = "queries.generated.jsonl"
CASE_QUERIES_FILE = "queries.cases.jsonl"
COMBINED_QUERIES_FILE = "queries.jsonl"
CASES_DIR_NAME = "cases"
REGISTRY_FILTER_KEYS = (
    "title_keyword",
    "title_keywords",
    "exclude_title_keywords",
    "all_matches",
    "min_chunks",
    "section_keyword",
    "content_role",
    "chunk_pattern",
)
ALL_MATCHES_LIMIT = 50
PREVIEW_TITLE_CHARS = 80

PAPER_IDS_SQL = """
    SELECT p.arxiv_id
    FROM papers p
    WHERE length(coalesce(p.abstract, '')) >= %s
      AND EXISTS (SELECT 1 FROM paper_chunks c WHERE c.arxiv_id = p.arxiv_id)
    ORDER BY p.arxiv_id
"""

CHUNK_IDS_SQL = """
    SELECT c.id
    FROM paper_chunks c
    WHERE coalesce(nullif(c.metadata->>'content_role', ''), 'body') = 'body'
      AND length(c.chunk_text) >= %s
      AND coalesce(c.section_title, '') !~* '^\\s*abstract\\s*$'
    ORDER BY c.id
"""

OVERVIEWS_TABLE_SQL = "SELECT to_regclass('public.paper_ai_overviews') IS NOT NULL AS present"

PAPER_SAMPLES_WITH_OVERVIEWS_SQL = """
    SELECT p.arxiv_id, p.title, p.abstract, coalesce(o.key_findings, '[]'::jsonb) AS key_findings
    FROM papers p
    LEFT JOIN paper_ai_overviews o ON o.arxiv_id = p.arxiv_id
    WHERE p.arxiv_id = ANY(%s)
"""

PAPER_SAMPLES_SQL = """
    SELECT arxiv_id, title, abstract, '[]'::jsonb AS key_findings FROM papers WHERE arxiv_id = ANY(%s)
"""

CHUNK_SAMPLES_SQL = """
    SELECT c.id, c.arxiv_id, c.chunk_index, coalesce(c.section_title, '') AS section_title, c.chunk_text,
           p.title AS paper_title
    FROM paper_chunks c
    JOIN papers p ON p.arxiv_id = c.arxiv_id
    WHERE c.id = ANY(%s)
"""

PAPER_TITLES_SQL = "SELECT arxiv_id, title FROM papers WHERE arxiv_id = ANY(%s)"
ABSENT_PAPER_SQL = "SELECT arxiv_id, title FROM papers WHERE title ILIKE %s ORDER BY arxiv_id LIMIT 1"

type Connection = AsyncConnection[Any]
type LogFn = Callable[[str], None]


@dataclass(frozen=True)
class PaperSample:
    arxiv_id: str
    title: str
    abstract: str
    key_findings: tuple[str, ...]


@dataclass(frozen=True)
class ChunkSample:
    chunk_id: int
    arxiv_id: str
    chunk_index: int
    section_title: str
    chunk_text: str
    paper_title: str
    lang: str


class KnownItemOutput(BaseModel):
    ko: str = Field(description="Korean query")
    en: str = Field(description="English query")


class ChunkQuestionOutput(BaseModel):
    question: str = Field(description="Question in the requested language, or empty string")


def sample_ids[T](ids: Sequence[T], count: int, *, seed: int, stream: str) -> list[T]:
    rng = random.Random(f"{seed}:{stream}")
    return rng.sample(list(ids), min(max(0, count), len(ids)))


def chunk_lang_for(index: int, policy: str) -> str:
    if policy in ("ko", "en"):
        return policy
    return "en" if index % 2 == 0 else "ko"


async def _rows(connection: Connection, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    async with connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(sql, params)
        return await cursor.fetchall()


async def fetch_paper_samples(
    connection: Connection, *, count: int, seed: int, min_abstract_chars: int
) -> list[PaperSample]:
    ids = [str(row["arxiv_id"]) for row in await _rows(connection, PAPER_IDS_SQL, (min_abstract_chars,))]
    sampled = sample_ids(ids, count, seed=seed, stream="papers")
    if not sampled:
        return []
    has_overviews = bool((await _rows(connection, OVERVIEWS_TABLE_SQL))[0]["present"])
    sql = PAPER_SAMPLES_WITH_OVERVIEWS_SQL if has_overviews else PAPER_SAMPLES_SQL
    rows = {str(row["arxiv_id"]): row for row in await _rows(connection, sql, (sampled,))}
    samples = []
    for arxiv_id in sampled:
        row = rows[arxiv_id]
        findings = row["key_findings"] if isinstance(row["key_findings"], list) else []
        samples.append(
            PaperSample(
                arxiv_id=arxiv_id,
                title=row["title"] or "",
                abstract=row["abstract"] or "",
                key_findings=tuple(str(item) for item in findings if isinstance(item, str) and item.strip()),
            )
        )
    return samples


async def fetch_chunk_samples(
    connection: Connection, *, count: int, seed: int, min_chunk_chars: int, lang_policy: str
) -> list[ChunkSample]:
    ids = [int(row["id"]) for row in await _rows(connection, CHUNK_IDS_SQL, (min_chunk_chars,))]
    sampled = sample_ids(ids, count, seed=seed, stream="chunks")
    if not sampled:
        return []
    rows = {int(row["id"]): row for row in await _rows(connection, CHUNK_SAMPLES_SQL, (sampled,))}
    return [
        ChunkSample(
            chunk_id=chunk_id,
            arxiv_id=str(rows[chunk_id]["arxiv_id"]),
            chunk_index=int(rows[chunk_id]["chunk_index"]),
            section_title=str(rows[chunk_id]["section_title"]),
            chunk_text=rows[chunk_id]["chunk_text"] or "",
            paper_title=rows[chunk_id]["paper_title"] or "",
            lang=chunk_lang_for(index, lang_policy),
        )
        for index, chunk_id in enumerate(sampled)
    ]


def paper_prompt(sample: PaperSample) -> str:
    return build_known_item_prompt(title=sample.title, abstract=sample.abstract, key_findings=sample.key_findings)


def chunk_prompt(sample: ChunkSample) -> str:
    return build_chunk_synth_prompt(
        title=sample.paper_title,
        section_title=sample.section_title,
        chunk_text=sample.chunk_text[:CHUNK_PROMPT_MAX_CHARS],
        lang=sample.lang,
    )


def _basis(paper: PaperSample) -> str:
    return "abstract+key_findings" if paper.key_findings else "abstract"


async def generate_queries(
    papers: Sequence[PaperSample],
    chunks: Sequence[ChunkSample],
    *,
    model: BaseChatModel,
    max_shared_words: int,
    log: LogFn | None = None,
) -> tuple[list[EvalQuery], list[str]]:
    known_item_model = model.with_structured_output(KnownItemOutput)
    chunk_model = model.with_structured_output(ChunkQuestionOutput)
    records: list[EvalQuery] = []
    rejected: list[str] = []

    def too_close(text: str, source: str) -> bool:
        return longest_shared_word_run(text, source) >= max_shared_words

    for index, paper in enumerate(papers, start=1):
        if log is not None:
            log(f"[known_item {index}/{len(papers)}] {paper.arxiv_id}")
        output = KnownItemOutput.model_validate(
            await known_item_model.ainvoke([("system", KNOWN_ITEM_SYSTEM_PROMPT), ("human", paper_prompt(paper))])
        )
        source_text = " ".join([paper.title, paper.abstract, *paper.key_findings])
        accepted: dict[str, str] = {}
        for lang, text in (("ko", output.ko), ("en", output.en)):
            if too_close(text, source_text):
                rejected.append(f"ki-{paper.arxiv_id}-{lang}: 입력과 {max_shared_words}단어 이상 일치 — {text}")
            else:
                accepted[lang] = text
        records.extend(
            known_item_queries(
                arxiv_id=paper.arxiv_id, title=paper.title, queries_by_lang=accepted, basis=_basis(paper)
            )
        )

    for index, chunk in enumerate(chunks, start=1):
        if log is not None:
            log(f"[chunk_synth {index}/{len(chunks)}] chunk {chunk.chunk_id} ({chunk.lang})")
        question_output = ChunkQuestionOutput.model_validate(
            await chunk_model.ainvoke([("system", CHUNK_SYNTH_SYSTEM_PROMPT), ("human", chunk_prompt(chunk))])
        )
        question = question_output.question.strip()
        if not question:
            rejected.append(f"cs-{chunk.chunk_id}: LLM이 답할 수 있는 내용이 없다고 판단")
            continue
        if too_close(question, f"{chunk.paper_title} {chunk.chunk_text}"):
            rejected.append(f"cs-{chunk.chunk_id}: 입력과 {max_shared_words}단어 이상 일치 — {question}")
            continue
        record = chunk_synth_query(
            chunk_id=chunk.chunk_id,
            arxiv_id=chunk.arxiv_id,
            chunk_index=chunk.chunk_index,
            section_title=chunk.section_title,
            question=question,
            lang=chunk.lang,
        )
        if record is not None:
            records.append(record)
    return records, rejected


def dry_run_lines(papers: Sequence[PaperSample], chunks: Sequence[ChunkSample], *, show_prompts: bool) -> list[str]:
    lines = [f"known_item 표본 {len(papers)}편 → 최대 {len(papers) * 2}개 질의 (ko/en)"]
    for index, paper in enumerate(papers):
        lines.append(
            f"  ki-{paper.arxiv_id}-{{ko,en}}  [{_basis(paper)}, abstract {len(paper.abstract)}자]  "
            f"{paper.title[:PREVIEW_TITLE_CHARS]}"
        )
        if show_prompts or index == 0:
            lines.append("    --- prompt ---")
            lines.append("    " + paper_prompt(paper).replace("\n", "\n    "))
    lines.append(f"\nchunk_synth 표본 {len(chunks)}개 → 최대 {len(chunks)}개 질의")
    for index, chunk in enumerate(chunks):
        lines.append(
            f"  cs-{chunk.chunk_id}  [{chunk.lang}] {chunk.arxiv_id} #{chunk.chunk_index} "
            f"'{chunk.section_title[:40]}' ({len(chunk.chunk_text)}자)"
        )
        if show_prompts or index == 0:
            lines.append("    --- prompt ---")
            lines.append("    " + chunk_prompt(chunk).replace("\n", "\n    "))
    lines.append("\ndry-run: LLM을 호출하지 않았고 파일을 쓰지 않았습니다. 생성하려면 --generate.")
    return lines


def merge_with_existing(out_path: Path, records: Sequence[EvalQuery], *, append: bool) -> tuple[list[EvalQuery], int]:
    if not append or not out_path.exists():
        return list(records), 0
    existing = load_queries(out_path)
    existing_ids = {query.id for query in existing}
    fresh = [record for record in records if record.id not in existing_ids]
    return existing + fresh, len(records) - len(fresh)


def combine_query_files(generated: Path, cases: Path, out: Path) -> list[EvalQuery]:
    payload = generated.read_bytes() + cases.read_bytes()
    combined = out.with_name(out.name + ".tmp")
    combined.write_bytes(payload)
    try:
        queries = load_queries(combined)
    except DatasetError:
        combined.unlink()
        raise
    combined.replace(out)
    return queries


def load_placeholder_registry(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    placeholders = payload.get("placeholders") if isinstance(payload, dict) else None
    absent = payload.get("absent", {}) if isinstance(payload, dict) else None
    if not isinstance(placeholders, dict) or not isinstance(absent, dict):
        raise DatasetError(f"{path}: 'placeholders'와 'absent'는 객체여야 합니다.")
    for key, entry in placeholders.items():
        if not PLACEHOLDER_PATTERN.fullmatch(key):
            raise DatasetError(f"{path}: 자리표시자 {key!r}는 0000.0000a 형식이어야 합니다.")
        if not isinstance(entry, dict) or not isinstance(entry.get("title_keyword", ""), str):
            raise DatasetError(f"{path}: {key}의 조건은 title_keyword(문자열)를 가진 객체여야 합니다.")
        unknown = set(entry) - {*REGISTRY_FILTER_KEYS, "need"}
        if unknown:
            raise DatasetError(f"{path}: {key}에 알 수 없는 키 {sorted(unknown)}")
        for list_key in ("title_keywords", "exclude_title_keywords"):
            values = entry.get(list_key, [])
            if not isinstance(values, list) or not all(isinstance(value, str) and value.strip() for value in values):
                raise DatasetError(f"{path}: {key}의 {list_key}는 비어 있지 않은 문자열 리스트여야 합니다.")
        if not isinstance(entry.get("all_matches", False), bool):
            raise DatasetError(f"{path}: {key}의 all_matches는 true/false여야 합니다.")
    if not all(isinstance(value, str) and value.strip() for value in absent.values()):
        raise DatasetError(f"{path}: 'absent' 값은 비어 있지 않은 제목 키워드여야 합니다.")
    return {"placeholders": placeholders, "absent": absent}


def candidate_sql(entry: Mapping[str, Any], limit: int) -> tuple[str, list[Any]]:
    keywords = list(entry.get("title_keywords") or []) or [entry.get("title_keyword") or ""]
    clauses = ["(" + " OR ".join("p.title ILIKE %s" for _ in keywords) + ")"]
    params: list[Any] = [f"%{keyword}%" for keyword in keywords]
    for keyword in entry.get("exclude_title_keywords") or []:
        clauses.append("p.title NOT ILIKE %s")
        params.append(f"%{keyword}%")
    clauses.append("(SELECT COUNT(*) FROM paper_chunks c WHERE c.arxiv_id = p.arxiv_id) >= %s")
    params.append(max(1, int(entry.get("min_chunks") or 1)))
    if entry.get("section_keyword"):
        clauses.append(
            "EXISTS (SELECT 1 FROM paper_chunks c WHERE c.arxiv_id = p.arxiv_id AND c.section_title ILIKE %s)"
        )
        params.append(f"%{entry['section_keyword']}%")
    if entry.get("content_role"):
        clauses.append(
            "EXISTS (SELECT 1 FROM paper_chunks c WHERE c.arxiv_id = p.arxiv_id "
            "AND coalesce(nullif(c.metadata->>'content_role', ''), 'body') = %s)"
        )
        params.append(entry["content_role"])
    if entry.get("chunk_pattern"):
        clauses.append("EXISTS (SELECT 1 FROM paper_chunks c WHERE c.arxiv_id = p.arxiv_id AND c.chunk_text ILIKE %s)")
        params.append(f"%{entry['chunk_pattern']}%")
    sql = (
        "SELECT p.arxiv_id, p.title FROM papers p WHERE "
        + " AND ".join(clauses)
        + " ORDER BY p.published_at DESC NULLS LAST, p.arxiv_id LIMIT %s"
    )
    params.append(limit)
    return sql, params


def choose_distinct(
    candidates: Mapping[str, Sequence[tuple[str, str]]], all_matches: Collection[str] = ()
) -> dict[str, PaperFill]:
    chosen: dict[str, PaperFill] = {}
    used: set[str] = set()
    for placeholder in sorted(candidates):
        if placeholder in all_matches:
            if candidates[placeholder]:
                chosen[placeholder] = list(candidates[placeholder])
            continue
        for arxiv_id, title in candidates[placeholder]:
            if arxiv_id not in used:
                chosen[placeholder] = (arxiv_id, title)
                used.add(arxiv_id)
                break
    return chosen


def attach_catalogue(
    payloads: Sequence[Mapping[str, Any]],
    papers: Mapping[str, PaperFill],
    *,
    absent_present: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[str]]:
    attached: list[dict[str, Any]] = []
    dropped: list[str] = []
    for payload in payloads:
        case_id = str(payload.get("id"))
        if case_id in absent_present:
            dropped.append(f"{case_id}: 코퍼스에 없어야 할 논문이 있습니다 ({absent_present[case_id]})")
            continue
        filled = fill_placeholders(payload, papers)
        remaining = sorted(find_placeholders(filled))
        if remaining:
            dropped.append(f"{case_id}: 채우지 못한 자리표시자 {remaining}")
            continue
        attached.append(filled)
    return attached, dropped


def read_case_payloads(path: str | Path) -> list[dict[str, Any]]:
    load_queries(path)
    payloads: list[dict[str, Any]] = []
    for file_path in dataset_files(path):
        with file_path.open(encoding="utf-8") as handle:
            payloads.extend(json.loads(line) for line in handle if line.strip())
    return payloads


def parse_overrides(values: Sequence[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        placeholder, separator, arxiv_id = value.partition("=")
        if not separator or not PLACEHOLDER_PATTERN.fullmatch(placeholder.strip()) or not arxiv_id.strip():
            raise DatasetError(f"--set 형식은 0000.0000a=ARXIV_ID 입니다: {value!r}")
        overrides[placeholder.strip()] = arxiv_id.strip()
    return overrides


class OverrideError(LookupError):
    pass


@dataclass(frozen=True)
class AttachCandidates:
    candidates: dict[str, list[tuple[str, str]]]
    absent_present: dict[str, str]


async def find_attach_candidates(
    connection: Connection,
    *,
    needed: Sequence[str],
    registry: Mapping[str, Any],
    overrides: Mapping[str, str],
    case_ids: Collection[str],
    limit: int,
) -> AttachCandidates:
    titles: dict[str, str] = {}
    if overrides:
        rows = await _rows(connection, PAPER_TITLES_SQL, (sorted(set(overrides.values())),))
        titles = {str(row["arxiv_id"]): row["title"] or "" for row in rows}
        missing = sorted({value for value in overrides.values() if value not in titles})
        if missing:
            raise OverrideError(f"--set으로 지정한 arxiv_id가 papers에 없습니다: {missing}")
    candidates: dict[str, list[tuple[str, str]]] = {}
    for placeholder in needed:
        if placeholder in overrides:
            candidates[placeholder] = [(overrides[placeholder], titles[overrides[placeholder]])]
            continue
        entry = registry["placeholders"][placeholder]
        sql, params = candidate_sql(entry, ALL_MATCHES_LIMIT if entry.get("all_matches") else limit)
        candidates[placeholder] = [
            (str(row["arxiv_id"]), row["title"] or "") for row in await _rows(connection, sql, params)
        ]
    absent_present: dict[str, str] = {}
    for case_id, keyword in registry["absent"].items():
        if case_id not in case_ids:
            continue
        rows = await _rows(connection, ABSENT_PAPER_SQL, (f"%{keyword}%",))
        if rows:
            absent_present[case_id] = f"{rows[0]['arxiv_id']} {rows[0]['title']}"
    return AttachCandidates(candidates=candidates, absent_present=absent_present)


def all_match_placeholders(
    needed: Sequence[str], registry: Mapping[str, Any], overrides: Mapping[str, str]
) -> set[str]:
    return {
        placeholder
        for placeholder in needed
        if placeholder not in overrides and registry["placeholders"][placeholder].get("all_matches")
    }


def attach_preview_lines(
    needed: Sequence[str],
    registry: Mapping[str, Any],
    candidates: Mapping[str, Sequence[tuple[str, str]]],
    papers: Mapping[str, PaperFill],
) -> list[str]:
    lines = ["자리표시자 → 선택한 논문 (후보는 최신 순, 사람이 확인할 것. 집합 자리표시자는 조건에 맞는 논문 전부)"]
    for placeholder in needed:
        need = registry["placeholders"].get(placeholder, {}).get("need", "--set 지정")
        chosen = papers.get(placeholder)
        if isinstance(chosen, list):
            lines.append(f"  {placeholder}  집합 {len(chosen)}편\n      조건: {need}")
            lines.extend(f"      포함: {arxiv_id}  {title[:PREVIEW_TITLE_CHARS]}" for arxiv_id, title in chosen)
            continue
        label = f"{chosen[0]}  {chosen[1][:90]}" if chosen else "(후보 없음)"
        lines.append(f"  {placeholder}  {label}\n      조건: {need}")
        lines.extend(
            f"      다른 후보: {arxiv_id}  {title[:PREVIEW_TITLE_CHARS]}"
            for arxiv_id, title in candidates.get(placeholder, [])[1:]
        )
    return lines
