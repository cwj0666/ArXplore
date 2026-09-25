"""검색 평가용 질의셋(eval/queries.jsonl)을 DB 표본과 LLM으로 만든다. 기본은 dry-run.

    python scripts/eval_build_queries.py                                   # 표본·프롬프트만 출력(LLM 미호출)
    python scripts/eval_build_queries.py --generate --papers 20 --chunks 10
    python scripts/eval_build_queries.py --generate --mode chunk_synth --chunks 10 --append
    python scripts/eval_build_queries.py --attach-ids                       # 케이스 카탈로그 자리표시자 후보만 출력
    python scripts/eval_build_queries.py --attach-ids --write               # eval/queries.cases.jsonl로 기록

known_item: 청크가 있는 논문을 표본 추출해 초록(+paper_ai_overviews.key_findings)을 바꿔 말한 질의를
            논문당 ko/en 1개씩 만든다. 정답은 해당 arxiv_id.
chunk_synth: 본문(content_role=body) 청크를 표본 추출해 그 청크로만 답할 수 있는 질문을 만든다.
             정답은 arxiv_id + chunk_id. chunk_id는 재청킹·백필 시 바뀌므로 백필 후에 만든다.

표본은 --seed로 결정적이다(같은 DB 상태 + 같은 seed → 같은 표본). LLM 출력은 결정적이지 않다.
입력과 5단어 이상 연속으로 겹치는 질의(--max-shared-words)는 버리고 개수를 보고한다.
PostgreSQL이 필요하고, --generate에는 OPENAI_API_KEY가 필요하다.

attach-ids: eval/cases/*.jsonl의 자리표시자 arxiv_id(0000.0000a 형식)를 eval/cases/placeholders.json의 조건
(제목 키워드, 최소 청크 수, 섹션·content_role·본문 패턴)에 맞는 DB 논문으로 채운다. 자리표시자마다 서로 다른 논문을
고르고(최신 순), 제목 템플릿 {{title:ID}}·{{title_head:ID:N}}도 실제 제목으로 바꾼다. --set 0000.0000a=2405.01234로
직접 지정할 수 있다. 채우지 못한 자리표시자가 남은 케이스와 코퍼스에 있으면 안 되는 논문(absent)이 실제로 있는
케이스는 빼고 보고한다. 원본 카탈로그는 고치지 않는다. 기본은 미리보기이고 --write일 때만 --attach-out에 쓴다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.dataset import (  # noqa: E402
    CHUNK_SYNTH_SYSTEM_PROMPT,
    KNOWN_ITEM_SYSTEM_PROMPT,
    PLACEHOLDER_PATTERN,
    DatasetError,
    EvalQuery,
    build_chunk_synth_prompt,
    build_known_item_prompt,
    chunk_synth_query,
    dataset_files,
    dump_queries,
    fill_placeholders,
    find_placeholders,
    known_item_queries,
    load_queries,
    longest_shared_word_run,
)

EXIT_PRECONDITION = 2
CHUNK_PROMPT_MAX_CHARS = 3000
DEFAULT_CASES_DIR = REPO_ROOT / "eval" / "cases"
DEFAULT_ATTACH_OUT = REPO_ROOT / "eval" / "queries.cases.jsonl"
PLACEHOLDER_REGISTRY_NAME = "placeholders.json"
REGISTRY_FILTER_KEYS = ("title_keyword", "min_chunks", "section_keyword", "content_role", "chunk_pattern")

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("known_item", "chunk_synth", "both"), default="both")
    parser.add_argument("--papers", type=int, default=20, help="known_item 논문 수(논문당 ko/en 2개)")
    parser.add_argument("--chunks", type=int, default=10, help="chunk_synth 청크 수")
    parser.add_argument("--chunk-lang", choices=("alternate", "ko", "en"), default="alternate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-abstract-chars", type=int, default=300)
    parser.add_argument("--min-chunk-chars", type=int, default=600)
    parser.add_argument("--max-shared-words", type=int, default=5, help="입력과 이 단어 수 이상 연속 일치하면 버림")
    parser.add_argument("--out", default=str(REPO_ROOT / "eval" / "queries.jsonl"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="generate", action="store_false", help="기본값. LLM을 호출하지 않는다")
    mode.add_argument("--generate", dest="generate", action="store_true", help="LLM으로 질의를 만들어 --out에 쓴다")
    write_mode = parser.add_mutually_exclusive_group()
    write_mode.add_argument("--overwrite", action="store_true")
    write_mode.add_argument("--append", action="store_true", help="기존 파일에 새 id만 추가")
    parser.add_argument("--show-prompts", action="store_true", help="dry-run에서 모든 프롬프트를 출력")
    attach = parser.add_argument_group("attach-ids (케이스 카탈로그 자리표시자 채우기)")
    attach.add_argument("--attach-ids", action="store_true", help="카탈로그 자리표시자를 DB 논문 id로 채운다")
    attach.add_argument("--cases", default=str(DEFAULT_CASES_DIR), help="카탈로그 JSONL 파일 또는 디렉터리")
    attach.add_argument(
        "--placeholders", default=None, help=f"자리표시자 조건 파일(기본: <cases>/{PLACEHOLDER_REGISTRY_NAME})"
    )
    attach.add_argument("--attach-out", default=str(DEFAULT_ATTACH_OUT))
    attach.add_argument("--set", action="append", default=[], metavar="PLACEHOLDER=ARXIV_ID", help="직접 지정")
    attach.add_argument("--candidates", type=int, default=5, help="자리표시자마다 보여 줄 후보 수")
    attach.add_argument("--write", action="store_true", help="--attach-ids 결과를 --attach-out에 쓴다")
    parser.set_defaults(generate=False)
    args = parser.parse_args(argv)
    if args.write and not args.attach_ids:
        parser.error("--write는 --attach-ids와 함께만 씁니다")
    if args.candidates < 1:
        parser.error("--candidates must be >= 1")
    return args


def sample_ids(ids: list[Any], count: int, *, seed: int, stream: str) -> list[Any]:
    """정렬된 id 목록에서 결정적으로 표본을 뽑는다. stream별로 난수열을 분리해 서로 영향을 주지 않는다."""
    rng = random.Random(f"{seed}:{stream}")
    return rng.sample(list(ids), min(max(0, count), len(ids)))


def chunk_lang_for(index: int, policy: str) -> str:
    if policy in ("ko", "en"):
        return policy
    return "en" if index % 2 == 0 else "ko"


def _connect():
    import psycopg2

    from src.shared import build_postgres_connection_params, get_settings

    return psycopg2.connect(connect_timeout=5, **build_postgres_connection_params(get_settings()))


def fetch_paper_samples(cursor, *, count: int, seed: int, min_abstract_chars: int) -> list[PaperSample]:
    cursor.execute(PAPER_IDS_SQL, (min_abstract_chars,))
    sampled = sample_ids([row[0] for row in cursor.fetchall()], count, seed=seed, stream="papers")
    if not sampled:
        return []
    cursor.execute("SELECT to_regclass('public.paper_ai_overviews') IS NOT NULL")
    has_overviews = bool(cursor.fetchone()[0])
    if has_overviews:
        cursor.execute(
            """
            SELECT p.arxiv_id, p.title, p.abstract, coalesce(o.key_findings, '[]'::jsonb)
            FROM papers p
            LEFT JOIN paper_ai_overviews o ON o.arxiv_id = p.arxiv_id
            WHERE p.arxiv_id = ANY(%s)
            """,
            (sampled,),
        )
    else:
        cursor.execute(
            "SELECT arxiv_id, title, abstract, '[]'::jsonb FROM papers WHERE arxiv_id = ANY(%s)",
            (sampled,),
        )
    rows = {row[0]: row for row in cursor.fetchall()}
    samples = []
    for arxiv_id in sampled:
        row = rows[arxiv_id]
        findings = row[3] if isinstance(row[3], list) else []
        samples.append(
            PaperSample(
                arxiv_id=arxiv_id,
                title=row[1] or "",
                abstract=row[2] or "",
                key_findings=tuple(str(item) for item in findings if isinstance(item, str) and item.strip()),
            )
        )
    return samples


def fetch_chunk_samples(cursor, *, count: int, seed: int, min_chunk_chars: int, lang_policy: str) -> list[ChunkSample]:
    cursor.execute(CHUNK_IDS_SQL, (min_chunk_chars,))
    sampled = sample_ids([int(row[0]) for row in cursor.fetchall()], count, seed=seed, stream="chunks")
    if not sampled:
        return []
    cursor.execute(
        """
        SELECT c.id, c.arxiv_id, c.chunk_index, coalesce(c.section_title, ''), c.chunk_text, p.title
        FROM paper_chunks c
        JOIN papers p ON p.arxiv_id = c.arxiv_id
        WHERE c.id = ANY(%s)
        """,
        (sampled,),
    )
    rows = {int(row[0]): row for row in cursor.fetchall()}
    return [
        ChunkSample(
            chunk_id=chunk_id,
            arxiv_id=rows[chunk_id][1],
            chunk_index=int(rows[chunk_id][2]),
            section_title=rows[chunk_id][3],
            chunk_text=rows[chunk_id][4] or "",
            paper_title=rows[chunk_id][5] or "",
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


def build_llm():
    from langchain_openai import ChatOpenAI

    from src.shared import get_settings

    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
    kwargs: dict[str, Any] = {"model": settings.openai_model, "api_key": settings.openai_api_key}
    if not settings.openai_model.startswith("gpt-5"):
        kwargs["temperature"] = 0.2
    return ChatOpenAI(**kwargs)


def generate_queries(
    papers: list[PaperSample], chunks: list[ChunkSample], *, max_shared_words: int
) -> tuple[list[EvalQuery], list[str]]:
    from pydantic import BaseModel, Field

    class KnownItemOutput(BaseModel):
        ko: str = Field(description="Korean query")
        en: str = Field(description="English query")

    class ChunkQuestionOutput(BaseModel):
        question: str = Field(description="Question in the requested language, or empty string")

    llm = build_llm()
    known_item_llm = llm.with_structured_output(KnownItemOutput)
    chunk_llm = llm.with_structured_output(ChunkQuestionOutput)
    records: list[EvalQuery] = []
    rejected: list[str] = []

    def too_close(text: str, source: str) -> bool:
        return longest_shared_word_run(text, source) >= max_shared_words

    for index, paper in enumerate(papers, start=1):
        print(f"[known_item {index}/{len(papers)}] {paper.arxiv_id}", file=sys.stderr)
        output = known_item_llm.invoke([("system", KNOWN_ITEM_SYSTEM_PROMPT), ("human", paper_prompt(paper))])
        source_text = " ".join([paper.title, paper.abstract, *paper.key_findings])
        accepted: dict[str, str] = {}
        for lang, text in (("ko", output.ko), ("en", output.en)):
            if too_close(text, source_text):
                rejected.append(f"ki-{paper.arxiv_id}-{lang}: 입력과 {max_shared_words}단어 이상 일치 — {text}")
            else:
                accepted[lang] = text
        basis = "abstract+key_findings" if paper.key_findings else "abstract"
        records.extend(
            known_item_queries(arxiv_id=paper.arxiv_id, title=paper.title, queries_by_lang=accepted, basis=basis)
        )

    for index, chunk in enumerate(chunks, start=1):
        print(f"[chunk_synth {index}/{len(chunks)}] chunk {chunk.chunk_id} ({chunk.lang})", file=sys.stderr)
        output = chunk_llm.invoke([("system", CHUNK_SYNTH_SYSTEM_PROMPT), ("human", chunk_prompt(chunk))])
        question = output.question.strip()
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


def print_dry_run(papers: list[PaperSample], chunks: list[ChunkSample], *, show_prompts: bool) -> None:
    print(f"known_item 표본 {len(papers)}편 → 최대 {len(papers) * 2}개 질의 (ko/en)")
    for index, paper in enumerate(papers):
        basis = "abstract+key_findings" if paper.key_findings else "abstract"
        print(f"  ki-{paper.arxiv_id}-{{ko,en}}  [{basis}, abstract {len(paper.abstract)}자]  {paper.title[:80]}")
        if show_prompts or index == 0:
            print("    --- prompt ---")
            print("    " + paper_prompt(paper).replace("\n", "\n    "))
    print(f"\nchunk_synth 표본 {len(chunks)}개 → 최대 {len(chunks)}개 질의")
    for index, chunk in enumerate(chunks):
        print(
            f"  cs-{chunk.chunk_id}  [{chunk.lang}] {chunk.arxiv_id} #{chunk.chunk_index} "
            f"'{chunk.section_title[:40]}' ({len(chunk.chunk_text)}자)"
        )
        if show_prompts or index == 0:
            print("    --- prompt ---")
            print("    " + chunk_prompt(chunk).replace("\n", "\n    "))
    print("\ndry-run: LLM을 호출하지 않았고 파일을 쓰지 않았습니다. 생성하려면 --generate.")


def merge_with_existing(out_path: Path, records: list[EvalQuery], *, append: bool) -> tuple[list[EvalQuery], int]:
    if not append or not out_path.exists():
        return records, 0
    existing = load_queries(out_path)
    existing_ids = {query.id for query in existing}
    fresh = [record for record in records if record.id not in existing_ids]
    return existing + fresh, len(records) - len(fresh)


def load_placeholder_registry(path: str | Path) -> dict[str, Any]:
    """자리표시자 조건 파일을 읽고 검증한다.

    형식: `{"placeholders": {"0000.0000a": {"title_keyword": "...", "need": "...", ...}}, "absent": {케이스 id: 제목 키워드}}`.
    """
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
    if not all(isinstance(value, str) and value.strip() for value in absent.values()):
        raise DatasetError(f"{path}: 'absent' 값은 비어 있지 않은 제목 키워드여야 합니다.")
    return {"placeholders": placeholders, "absent": absent}


def candidate_sql(entry: dict[str, Any], limit: int) -> tuple[str, list[Any]]:
    """자리표시자 조건에 맞는 논문(arxiv_id, title)을 최신 순으로 고르는 SQL과 인자.

    title_keyword는 제목 ILIKE `%키워드%`(키워드 안의 `%`는 와일드카드), min_chunks는 청크 수 하한(기본 1),
    section_keyword·content_role·chunk_pattern은 조건을 만족하는 청크가 하나 이상 있어야 한다는 뜻이다.
    """
    clauses = ["p.title ILIKE %s", "(SELECT COUNT(*) FROM paper_chunks c WHERE c.arxiv_id = p.arxiv_id) >= %s"]
    params: list[Any] = [f"%{entry.get('title_keyword') or ''}%", max(1, int(entry.get("min_chunks") or 1))]
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


def choose_distinct(candidates: dict[str, list[tuple[str, str]]]) -> dict[str, tuple[str, str]]:
    """자리표시자 이름순으로 후보 목록의 앞에서부터, 앞선 자리표시자가 이미 고른 논문은 피해 하나씩 고른다."""
    chosen: dict[str, tuple[str, str]] = {}
    used: set[str] = set()
    for placeholder in sorted(candidates):
        for arxiv_id, title in candidates[placeholder]:
            if arxiv_id not in used:
                chosen[placeholder] = (arxiv_id, title)
                used.add(arxiv_id)
                break
    return chosen


def attach_catalogue(
    payloads: list[dict[str, Any]], papers: dict[str, tuple[str, str]], *, absent_present: dict[str, str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """카탈로그 케이스의 자리표시자를 채운 사본과 뺀 케이스 설명을 돌려준다.

    `absent_present`(케이스 id → 코퍼스에서 발견된 논문 설명)에 있는 케이스와, 채운 뒤에도 자리표시자가 남은 케이스는 뺀다.
    """
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
    """카탈로그를 검증(`load_queries`)한 뒤 원본 JSON 객체를 파일·줄 순서대로 읽는다."""
    load_queries(path)
    payloads: list[dict[str, Any]] = []
    for file_path in dataset_files(path):
        with file_path.open(encoding="utf-8") as handle:
            payloads.extend(json.loads(line) for line in handle if line.strip())
    return payloads


def parse_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        placeholder, sep, arxiv_id = value.partition("=")
        if not sep or not PLACEHOLDER_PATTERN.fullmatch(placeholder.strip()) or not arxiv_id.strip():
            raise DatasetError(f"--set 형식은 0000.0000a=ARXIV_ID 입니다: {value!r}")
        overrides[placeholder.strip()] = arxiv_id.strip()
    return overrides


def run_attach(args: argparse.Namespace) -> int:
    cases_path = Path(args.cases)
    registry_path = Path(args.placeholders) if args.placeholders else cases_path / PLACEHOLDER_REGISTRY_NAME
    out_path = Path(args.attach_out)
    try:
        payloads = read_case_payloads(cases_path)
        registry = load_placeholder_registry(registry_path)
        overrides = parse_overrides(args.set)
    except (FileNotFoundError, DatasetError, json.JSONDecodeError) as exc:
        print(f"카탈로그를 읽을 수 없습니다: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION
    if args.write and out_path.exists() and not args.overwrite:
        print(f"{out_path}가 이미 있습니다. --overwrite를 지정하세요.", file=sys.stderr)
        return EXIT_PRECONDITION

    needed = sorted(find_placeholders(payloads))
    unknown = [value for value in needed if value not in registry["placeholders"] and value not in overrides]
    if unknown:
        print(f"조건 파일에 없는 자리표시자: {unknown}", file=sys.stderr)
        return EXIT_PRECONDITION
    case_ids = {str(payload.get("id")) for payload in payloads}

    try:
        connection = _connect()
    except Exception as exc:
        print(f"PostgreSQL에 연결할 수 없습니다: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION
    candidates: dict[str, list[tuple[str, str]]] = {}
    absent_present: dict[str, str] = {}
    try:
        with connection, connection.cursor() as cursor:
            if overrides:
                cursor.execute(
                    "SELECT arxiv_id, title FROM papers WHERE arxiv_id = ANY(%s)", (sorted(set(overrides.values())),)
                )
                titles = {row[0]: row[1] or "" for row in cursor.fetchall()}
                missing = sorted({value for value in overrides.values() if value not in titles})
                if missing:
                    print(f"--set으로 지정한 arxiv_id가 papers에 없습니다: {missing}", file=sys.stderr)
                    return EXIT_PRECONDITION
            for placeholder in needed:
                if placeholder in overrides:
                    candidates[placeholder] = [(overrides[placeholder], titles[overrides[placeholder]])]
                    continue
                sql, params = candidate_sql(registry["placeholders"][placeholder], args.candidates)
                cursor.execute(sql, params)
                candidates[placeholder] = [(row[0], row[1] or "") for row in cursor.fetchall()]
            for case_id, keyword in registry["absent"].items():
                if case_id not in case_ids:
                    continue
                cursor.execute(
                    "SELECT arxiv_id, title FROM papers WHERE title ILIKE %s ORDER BY arxiv_id LIMIT 1",
                    (f"%{keyword}%",),
                )
                row = cursor.fetchone()
                if row:
                    absent_present[case_id] = f"{row[0]} {row[1]}"
    except Exception as exc:
        print(f"후보 조회가 실패했습니다: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION
    finally:
        connection.close()

    papers = choose_distinct(candidates)
    print("자리표시자 → 선택한 논문 (후보는 최신 순, 사람이 확인할 것)")
    for placeholder in needed:
        need = registry["placeholders"].get(placeholder, {}).get("need", "--set 지정")
        chosen = papers.get(placeholder)
        label = f"{chosen[0]}  {chosen[1][:90]}" if chosen else "(후보 없음)"
        print(f"  {placeholder}  {label}\n      조건: {need}")
        for arxiv_id, title in candidates.get(placeholder, [])[1:]:
            print(f"      다른 후보: {arxiv_id}  {title[:80]}")

    attached, dropped = attach_catalogue(payloads, papers, absent_present=absent_present)
    print(f"\n케이스 {len(payloads)}개 중 {len(attached)}개를 채웠고 {len(dropped)}개를 뺐습니다.")
    for line in dropped:
        print(f"  뺌: {line}")
    if not args.write:
        print("\n미리보기: 파일을 쓰지 않았습니다. 쓰려면 --write.")
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in attached), encoding="utf-8"
    )
    load_queries(out_path)
    print(f"{out_path}에 {len(attached)}개 케이스를 썼습니다. 답이 실제 논문과 맞는지 훑어보고 쓰세요.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.attach_ids:
        return run_attach(args)
    out_path = Path(args.out)
    if args.generate and out_path.exists() and not (args.overwrite or args.append):
        print(f"{out_path}가 이미 있습니다. --overwrite 또는 --append를 지정하세요.", file=sys.stderr)
        return EXIT_PRECONDITION

    paper_count = args.papers if args.mode in ("known_item", "both") else 0
    chunk_count = args.chunks if args.mode in ("chunk_synth", "both") else 0
    try:
        connection = _connect()
    except Exception as exc:
        print(f"PostgreSQL에 연결할 수 없습니다: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION
    try:
        with connection, connection.cursor() as cursor:
            papers = fetch_paper_samples(
                cursor, count=paper_count, seed=args.seed, min_abstract_chars=args.min_abstract_chars
            )
            chunks = fetch_chunk_samples(
                cursor,
                count=chunk_count,
                seed=args.seed,
                min_chunk_chars=args.min_chunk_chars,
                lang_policy=args.chunk_lang,
            )
    except Exception as exc:
        print(f"표본 조회가 실패했습니다: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION
    finally:
        connection.close()

    if paper_count and len(papers) < paper_count:
        print(f"경고: 조건을 만족하는 논문이 {len(papers)}편뿐입니다(요청 {paper_count}).", file=sys.stderr)
    if chunk_count and len(chunks) < chunk_count:
        print(f"경고: 조건을 만족하는 본문 청크가 {len(chunks)}개뿐입니다(요청 {chunk_count}).", file=sys.stderr)
    if not papers and not chunks:
        print("표본이 비어 있습니다. 수집·prepare 상태를 확인하세요.", file=sys.stderr)
        return EXIT_PRECONDITION

    if not args.generate:
        print_dry_run(papers, chunks, show_prompts=args.show_prompts)
        return 0

    try:
        records, rejected = generate_queries(papers, chunks, max_shared_words=args.max_shared_words)
        merged, skipped = merge_with_existing(out_path, records, append=args.append)
    except (RuntimeError, DatasetError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_PRECONDITION

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dump_queries(merged), encoding="utf-8")
    load_queries(out_path)
    print(f"{len(records) - skipped}개 질의를 {out_path}에 썼습니다(전체 {len(merged)}개, 중복 id {skipped}개 건너뜀).")
    if rejected:
        print(f"버린 질의 {len(rejected)}개:")
        for line in rejected:
            print(f"  {line}")
    print("측정 전에 파일을 직접 훑어 정답이 모호한 질의를 지우고, 손으로 고친 항목은 source를 manual로 바꾸세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
