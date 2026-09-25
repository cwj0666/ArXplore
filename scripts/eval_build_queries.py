"""검색 평가용 질의셋(eval/queries.jsonl)을 DB 표본과 LLM으로 만든다. 기본은 dry-run.

    python scripts/eval_build_queries.py                                   # 표본·프롬프트만 출력(LLM 미호출)
    python scripts/eval_build_queries.py --generate --papers 20 --chunks 10
    python scripts/eval_build_queries.py --generate --mode chunk_synth --chunks 10 --append

known_item: 청크가 있는 논문을 표본 추출해 초록(+paper_ai_overviews.key_findings)을 바꿔 말한 질의를
            논문당 ko/en 1개씩 만든다. 정답은 해당 arxiv_id.
chunk_synth: 본문(content_role=body) 청크를 표본 추출해 그 청크로만 답할 수 있는 질문을 만든다.
             정답은 arxiv_id + chunk_id. chunk_id는 재청킹·백필 시 바뀌므로 백필 후에 만든다.

표본은 --seed로 결정적이다(같은 DB 상태 + 같은 seed → 같은 표본). LLM 출력은 결정적이지 않다.
입력과 5단어 이상 연속으로 겹치는 질의(--max-shared-words)는 버리고 개수를 보고한다.
PostgreSQL이 필요하고, --generate에는 OPENAI_API_KEY가 필요하다.
"""

from __future__ import annotations

import argparse
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
    DatasetError,
    EvalQuery,
    build_chunk_synth_prompt,
    build_known_item_prompt,
    chunk_synth_query,
    dump_queries,
    known_item_queries,
    load_queries,
    longest_shared_word_run,
)

EXIT_PRECONDITION = 2
CHUNK_PROMPT_MAX_CHARS = 3000

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
    parser.set_defaults(generate=False)
    return parser.parse_args(argv)


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
