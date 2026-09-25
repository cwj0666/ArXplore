"""lexical / vector 검색 SQL에 `EXPLAIN (ANALYZE, BUFFERS)`를 실행해 인덱스 사용 여부와 실행 시간을 출력한다.

실제 DB에 붙어 제품 경로와 같은 SQL(`build_lexical_candidates_query`, `build_vector_search_query`)을 실행한다.
결과를 README에 옮길 때는 출력된 값을 그대로 쓴다.

    python scripts/explain_retrieval.py --query "direct preference optimization"
    python scripts/explain_retrieval.py --query "..." --arxiv-id 2401.00001 --runs 3
    python scripts/explain_retrieval.py --query "..." --embed      # 질의 임베딩을 OpenAI API로 만든다
    python scripts/explain_retrieval.py --query "..." --text       # PostgreSQL 텍스트 플랜도 출력

`--embed`가 없으면 저장된 임베딩 하나를 질의 벡터로 쓴다(API 키 불필요). 플랜과 시간 측정에는 충분하지만
검색 결과 자체는 의미가 없다.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integrations.db import close_all_pools  # noqa: E402
from src.integrations.paper_repository import PaperRepository, build_lexical_candidates_query  # noqa: E402
from src.integrations.vector_repository import (  # noqa: E402
    VectorRepository,
    build_vector_search_query,
    resolve_hnsw_ef_search,
)

LEXICAL_INDEXES = ("idx_paper_chunks_chunk_vector", "idx_papers_title_abstract_vector")
VECTOR_INDEXES = ("paper_embeddings_embedding_hnsw",)


def walk_plan(node: dict[str, Any], depth: int = 0):
    yield depth, node
    for child in node.get("Plans", []) or []:
        yield from walk_plan(child, depth + 1)


def summarize_plan(explain_output: Any) -> dict[str, Any]:
    document = explain_output[0] if isinstance(explain_output, list) else explain_output
    root = document["Plan"]
    indexes: set[str] = set()
    seq_scans: set[str] = set()
    for _, node in walk_plan(root):
        if node.get("Index Name"):
            indexes.add(node["Index Name"])
        if node.get("Node Type") == "Seq Scan" and node.get("Relation Name"):
            seq_scans.add(node["Relation Name"])
    return {
        "planning_ms": document.get("Planning Time"),
        "execution_ms": document.get("Execution Time"),
        "indexes": indexes,
        "seq_scans": seq_scans,
        "rows": root.get("Actual Rows"),
        "shared_hit": root.get("Shared Hit Blocks"),
        "shared_read": root.get("Shared Read Blocks"),
        "root": root,
    }


def render_plan(root: dict[str, Any]) -> str:
    lines = []
    for depth, node in walk_plan(root):
        target = node.get("Index Name") or node.get("Relation Name") or node.get("CTE Name") or ""
        label = f"{node.get('Node Type')}" + (f" [{target}]" if target else "")
        lines.append(
            f"{'  ' * depth}- {label}: rows={node.get('Actual Rows')} loops={node.get('Actual Loops')} "
            f"time={node.get('Actual Total Time')}ms"
        )
    return "\n".join(lines)


def explain(repository: PaperRepository, sql: str, params: dict[str, Any], *, setup: list[tuple[str, Any]], text: bool):
    with repository._connection() as connection, connection.cursor() as cursor:
        for statement, statement_params in setup:
            cursor.execute(statement, statement_params)
        cursor.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, params)
        plan = cursor.fetchone()[0]
        text_plan = None
        if text:
            cursor.execute("EXPLAIN (ANALYZE, BUFFERS) " + sql, params)
            text_plan = "\n".join(row[0] for row in cursor.fetchall())
        connection.rollback()
    if isinstance(plan, str):
        plan = json.loads(plan)
    return summarize_plan(plan), text_plan


def report(title: str, runs: list[dict[str, Any]], expected_indexes: tuple[str, ...], text_plan: str | None) -> None:
    last = runs[-1]
    print(f"\n=== {title} ===")
    print(render_plan(last["root"]))
    if text_plan:
        print("\n" + text_plan)
    print()
    for name in expected_indexes:
        print(f"  {name}: {'USED' if name in last['indexes'] else 'not used'}")
    print(f"  other indexes: {', '.join(sorted(last['indexes'] - set(expected_indexes))) or '-'}")
    print(f"  seq scans: {', '.join(sorted(last['seq_scans'])) or '-'}")
    if any(name not in last["indexes"] for name in expected_indexes):
        print("  note: 행 수가 적은 테이블은 플래너가 인덱스 대신 순차 스캔을 고른다. 행 수와 함께 해석한다")
    times = [run["execution_ms"] for run in runs if run["execution_ms"] is not None]
    print(f"  planning ms (last): {last['planning_ms']}")
    print(f"  execution ms per run: {', '.join(f'{value:.3f}' for value in times)}")
    if len(times) > 1:
        print(f"  execution ms median: {statistics.median(times):.3f}")
    print(f"  result rows: {last['rows']}  shared buffers hit/read: {last['shared_hit']}/{last['shared_read']}")


def database_context(repository: PaperRepository) -> dict[str, Any]:
    with repository._connection() as connection, connection.cursor() as cursor:
        cursor.execute("SHOW server_version")
        server_version = cursor.fetchone()[0]
        cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cursor.fetchone()
        counts = {}
        for table in ("papers", "paper_chunks", "paper_embeddings"):
            cursor.execute(f"SELECT count(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
    return {"server_version": server_version, "pgvector": row[0] if row else None, **counts}


def query_embedding(repository: PaperRepository, query: str, *, embed: bool) -> tuple[list[float] | None, str]:
    if embed:
        from src.integrations.embedding_client import EmbeddingClient

        client = EmbeddingClient()
        return client.embed_texts([query])[0], f"OpenAI {client.model_name}"
    with repository._connection() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT chunk_id, embedding::text FROM paper_embeddings ORDER BY chunk_id LIMIT 1")
        row = cursor.fetchone()
    if row is None:
        return None, "no stored embeddings"
    return json.loads(row[1]), f"stored embedding of chunk {row[0]}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query", default="direct preference optimization")
    parser.add_argument("--limit", type=int, default=15, help="SQL LIMIT (retriever는 limit 5일 때 15개를 가져온다)")
    parser.add_argument("--arxiv-id", default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--skip-vector", action="store_true")
    parser.add_argument("--text", action="store_true")
    args = parser.parse_args(argv)
    runs = max(1, args.runs)

    try:
        repository = PaperRepository()
        context = database_context(repository)
        print("database: " + ", ".join(f"{key}={value}" for key, value in context.items()))
        print(f"query: {args.query!r} limit={args.limit} arxiv_id={args.arxiv_id}")

        built = build_lexical_candidates_query(args.query, limit=args.limit, arxiv_id=args.arxiv_id)
        if built is None:
            print("empty query")
            return 1
        sql, params = built
        results = [explain(repository, sql, params, setup=[], text=False)[0] for _ in range(runs - 1)]
        summary, text_plan = explain(repository, sql, params, setup=[], text=args.text)
        report("lexical", [*results, summary], LEXICAL_INDEXES, text_plan)

        if args.skip_vector:
            return 0
        embedding, source = query_embedding(repository, args.query, embed=args.embed)
        print(f"\nvector query embedding: {source}")
        if embedding is None:
            return 0
        vector_repository = VectorRepository()
        sql, params = build_vector_search_query(
            embedding,
            limit=args.limit,
            arxiv_id=args.arxiv_id,
            model_name=getattr(vector_repository.settings, "openai_embedding_model", None),
            min_similarity=vector_repository._min_similarity(),
        )
        setup: list[tuple[str, Any]] = []
        ef_search = None if args.arxiv_id else resolve_hnsw_ef_search(params["candidate_limit"])
        if ef_search is not None:
            setup.append(("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),)))
        results = [explain(repository, sql, params, setup=setup, text=False)[0] for _ in range(runs - 1)]
        summary, text_plan = explain(repository, sql, params, setup=setup, text=args.text)
        expected = () if args.arxiv_id else VECTOR_INDEXES
        report(
            "vector" + (" (paper-scoped, exact scan by design)" if args.arxiv_id else ""),
            [*results, summary],
            expected,
            text_plan,
        )
        return 0
    finally:
        close_all_pools()


if __name__ == "__main__":
    raise SystemExit(main())
