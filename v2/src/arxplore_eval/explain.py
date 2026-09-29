import json
import statistics
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row, tuple_row

from arxplore.db.migrate import SEARCH_INDEXES
from arxplore.retrieval.lexical import build_lexical_candidates_query
from arxplore.retrieval.query import candidate_fetch_limit, hybrid_branch_limit
from arxplore.retrieval.service import RetrievalConfig
from arxplore.retrieval.vector import build_vector_search_query, resolve_hnsw_ef_search

LEXICAL = "lexical"
VECTOR = "vector"
CHANNELS = (LEXICAL, VECTOR)
CHANNEL_TABLES = {LEXICAL: frozenset({"papers", "paper_chunks"}), VECTOR: frozenset({"paper_embeddings"})}
EXPLAIN_JSON = "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) "
EXPLAIN_TEXT = "EXPLAIN (ANALYZE, BUFFERS) "
READ_ONLY_SQL = "SET TRANSACTION READ ONLY"
HNSW_EF_SEARCH_SQL = "SELECT set_config('hnsw.ef_search', %s, true)"
SEARCH_INDEX_TABLES_SQL = """
    SELECT indexname, tablename
    FROM pg_indexes
    WHERE schemaname = current_schema() AND indexname = ANY(%s)
"""
STORED_EMBEDDING_SQL = "SELECT chunk_id, embedding::text AS embedding FROM paper_embeddings ORDER BY chunk_id LIMIT 1"
SEQ_SCAN = "Seq Scan"
NO_ITEMS = "-"
SEQ_SCAN_NOTE = "  note: 플래너가 인덱스 대신 순차 스캔을 골랐다. 테이블 행 수와 후보 수(LIMIT)를 함께 보고 해석한다"

type Connection = AsyncConnection[Any]
type Statement = tuple[str, tuple[Any, ...]]


def channel_limit(limit: int, *, hybrid: bool) -> int:
    normalized = max(1, limit)
    return hybrid_branch_limit(normalized) if hybrid else normalized


def sql_fetch_limit(limit: int, *, arxiv_id: str | None) -> int:
    normalized = max(1, limit)
    return normalized if arxiv_id else candidate_fetch_limit(normalized)


@dataclass(frozen=True)
class ExplainTarget:
    channel: str
    sql: str
    params: dict[str, Any]
    setup: tuple[Statement, ...] = ()

    def describe(self) -> str:
        if self.channel == LEXICAL:
            return f"SQL LIMIT {self.params['limit']}, 부분 일치 후보 상한 {self.params['partial_pool']}"
        ef_search = self.setup[0][1][0] if self.setup else "기본값(논문 범위는 정확 검색)"
        return (
            f"SQL LIMIT {self.params['limit']}, 1단계 후보 {self.params['candidate_limit']}, hnsw.ef_search {ef_search}"
        )


def lexical_target(query: str, *, sql_limit: int, arxiv_id: str | None = None) -> ExplainTarget | None:
    built = build_lexical_candidates_query(query, limit=sql_limit, arxiv_id=arxiv_id)
    if built is None:
        return None
    sql, params = built
    return ExplainTarget(channel=LEXICAL, sql=sql, params=params)


def vector_target(
    embedding: Sequence[float], *, sql_limit: int, config: RetrievalConfig, arxiv_id: str | None = None
) -> ExplainTarget:
    sql, params = build_vector_search_query(
        embedding,
        limit=sql_limit,
        arxiv_id=arxiv_id,
        model_name=config.embedding_model,
        min_similarity=max(0.0, config.vector_min_similarity),
    )
    ef_search = None if arxiv_id else resolve_hnsw_ef_search(params["candidate_limit"])
    setup: tuple[Statement, ...] = ((HNSW_EF_SEARCH_SQL, (str(ef_search),)),) if ef_search is not None else ()
    return ExplainTarget(channel=VECTOR, sql=sql, params=params, setup=setup)


def walk_plan(node: Mapping[str, Any], depth: int = 0) -> Iterator[tuple[int, Mapping[str, Any]]]:
    yield depth, node
    for child in node.get("Plans") or []:
        yield from walk_plan(child, depth + 1)


@dataclass(frozen=True)
class PlanSummary:
    planning_ms: float | None
    execution_ms: float | None
    indexes: frozenset[str]
    seq_scans: frozenset[str]
    rows: Any
    shared_hit: Any
    shared_read: Any
    root: Mapping[str, Any]


def summarize_plan(explain_output: Any) -> PlanSummary:
    document = explain_output[0] if isinstance(explain_output, list) else explain_output
    root = document["Plan"]
    indexes: set[str] = set()
    seq_scans: set[str] = set()
    for _, node in walk_plan(root):
        if node.get("Index Name"):
            indexes.add(str(node["Index Name"]))
        if node.get("Node Type") == SEQ_SCAN and node.get("Relation Name"):
            seq_scans.add(str(node["Relation Name"]))
    return PlanSummary(
        planning_ms=document.get("Planning Time"),
        execution_ms=document.get("Execution Time"),
        indexes=frozenset(indexes),
        seq_scans=frozenset(seq_scans),
        rows=root.get("Actual Rows"),
        shared_hit=root.get("Shared Hit Blocks"),
        shared_read=root.get("Shared Read Blocks"),
        root=root,
    )


def render_plan(root: Mapping[str, Any]) -> str:
    lines = []
    for depth, node in walk_plan(root):
        target = node.get("Index Name") or node.get("Relation Name") or node.get("CTE Name") or ""
        label = f"{node.get('Node Type')}" + (f" [{target}]" if target else "")
        lines.append(
            f"{'  ' * depth}- {label}: rows={node.get('Actual Rows')} loops={node.get('Actual Loops')} "
            f"time={node.get('Actual Total Time')}ms"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class ExplainRun:
    summary: PlanSummary
    text_plan: str | None


async def explain_target(connection: Connection, target: ExplainTarget, *, text: bool = False) -> ExplainRun:
    async with connection.transaction(force_rollback=True), connection.cursor(row_factory=tuple_row) as cursor:
        await cursor.execute(READ_ONLY_SQL)
        for statement, params in target.setup:
            await cursor.execute(statement, params, prepare=False)
        await cursor.execute(EXPLAIN_JSON + target.sql, target.params, prepare=False)
        row = await cursor.fetchone()
        text_plan = None
        if text:
            await cursor.execute(EXPLAIN_TEXT + target.sql, target.params, prepare=False)
            text_plan = "\n".join(str(line[0]) for line in await cursor.fetchall())
    if row is None:
        raise RuntimeError("EXPLAIN이 계획을 돌려주지 않았습니다.")
    plan = json.loads(row[0]) if isinstance(row[0], str) else row[0]
    return ExplainRun(summary=summarize_plan(plan), text_plan=text_plan)


async def channel_indexes(connection: Connection) -> dict[str, tuple[str, ...]]:
    async with connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(SEARCH_INDEX_TABLES_SQL, (list(SEARCH_INDEXES),))
        tables = {str(row["indexname"]): str(row["tablename"]) for row in await cursor.fetchall()}
    return {
        channel: tuple(name for name in SEARCH_INDEXES if tables.get(name) in CHANNEL_TABLES[channel])
        for channel in CHANNELS
    }


async def stored_query_embedding(connection: Connection) -> tuple[list[float], str] | None:
    async with connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(STORED_EMBEDDING_SQL)
        row = await cursor.fetchone()
    if row is None:
        return None
    return [float(value) for value in json.loads(row["embedding"])], f"stored embedding of chunk {row['chunk_id']}"


def _joined(values: frozenset[str] | set[str]) -> str:
    return ", ".join(sorted(values)) or NO_ITEMS


def report_lines(
    title: str, runs: Sequence[PlanSummary], expected: Sequence[str], text_plan: str | None = None
) -> list[str]:
    last = runs[-1]
    lines = ["", f"=== {title} ===", render_plan(last.root)]
    if text_plan:
        lines += ["", text_plan]
    lines.append("")
    lines.extend(f"  {name}: {'USED' if name in last.indexes else 'not used'}" for name in expected)
    lines.append(f"  other indexes: {_joined(set(last.indexes) - set(expected))}")
    lines.append(f"  seq scans: {_joined(last.seq_scans)}")
    if any(name not in last.indexes for name in expected):
        lines.append(SEQ_SCAN_NOTE)
    times = [run.execution_ms for run in runs if run.execution_ms is not None]
    lines.append(f"  planning ms (last): {last.planning_ms}")
    lines.append(f"  execution ms per run: {', '.join(f'{value:.3f}' for value in times)}")
    if len(times) > 1:
        lines.append(f"  execution ms median: {statistics.median(times):.3f}")
    lines.append(f"  result rows: {last.rows}  shared buffers hit/read: {last.shared_hit}/{last.shared_read}")
    return lines
