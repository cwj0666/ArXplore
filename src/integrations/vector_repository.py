from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from psycopg2.extras import execute_values

from src.integrations.db import get_connection
from src.shared import AppSettings, build_postgres_connection_params, get_settings

MIN_VECTOR_CANDIDATES = 40
MAX_HNSW_EF_SEARCH = 1000
DEFAULT_HNSW_EF_SEARCH = 40

_CONTENT_ROLE_ADJUSTMENT_SQL = """
    CASE
        WHEN COALESCE(c.metadata->>'content_role', '') = 'references' THEN -0.22
        WHEN COALESCE(c.metadata->>'content_role', '') = 'toc' THEN -0.28
        WHEN COALESCE(c.metadata->>'content_role', '') = 'front_matter' THEN -0.16
        WHEN COALESCE(c.metadata->>'content_role', '') = 'table_like' THEN -0.1
        WHEN COALESCE(c.metadata->>'content_role', '') = 'figure_caption' THEN -0.08
        WHEN COALESCE(c.metadata->>'content_role', '') = 'appendix' THEN -0.1
        ELSE 0
    END
"""

_SECTION_BOOST_SQL = """
    CASE
        WHEN c.section_title ILIKE 'Abstract' THEN 0.12
        WHEN c.section_title ILIKE '%%Introduction%%' THEN 0.09
        WHEN c.section_title ILIKE '%%Method%%' OR c.section_title ILIKE '%%Approach%%' THEN 0.03
        WHEN c.section_title ILIKE '%%Related Work%%' THEN 0.02
        WHEN c.section_title ILIKE '%%Conclusion%%' THEN -0.03
        WHEN c.section_title ILIKE '%%Discussion%%' THEN -0.03
        WHEN c.section_title ILIKE '%%Appendix%%' THEN -0.1
        WHEN c.section_title ILIKE '%%Additional Analysis%%' THEN -0.1
        WHEN c.section_title ILIKE '%%Experimental Details%%' THEN -0.08
        WHEN c.section_title ILIKE '%%Implementation Details%%' THEN -0.08
        ELSE 0
    END
"""


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.12f}" for value in values) + "]"


def resolve_candidate_limit(limit: int) -> int:
    return max(max(1, int(limit)) * 4, MIN_VECTOR_CANDIDATES)


def resolve_hnsw_ef_search(candidate_limit: int) -> int | None:
    """HNSW 스캔은 ef_search개까지만 후보를 돌려준다. 후보 수가 기본값(40)보다 크면 그만큼 올린다."""
    if candidate_limit <= DEFAULT_HNSW_EF_SEARCH:
        return None
    return min(int(candidate_limit), MAX_HNSW_EF_SEARCH)


def build_vector_search_query(
    query_embedding: Sequence[float],
    *,
    limit: int,
    arxiv_id: str | None = None,
    model_name: str | None = None,
    min_similarity: float = 0.0,
) -> tuple[str, dict[str, Any]]:
    """2단계 벡터 검색 SQL과 파라미터를 만든다.

    1단계(candidates)는 순수 코사인 거리 순으로 `candidate_limit`개를 뽑는다. 전체 검색에서는
    `ORDER BY e.embedding <=> query`가 HNSW 인덱스를 탄다. 논문 범위 검색은 HNSW 사후 필터가 결과를
    잃을 수 있어 그 논문의 청크만 MATERIALIZED CTE로 모은 뒤 정확히 정렬한다.
    2단계는 후보에만 섹션·content_role 보정을 적용하고 최종 정렬한다.
    """
    candidate_limit = resolve_candidate_limit(limit)
    params: dict[str, Any] = {
        "query": vector_literal(query_embedding),
        "candidate_limit": candidate_limit,
        "limit": max(1, int(limit)),
    }
    if model_name:
        params["model_name"] = model_name

    if arxiv_id:
        params["arxiv_id"] = arxiv_id
        candidates_sql = f"""
            scoped AS MATERIALIZED (
                SELECT e.chunk_id, e.embedding <=> %(query)s::vector AS distance
                FROM paper_chunks sc
                JOIN paper_embeddings e ON e.chunk_id = sc.id
                WHERE sc.arxiv_id = %(arxiv_id)s
                    {"AND e.model_name = %(model_name)s" if model_name else ""}
            ),
            candidates AS (
                SELECT chunk_id, distance
                FROM scoped
                ORDER BY distance ASC, chunk_id DESC
                LIMIT %(candidate_limit)s
            )"""
    else:
        candidates_sql = f"""
            candidates AS (
                SELECT e.chunk_id, e.embedding <=> %(query)s::vector AS distance
                FROM paper_embeddings e
                {"WHERE e.model_name = %(model_name)s" if model_name else ""}
                ORDER BY e.embedding <=> %(query)s::vector
                LIMIT %(candidate_limit)s
            )"""

    min_similarity_sql = ""
    if min_similarity and float(min_similarity) > 0:
        min_similarity_sql = "AND raw_similarity_score >= %(min_similarity)s"
        params["min_similarity"] = float(min_similarity)

    sql = f"""
        WITH {candidates_sql},
        ranked AS (
            SELECT
                c.id,
                c.arxiv_id,
                p.title,
                p.abstract,
                c.chunk_text,
                c.chunk_index,
                c.section_title,
                COALESCE(c.metadata->>'content_role', '') AS content_role,
                1 - k.distance AS raw_similarity_score,
                {_CONTENT_ROLE_ADJUSTMENT_SQL} AS content_role_adjustment,
                {_SECTION_BOOST_SQL} AS section_boost
            FROM candidates k
            JOIN paper_chunks c ON c.id = k.chunk_id
            JOIN papers p ON p.arxiv_id = c.arxiv_id
        )
        SELECT
            id,
            arxiv_id,
            title,
            abstract,
            chunk_text,
            chunk_index,
            section_title,
            (raw_similarity_score + content_role_adjustment + section_boost) AS similarity_score,
            raw_similarity_score,
            content_role,
            content_role_adjustment,
            section_boost
        FROM ranked
        WHERE content_role <> 'toc'
            {min_similarity_sql}
        ORDER BY (raw_similarity_score + content_role_adjustment + section_boost) DESC, id DESC
        LIMIT %(limit)s
    """
    return sql, params


class VectorRepository:
    """논문 청크 임베딩 저장과 유사도 검색을 담당하는 진입점."""

    def __init__(self, *, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def list_chunks_missing_embeddings(
        self,
        *,
        limit: int = 200,
        arxiv_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """아직 임베딩이 없는 청크를 조회한다."""
        query = """
            SELECT
                c.id,
                c.arxiv_id,
                c.chunk_index,
                c.chunk_text,
                c.section_title,
                c.metadata,
                p.title
            FROM paper_chunks c
            JOIN papers p ON p.arxiv_id = c.arxiv_id
            LEFT JOIN paper_embeddings e ON e.chunk_id = c.id
            WHERE
                e.chunk_id IS NULL
                AND COALESCE(c.metadata->>'content_role', '') <> 'references'
        """
        params: list[Any] = []
        if arxiv_id:
            query += " AND c.arxiv_id = %s"
            params.append(arxiv_id)
        query += " ORDER BY c.arxiv_id ASC, c.chunk_index ASC LIMIT %s"
        params.append(max(1, limit))

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

        return [
            {
                "chunk_id": row[0],
                "arxiv_id": row[1],
                "chunk_index": row[2],
                "chunk_text": self._sanitize_text(row[3] or ""),
                "section_title": row[4],
                "metadata": row[5] or {},
                "paper_title": row[6] or "",
            }
            for row in rows
        ]

    def upsert_paper_embeddings(self, rows: list[dict[str, Any]]) -> None:
        """논문 청크와 임베딩 벡터를 저장하거나 갱신한다."""
        if not rows:
            return

        deduplicated: dict[int, dict[str, Any]] = {int(row["chunk_id"]): row for row in rows}
        with self._connection() as connection, connection.cursor() as cursor:
            execute_values(
                cursor,
                """
                INSERT INTO paper_embeddings (chunk_id, embedding, model_name, updated_at)
                VALUES %s
                ON CONFLICT (chunk_id)
                DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    model_name = EXCLUDED.model_name,
                    updated_at = NOW()
                """,
                [
                    (chunk_id, self._vector_literal(row["embedding"]), row["model_name"])
                    for chunk_id, row in deduplicated.items()
                ],
                template="(%s, %s::vector, %s, NOW())",
                page_size=64,
            )

    def search_paper_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 5,
        arxiv_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """질문 임베딩과 유사한 논문 청크 목록을 반환한다. SQL은 `build_vector_search_query` 참고."""
        sql, params = build_vector_search_query(
            query_embedding,
            limit=limit,
            arxiv_id=arxiv_id,
            model_name=getattr(self.settings, "openai_embedding_model", None),
            min_similarity=self._min_similarity(),
        )
        ef_search = None if arxiv_id else resolve_hnsw_ef_search(params["candidate_limit"])

        with self._connection() as connection, connection.cursor() as cursor:
            if ef_search is not None:
                cursor.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
            cursor.execute(sql, params)
            rows = cursor.fetchall()

        return [
            {
                "chunk_id": row[0],
                "arxiv_id": row[1],
                "paper_title": row[2],
                "paper_abstract": row[3] or "",
                "chunk_text": row[4] or "",
                "chunk_index": row[5],
                "section_title": row[6],
                "score": float(row[7]) if row[7] is not None else 0.0,
                "similarity_score": float(row[7]) if row[7] is not None else 0.0,
                "raw_similarity_score": float(row[8]) if row[8] is not None else 0.0,
                "content_role": row[9] or "",
                "retrieval_method": "vector",
                "score_breakdown": {
                    "raw_similarity_score": float(row[8]) if row[8] is not None else 0.0,
                    "content_role_adjustment": float(row[10]) if row[10] is not None else 0.0,
                    "section_boost": float(row[11]) if row[11] is not None else 0.0,
                },
            }
            for row in rows
        ]

    def _connection(self):
        return get_connection(self._build_postgres_connection_params(), settings=self.settings)

    def _min_similarity(self) -> float:
        try:
            return max(0.0, float(getattr(self.settings, "vector_min_similarity", 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _build_postgres_connection_params(self) -> dict[str, Any]:
        return build_postgres_connection_params(self.settings)

    @staticmethod
    def _vector_literal(values: Sequence[float]) -> str:
        return vector_literal(values)

    @staticmethod
    def _sanitize_text(value: str) -> str:
        return "".join(char for char in value if not 0xD800 <= ord(char) <= 0xDFFF)
