from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import psycopg2

from src.shared import AppSettings, build_postgres_connection_params, get_settings


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

        with self._connection() as connection, connection.cursor() as cursor:
            for row in rows:
                cursor.execute(
                    """
                    INSERT INTO paper_embeddings (chunk_id, embedding, model_name, updated_at)
                    VALUES (%s, %s::vector, %s, NOW())
                    ON CONFLICT (chunk_id)
                    DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        model_name = EXCLUDED.model_name,
                        updated_at = NOW()
                    """,
                    (
                        int(row["chunk_id"]),
                        self._vector_literal(row["embedding"]),
                        row["model_name"],
                    ),
                )

    def search_paper_chunks(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 5,
        arxiv_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """질문 임베딩과 유사한 논문 청크 목록을 반환한다."""
        query = """
            WITH ranked AS (
                SELECT
                    c.id,
                    c.arxiv_id,
                    p.title,
                    p.abstract,
                    c.chunk_text,
                    c.chunk_index,
                    c.section_title,
                    COALESCE(c.metadata->>'content_role', '') AS content_role,
                    1 - (e.embedding <=> %s::vector) AS raw_similarity_score,
                    CASE
                        WHEN COALESCE(c.metadata->>'content_role', '') = 'references' THEN -0.22
                        WHEN COALESCE(c.metadata->>'content_role', '') = 'toc' THEN -0.28
                        WHEN COALESCE(c.metadata->>'content_role', '') = 'front_matter' THEN -0.16
                        WHEN COALESCE(c.metadata->>'content_role', '') = 'table_like' THEN -0.1
                        WHEN COALESCE(c.metadata->>'content_role', '') = 'figure_caption' THEN -0.08
                        WHEN COALESCE(c.metadata->>'content_role', '') = 'appendix' THEN -0.1
                        ELSE 0
                    END AS content_role_adjustment,
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
                    END AS section_boost
                FROM paper_embeddings e
                JOIN paper_chunks c ON c.id = e.chunk_id
                JOIN papers p ON p.arxiv_id = c.arxiv_id
        """
        params: list[Any] = [self._vector_literal(query_embedding)]
        if arxiv_id:
            query += " WHERE c.arxiv_id = %s"
            params.append(arxiv_id)
        query += """
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
            ORDER BY (raw_similarity_score + content_role_adjustment + section_boost) DESC, id DESC
            LIMIT %s
        """
        params.append(max(1, limit))

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(query, tuple(params))
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

    @contextmanager
    def _connection(self):
        connection = psycopg2.connect(**self._build_postgres_connection_params())
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _build_postgres_connection_params(self) -> dict[str, Any]:
        return build_postgres_connection_params(self.settings)

    @staticmethod
    def _vector_literal(values: Sequence[float]) -> str:
        return "[" + ",".join(f"{float(value):.12f}" for value in values) + "]"

    @staticmethod
    def _sanitize_text(value: str) -> str:
        return "".join(char for char in value if not 0xD800 <= ord(char) <= 0xDFFF)
