from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import psycopg2
from psycopg2.extras import Json, execute_values

from src.integrations.db import get_connection
from src.shared import AppSettings, build_postgres_connection_params, get_settings

logger = logging.getLogger(__name__)

# src/integrations/pdf_parser/section_roles.is_references_section_title의 SQL(~*) 버전.
# 제목 전체가 참고문헌 제목일 때만 매치하므로 "Reference Model", "Direct Preference Optimization"은 제외된다.
# PostgreSQL ARE에서 \b는 backspace라 단어 경계로 쓸 수 없어 전체 매치(^...$)로 표현한다.
REFERENCES_SECTION_TITLE_SQL_REGEX = (
    r"^\s*(?:(?:\d+(?:\.\d+)*[.)]?|[ivxlc]+[.)]?)\s+)?[\s.:]*"
    r"(?:references?|bibliography|works\s+cited|literature\s+cited)(?:\s+and\s+notes)?[\s.:]*$"
)

PAPER_TITLE_ABSTRACT_VECTOR_SQL = (
    "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
    "setweight(to_tsvector('english', coalesce(abstract, '')), 'B')"
)
PAPER_CHUNK_VECTOR_SQL = "setweight(to_tsvector('english', coalesce(chunk_text, '')), 'C')"

LEXICAL_INDEX_DDL = (
    f"""
    ALTER TABLE papers
        ADD COLUMN IF NOT EXISTS title_abstract_vector tsvector
        GENERATED ALWAYS AS ({PAPER_TITLE_ABSTRACT_VECTOR_SQL}) STORED
    """,
    f"""
    ALTER TABLE paper_chunks
        ADD COLUMN IF NOT EXISTS chunk_vector tsvector
        GENERATED ALWAYS AS ({PAPER_CHUNK_VECTOR_SQL}) STORED
    """,
    "CREATE INDEX IF NOT EXISTS idx_papers_title_abstract_vector ON papers USING GIN (title_abstract_vector)",
    "CREATE INDEX IF NOT EXISTS idx_paper_chunks_chunk_vector ON paper_chunks USING GIN (chunk_vector)",
    # chunk_vector GIN으로 대체된 표현식 인덱스. 어떤 쿼리도 쓰지 않으므로 쓰기 비용만 든다.
    "DROP INDEX IF EXISTS idx_paper_chunks_fts",
)

# pgvector 0.5.0 미만에는 hnsw 접근 방식이 없다. 실패해도 스키마 생성은 계속한다.
VECTOR_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS paper_embeddings_embedding_hnsw "
    "ON paper_embeddings USING hnsw (embedding vector_cosine_ops)"
)

_LIKE_SPECIAL_CHARS = re.compile(r"([\\%_])")


def escape_like(value: str) -> str:
    return _LIKE_SPECIAL_CHARS.sub(r"\\\1", value)


# 질의의 모든 lexeme을 OR로 묶은 tsquery. lexeme은 이미 정규화된 값이라 text -> tsquery 캐스트로 그대로 쓴다.
_ANY_QUERY_TERM_SQL = r"""
    SELECT string_agg('''' || replace(replace(lexeme, '\', '\\'), '''', '''''') || '''', ' | ')::tsquery AS any_term
    FROM unnest(
        tsvector_to_array(to_tsvector('english', %(query)s) || to_tsvector('english', %(fts_query)s))
    ) AS lexeme
"""


def build_lexical_candidates_query(
    query: str,
    *,
    limit: int,
    arxiv_id: str | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """lexical 후보 조회 SQL과 파라미터를 만든다. 질의가 비어 있으면 None.

    1. 후보: 질의 lexeme 중 하나라도 가진 청크. GIN 인덱스가 걸린 두 생성 컬럼
       (`papers.title_abstract_vector`, `paper_chunks.chunk_vector`)을 각각 조회해 UNION한다.
       AND 질의는 제목과 청크에 걸쳐 만족될 수 있어 컬럼별 `@@ 원래 질의`로는 후보가 빠진다.
    2. 판정·점수: 두 벡터를 이어 붙인 tsvector(제목 A, 초록 B, 청크 C)에 원래 질의를 적용한다.
       이전의 행별 to_tsvector 식과 같은 tsvector라 ts_rank_cd 값도 같다.
    전체 질의 ILIKE는 후보 필터가 아니라 점수 보너스로만 쓴다.
    """
    normalized_query = " ".join(query.split())
    if not normalized_query:
        return None

    chunk_scope_sql = ""
    paper_scope_sql = ""
    params: dict[str, Any] = {
        "query": normalized_query,
        "fts_query": re.sub(r"[-/]+", " ", normalized_query),
        "like_pattern": f"%{escape_like(normalized_query)}%",
        "limit": max(1, int(limit)),
    }
    if arxiv_id:
        chunk_scope_sql = "AND c.arxiv_id = %(arxiv_id)s"
        paper_scope_sql = "AND p.arxiv_id = %(arxiv_id)s"
        params["arxiv_id"] = arxiv_id

    sql = f"""
        WITH query_terms AS ({_ANY_QUERY_TERM_SQL}),
        matched AS (
            SELECT c.id
            FROM paper_chunks c
            WHERE c.chunk_vector @@ (SELECT any_term FROM query_terms)
                {chunk_scope_sql}
            UNION
            SELECT c.id
            FROM papers p
            JOIN paper_chunks c ON c.arxiv_id = p.arxiv_id
            WHERE p.title_abstract_vector @@ (SELECT any_term FROM query_terms)
                {paper_scope_sql}
        ),
        ranked AS (
            SELECT
                c.id AS chunk_id,
                c.arxiv_id,
                p.title AS paper_title,
                p.abstract AS paper_abstract,
                c.chunk_text,
                c.chunk_index,
                c.section_title,
                COALESCE(c.metadata->>'content_role', '') AS content_role,
                (
                    ts_rank_cd(
                        p.title_abstract_vector || c.chunk_vector,
                        websearch_to_tsquery('english', %(query)s)
                    )
                    +
                    0.65 * ts_rank_cd(
                        p.title_abstract_vector || c.chunk_vector,
                        plainto_tsquery('english', %(fts_query)s)
                    )
                ) AS fts_score,
                CASE
                    WHEN p.title ILIKE %(like_pattern)s THEN 0.45
                    WHEN p.abstract ILIKE %(like_pattern)s THEN 0.2
                    WHEN c.chunk_text ILIKE %(like_pattern)s THEN 0.15
                    ELSE 0
                END AS ilike_bonus,
                CASE
                    WHEN coalesce(c.metadata->>'content_role', '') = 'references' THEN -0.24
                    WHEN coalesce(c.metadata->>'content_role', '') = 'toc' THEN -0.28
                    WHEN coalesce(c.metadata->>'content_role', '') = 'front_matter' THEN -0.14
                    WHEN coalesce(c.metadata->>'content_role', '') = 'table_like' THEN -0.12
                    WHEN coalesce(c.metadata->>'content_role', '') = 'figure_caption' THEN -0.08
                    WHEN coalesce(c.metadata->>'content_role', '') = 'appendix' THEN -0.08
                    ELSE 0
                END AS content_role_adjustment,
                CASE
                    WHEN c.section_title ILIKE 'Abstract' THEN 0.16
                    WHEN c.section_title ILIKE '%%Introduction%%' THEN 0.1
                    WHEN c.section_title ILIKE '%%Method%%' OR c.section_title ILIKE '%%Approach%%' THEN 0.04
                    WHEN c.section_title ILIKE '%%Related Work%%' THEN 0.02
                    WHEN c.section_title ILIKE '%%Conclusion%%' THEN -0.02
                    WHEN c.section_title ILIKE '%%Discussion%%' THEN -0.02
                    WHEN c.section_title ILIKE '%%Appendix%%' THEN -0.08
                    WHEN c.section_title ILIKE '%%Additional Analysis%%' THEN -0.08
                    WHEN c.section_title ILIKE '%%Experimental Details%%' THEN -0.06
                    WHEN c.section_title ILIKE '%%Implementation Details%%' THEN -0.06
                    ELSE 0
                END AS section_boost,
                CASE
                    WHEN c.section_title ILIKE '%%Table of Contents%%' THEN -0.12
                    WHEN c.section_title ~* '{REFERENCES_SECTION_TITLE_SQL_REGEX}' THEN -0.08
                    WHEN c.section_title = 'Front Matter' THEN -0.03
                    ELSE 0
                END AS structural_adjustment
            FROM matched m
            JOIN paper_chunks c ON c.id = m.id
            JOIN papers p ON p.arxiv_id = c.arxiv_id
            WHERE
                coalesce(c.metadata->>'content_role', '') <> 'toc'
                AND (
                    (p.title_abstract_vector || c.chunk_vector) @@ websearch_to_tsquery('english', %(query)s)
                    OR (p.title_abstract_vector || c.chunk_vector) @@ plainto_tsquery('english', %(fts_query)s)
                )
        )
        SELECT
            chunk_id,
            arxiv_id,
            paper_title,
            paper_abstract,
            chunk_text,
            chunk_index,
            section_title,
            content_role,
            fts_score,
            ilike_bonus,
            content_role_adjustment,
            section_boost,
            structural_adjustment,
            (fts_score + ilike_bonus + content_role_adjustment + section_boost + structural_adjustment) AS score
        FROM ranked
        WHERE (fts_score + ilike_bonus + content_role_adjustment + section_boost + structural_adjustment) > 0.01
        ORDER BY score DESC, chunk_id DESC
        LIMIT %(limit)s
    """
    return sql, params


class PaperRepository:
    """정제 논문과 논문 청크를 PostgreSQL에 저장하고 조회하는 진입점.

    생성자는 DDL을 실행하지 않는다. 스키마 생성은 `ensure_schema()`를 명시적으로 호출하거나
    `scripts/migrate_schema.py`로 수행한다.
    """

    def __init__(self, *, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def save_paper(self, paper: dict[str, Any]) -> str:
        """정제 논문 1건을 저장하고 arxiv_id를 반환한다."""
        sanitized_paper = self._sanitize_json_value(paper)
        arxiv_id = str(sanitized_paper["arxiv_id"]).strip()
        if not arxiv_id:
            raise ValueError("paper['arxiv_id']는 비어 있을 수 없습니다.")

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO papers (
                    arxiv_id, title, authors, abstract, primary_category,
                    categories, pdf_url, published_at, updated_at,
                    upvotes, github_url, github_stars, citation_count, source, updated_at_utc
                ) VALUES (
                    %(arxiv_id)s, %(title)s, %(authors)s, %(abstract)s, %(primary_category)s,
                    %(categories)s, %(pdf_url)s, %(published_at)s, %(arxiv_updated_at)s,
                    COALESCE(%(upvotes)s::integer, 0), %(github_url)s, %(github_stars)s, %(citation_count)s, %(source)s, NOW()
                )
                ON CONFLICT (arxiv_id)
                DO UPDATE SET
                    title = EXCLUDED.title,
                    authors = EXCLUDED.authors,
                    abstract = EXCLUDED.abstract,
                    primary_category = COALESCE(EXCLUDED.primary_category, papers.primary_category),
                    categories = CASE
                        WHEN EXCLUDED.categories = '[]'::jsonb THEN papers.categories
                        ELSE EXCLUDED.categories
                    END,
                    pdf_url = EXCLUDED.pdf_url,
                    published_at = COALESCE(EXCLUDED.published_at, papers.published_at),
                    updated_at = COALESCE(EXCLUDED.updated_at, papers.updated_at),
                    upvotes = COALESCE(%(upvotes)s::integer, papers.upvotes),
                    github_url = COALESCE(EXCLUDED.github_url, papers.github_url),
                    github_stars = COALESCE(EXCLUDED.github_stars, papers.github_stars),
                    citation_count = COALESCE(EXCLUDED.citation_count, papers.citation_count),
                    source = EXCLUDED.source,
                    updated_at_utc = NOW()
                """,
                {
                    "arxiv_id": arxiv_id,
                    "title": sanitized_paper.get("title", ""),
                    "authors": Json(sanitized_paper.get("authors", [])),
                    "abstract": sanitized_paper.get("abstract", ""),
                    "primary_category": sanitized_paper.get("primary_category"),
                    "categories": Json(sanitized_paper.get("categories", [])),
                    "pdf_url": sanitized_paper.get("pdf_url"),
                    "published_at": self._to_datetime(sanitized_paper.get("published_at")),
                    "arxiv_updated_at": self._to_datetime(sanitized_paper.get("updated_at")),
                    "upvotes": self._to_int_or_none(sanitized_paper.get("upvotes")),
                    "github_url": sanitized_paper.get("github_url"),
                    "github_stars": self._to_int_or_none(sanitized_paper.get("github_stars")),
                    "citation_count": self._to_int_or_none(sanitized_paper.get("citation_count")),
                    "source": sanitized_paper.get("source", "hf_daily_papers"),
                },
            )
        return arxiv_id

    def save_paper_fulltext(
        self,
        arxiv_id: str,
        *,
        text: str,
        sections: list[dict[str, Any]] | None = None,
        source: str = "pdf",
        quality_metrics: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        parser_metadata: dict[str, Any] | None = None,
    ) -> None:
        """논문 본문 텍스트를 저장한다."""
        sanitized_text = self._sanitize_text(text)
        sanitized_sections = self._sanitize_json_value(sections or [])
        sanitized_quality_metrics = self._sanitize_json_value(quality_metrics or {})
        sanitized_artifacts = self._sanitize_json_value(artifacts or {})
        sanitized_parser_metadata = self._sanitize_json_value(parser_metadata or {})
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO paper_fulltexts (
                    arxiv_id, text, sections, source, quality_metrics, artifacts, parser_metadata, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (arxiv_id)
                DO UPDATE SET
                    text = EXCLUDED.text,
                    sections = EXCLUDED.sections,
                    source = EXCLUDED.source,
                    quality_metrics = EXCLUDED.quality_metrics,
                    artifacts = EXCLUDED.artifacts,
                    parser_metadata = EXCLUDED.parser_metadata,
                    updated_at = NOW()
                """,
                (
                    arxiv_id,
                    sanitized_text,
                    Json(sanitized_sections),
                    source,
                    Json(sanitized_quality_metrics),
                    Json(sanitized_artifacts),
                    Json(sanitized_parser_metadata),
                ),
            )

    def save_paper_chunks(self, arxiv_id: str, chunks: list[dict[str, Any]]) -> None:
        """논문 청크 목록을 저장한다."""
        if not chunks:
            return
        sanitized_chunks = self._sanitize_json_value(chunks)

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM paper_chunks WHERE arxiv_id = %s", (arxiv_id,))
            execute_values(
                cursor,
                """
                INSERT INTO paper_chunks (arxiv_id, chunk_index, chunk_text, section_title, token_count, metadata, updated_at)
                VALUES %s
                """,
                [
                    (
                        arxiv_id,
                        int(chunk.get("chunk_index", 0)),
                        chunk.get("chunk_text", ""),
                        chunk.get("section_title"),
                        int(chunk.get("token_count", 0)),
                        Json(chunk.get("metadata", {})),
                    )
                    for chunk in sanitized_chunks
                ],
                template="(%s, %s, %s, %s, %s, %s, NOW())",
            )

    def list_recent_papers(self, *, limit: int = 200) -> list[dict[str, Any]]:
        """최근 저장 논문을 조회한다."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT arxiv_id, title, authors, abstract, primary_category, categories, pdf_url,
                       published_at, updated_at, upvotes, github_url, github_stars, citation_count
                FROM papers
                ORDER BY COALESCE(published_at, updated_at_utc) DESC
                LIMIT %s
                """,
                (max(1, limit),),
            )
            rows = cursor.fetchall()

        papers: list[dict[str, Any]] = []
        for row in rows:
            papers.append(
                {
                    "arxiv_id": row[0],
                    "title": row[1],
                    "authors": row[2] or [],
                    "abstract": row[3] or "",
                    "primary_category": row[4],
                    "categories": row[5] or [],
                    "pdf_url": row[6],
                    "published_at": row[7].isoformat() if row[7] else None,
                    "updated_at": row[8].isoformat() if row[8] else None,
                    "upvotes": row[9] or 0,
                    "github_url": row[10],
                    "github_stars": row[11],
                    "citation_count": row[12],
                }
            )
        return papers

    def list_recent_paper_cards(self, *, limit: int = 12) -> list[dict[str, Any]]:
        """메인 목록 렌더링에 필요한 최소 논문 카드 정보만 조회한다."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT arxiv_id, title, pdf_url, published_at, updated_at
                FROM papers
                ORDER BY COALESCE(published_at, updated_at_utc) DESC
                LIMIT %s
                """,
                (max(1, limit),),
            )
            rows = cursor.fetchall()

        return [
            {
                "arxiv_id": row[0],
                "title": row[1],
                "pdf_url": row[2],
                "published_at": row[3].isoformat() if row[3] else None,
                "updated_at": row[4].isoformat() if row[4] else None,
            }
            for row in rows
        ]

    def list_papers_missing_arxiv_metadata(self, *, limit: int = 200) -> list[dict[str, Any]]:
        """arXiv 보강이 아직 충분히 적용되지 않은 논문 목록을 조회한다."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT arxiv_id, title, authors, abstract, primary_category, categories, pdf_url,
                       published_at, updated_at, upvotes, github_url, github_stars, citation_count, source
                FROM papers
                WHERE
                    primary_category IS NULL
                    OR categories = '[]'::jsonb
                    OR source = 'hf_daily_papers_raw'
                ORDER BY COALESCE(published_at, updated_at_utc) DESC
                LIMIT %s
                """,
                (max(1, limit),),
            )
            rows = cursor.fetchall()

        return [
            {
                "arxiv_id": row[0],
                "title": row[1],
                "authors": row[2] or [],
                "abstract": row[3] or "",
                "primary_category": row[4],
                "categories": row[5] or [],
                "pdf_url": row[6],
                "published_at": row[7].isoformat() if row[7] else None,
                "updated_at": row[8].isoformat() if row[8] else None,
                "upvotes": row[9] or 0,
                "github_url": row[10],
                "github_stars": row[11],
                "citation_count": row[12],
                "source": row[13] or "hf_daily_papers",
            }
            for row in rows
        ]

    def get_paper(self, arxiv_id: str) -> dict[str, Any] | None:
        """단일 논문 메타데이터를 조회한다."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT arxiv_id, title, authors, abstract, primary_category, categories, pdf_url,
                       published_at, updated_at, upvotes, github_url, github_stars, citation_count
                FROM papers
                WHERE arxiv_id = %s
                """,
                (arxiv_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return {
            "arxiv_id": row[0],
            "title": row[1],
            "authors": row[2] or [],
            "abstract": row[3] or "",
            "primary_category": row[4],
            "categories": row[5] or [],
            "pdf_url": row[6],
            "published_at": row[7].isoformat() if row[7] else None,
            "updated_at": row[8].isoformat() if row[8] else None,
            "upvotes": row[9] or 0,
            "github_url": row[10],
            "github_stars": row[11],
            "citation_count": row[12],
        }

    def get_paper_fulltext(self, arxiv_id: str) -> dict[str, Any] | None:
        """단일 논문의 fulltext와 섹션 정보를 조회한다."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT arxiv_id, text, sections, source, quality_metrics, artifacts, parser_metadata, updated_at
                FROM paper_fulltexts
                WHERE arxiv_id = %s
                """,
                (arxiv_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return {
            "arxiv_id": row[0],
            "text": row[1] or "",
            "sections": row[2] or [],
            "source": row[3],
            "quality_metrics": row[4] or {},
            "artifacts": row[5] or {},
            "parser_metadata": row[6] or {},
            "updated_at": row[7].isoformat() if row[7] else None,
        }

    def get_paper_fulltext_source(self, arxiv_id: str) -> str | None:
        """저장된 fulltext의 source 값만 조회한다. fulltext가 없으면 None."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT source FROM paper_fulltexts WHERE arxiv_id = %s",
                (arxiv_id,),
            )
            row = cursor.fetchone()
        return row[0] if row is not None else None

    def list_paper_chunks(self, arxiv_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        """단일 논문의 청크 목록을 chunk_index 순으로 조회한다."""
        query = """
            SELECT id, arxiv_id, chunk_index, chunk_text, section_title, token_count, metadata, updated_at
            FROM paper_chunks
            WHERE arxiv_id = %s
            ORDER BY chunk_index ASC
        """
        params: tuple[Any, ...] = (arxiv_id,)
        if limit is not None:
            query += " LIMIT %s"
            params = (arxiv_id, max(1, limit))

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [
            {
                "chunk_id": row[0],
                "arxiv_id": row[1],
                "chunk_index": row[2],
                "chunk_text": row[3] or "",
                "section_title": row[4],
                "token_count": row[5] or 0,
                "metadata": row[6] or {},
                "updated_at": row[7].isoformat() if row[7] else None,
            }
            for row in rows
        ]

    def list_chunk_window(self, arxiv_id: str, center_chunk_index: int, *, window: int = 1) -> list[dict[str, Any]]:
        """중심 청크를 기준으로 앞뒤 문맥 청크를 함께 조회한다."""
        normalized_window = max(0, int(window))
        start_index = max(0, int(center_chunk_index) - normalized_window)
        end_index = int(center_chunk_index) + normalized_window

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, arxiv_id, chunk_index, chunk_text, section_title, token_count, metadata, updated_at
                FROM paper_chunks
                WHERE arxiv_id = %s
                  AND chunk_index BETWEEN %s AND %s
                ORDER BY chunk_index ASC
                """,
                (arxiv_id, start_index, end_index),
            )
            rows = cursor.fetchall()

        return [
            {
                "chunk_id": row[0],
                "arxiv_id": row[1],
                "chunk_index": row[2],
                "chunk_text": row[3] or "",
                "section_title": row[4],
                "token_count": row[5] or 0,
                "metadata": row[6] or {},
                "updated_at": row[7].isoformat() if row[7] else None,
            }
            for row in rows
        ]

    def list_chunk_windows(
        self,
        centers: Sequence[tuple[str, int]],
        *,
        window: int = 1,
    ) -> list[list[dict[str, Any]]]:
        """여러 중심 청크의 앞뒤 문맥 창을 한 번의 쿼리로 조회한다. 반환 순서는 `centers` 순서와 같다."""
        if not centers:
            return []
        normalized_window = max(0, int(window))

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT w.ord, c.id, c.arxiv_id, c.chunk_index, c.chunk_text, c.section_title,
                       c.token_count, c.metadata, c.updated_at
                FROM unnest(%(arxiv_ids)s::text[], %(center_indexes)s::integer[])
                    WITH ORDINALITY AS w(arxiv_id, center_index, ord)
                JOIN paper_chunks c
                  ON c.arxiv_id = w.arxiv_id
                 AND c.chunk_index BETWEEN GREATEST(0, w.center_index - %(window)s) AND w.center_index + %(window)s
                ORDER BY w.ord ASC, c.chunk_index ASC
                """,
                {
                    "arxiv_ids": [str(arxiv_id) for arxiv_id, _ in centers],
                    "center_indexes": [int(center_index) for _, center_index in centers],
                    "window": normalized_window,
                },
            )
            rows = cursor.fetchall()

        windows: list[list[dict[str, Any]]] = [[] for _ in centers]
        for row in rows:
            windows[int(row[0]) - 1].append(
                {
                    "chunk_id": row[1],
                    "arxiv_id": row[2],
                    "chunk_index": row[3],
                    "chunk_text": row[4] or "",
                    "section_title": row[5],
                    "token_count": row[6] or 0,
                    "metadata": row[7] or {},
                    "updated_at": row[8].isoformat() if row[8] else None,
                }
            )
        return windows

    def list_papers_for_topic(self, topic_id: int) -> list[dict[str, Any]]:
        """토픽 문서 생성에 사용할 논문 묶음을 반환한다."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT p.arxiv_id, p.title, p.authors, p.abstract, p.pdf_url, p.published_at,
                       p.upvotes, p.github_url, p.github_stars, p.citation_count
                FROM topic_papers tp
                JOIN papers p ON p.arxiv_id = tp.arxiv_id
                WHERE tp.topic_id = %s
                ORDER BY p.published_at DESC NULLS LAST, p.updated_at_utc DESC
                """,
                (topic_id,),
            )
            rows = cursor.fetchall()

        return [
            {
                "arxiv_id": row[0],
                "title": row[1],
                "authors": row[2] or [],
                "abstract": row[3] or "",
                "pdf_url": row[4] or "",
                "published_at": row[5].isoformat() if row[5] else None,
                "upvotes": row[6] or 0,
                "github_url": row[7],
                "github_stars": row[8],
                "citation_count": row[9],
            }
            for row in rows
        ]

    def list_chunk_candidates_by_query(
        self,
        query: str,
        *,
        limit: int = 5,
        arxiv_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """FTS 기반 청크 후보를 조회한다. SQL은 `build_lexical_candidates_query` 참고."""
        built = build_lexical_candidates_query(query, limit=limit, arxiv_id=arxiv_id)
        if built is None:
            return []
        sql, params = built
        normalized_query = params["query"]

        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall()

        return [
            {
                "chunk_id": row[0],
                "arxiv_id": row[1],
                "paper_title": row[2],
                "chunk_text": row[4],
                "chunk_index": row[5],
                "section_title": row[6],
                "content_role": row[7] or "",
                "score": float(row[13]) if row[13] is not None else 0.0,
                "similarity_score": float(row[13]) if row[13] is not None else 0.0,
                "retrieval_method": "lexical",
                "score_breakdown": {
                    "fts_score": float(row[8]) if row[8] is not None else 0.0,
                    "ilike_bonus": float(row[9]) if row[9] is not None else 0.0,
                    "content_role_adjustment": float(row[10]) if row[10] is not None else 0.0,
                    "section_boost": float(row[11]) if row[11] is not None else 0.0,
                    "structural_adjustment": float(row[12]) if row[12] is not None else 0.0,
                },
                "snippet": self._build_search_snippet(
                    normalized_query,
                    chunk_text=row[4] or "",
                    abstract=row[3] or "",
                    title=row[2] or "",
                ),
            }
            for row in rows
        ]

    def _connection(self):
        return get_connection(self._build_postgres_connection_params(), settings=self.settings)

    def ensure_schema(self) -> None:
        """테이블·인덱스를 멱등하게 생성한다. 요청 경로가 아닌 마이그레이션/프로세스 시작 시 1회만 호출한다."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS papers (
                    arxiv_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors JSONB NOT NULL DEFAULT '[]'::jsonb,
                    abstract TEXT NOT NULL DEFAULT '',
                    primary_category TEXT,
                    categories JSONB NOT NULL DEFAULT '[]'::jsonb,
                    pdf_url TEXT,
                    published_at TIMESTAMPTZ NULL,
                    updated_at TIMESTAMPTZ NULL,
                    upvotes INTEGER NOT NULL DEFAULT 0,
                    github_url TEXT,
                    github_stars INTEGER NULL,
                    citation_count INTEGER NULL,
                    source TEXT NOT NULL DEFAULT 'hf_daily_papers',
                    updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_fulltexts (
                    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                    text TEXT NOT NULL,
                    sections JSONB NOT NULL DEFAULT '[]'::jsonb,
                    source TEXT NOT NULL DEFAULT 'pdf',
                    quality_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
                    artifacts JSONB NOT NULL DEFAULT '{}'::jsonb,
                    parser_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cursor.execute(
                """
                ALTER TABLE paper_fulltexts
                ADD COLUMN IF NOT EXISTS quality_metrics JSONB NOT NULL DEFAULT '{}'::jsonb;
                """
            )
            cursor.execute(
                """
                ALTER TABLE paper_fulltexts
                ADD COLUMN IF NOT EXISTS artifacts JSONB NOT NULL DEFAULT '{}'::jsonb;
                """
            )
            cursor.execute(
                """
                ALTER TABLE paper_fulltexts
                ADD COLUMN IF NOT EXISTS parser_metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    section_title TEXT NULL,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(arxiv_id, chunk_index)
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_embeddings (
                    chunk_id BIGINT PRIMARY KEY REFERENCES paper_chunks(id) ON DELETE CASCADE,
                    embedding VECTOR(1536) NOT NULL,
                    model_name TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS topics (
                    topic_id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL,
                    keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS topic_papers (
                    topic_id INTEGER NOT NULL,
                    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                    PRIMARY KEY (topic_id, arxiv_id)
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS topic_documents (
                    topic_id INTEGER PRIMARY KEY,
                    document JSONB NOT NULL,
                    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            for statement in LEXICAL_INDEX_DDL:
                cursor.execute(statement)
            self._ensure_vector_index(cursor)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_ai_overviews (
                    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                    overview TEXT,
                    key_findings JSONB NOT NULL DEFAULT '[]'::jsonb,
                    model TEXT NOT NULL,
                    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_ai_detailed_summaries (
                    id BIGSERIAL PRIMARY KEY,
                    arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
                    model TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    created_by_user_id BIGINT NULL,
                    UNIQUE (arxiv_id, model)
                );
                """
            )

    @staticmethod
    def _ensure_vector_index(cursor) -> bool:
        cursor.execute("SAVEPOINT paper_embeddings_hnsw")
        try:
            cursor.execute(VECTOR_INDEX_DDL)
        except psycopg2.Error as exc:
            cursor.execute("ROLLBACK TO SAVEPOINT paper_embeddings_hnsw")
            logger.warning(
                "paper_embeddings HNSW 인덱스를 만들지 못했습니다(pgvector 0.5.0 이상 필요). 벡터 검색은 순차 스캔으로 동작합니다: %s",
                exc,
            )
            return False
        cursor.execute("RELEASE SAVEPOINT paper_embeddings_hnsw")
        return True

    def get_paper_overview(self, arxiv_id: str) -> dict[str, Any] | None:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT overview, key_findings, model, generated_at
                FROM paper_ai_overviews
                WHERE arxiv_id = %s
                """,
                (arxiv_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return {
            "overview": row[0],
            "key_findings": row[1] or [],
            "model": row[2],
            "generated_at": row[3].isoformat() if row[3] else None,
        }

    def get_detailed_summary(self, arxiv_id: str, model_name: str) -> dict[str, Any] | None:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT summary, model, generated_at, created_by_user_id
                FROM paper_ai_detailed_summaries
                WHERE arxiv_id = %s AND model = %s
                """,
                (arxiv_id, model_name),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return {
            "summary": row[0],
            "model": row[1],
            "generated_at": row[2].isoformat() if row[2] else None,
            "created_by_user_id": row[3],
        }

    def upsert_paper_overview(
        self,
        arxiv_id: str,
        overview: str,
        key_findings: list[str],
        model_name: str,
    ) -> None:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO paper_ai_overviews (arxiv_id, overview, key_findings, model, generated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (arxiv_id) DO UPDATE SET
                    overview = EXCLUDED.overview,
                    key_findings = EXCLUDED.key_findings,
                    model = EXCLUDED.model,
                    generated_at = NOW()
                """,
                (arxiv_id, overview, Json(key_findings), model_name),
            )

    def upsert_detailed_summary(
        self,
        arxiv_id: str,
        summary: str,
        model_name: str,
        *,
        created_by_user_id: int | None = None,
    ) -> None:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO paper_ai_detailed_summaries (arxiv_id, model, summary, generated_at, created_by_user_id)
                VALUES (%s, %s, %s, NOW(), %s)
                ON CONFLICT (arxiv_id, model) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    generated_at = NOW(),
                    created_by_user_id = EXCLUDED.created_by_user_id
                """,
                (arxiv_id, model_name, summary, created_by_user_id),
            )

    def _build_postgres_connection_params(self) -> dict[str, Any]:
        return build_postgres_connection_params(self.settings)

    @classmethod
    def _sanitize_json_value(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cls._sanitize_text(value)
        if isinstance(value, list):
            return [cls._sanitize_json_value(item) for item in value]
        if isinstance(value, tuple):
            return [cls._sanitize_json_value(item) for item in value]
        if isinstance(value, dict):
            return {cls._sanitize_text(str(key)): cls._sanitize_json_value(item) for key, item in value.items()}
        return value

    @staticmethod
    def _sanitize_text(value: str) -> str:
        return "".join(char for char in value if not 0xD800 <= ord(char) <= 0xDFFF)

    @staticmethod
    def _to_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            return datetime.fromisoformat(normalized)
        return None

    @staticmethod
    def _to_int_or_none(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_search_snippet(query: str, *, chunk_text: str, abstract: str, title: str, max_chars: int = 280) -> str:
        terms = [term for term in re.split(r"\W+", query.lower()) if len(term) >= 3]
        candidates = [chunk_text, abstract, title]

        for candidate in candidates:
            if not candidate:
                continue
            lowered = candidate.lower()
            for term in terms:
                index = lowered.find(term)
                if index != -1:
                    start = max(0, index - max_chars // 3)
                    end = min(len(candidate), start + max_chars)
                    snippet = candidate[start:end].strip()
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(candidate):
                        snippet = snippet + "..."
                    return snippet

        fallback = next((candidate for candidate in candidates if candidate), "")
        compact = " ".join(fallback.split())
        return compact[:max_chars] + ("..." if len(compact) > max_chars else "")
