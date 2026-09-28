from collections.abc import Sequence
from typing import Any

from psycopg.rows import dict_row

from arxplore.db.pool import ConnectionSource, use_connection
from arxplore.domain.chunks import ChunkEmbedding, EmbeddingCandidate
from arxplore.text import strip_surrogates

MISSING_EMBEDDINGS_SQL = """
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

UPSERT_EMBEDDING_SQL = """
INSERT INTO paper_embeddings (chunk_id, embedding, model_name, updated_at)
VALUES (%s, %s::vector, %s, NOW())
ON CONFLICT (chunk_id)
DO UPDATE SET
    embedding = EXCLUDED.embedding,
    model_name = EXCLUDED.model_name,
    updated_at = NOW()
"""


def vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.12f}" for value in values) + "]"


async def list_chunks_missing_embeddings(
    db: ConnectionSource,
    *,
    limit: int = 200,
    arxiv_id: str | None = None,
) -> list[EmbeddingCandidate]:
    query = MISSING_EMBEDDINGS_SQL
    params: list[Any] = []
    if arxiv_id:
        query += " AND c.arxiv_id = %s"
        params.append(arxiv_id)
    query += " ORDER BY c.arxiv_id ASC, c.chunk_index ASC LIMIT %s"
    params.append(max(1, limit))
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(query, params)
        rows = await cursor.fetchall()
    return [
        EmbeddingCandidate(
            chunk_id=row["id"],
            arxiv_id=row["arxiv_id"],
            chunk_index=row["chunk_index"],
            chunk_text=strip_surrogates(row["chunk_text"] or ""),
            section_title=row["section_title"],
            metadata=row["metadata"] or {},
            paper_title=row["title"] or "",
        )
        for row in rows
    ]


async def upsert_embeddings(db: ConnectionSource, rows: Sequence[ChunkEmbedding]) -> None:
    if not rows:
        return
    deduplicated = {int(row.chunk_id): row for row in rows}
    async with use_connection(db) as connection, connection.cursor() as cursor:
        await cursor.executemany(
            UPSERT_EMBEDDING_SQL,
            [(chunk_id, vector_literal(row.embedding), row.model_name) for chunk_id, row in deduplicated.items()],
        )
