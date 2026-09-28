from collections.abc import Mapping, Sequence
from typing import Any

from psycopg.rows import dict_row

from arxplore.db.pool import ConnectionSource, use_connection
from arxplore.domain.retrieval import ContextChunk, RetrievalContext, RetrievalHit

DEFAULT_ADJACENCY_WINDOW = 1
CONTEXT_SEPARATOR = "\n\n"

CHUNK_WINDOWS_SQL = """
    SELECT w.ord, c.id, c.arxiv_id, c.chunk_index, c.chunk_text, c.section_title,
           COALESCE(c.metadata->>'content_role', '') AS content_role
    FROM unnest(%(arxiv_ids)s::text[], %(center_indexes)s::integer[])
        WITH ORDINALITY AS w(arxiv_id, center_index, ord)
    JOIN paper_chunks c
      ON c.arxiv_id = w.arxiv_id
     AND c.chunk_index BETWEEN GREATEST(0, w.center_index - %(window)s) AND w.center_index + %(window)s
    ORDER BY w.ord ASC, c.chunk_index ASC
"""


def context_chunk_from_row(row: Mapping[str, Any]) -> ContextChunk:
    return ContextChunk(
        chunk_id=int(row["id"]),
        arxiv_id=str(row["arxiv_id"]),
        chunk_index=int(row["chunk_index"]),
        chunk_text=str(row["chunk_text"] or ""),
        section_title=str(row["section_title"] or ""),
        content_role=str(row["content_role"] or ""),
    )


async def fetch_chunk_windows(
    db: ConnectionSource,
    centers: Sequence[tuple[str, int]],
    *,
    window: int = DEFAULT_ADJACENCY_WINDOW,
) -> list[list[ContextChunk]]:
    if not centers:
        return []
    params = {
        "arxiv_ids": [str(arxiv_id) for arxiv_id, _ in centers],
        "center_indexes": [int(center_index) for _, center_index in centers],
        "window": max(0, int(window)),
    }
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(CHUNK_WINDOWS_SQL, params, prepare=False)
        rows = await cursor.fetchall()
    windows: list[list[ContextChunk]] = [[] for _ in centers]
    for row in rows:
        windows[int(row["ord"]) - 1].append(context_chunk_from_row(row))
    return windows


def join_context_text(chunks: Sequence[ContextChunk]) -> str:
    return CONTEXT_SEPARATOR.join(chunk.chunk_text for chunk in chunks if chunk.chunk_text)


def attach_contexts(hits: Sequence[RetrievalHit], windows: Sequence[Sequence[ContextChunk]]) -> list[RetrievalContext]:
    return [
        RetrievalContext(hit=hit, chunks=tuple(chunks), text=join_context_text(chunks))
        for hit, chunks in zip(hits, windows, strict=True)
    ]
