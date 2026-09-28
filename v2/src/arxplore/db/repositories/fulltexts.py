from collections.abc import Sequence
from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from arxplore.db.pool import ConnectionSource, use_connection
from arxplore.domain.chunks import ChunkInput, StoredChunk, chunk_text_hash
from arxplore.domain.papers import DEFAULT_FULLTEXT_SOURCE, FulltextState, PaperFulltext
from arxplore.text import strip_surrogates, strip_surrogates_deep

SAVE_FULLTEXT_SQL = """
INSERT INTO paper_fulltexts (
    arxiv_id, text, sections, source, quality_metrics, artifacts, parser_metadata, content_hash, updated_at
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
ON CONFLICT (arxiv_id)
DO UPDATE SET
    text = EXCLUDED.text,
    sections = EXCLUDED.sections,
    source = EXCLUDED.source,
    quality_metrics = EXCLUDED.quality_metrics,
    artifacts = EXCLUDED.artifacts,
    parser_metadata = EXCLUDED.parser_metadata,
    content_hash = EXCLUDED.content_hash,
    updated_at = NOW()
"""

GET_FULLTEXT_SQL = """
SELECT arxiv_id, text, sections, source, quality_metrics, artifacts, parser_metadata, updated_at
FROM paper_fulltexts
WHERE arxiv_id = %s
"""

GET_FULLTEXT_STATE_SQL = """
SELECT f.source, f.content_hash,
       (SELECT COUNT(*) FROM paper_chunks c WHERE c.arxiv_id = f.arxiv_id) AS chunk_count
FROM paper_fulltexts f
WHERE f.arxiv_id = %s
"""

UPDATE_CONTENT_HASH_SQL = "UPDATE paper_fulltexts SET content_hash = %s WHERE arxiv_id = %s"

EXISTING_CHUNK_HASHES_SQL = (
    "SELECT chunk_index, md5(chunk_text) AS text_hash FROM paper_chunks WHERE arxiv_id = %s ORDER BY chunk_index ASC"
)

UPDATE_CHUNK_METADATA_SQL = """
UPDATE paper_chunks AS c
SET
    section_title = v.section_title,
    token_count = v.token_count,
    metadata = v.metadata,
    updated_at = NOW()
FROM (VALUES (%s::text, %s::integer, %s::text, %s::integer, %s::jsonb))
    AS v(arxiv_id, chunk_index, section_title, token_count, metadata)
WHERE c.arxiv_id = v.arxiv_id
  AND c.chunk_index = v.chunk_index
  AND (
    c.section_title IS DISTINCT FROM v.section_title
    OR c.token_count <> v.token_count
    OR c.metadata <> v.metadata
  )
"""

DELETE_CHUNKS_SQL = "DELETE FROM paper_chunks WHERE arxiv_id = %s"

INSERT_CHUNK_SQL = """
INSERT INTO paper_chunks (arxiv_id, chunk_index, chunk_text, section_title, token_count, metadata, updated_at)
VALUES (%s, %s, %s, %s, %s, %s, NOW())
"""

LIST_CHUNKS_SQL = """
SELECT id, arxiv_id, chunk_index, chunk_text, section_title, token_count, metadata, updated_at
FROM paper_chunks
WHERE arxiv_id = %s
ORDER BY chunk_index ASC
"""


def _clean_chunk(chunk: ChunkInput) -> ChunkInput:
    return ChunkInput(
        chunk_index=chunk.chunk_index,
        chunk_text=strip_surrogates(chunk.chunk_text) if isinstance(chunk.chunk_text, str) else chunk.chunk_text,
        section_title=strip_surrogates(chunk.section_title)
        if isinstance(chunk.section_title, str)
        else chunk.section_title,
        token_count=chunk.token_count,
        metadata=strip_surrogates_deep(chunk.metadata),
    )


def _stored_chunk(row: dict[str, Any]) -> StoredChunk:
    return StoredChunk(
        chunk_id=row["id"],
        arxiv_id=row["arxiv_id"],
        chunk_index=row["chunk_index"],
        chunk_text=row["chunk_text"] or "",
        section_title=row["section_title"],
        token_count=row["token_count"] or 0,
        metadata=row["metadata"] or {},
        updated_at=row["updated_at"],
    )


async def save_fulltext(
    db: ConnectionSource,
    arxiv_id: str,
    *,
    text: str,
    sections: list[Any] | None = None,
    source: str = DEFAULT_FULLTEXT_SOURCE,
    quality_metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    parser_metadata: dict[str, Any] | None = None,
    content_hash: str | None = None,
) -> None:
    params = (
        arxiv_id,
        strip_surrogates(text),
        Jsonb(strip_surrogates_deep(sections or [])),
        source,
        Jsonb(strip_surrogates_deep(quality_metrics or {})),
        Jsonb(strip_surrogates_deep(artifacts or {})),
        Jsonb(strip_surrogates_deep(parser_metadata or {})),
        content_hash,
    )
    async with use_connection(db) as connection, connection.cursor() as cursor:
        await cursor.execute(SAVE_FULLTEXT_SQL, params)


async def get_fulltext(db: ConnectionSource, arxiv_id: str) -> PaperFulltext | None:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(GET_FULLTEXT_SQL, (arxiv_id,))
        row = await cursor.fetchone()
    if row is None:
        return None
    return PaperFulltext(
        arxiv_id=row["arxiv_id"],
        text=row["text"] or "",
        sections=row["sections"] or [],
        source=row["source"],
        quality_metrics=row["quality_metrics"] or {},
        artifacts=row["artifacts"] or {},
        parser_metadata=row["parser_metadata"] or {},
        updated_at=row["updated_at"],
    )


async def get_fulltext_state(db: ConnectionSource, arxiv_id: str) -> FulltextState | None:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(GET_FULLTEXT_STATE_SQL, (arxiv_id,))
        row = await cursor.fetchone()
    if row is None:
        return None
    return FulltextState(
        source=row["source"], content_hash=row["content_hash"], chunk_count=int(row["chunk_count"] or 0)
    )


async def update_content_hash(db: ConnectionSource, arxiv_id: str, content_hash: str | None) -> None:
    async with use_connection(db) as connection, connection.cursor() as cursor:
        await cursor.execute(UPDATE_CONTENT_HASH_SQL, (content_hash, arxiv_id))


async def save_chunks(db: ConnectionSource, arxiv_id: str, chunks: Sequence[ChunkInput]) -> bool:
    if not chunks:
        return False
    cleaned = [_clean_chunk(chunk) for chunk in chunks]
    ordered = sorted(cleaned, key=lambda chunk: chunk.chunk_index)
    new_indexes = [chunk.chunk_index for chunk in ordered]
    new_hashes = [chunk_text_hash(chunk.chunk_text) for chunk in ordered]

    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(EXISTING_CHUNK_HASHES_SQL, (arxiv_id,))
        existing = await cursor.fetchall()
        existing_indexes = [int(row["chunk_index"]) for row in existing]
        existing_hashes = [row["text_hash"] for row in existing]
        if existing and existing_indexes == new_indexes and existing_hashes == new_hashes:
            await cursor.executemany(
                UPDATE_CHUNK_METADATA_SQL,
                [
                    (arxiv_id, chunk.chunk_index, chunk.section_title, chunk.token_count, Jsonb(chunk.metadata))
                    for chunk in ordered
                ],
            )
            return False
        await cursor.execute(DELETE_CHUNKS_SQL, (arxiv_id,))
        await cursor.executemany(
            INSERT_CHUNK_SQL,
            [
                (
                    arxiv_id,
                    chunk.chunk_index,
                    chunk.chunk_text,
                    chunk.section_title,
                    chunk.token_count,
                    Jsonb(chunk.metadata),
                )
                for chunk in cleaned
            ],
        )
    return True


async def list_chunks(db: ConnectionSource, arxiv_id: str, *, limit: int | None = None) -> list[StoredChunk]:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        if limit is None:
            await cursor.execute(LIST_CHUNKS_SQL, (arxiv_id,))
        else:
            await cursor.execute(LIST_CHUNKS_SQL + " LIMIT %s", (arxiv_id, max(1, limit)))
        rows = await cursor.fetchall()
    return [_stored_chunk(row) for row in rows]
