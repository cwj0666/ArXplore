import json
import math
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from arxplore.db.migrate import migrate
from arxplore.db.pool import Pool, create_pool, open_pool
from arxplore.db.repositories.embeddings import vector_literal

EMBEDDING_MODEL = "text-embedding-3-large"
DIMENSIONS = 1536

type ChunkIds = dict[tuple[str, int], int]
type BasisVector = Callable[..., list[float]]


def basis_vector(*weights: tuple[int, float]) -> list[float]:
    vector = [0.0] * DIMENSIONS
    for index, value in weights:
        vector[index] = value
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


@dataclass
class FakeEmbedder:
    vectors: Mapping[str, Sequence[float]] = field(default_factory=dict)
    default: Sequence[float] = field(default_factory=lambda: basis_vector((0, 1.0)))
    error: BaseException | None = None
    available: bool = True
    calls: list[list[str]] = field(default_factory=list)

    def is_available(self) -> bool:
        return self.available

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return [list(self.vectors.get(text, self.default)) for text in texts]


class Seeder:
    def __init__(self, connection: psycopg.AsyncConnection[Any]) -> None:
        self.connection = connection

    async def execute(self, sql: str, params: Any = None) -> list[dict[str, Any]]:
        async with self.connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute(sql, params)
            return await cursor.fetchall() if cursor.description else []

    async def papers(self, papers: Sequence[Mapping[str, Any]]) -> ChunkIds:
        ids: ChunkIds = {}
        for paper in papers:
            await self.execute(
                "INSERT INTO papers (arxiv_id, title, abstract, pdf_url) VALUES (%s, %s, %s, %s)",
                (paper["arxiv_id"], paper["title"], paper["abstract"], paper.get("pdf_url")),
            )
            for index, (section, text, role) in enumerate(paper["chunks"]):
                rows = await self.execute(
                    """
                    INSERT INTO paper_chunks (arxiv_id, chunk_index, chunk_text, section_title, token_count, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING id
                    """,
                    (paper["arxiv_id"], index, text, section, len(text.split()), json.dumps({"content_role": role})),
                )
                ids[(paper["arxiv_id"], index)] = int(rows[0]["id"])
        return ids

    async def embeddings(self, vectors: Mapping[int, Sequence[float]], *, model_name: str = EMBEDDING_MODEL) -> None:
        for chunk_id, vector in vectors.items():
            await self.execute(
                """
                INSERT INTO paper_embeddings (chunk_id, embedding, model_name) VALUES (%s, %s::vector, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding, model_name = EXCLUDED.model_name
                """,
                (chunk_id, vector_literal(vector), model_name),
            )

    async def explain(self, sql: str, params: Mapping[str, Any]) -> str:
        async with self.connection.transaction(force_rollback=True):
            await self.execute("SET LOCAL enable_seqscan = off")
            rows = await self.execute("EXPLAIN (FORMAT JSON) " + sql, params)
        return json.dumps(rows[0]["QUERY PLAN"])


@pytest.fixture
async def retrieval_database(empty_database: str) -> str:
    async with await psycopg.AsyncConnection.connect(empty_database, autocommit=True) as connection:
        await migrate(connection)
    return empty_database


@pytest.fixture
async def seeder(retrieval_database: str) -> AsyncIterator[Seeder]:
    async with await psycopg.AsyncConnection.connect(retrieval_database, autocommit=True) as connection:
        yield Seeder(connection)


@pytest.fixture
async def retrieval_pool(retrieval_database: str) -> AsyncIterator[Pool]:
    pool = create_pool(retrieval_database, role="test", max_size=4, timeout=5, connect_timeout=5)
    async with open_pool(pool):
        yield pool


@pytest.fixture
def basis() -> BasisVector:
    return basis_vector


@pytest.fixture
def fake_embedder() -> type[FakeEmbedder]:
    return FakeEmbedder
