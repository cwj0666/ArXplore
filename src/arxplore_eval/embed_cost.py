from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

import psycopg

from arxplore.db.repositories.embeddings import MISSING_EMBEDDINGS_SQL
from arxplore.embedding import clean_embedding_input

TOKENS_PER_MILLION = 1_000_000
TARGET_ORDER_SQL = " ORDER BY c.arxiv_id ASC, c.chunk_index ASC"
ALL_TARGETS_SQL = """
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
WHERE
    COALESCE(c.metadata->>'content_role', '') <> 'references'
"""
EMBEDDINGS_BY_MODEL_SQL = "SELECT model_name, count(*) FROM paper_embeddings GROUP BY model_name ORDER BY model_name"
CURSOR_NAME = "arxplore_eval_embed_cost"
FETCH_SIZE = 1000

type TokenCounter = Callable[[str], int]


def target_sql(*, all_chunks: bool) -> str:
    base = ALL_TARGETS_SQL if all_chunks else MISSING_EMBEDDINGS_SQL
    return f"SELECT chunk_text FROM ({base}{TARGET_ORDER_SQL}) AS targets"


@dataclass(frozen=True)
class CostSummary:
    chunks: int
    empty_chunks: int
    characters: int
    tokens: int
    token_method: str

    def usd(self, price_per_million: float) -> float:
        return self.tokens / TOKENS_PER_MILLION * price_per_million

    def as_dict(self, price_per_million: float) -> dict[str, Any]:
        return {**asdict(self), "price_per_million_usd": price_per_million, "usd": self.usd(price_per_million)}


def summarize_texts(texts: Iterable[str], counter: TokenCounter, token_method: str) -> CostSummary:
    chunks = 0
    empty = 0
    characters = 0
    tokens = 0
    for text in texts:
        cleaned = clean_embedding_input(text or "")
        chunks += 1
        characters += len(cleaned)
        if not cleaned:
            empty += 1
            continue
        tokens += counter(cleaned)
    return CostSummary(
        chunks=chunks, empty_chunks=empty, characters=characters, tokens=tokens, token_method=token_method
    )


def stream_target_texts(connection: psycopg.Connection[Any], *, all_chunks: bool) -> Iterator[str]:
    with connection.cursor(name=CURSOR_NAME) as cursor:
        cursor.itersize = FETCH_SIZE
        cursor.execute(target_sql(all_chunks=all_chunks).encode())
        for (text,) in cursor:
            yield text or ""


def embeddings_by_model(connection: psycopg.Connection[Any]) -> dict[str, int]:
    with connection.cursor() as cursor:
        cursor.execute(EMBEDDINGS_BY_MODEL_SQL.encode())
        return {str(model): int(count) for model, count in cursor.fetchall()}


@dataclass(frozen=True)
class CostReport:
    summary: CostSummary
    embeddings: dict[str, int]


def measure_cost(dsn: str, *, all_chunks: bool, counter: TokenCounter, token_method: str) -> CostReport:
    with psycopg.connect(dsn) as connection:
        connection.read_only = True
        embeddings = embeddings_by_model(connection)
        summary = summarize_texts(stream_target_texts(connection, all_chunks=all_chunks), counter, token_method)
        connection.rollback()
    return CostReport(summary=summary, embeddings=embeddings)
