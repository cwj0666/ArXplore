import pytest

from arxplore.db.repositories import embeddings, fulltexts, papers
from arxplore.db.repositories.embeddings import vector_literal
from arxplore.domain.chunks import ChunkEmbedding, ChunkInput, EmbeddingCandidate
from arxplore.domain.papers import PaperUpsert

pytestmark = pytest.mark.integration

MODEL = "text-embedding-3-large"
DIMENSIONS = 1536


def _vector(*weights: tuple[int, float]) -> list[float]:
    vector = [0.0] * DIMENSIONS
    for index, value in weights:
        vector[index] = value
    return vector


async def _seed(pool, arxiv_id: str, title: str, roles: list[str]) -> list[int]:
    await papers.save_paper(pool, PaperUpsert(arxiv_id=arxiv_id, title=title))
    await fulltexts.save_chunks(
        pool,
        arxiv_id,
        [
            ChunkInput(chunk_index=index, chunk_text=f"{arxiv_id} chunk {index}", metadata={"content_role": role})
            for index, role in enumerate(roles)
        ]
        + [ChunkInput(chunk_index=len(roles), chunk_text=f"{arxiv_id} bare", section_title="Notes")],
    )
    return [chunk.chunk_id for chunk in await fulltexts.list_chunks(pool, arxiv_id)]


async def test_missing_embeddings_skip_references_and_embedded_chunks(pool) -> None:
    second = await _seed(pool, "2601.00002", "Second", ["body", "references"])
    first = await _seed(pool, "2601.00001", "First", ["references", "body", "table_like"])
    await embeddings.upsert_embeddings(
        pool, [ChunkEmbedding(chunk_id=first[1], embedding=_vector((0, 1.0)), model_name=MODEL)]
    )

    missing = await embeddings.list_chunks_missing_embeddings(pool)

    assert [(chunk.arxiv_id, chunk.chunk_index) for chunk in missing] == [
        ("2601.00001", 2),
        ("2601.00001", 3),
        ("2601.00002", 0),
        ("2601.00002", 2),
    ]
    assert missing[0] == EmbeddingCandidate(
        chunk_id=first[2],
        arxiv_id="2601.00001",
        chunk_index=2,
        chunk_text="2601.00001 chunk 2",
        section_title=None,
        metadata={"content_role": "table_like"},
        paper_title="First",
    )
    assert (missing[1].metadata, missing[1].section_title) == ({}, "Notes")
    assert missing[2].chunk_id == second[0]


async def test_missing_embeddings_filter_by_paper_and_limit(pool) -> None:
    await _seed(pool, "2601.00001", "First", ["body", "body"])
    await _seed(pool, "2601.00002", "Second", ["body"])

    scoped = await embeddings.list_chunks_missing_embeddings(pool, arxiv_id="2601.00002")
    limited = await embeddings.list_chunks_missing_embeddings(pool, limit=2)
    floored = await embeddings.list_chunks_missing_embeddings(pool, limit=0)

    assert [(chunk.arxiv_id, chunk.chunk_index) for chunk in scoped] == [("2601.00002", 0), ("2601.00002", 1)]
    assert [(chunk.arxiv_id, chunk.chunk_index) for chunk in limited] == [("2601.00001", 0), ("2601.00001", 1)]
    assert len(floored) == 1


async def test_upsert_keeps_the_last_row_per_chunk_and_overwrites(pool, probe) -> None:
    chunk_ids = await _seed(pool, "2601.00001", "First", ["body", "body"])

    await embeddings.upsert_embeddings(
        pool,
        [
            ChunkEmbedding(chunk_id=chunk_ids[0], embedding=_vector((0, 1.0)), model_name="old"),
            ChunkEmbedding(chunk_id=chunk_ids[1], embedding=_vector((1, 1.0)), model_name=MODEL),
            ChunkEmbedding(chunk_id=chunk_ids[0], embedding=_vector((2, 1.0)), model_name=MODEL),
        ],
    )
    await probe.execute("UPDATE paper_embeddings SET updated_at = '2000-01-01'")
    await embeddings.upsert_embeddings(
        pool, [ChunkEmbedding(chunk_id=chunk_ids[1], embedding=_vector((3, 0.5)), model_name="new")]
    )

    rows = await probe.rows(
        "SELECT chunk_id, model_name, embedding::text AS embedding, updated_at > '2000-01-01' AS touched "
        "FROM paper_embeddings ORDER BY chunk_id"
    )
    assert [(row["chunk_id"], row["model_name"], row["touched"]) for row in rows] == [
        (chunk_ids[0], MODEL, False),
        (chunk_ids[1], "new", True),
    ]
    assert rows[0]["embedding"] == vector_literal(_vector((2, 1.0))).replace(".000000000000", "")
    await embeddings.upsert_embeddings(pool, [])


async def test_vectors_are_sent_as_twelve_decimal_text(pool, probe) -> None:
    chunk_ids = await _seed(pool, "2601.00001", "First", ["body"])
    values = _vector((0, 1 / 3), (1, -2 / 7), (2, 1e-13))

    await embeddings.upsert_embeddings(
        pool, [ChunkEmbedding(chunk_id=chunk_ids[0], embedding=values, model_name=MODEL)]
    )

    literal = vector_literal(values)
    assert literal.startswith("[0.333333333333,-0.285714285714,0.000000000000,0.000000000000,")
    stored = await probe.scalar("SELECT embedding = %s::vector FROM paper_embeddings", (literal,))
    assert stored is True
