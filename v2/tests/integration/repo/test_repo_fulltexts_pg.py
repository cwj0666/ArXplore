import math
from typing import Any

import psycopg
import pytest

from arxplore.db.repositories import embeddings, fulltexts, papers
from arxplore.domain.chunks import ChunkEmbedding, ChunkInput
from arxplore.domain.papers import FulltextState, PaperUpsert

pytestmark = pytest.mark.integration

EMBEDDING_MODEL = "text-embedding-3-large"
DIMENSIONS = 1536
ARXIV_ID = "2601.00001"
OTHER_ID = "2601.00002"
SEEDED_CHUNKS = [
    ("Abstract", "Latent diffusion transformers generate long videos.", "body"),
    ("3 Method", "Patch embeddings are denoised by a transformer.", "body"),
    ("Contents", "1 Introduction 2 Method 3 Experiments", "toc"),
]
OTHER_CHUNKS = [
    ("1 Introduction", "Reinforcement learning from human feedback is costly.", "body"),
    ("4 Experiments", "The 50%_off baseline uses the reward model only.", "body"),
]


def _basis_vector(*weights: tuple[int, float]) -> list[float]:
    vector = [0.0] * DIMENSIONS
    for index, value in weights:
        vector[index] = value
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


def _chunks(rows: list[tuple[str, str, str]], *, role: str | None = None) -> list[ChunkInput]:
    return [
        ChunkInput(
            chunk_index=index,
            chunk_text=text,
            section_title=section,
            token_count=len(text.split()),
            metadata={"content_role": role or chunk_role},
        )
        for index, (section, text, chunk_role) in enumerate(rows)
    ]


async def _chunk_ids(probe, arxiv_id: str | None = None) -> dict[tuple[str, int], int]:
    rows = await probe.rows("SELECT arxiv_id, chunk_index, id FROM paper_chunks")
    return {(row["arxiv_id"], row["chunk_index"]): row["id"] for row in rows if arxiv_id in (None, row["arxiv_id"])}


async def _embedded_chunk_ids(probe, arxiv_id: str) -> set[int]:
    rows = await probe.rows(
        "SELECT e.chunk_id FROM paper_embeddings e JOIN paper_chunks c ON c.id = e.chunk_id WHERE c.arxiv_id = %s",
        (arxiv_id,),
    )
    return {row["chunk_id"] for row in rows}


async def _seed_embeddings(pool, probe) -> dict[tuple[str, int], int]:
    for arxiv_id, rows in ((ARXIV_ID, SEEDED_CHUNKS), (OTHER_ID, OTHER_CHUNKS)):
        await papers.save_paper(pool, PaperUpsert(arxiv_id=arxiv_id, title=f"Paper {arxiv_id}", abstract="A"))
        await fulltexts.save_chunks(pool, arxiv_id, _chunks(rows))
    ids = await _chunk_ids(probe)
    await embeddings.upsert_embeddings(
        pool,
        [
            ChunkEmbedding(chunk_id=chunk_id, embedding=_basis_vector((index % 7, 1.0)), model_name=EMBEDDING_MODEL)
            for index, chunk_id in enumerate(ids.values())
        ],
    )
    return ids


async def test_identical_chunk_texts_keep_ids_and_embeddings_but_refresh_metadata(pool, probe) -> None:
    ids = await _seed_embeddings(pool, probe)
    before_ids = {key: value for key, value in ids.items() if key[0] == ARXIV_ID}
    before_embedded = await _embedded_chunk_ids(probe, ARXIV_ID)
    assert before_embedded == set(before_ids.values())

    replaced = await fulltexts.save_chunks(pool, ARXIV_ID, _chunks(SEEDED_CHUNKS, role="body"))

    assert replaced is False
    assert await _chunk_ids(probe, ARXIV_ID) == before_ids
    assert await _embedded_chunk_ids(probe, ARXIV_ID) == before_embedded
    roles = await probe.rows(
        "SELECT metadata->>'content_role' AS role FROM paper_chunks WHERE arxiv_id = %s ORDER BY chunk_index",
        (ARXIV_ID,),
    )
    assert [row["role"] for row in roles] == ["body", "body", "body"]


async def test_unchanged_texts_only_touch_rows_whose_metadata_changed(pool, probe) -> None:
    await papers.save_paper(pool, PaperUpsert(arxiv_id=ARXIV_ID, title="T"))
    await fulltexts.save_chunks(pool, ARXIV_ID, _chunks(SEEDED_CHUNKS))
    await probe.execute("UPDATE paper_chunks SET updated_at = '2000-01-01'")
    changed = _chunks(SEEDED_CHUNKS)
    changed[1] = ChunkInput(
        chunk_index=1,
        chunk_text=SEEDED_CHUNKS[1][1],
        section_title="3 Approach",
        token_count=99,
        metadata={"content_role": "body", "section_index": 2},
    )

    assert await fulltexts.save_chunks(pool, ARXIV_ID, list(reversed(changed))) is False

    rows = await probe.rows(
        "SELECT chunk_index, section_title, token_count, metadata, updated_at > '2000-01-01' AS touched "
        "FROM paper_chunks WHERE arxiv_id = %s ORDER BY chunk_index",
        (ARXIV_ID,),
    )
    assert [row["touched"] for row in rows] == [False, True, False]
    assert (rows[1]["section_title"], rows[1]["token_count"], rows[1]["metadata"]) == (
        "3 Approach",
        99,
        {"content_role": "body", "section_index": 2},
    )


async def test_changed_chunk_texts_replace_rows_and_drop_embeddings(pool, probe) -> None:
    ids = await _seed_embeddings(pool, probe)
    rows = list(SEEDED_CHUNKS)
    rows[1] = ("3 Method", "Patch embeddings are denoised by a bigger transformer.", "body")

    assert await fulltexts.save_chunks(pool, ARXIV_ID, _chunks(rows)) is True

    assert await _embedded_chunk_ids(probe, ARXIV_ID) == set()
    new_ids = await _chunk_ids(probe, ARXIV_ID)
    assert set(new_ids.values()).isdisjoint({ids[(ARXIV_ID, index)] for index in range(3)})
    assert await _embedded_chunk_ids(probe, OTHER_ID)


async def test_a_different_chunk_count_replaces_the_rows(pool, probe) -> None:
    ids = await _seed_embeddings(pool, probe)

    assert await fulltexts.save_chunks(pool, ARXIV_ID, _chunks(SEEDED_CHUNKS[:2])) is True

    new_ids = await _chunk_ids(probe, ARXIV_ID)
    assert sorted(key[1] for key in new_ids) == [0, 1]
    assert set(new_ids.values()).isdisjoint(ids.values())


async def test_new_chunks_are_inserted_in_input_order(pool, probe) -> None:
    await papers.save_paper(pool, PaperUpsert(arxiv_id=ARXIV_ID, title="T"))
    chunks = _chunks(SEEDED_CHUNKS)

    assert await fulltexts.save_chunks(pool, ARXIV_ID, [chunks[2], chunks[0], chunks[1]]) is True

    ids = await _chunk_ids(probe, ARXIV_ID)
    assert sorted(ids, key=ids.__getitem__) == [(ARXIV_ID, 2), (ARXIV_ID, 0), (ARXIV_ID, 1)]
    listed = await fulltexts.list_chunks(pool, ARXIV_ID)
    assert [chunk.chunk_index for chunk in listed] == [0, 1, 2]
    assert [chunk.chunk_id for chunk in listed] == [ids[(ARXIV_ID, 0)], ids[(ARXIV_ID, 1)], ids[(ARXIV_ID, 2)]]


async def test_failed_replacement_rolls_back_the_delete(pool, probe) -> None:
    ids = await _seed_embeddings(pool, probe)
    duplicate = [
        ChunkInput(chunk_index=0, chunk_text="new text"),
        ChunkInput(chunk_index=0, chunk_text="new text again"),
    ]

    with pytest.raises(psycopg.errors.UniqueViolation):
        await fulltexts.save_chunks(pool, ARXIV_ID, duplicate)

    assert await _chunk_ids(probe) == ids
    assert await _embedded_chunk_ids(probe, ARXIV_ID) == {ids[(ARXIV_ID, index)] for index in range(3)}


async def test_empty_chunk_list_changes_nothing(pool, probe) -> None:
    ids = await _seed_embeddings(pool, probe)

    assert await fulltexts.save_chunks(pool, ARXIV_ID, []) is False

    assert await _chunk_ids(probe) == ids


async def test_chunk_text_is_cleaned_and_mapping_input_is_coerced(pool) -> None:
    await papers.save_paper(pool, PaperUpsert(arxiv_id=ARXIV_ID, title="T"))
    loose: list[dict[str, Any]] = [
        {"chunk_index": "1", "chunk_text": "b\ud800"},
        {
            "chunk_index": 0,
            "chunk_text": "a",
            "section_title": "Abs\udfff",
            "token_count": "3",
            "metadata": {"k\ud800": 1},
        },
    ]

    assert await fulltexts.save_chunks(pool, ARXIV_ID, [ChunkInput.from_mapping(chunk) for chunk in loose]) is True

    listed = await fulltexts.list_chunks(pool, ARXIV_ID)
    assert [(chunk.chunk_index, chunk.chunk_text, chunk.section_title, chunk.token_count) for chunk in listed] == [
        (0, "a", "Abs", 3),
        (1, "b", None, 0),
    ]
    assert [chunk.metadata for chunk in listed] == [{"k": 1}, {}]
    assert await fulltexts.save_chunks(pool, ARXIV_ID, [ChunkInput.from_mapping(chunk) for chunk in loose]) is False


async def test_list_chunks_limits_with_a_floor_of_one(pool) -> None:
    await papers.save_paper(pool, PaperUpsert(arxiv_id=ARXIV_ID, title="T"))
    await fulltexts.save_chunks(pool, ARXIV_ID, _chunks(SEEDED_CHUNKS))

    assert [chunk.chunk_index for chunk in await fulltexts.list_chunks(pool, ARXIV_ID, limit=2)] == [0, 1]
    assert [chunk.chunk_index for chunk in await fulltexts.list_chunks(pool, ARXIV_ID, limit=0)] == [0]
    assert await fulltexts.list_chunks(pool, "missing") == []
    first = (await fulltexts.list_chunks(pool, ARXIV_ID))[0]
    assert (first.arxiv_id, first.section_title, first.token_count, first.metadata) == (
        ARXIV_ID,
        "Abstract",
        6,
        {"content_role": "body"},
    )
    assert first.updated_at is not None


async def test_fulltext_round_trip_and_state(pool) -> None:
    await papers.save_paper(pool, PaperUpsert(arxiv_id=ARXIV_ID, title="T"))
    assert await fulltexts.get_fulltext(pool, ARXIV_ID) is None
    assert await fulltexts.get_fulltext_state(pool, ARXIV_ID) is None

    await fulltexts.save_fulltext(
        pool,
        ARXIV_ID,
        text="Body \ud800text",
        sections=[{"title": "Abstract\udfff", "text": "x"}],
        source="layout_pdf",
        quality_metrics={"text_length": 9},
        artifacts={"tables": []},
        parser_metadata={"provider": "huridocs"},
        content_hash="ignored-later",
    )
    stored = await fulltexts.get_fulltext(pool, ARXIV_ID)
    assert stored is not None
    assert (stored.text, stored.sections, stored.source) == (
        "Body text",
        [{"title": "Abstract", "text": "x"}],
        "layout_pdf",
    )
    assert (stored.quality_metrics, stored.artifacts, stored.parser_metadata) == (
        {"text_length": 9},
        {"tables": []},
        {"provider": "huridocs"},
    )
    assert await fulltexts.get_fulltext_state(pool, ARXIV_ID) == FulltextState(
        source="layout_pdf", content_hash="ignored-later", chunk_count=0
    )

    await fulltexts.save_fulltext(pool, ARXIV_ID, text="Plain")
    plain = await fulltexts.get_fulltext(pool, ARXIV_ID)
    assert plain is not None
    assert (plain.source, plain.sections, plain.quality_metrics, plain.artifacts, plain.parser_metadata) == (
        "pdf",
        [],
        {},
        {},
        {},
    )
    assert await fulltexts.get_fulltext_state(pool, ARXIV_ID) == FulltextState(
        source="pdf", content_hash=None, chunk_count=0
    )


async def test_chunk_write_failure_leaves_hash_unset_so_a_retry_rewrites_chunks(pool) -> None:
    arxiv_id = "2601.00009"
    await papers.save_paper(pool, PaperUpsert(arxiv_id=arxiv_id, title="T", abstract="A"))
    chunks = _chunks(SEEDED_CHUNKS)

    await fulltexts.save_fulltext(pool, arxiv_id, text="Body", source="layout_pdf", content_hash=None)
    with pytest.raises(psycopg.errors.UniqueViolation):
        await fulltexts.save_chunks(pool, arxiv_id, [*chunks, chunks[0]])
    assert await fulltexts.get_fulltext_state(pool, arxiv_id) == FulltextState(
        source="layout_pdf", content_hash=None, chunk_count=0
    )

    await fulltexts.save_fulltext(pool, arxiv_id, text="Body", source="layout_pdf", content_hash=None)
    assert await fulltexts.save_chunks(pool, arxiv_id, chunks) is True
    assert await fulltexts.get_fulltext_state(pool, arxiv_id) == FulltextState(
        source="layout_pdf", content_hash=None, chunk_count=3
    )
    await fulltexts.update_content_hash(pool, arxiv_id, "hash-1")
    assert await fulltexts.get_fulltext_state(pool, arxiv_id) == FulltextState(
        source="layout_pdf", content_hash="hash-1", chunk_count=3
    )
