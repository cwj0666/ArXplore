import math
from collections import Counter

import pytest

from arxplore.db.pool import create_pool, open_pool
from arxplore.retrieval.vector import build_vector_search_query, fetch_vector_candidates

pytestmark = pytest.mark.integration

EMBEDDING_MODEL = "text-embedding-3-large"

PAPERS = [
    {
        "arxiv_id": "2601.00001",
        "title": "Diffusion Transformers for Video Generation",
        "abstract": "We scale latent diffusion with transformer backbones.",
        "chunks": [
            ("Abstract", "Latent diffusion transformers generate long videos.", "body"),
            ("3 Method", "Patch embeddings are denoised by a transformer.", "body"),
            ("Contents", "1 Introduction 2 Method 3 Experiments", "toc"),
        ],
    },
    {
        "arxiv_id": "2601.00002",
        "title": "Preference Optimization Without Rewards",
        "abstract": "Direct preference optimization aligns language models.",
        "chunks": [
            ("1 Introduction", "Reinforcement learning from human feedback is costly.", "body"),
            ("4 Experiments", "The 50%_off baseline uses the reward model only.", "body"),
        ],
    },
]

QUANT_PAPERS = [
    {
        "arxiv_id": f"2604.0000{paper}",
        "title": f"Quantization study {paper}",
        "abstract": "We study post-training quantization of neural networks.",
        "chunks": [("3 Method", f"Step {index} keeps activations stable.", "body") for index in range(8)],
    }
    for paper in (1, 2, 3)
]


async def _seed_embeddings(seeder, basis) -> tuple[dict[tuple[str, int], int], list[float]]:
    ids = await seeder.papers(PAPERS)
    await seeder.embeddings(
        {
            ids[("2601.00001", 0)]: basis((0, 1.0), (1, 0.2)),
            ids[("2601.00001", 1)]: basis((0, 1.0), (1, 1.0)),
            ids[("2601.00001", 2)]: basis((0, 1.0)),
            ids[("2601.00002", 0)]: basis((1, 1.0), (2, 0.1)),
            ids[("2601.00002", 1)]: basis((5, 1.0)),
        }
    )
    await seeder.embeddings({ids[("2601.00002", 1)]: basis((0, 1.0))}, model_name="old-model")
    return ids, basis((0, 1.0))


async def test_vector_search_ranks_by_similarity_then_adjusts(seeder, retrieval_pool, basis) -> None:
    ids, query = await _seed_embeddings(seeder, basis)

    async with retrieval_pool.connection() as connection:
        old_model = await fetch_vector_candidates(connection, query, query="q", limit=5, model_name="old-model")
        results = await fetch_vector_candidates(connection, query, query="q", limit=5, model_name=EMBEDDING_MODEL)
        wide = await fetch_vector_candidates(connection, query, query="q", limit=30, model_name=EMBEDDING_MODEL)
        filtered = await fetch_vector_candidates(
            connection, query, query="q", limit=5, model_name=EMBEDDING_MODEL, min_similarity=0.5
        )
        scoped = await fetch_vector_candidates(
            connection, query, query="q", limit=5, model_name=EMBEDDING_MODEL, arxiv_id="2601.00002"
        )

    assert [hit.chunk_id for hit in old_model] == [ids[("2601.00002", 1)]]
    assert old_model[0].vector is not None
    assert old_model[0].vector.raw_similarity_score == pytest.approx(1.0)

    assert [hit.chunk_id for hit in results] == [
        ids[("2601.00001", 0)],
        ids[("2601.00001", 1)],
        ids[("2601.00002", 0)],
    ]
    top = results[0]
    assert top.vector is not None
    assert top.vector.raw_similarity_score == pytest.approx(1 / math.sqrt(1.04), abs=1e-6)
    assert top.score == pytest.approx(top.vector.raw_similarity_score + 0.12, abs=1e-6)
    assert top.vector.section_boost == pytest.approx(0.12)
    assert top.paper_title == "Diffusion Transformers for Video Generation"
    assert top.paper_abstract.startswith("We scale")
    assert top.method == "vector"
    assert results[1].vector is not None
    assert results[1].vector.section_boost == pytest.approx(0.03)

    assert [hit.chunk_id for hit in wide] == [hit.chunk_id for hit in results]
    assert [hit.chunk_id for hit in filtered] == [ids[("2601.00001", 0)], ids[("2601.00001", 1)]]
    assert [hit.chunk_id for hit in scoped] == [ids[("2601.00002", 0)]]


async def test_vector_per_paper_cap_spreads_top_k_across_papers(seeder, retrieval_pool, basis) -> None:
    ids = await seeder.papers(QUANT_PAPERS)
    await seeder.embeddings(
        {
            chunk_id: basis((0, 1.0), (int(arxiv_id[-1]), 0.1 * int(arxiv_id[-1]) + 0.01 * index))
            for (arxiv_id, index), chunk_id in ids.items()
        }
    )
    query = basis((0, 1.0))

    async with retrieval_pool.connection() as connection:
        results = await fetch_vector_candidates(connection, query, query="q", limit=10, model_name=EMBEDDING_MODEL)
        scoped = await fetch_vector_candidates(
            connection, query, query="q", limit=10, model_name=EMBEDDING_MODEL, arxiv_id="2604.00001"
        )

    assert Counter(hit.arxiv_id for hit in results) == {"2604.00001": 3, "2604.00002": 3, "2604.00003": 3}
    assert [hit.arxiv_id for hit in results[:3]] == ["2604.00001"] * 3
    assert len(scoped) == 8


async def test_global_candidates_can_use_the_hnsw_index(seeder, basis) -> None:
    _, query = await _seed_embeddings(seeder, basis)
    await seeder.execute("ANALYZE")

    global_plan = await seeder.explain(*build_vector_search_query(query, limit=5, model_name=EMBEDDING_MODEL))
    scoped_plan = await seeder.explain(
        *build_vector_search_query(query, limit=5, arxiv_id="2601.00001", model_name=EMBEDDING_MODEL)
    )

    assert "paper_embeddings_embedding_hnsw" in global_plan
    assert "paper_embeddings_embedding_hnsw" not in scoped_plan


async def test_ef_search_is_raised_only_inside_the_search_transaction(seeder, retrieval_database, basis) -> None:
    _, query = await _seed_embeddings(seeder, basis)
    pool = create_pool(retrieval_database, role="test", max_size=1, timeout=5, connect_timeout=5)

    async with open_pool(pool):
        async with pool.connection() as connection:
            await connection.execute("SELECT 1")
            await fetch_vector_candidates(connection, query, query="q", limit=75, model_name=EMBEDDING_MODEL)
            inside_caller_transaction = await _setting_and_pid(connection)
        async with pool.connection() as connection:
            after_commit = await _setting_and_pid(connection)
        await fetch_vector_candidates(pool, query, query="q", limit=75, model_name=EMBEDDING_MODEL)
        async with pool.connection() as connection:
            after_pool_search = await _setting_and_pid(connection)
            await connection.execute("SELECT 1")
            await fetch_vector_candidates(
                connection, query, query="q", limit=75, model_name=EMBEDDING_MODEL, arxiv_id="2601.00001"
            )
            scoped = await _setting_and_pid(connection)

    pid = inside_caller_transaction["pid"]
    assert inside_caller_transaction == {"ef_search": "600", "pid": pid}
    assert after_commit == {"ef_search": "40", "pid": pid}
    assert after_pool_search == {"ef_search": "40", "pid": pid}
    assert scoped == {"ef_search": "40", "pid": pid}


async def _setting_and_pid(connection) -> dict[str, object]:
    cursor = await connection.execute("SELECT current_setting('hnsw.ef_search') AS ef_search, pg_backend_pid() AS pid")
    row = await cursor.fetchone()
    assert row is not None
    return dict(row)
