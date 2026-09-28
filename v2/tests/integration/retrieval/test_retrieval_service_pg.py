import psycopg
import pytest
from openai import OpenAIError

from arxplore.context import observe_retrieval
from arxplore.db.pool import create_pool, open_pool
from arxplore.retrieval.service import RetrievalConfig, Retriever
from arxplore.retrieval.windows import fetch_chunk_windows

pytestmark = pytest.mark.integration

PAPERS = [
    {
        "arxiv_id": "2601.00001",
        "title": "Diffusion Transformers for Video Generation",
        "abstract": "We scale latent diffusion with transformer backbones.",
        "chunks": [
            ("Abstract", "Latent diffusion transformers generate long videos.", "body"),
            ("3 Method", "Patch embeddings are denoised by a transformer.", "body"),
            ("Contents", "1 Introduction 2 Method 3 Experiments", "toc"),
            ("4 Results", "", "body"),
            ("5 Discussion", "Diffusion transformers scale with compute.", "body"),
        ],
    },
    {
        "arxiv_id": "2601.00002",
        "title": "Preference Optimization Without Rewards",
        "abstract": "Direct preference optimization aligns language models.",
        "chunks": [
            ("1 Introduction", "Reinforcement learning from human feedback is costly.", "body"),
            ("4 Experiments", "The reward model baseline uses a diffusion transformer.", "body"),
        ],
    },
]


class EmbeddingDown(OpenAIError):
    pass


async def _seed(seeder, basis) -> dict[tuple[str, int], int]:
    ids = await seeder.papers(PAPERS)
    await seeder.embeddings(
        {
            ids[("2601.00001", 0)]: basis((0, 1.0), (1, 0.2)),
            ids[("2601.00001", 1)]: basis((0, 1.0), (1, 1.0)),
            ids[("2601.00001", 4)]: basis((0, 1.0), (2, 0.5)),
            ids[("2601.00002", 0)]: basis((1, 1.0), (2, 0.1)),
            ids[("2601.00002", 1)]: basis((0, 1.0), (3, 0.3)),
        }
    )
    return ids


async def test_chunk_windows_keep_the_center_order(seeder, retrieval_pool, basis) -> None:
    await _seed(seeder, basis)
    centers = [("2601.00001", 1), ("2601.00002", 0), ("2601.00001", 0), ("missing", 0)]

    async with retrieval_pool.connection() as connection:
        windows = await fetch_chunk_windows(connection, centers, window=1)
        wide = await fetch_chunk_windows(connection, [("2601.00001", 3)], window=2)
        none = await fetch_chunk_windows(connection, [], window=1)

    assert [[chunk.chunk_index for chunk in window] for window in windows] == [[0, 1, 2], [0, 1], [0, 1], []]
    assert windows[0][2].content_role == "toc"
    assert [chunk.chunk_index for chunk in wide[0]] == [1, 2, 3, 4]
    assert none == []


async def test_contexts_join_neighbouring_chunks_and_skip_empty_text(seeder, retrieval_pool, basis) -> None:
    await _seed(seeder, basis)
    retriever = Retriever(retrieval_pool)
    hits = await retriever.search_lexical("diffusion compute scale", limit=5, arxiv_id="2601.00001")

    contexts = await retriever.build_contexts(hits, adjacency_window=1)

    by_index = {context.hit.chunk_index: context for context in contexts}
    assert by_index[4].text == "Diffusion transformers scale with compute."
    assert [chunk.chunk_index for chunk in by_index[4].chunks] == [3, 4]
    assert await retriever.build_contexts([], adjacency_window=1) == []


async def test_hybrid_is_used_when_an_embedder_is_available(seeder, retrieval_pool, basis, fake_embedder) -> None:
    ids = await _seed(seeder, basis)
    embedder = fake_embedder(default=basis((0, 1.0)))
    retriever = Retriever(retrieval_pool, embedder, RetrievalConfig())

    with observe_retrieval() as observations:
        result = await retriever.retrieve_contexts("  diffusion\x00 transformer ", limit=3)

    assert result.mode == "hybrid"
    assert embedder.calls == [["diffusion transformer"]]
    assert [context.hit.method for context in result.contexts] == ["hybrid"] * len(result.contexts)
    assert result.contexts[0].hit.chunk_id == ids[("2601.00001", 0)]
    assert result.contexts[0].hit.fusion is not None
    assert result.contexts[0].hit.fusion.matched_methods == ("lexical", "vector")
    assert [observation.result for observation in observations] == [result]


async def test_lexical_setting_never_embeds(seeder, retrieval_pool, basis, fake_embedder) -> None:
    await _seed(seeder, basis)
    embedder = fake_embedder()
    retriever = Retriever(retrieval_pool, embedder, RetrievalConfig(mode="lexical"))

    result = await retriever.retrieve_contexts("diffusion transformer", limit=3)

    assert result.mode == "lexical"
    assert embedder.calls == []
    assert result.contexts == tuple(await retriever.search_lexical_contexts("diffusion transformer", limit=3))


async def test_embedding_failure_falls_back_to_a_fresh_lexical_search(
    seeder, retrieval_pool, basis, fake_embedder
) -> None:
    await _seed(seeder, basis)
    embedder = fake_embedder(error=EmbeddingDown("embedding service unavailable"))
    retriever = Retriever(retrieval_pool, embedder, RetrievalConfig())

    result = await retriever.retrieve_contexts("diffusion transformer", limit=3)
    scoped = await retriever.retrieve_contexts("diffusion transformer", limit=3, arxiv_id="2601.00002")

    assert result.mode == "lexical"
    assert result.contexts == tuple(await retriever.search_lexical_contexts("diffusion transformer", limit=3))
    assert [context.hit.method for context in result.contexts] == ["lexical"] * len(result.contexts)
    assert scoped.mode == "lexical"
    assert {context.hit.arxiv_id for context in scoped.contexts} == {"2601.00002"}
    assert len(embedder.calls) == 2


async def test_database_errors_are_not_treated_as_embedding_failures(
    seeder, retrieval_pool, basis, fake_embedder
) -> None:
    await _seed(seeder, basis)
    embedder = fake_embedder(default=[1.0, 0.0, 0.0])
    retriever = Retriever(retrieval_pool, embedder, RetrievalConfig())

    with pytest.raises(psycopg.errors.DataException):
        await retriever.retrieve_contexts("diffusion transformer", limit=3)


async def test_failed_search_rolls_back_and_the_connection_is_reused(
    seeder, retrieval_database, basis, fake_embedder
) -> None:
    await _seed(seeder, basis)
    pool = create_pool(retrieval_database, role="test", max_size=1, timeout=5, connect_timeout=5)

    async with open_pool(pool):
        broken = Retriever(pool, fake_embedder(default=[1.0, 0.0, 0.0]), RetrievalConfig())
        with pytest.raises(psycopg.errors.DataException):
            await broken.search_vector("diffusion transformer", limit=5)
        async with pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT current_setting('hnsw.ef_search') AS ef_search, pg_backend_pid() AS pid"
            )
            first = await cursor.fetchone()
        healthy = Retriever(pool, fake_embedder(default=basis((0, 1.0))), RetrievalConfig())
        hits = await healthy.search_vector("diffusion transformer", limit=5)
        async with pool.connection() as connection:
            cursor = await connection.execute("SELECT pg_backend_pid() AS pid")
            second = await cursor.fetchone()

    assert first is not None
    assert first["ef_search"] == "40"
    assert hits
    assert second == {"pid": first["pid"]}


async def test_paper_scope_restricts_every_channel(seeder, retrieval_pool, basis, fake_embedder) -> None:
    await _seed(seeder, basis)
    retriever = Retriever(retrieval_pool, fake_embedder(default=basis((0, 1.0))), RetrievalConfig())

    result = await retriever.retrieve_contexts("diffusion transformer", limit=5, arxiv_id="2601.00001")

    assert result.mode == "hybrid"
    assert result.contexts
    assert {context.hit.arxiv_id for context in result.contexts} == {"2601.00001"}
    assert len(result.contexts) <= 5
