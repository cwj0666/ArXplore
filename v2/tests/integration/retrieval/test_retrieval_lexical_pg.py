from collections import Counter
from dataclasses import fields

import pytest

from arxplore.parsing.roles import is_references_section_title
from arxplore.retrieval.lexical import (
    REFERENCES_SECTION_TITLE_SQL_REGEX,
    build_lexical_candidates_query,
    fetch_lexical_candidates,
)
from arxplore.retrieval.service import Retriever

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

CREDIT_PAPERS = [
    {
        "arxiv_id": "2602.00001",
        "title": "Token-Level Credit Assignment for Critic Alignment",
        "abstract": "We assign token-level credit to align the critic in reinforcement learning.",
        "chunks": [
            ("1 Introduction", "Credit assignment over tokens improves the critic in reinforcement learning.", "body"),
        ],
    },
    {
        "arxiv_id": "2602.00002",
        "title": "Graph Partitioning with Caches",
        "abstract": "Sorting networks and cache-aware partitioning.",
        "chunks": [("1 Introduction", "Partitioning large graphs with caches.", "body")],
    },
]

ROUTING_PAPERS = [
    {
        "arxiv_id": "2603.00001",
        "title": "Sparse Routing",
        "abstract": "Experts are routed sparsely.",
        "chunks": [("2 Method", "Sparse expert routing with a load balancing loss.", "body")],
    },
    {
        "arxiv_id": "2603.00002",
        "title": "Sparse Expert Routing Everywhere",
        "abstract": "Sparse experts and routing, routing, routing for sparse expert models.",
        "chunks": [("Abstract", "Sparse expert routing sparse expert routing sparse expert routing.", "body")],
    },
]

QUANT_PAPERS = [
    {
        "arxiv_id": f"2604.0000{paper}",
        "title": title,
        "abstract": "We study post-training quantization of neural networks.",
        "chunks": [
            (
                "3 Method",
                f"Step {index} keeps activations stable. " + "Quantization error shrinks. " * (4 - paper),
                "body",
            )
            for index in range(8)
        ],
    }
    for paper, title in enumerate(
        ("Quantization for Attention", "Stable Vector Quantization", "Cellpose Post-Training Methods"), start=1
    )
]

LEXICAL_BREAKDOWN_FIELDS = {
    "fts_score",
    "ilike_bonus",
    "content_role_adjustment",
    "section_boost",
    "structural_adjustment",
    "coverage",
    "strict_match",
    "section_intent_bonus",
}


async def test_migrated_schema_has_generated_search_columns(seeder) -> None:
    await seeder.papers(PAPERS[:1])

    generated = await seeder.execute(
        """
        SELECT table_name, column_name, is_generated
        FROM information_schema.columns
        WHERE column_name IN ('title_abstract_vector', 'chunk_vector')
        ORDER BY table_name
        """
    )
    vectors = await seeder.execute(
        """
        SELECT p.title_abstract_vector::text AS paper_vector, c.chunk_vector::text AS chunk_vector
        FROM papers p JOIN paper_chunks c ON c.arxiv_id = p.arxiv_id AND c.chunk_index = 1
        """
    )

    assert [(row["table_name"], row["column_name"], row["is_generated"]) for row in generated] == [
        ("paper_chunks", "chunk_vector", "ALWAYS"),
        ("papers", "title_abstract_vector", "ALWAYS"),
    ]
    assert "'diffus':1A" in vectors[0]["paper_vector"]
    assert "'backbon':12B" in vectors[0]["paper_vector"]
    assert "'denois':4C" in vectors[0]["chunk_vector"]


async def test_lexical_candidates_come_from_the_generated_columns(seeder, retrieval_pool) -> None:
    ids = await seeder.papers(PAPERS)

    async with retrieval_pool.connection() as connection:
        results = await fetch_lexical_candidates(connection, "diffusion transformer", limit=5)
        title_only = await fetch_lexical_candidates(connection, "rewards", limit=5)
        scoped = await fetch_lexical_candidates(connection, "transformer", limit=5, arxiv_id="2601.00002")
        escaped = await fetch_lexical_candidates(connection, "50%_off", limit=5)
        unescaped = await fetch_lexical_candidates(connection, "50%off", limit=5)

    assert [hit.arxiv_id for hit in results] == ["2601.00001", "2601.00001"]
    assert results[0].chunk_id == ids[("2601.00001", 0)]
    assert results[0].score > results[1].score
    assert results[0].lexical is not None
    assert {field.name for field in fields(results[0].lexical)} == LEXICAL_BREAKDOWN_FIELDS
    assert results[0].lexical.strict_match is True
    assert results[0].lexical.coverage == pytest.approx(1.0)
    assert ids[("2601.00001", 2)] not in {hit.chunk_id for hit in results}

    assert {hit.chunk_id for hit in title_only} == {ids[("2601.00002", 0)], ids[("2601.00002", 1)]}
    assert scoped == []
    assert escaped[0].lexical is not None
    assert escaped[0].lexical.ilike_bonus == 0.15
    assert unescaped[0].lexical is not None
    assert unescaped[0].lexical.ilike_bonus == 0.0


async def test_long_query_returns_partial_matches(seeder, retrieval_pool) -> None:
    await seeder.papers(CREDIT_PAPERS)
    query = "Study defining token-level credit and improving reinforcement learning with PACT method"

    async with retrieval_pool.connection() as connection:
        results = await fetch_lexical_candidates(connection, query, limit=5)
    contexts = await Retriever(retrieval_pool).search_lexical_contexts(query, limit=3)

    assert [hit.arxiv_id for hit in results] == ["2602.00001"]
    assert results[0].lexical is not None
    assert results[0].lexical.strict_match is False
    assert 0.3 < results[0].lexical.coverage < 1.0
    assert results[0].lexical.ilike_bonus == 0.0
    assert results[0].score > 0.01
    assert contexts[0].hit.arxiv_id == "2602.00001"


async def test_strict_match_ranks_above_partial_match(seeder, retrieval_pool) -> None:
    await seeder.papers(ROUTING_PAPERS)

    async with retrieval_pool.connection() as connection:
        results = await fetch_lexical_candidates(connection, "sparse expert routing balancing loss", limit=5)

    assert [hit.arxiv_id for hit in results] == ["2603.00001", "2603.00002"]
    assert [hit.lexical.strict_match for hit in results if hit.lexical is not None] == [True, False]
    assert results[1].lexical is not None
    assert results[1].lexical.coverage == pytest.approx(0.6)
    assert results[0].score > results[1].score


async def test_per_paper_cap_spreads_top_k_across_papers(seeder, retrieval_pool) -> None:
    await seeder.papers(QUANT_PAPERS)

    async with retrieval_pool.connection() as connection:
        capped = await fetch_lexical_candidates(connection, "quantization", limit=10)
        uncapped = await fetch_lexical_candidates(connection, "quantization", limit=10, per_paper_cap=None)
        scoped = await fetch_lexical_candidates(connection, "quantization", limit=10, arxiv_id="2604.00002")
    contexts = await Retriever(retrieval_pool).search_lexical_contexts("quantization", limit=10)

    assert Counter(hit.arxiv_id for hit in capped) == {"2604.00001": 3, "2604.00002": 3, "2604.00003": 3}
    assert [hit.arxiv_id for hit in uncapped[:8]] == ["2604.00001"] * 8
    assert len(scoped) == 8
    assert len({context.hit.arxiv_id for context in contexts}) >= 3
    assert contexts[0].hit.arxiv_id == "2604.00001"


async def test_lexical_candidate_query_can_use_gin_indexes(seeder) -> None:
    await seeder.papers(PAPERS)
    await seeder.execute(
        """
        INSERT INTO papers (arxiv_id, title, abstract)
        SELECT 'filler-' || g, 'Filler study ' || g, 'Graph partitioning benchmark ' || g FROM generate_series(1, 500) g
        """
    )
    await seeder.execute(
        """
        INSERT INTO paper_chunks (arxiv_id, chunk_index, chunk_text)
        SELECT 'filler-' || g, 0, 'Sorting networks and caches ' || g FROM generate_series(1, 500) g
        """
    )
    await seeder.execute("ANALYZE")

    built = build_lexical_candidates_query("diffusion transformer", limit=5)
    assert built is not None
    plan = await seeder.explain(*built)

    assert "idx_paper_chunks_chunk_vector" in plan
    assert "idx_papers_title_abstract_vector" in plan


REFERENCE_TITLE_CASES = [
    "References",
    "7 References",
    "IV. REFERENCES",
    "iii) references",
    "iv references",
    "Civil References",
    "cl Bibliography",
    "3.2 Reference Model",
    "Works  Cited",
    "Method",
]


async def test_structural_penalty_follows_the_chunk_role_rule(seeder, retrieval_pool) -> None:
    await seeder.papers(
        [
            {
                "arxiv_id": "2605.00001",
                "title": "Quantization Notes",
                "abstract": "A study of quantization.",
                "chunks": [
                    (title, f"Quantization detail number {index}.", "body")
                    for index, title in enumerate(REFERENCE_TITLE_CASES)
                ],
            }
        ]
    )

    async with retrieval_pool.connection() as connection:
        hits = await fetch_lexical_candidates(connection, "quantization", limit=20, arxiv_id="2605.00001")

    penalties = {hit.section_title: hit.lexical.structural_adjustment for hit in hits if hit.lexical is not None}
    assert set(penalties) == set(REFERENCE_TITLE_CASES)
    assert penalties == {title: -0.08 if is_references_section_title(title) else 0.0 for title in REFERENCE_TITLE_CASES}
    assert penalties["iv references"] == 0.0
    assert penalties["Civil References"] == 0.0
    assert penalties["iii) references"] == -0.08


async def test_postgres_evaluates_the_title_rule_like_python(seeder) -> None:
    titles = [
        *REFERENCE_TITLE_CASES,
        "",
        "  References:  ",
        "1.References",
        "A. References",
        "C References",
        "7\nReferences",
    ]

    rows = await seeder.execute(
        "SELECT title, title ~ %s AS matched FROM unnest(%s::text[]) WITH ORDINALITY AS t(title, position) "
        "ORDER BY position",
        (REFERENCES_SECTION_TITLE_SQL_REGEX, titles),
    )

    assert [row["matched"] for row in rows] == [is_references_section_title(title) for title in titles]
