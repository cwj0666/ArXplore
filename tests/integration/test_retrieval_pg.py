from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any

import psycopg2
import pytest

from src.integrations import db
from src.integrations.paper_repository import PaperRepository, build_lexical_candidates_query
from src.integrations.paper_retriever import PaperRetriever
from src.integrations.vector_repository import VectorRepository, build_vector_search_query

pytestmark = pytest.mark.integration

EMBEDDING_MODEL = "text-embedding-3-large"
DIMENSIONS = 1536

PAPER_TABLES = (
    "paper_ai_detailed_summaries",
    "paper_ai_overviews",
    "paper_embeddings",
    "paper_chunks",
    "paper_fulltexts",
    "topic_papers",
    "topic_documents",
    "topics",
    "papers",
)

LEGACY_SCHEMA_DDL = (
    """
    CREATE TABLE papers (
        arxiv_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        authors JSONB NOT NULL DEFAULT '[]'::jsonb,
        abstract TEXT NOT NULL DEFAULT '',
        primary_category TEXT,
        categories JSONB NOT NULL DEFAULT '[]'::jsonb,
        pdf_url TEXT,
        published_at TIMESTAMPTZ NULL,
        updated_at TIMESTAMPTZ NULL,
        upvotes INTEGER NOT NULL DEFAULT 0,
        github_url TEXT,
        github_stars INTEGER NULL,
        citation_count INTEGER NULL,
        source TEXT NOT NULL DEFAULT 'hf_daily_papers',
        updated_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE paper_chunks (
        id BIGSERIAL PRIMARY KEY,
        arxiv_id TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
        chunk_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        section_title TEXT NULL,
        token_count INTEGER NOT NULL DEFAULT 0,
        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        UNIQUE(arxiv_id, chunk_index)
    )
    """,
    """
    CREATE TABLE paper_embeddings (
        chunk_id BIGINT PRIMARY KEY REFERENCES paper_chunks(id) ON DELETE CASCADE,
        embedding VECTOR(1536) NOT NULL,
        model_name TEXT NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX idx_paper_chunks_fts ON paper_chunks USING GIN (to_tsvector('english', chunk_text))",
)


def _execute(dsn: str, sql: str, params: Any = None) -> list[tuple[Any, ...]]:
    connection = psycopg2.connect(dsn=dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            rows = cursor.fetchall() if cursor.description else []
        connection.commit()
        return rows
    finally:
        connection.close()


def _drop_paper_tables(dsn: str) -> None:
    for table in PAPER_TABLES:
        _execute(dsn, f"DROP TABLE IF EXISTS {table} CASCADE")


@pytest.fixture(scope="module")
def pgvector_dsn(test_database_url: str) -> str:
    try:
        _execute(test_database_url, "CREATE EXTENSION IF NOT EXISTS vector")
    except psycopg2.Error as exc:
        pytest.skip(f"pgvector extension is not available: {exc}")
    return test_database_url


@pytest.fixture
def dsn(pgvector_dsn: str):
    _drop_paper_tables(pgvector_dsn)
    yield pgvector_dsn
    db.close_all_pools()
    _drop_paper_tables(pgvector_dsn)


def _settings(**overrides: Any) -> SimpleNamespace:
    values = {"openai_embedding_model": EMBEDDING_MODEL, "vector_min_similarity": 0.0}
    values.update(overrides)
    return SimpleNamespace(**values)


class DsnPaperRepository(PaperRepository):
    def __init__(self, dsn: str, **settings: Any) -> None:
        super().__init__(settings=_settings(**settings))
        self.dsn = dsn

    def _build_postgres_connection_params(self) -> dict[str, Any]:
        return {"dsn": self.dsn}


class DsnVectorRepository(VectorRepository):
    def __init__(self, dsn: str, **settings: Any) -> None:
        super().__init__(settings=_settings(**settings))
        self.dsn = dsn

    def _build_postgres_connection_params(self) -> dict[str, Any]:
        return {"dsn": self.dsn}


def _basis_vector(*weights: tuple[int, float]) -> list[float]:
    vector = [0.0] * DIMENSIONS
    for index, value in weights:
        vector[index] = value
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


def _explain(dsn: str, sql: str, params: dict[str, Any]) -> str:
    connection = psycopg2.connect(dsn=dsn)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL enable_seqscan = off")
            cursor.execute("EXPLAIN (FORMAT JSON) " + sql, params)
            plan = cursor.fetchone()[0]
        connection.rollback()
    finally:
        connection.close()
    return json.dumps(plan)


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


def _seed(repository: PaperRepository, papers: list[dict[str, Any]] | None = None) -> None:
    for paper in PAPERS if papers is None else papers:
        repository.save_paper({"arxiv_id": paper["arxiv_id"], "title": paper["title"], "abstract": paper["abstract"]})
        repository.save_paper_chunks(
            paper["arxiv_id"],
            [
                {
                    "chunk_index": index,
                    "chunk_text": text,
                    "section_title": section,
                    "token_count": len(text.split()),
                    "metadata": {"content_role": role},
                }
                for index, (section, text, role) in enumerate(paper["chunks"])
            ],
        )


def _chunk_ids(dsn: str) -> dict[tuple[str, int], int]:
    rows = _execute(dsn, "SELECT arxiv_id, chunk_index, id FROM paper_chunks")
    return {(row[0], row[1]): row[2] for row in rows}


def _index_names(dsn: str) -> set[str]:
    return {row[0] for row in _execute(dsn, "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema()")}


def test_ensure_schema_upgrades_legacy_tables(dsn):
    for statement in LEGACY_SCHEMA_DDL:
        _execute(dsn, statement)
    _execute(
        dsn,
        "INSERT INTO papers (arxiv_id, title, abstract) VALUES ('2501.00001', 'Sparse Mixture', 'Experts route tokens.')",
    )
    _execute(
        dsn,
        "INSERT INTO paper_chunks (arxiv_id, chunk_index, chunk_text) VALUES ('2501.00001', 0, 'Routing collapse.')",
    )

    repository = DsnPaperRepository(dsn)
    repository.ensure_schema()
    repository.ensure_schema()

    paper_vector = _execute(dsn, "SELECT title_abstract_vector::text FROM papers")[0][0]
    chunk_vector = _execute(dsn, "SELECT chunk_vector::text FROM paper_chunks")[0][0]
    assert "'spars':1A" in paper_vector
    assert "'expert':3B" in paper_vector
    assert "'collaps':2C" in chunk_vector

    indexes = _index_names(dsn)
    assert {
        "idx_papers_title_abstract_vector",
        "idx_paper_chunks_chunk_vector",
        "paper_embeddings_embedding_hnsw",
    } <= indexes
    assert "idx_paper_chunks_fts" not in indexes

    generated = _execute(
        dsn,
        """
        SELECT table_name, column_name, is_generated
        FROM information_schema.columns
        WHERE column_name IN ('title_abstract_vector', 'chunk_vector')
        ORDER BY table_name
        """,
    )
    assert generated == [("paper_chunks", "chunk_vector", "ALWAYS"), ("papers", "title_abstract_vector", "ALWAYS")]


def test_lexical_search_uses_generated_columns(dsn):
    repository = DsnPaperRepository(dsn)
    repository.ensure_schema()
    _seed(repository)
    ids = _chunk_ids(dsn)

    results = repository.list_chunk_candidates_by_query("diffusion transformer", limit=5)
    assert [row["arxiv_id"] for row in results] == ["2601.00001", "2601.00001"]
    assert results[0]["chunk_id"] == ids[("2601.00001", 0)]
    assert results[0]["score"] > results[1]["score"]
    assert set(results[0]["score_breakdown"]) == {
        "fts_score",
        "ilike_bonus",
        "content_role_adjustment",
        "section_boost",
        "structural_adjustment",
        "coverage",
        "strict_match",
    }
    assert results[0]["score_breakdown"]["strict_match"] is True
    assert results[0]["score_breakdown"]["coverage"] == pytest.approx(1.0)
    assert ids[("2601.00001", 2)] not in {row["chunk_id"] for row in results}

    title_only = repository.list_chunk_candidates_by_query("rewards", limit=5)
    assert {row["chunk_id"] for row in title_only} == {ids[("2601.00002", 0)], ids[("2601.00002", 1)]}

    scoped = repository.list_chunk_candidates_by_query("transformer", limit=5, arxiv_id="2601.00002")
    assert scoped == []

    assert repository.list_chunk_candidates_by_query("50%_off", limit=5)[0]["score_breakdown"]["ilike_bonus"] == 0.15
    assert repository.list_chunk_candidates_by_query("50%off", limit=5)[0]["score_breakdown"]["ilike_bonus"] == 0.0


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


def test_long_query_returns_partial_matches(dsn):
    repository = DsnPaperRepository(dsn)
    repository.ensure_schema()
    _seed(repository, CREDIT_PAPERS)
    query = "Study defining token-level credit and improving reinforcement learning with PACT method"

    results = repository.list_chunk_candidates_by_query(query, limit=5)

    assert [row["arxiv_id"] for row in results] == ["2602.00001"]
    breakdown = results[0]["score_breakdown"]
    assert breakdown["strict_match"] is False
    assert 0.3 < breakdown["coverage"] < 1.0
    assert breakdown["ilike_bonus"] == 0.0
    assert results[0]["score"] > 0.01

    retriever = PaperRetriever(repository=repository, embedding_client=object(), vector_repository=object())
    contexts = retriever.search_paper_contexts(query, limit=3)
    assert contexts[0]["arxiv_id"] == "2602.00001"


def test_strict_match_ranks_above_partial_match(dsn):
    repository = DsnPaperRepository(dsn)
    repository.ensure_schema()
    _seed(
        repository,
        [
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
        ],
    )

    results = repository.list_chunk_candidates_by_query("sparse expert routing balancing loss", limit=5)

    assert [row["arxiv_id"] for row in results] == ["2603.00001", "2603.00002"]
    assert [row["score_breakdown"]["strict_match"] for row in results] == [True, False]
    assert results[1]["score_breakdown"]["coverage"] == pytest.approx(0.6)
    assert results[0]["score"] > results[1]["score"]


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


def test_per_paper_cap_spreads_top_k_across_papers(dsn):
    repository = DsnPaperRepository(dsn)
    repository.ensure_schema()
    _seed(repository, QUANT_PAPERS)

    capped = repository.list_chunk_candidates_by_query("quantization", limit=10)
    counts: dict[str, int] = {}
    for row in capped:
        counts[row["arxiv_id"]] = counts.get(row["arxiv_id"], 0) + 1
    assert counts == {"2604.00001": 3, "2604.00002": 3, "2604.00003": 3}

    uncapped = repository.list_chunk_candidates_by_query("quantization", limit=10, per_paper_cap=None)
    assert [row["arxiv_id"] for row in uncapped[:8]] == ["2604.00001"] * 8

    scoped = repository.list_chunk_candidates_by_query("quantization", limit=10, arxiv_id="2604.00002")
    assert len(scoped) == 8

    retriever = PaperRetriever(repository=repository, embedding_client=object(), vector_repository=object())
    contexts = retriever.search_paper_contexts("quantization", limit=10)
    assert len({context["arxiv_id"] for context in contexts}) >= 3
    assert contexts[0]["arxiv_id"] == "2604.00001"


def test_vector_per_paper_cap_spreads_top_k_across_papers(dsn):
    repository = DsnPaperRepository(dsn)
    repository.ensure_schema()
    _seed(repository, QUANT_PAPERS)
    ids = _chunk_ids(dsn)
    rows = []
    for (arxiv_id, index), chunk_id in ids.items():
        paper = int(arxiv_id[-1])
        rows.append(
            {
                "chunk_id": chunk_id,
                "embedding": _basis_vector((0, 1.0), (paper, 0.1 * paper + 0.01 * index)),
                "model_name": EMBEDDING_MODEL,
            }
        )
    DsnVectorRepository(dsn).upsert_paper_embeddings(rows)
    query = _basis_vector((0, 1.0))

    results = DsnVectorRepository(dsn).search_paper_chunks(query, limit=10)
    counts: dict[str, int] = {}
    for row in results:
        counts[row["arxiv_id"]] = counts.get(row["arxiv_id"], 0) + 1
    assert counts == {"2604.00001": 3, "2604.00002": 3, "2604.00003": 3}
    assert [row["arxiv_id"] for row in results[:3]] == ["2604.00001"] * 3

    scoped = DsnVectorRepository(dsn).search_paper_chunks(query, limit=10, arxiv_id="2604.00001")
    assert len(scoped) == 8


def test_lexical_candidate_query_can_use_gin_indexes(dsn):
    repository = DsnPaperRepository(dsn)
    repository.ensure_schema()
    _seed(repository)
    _execute(
        dsn,
        """
        INSERT INTO papers (arxiv_id, title, abstract)
        SELECT 'filler-' || g, 'Filler study ' || g, 'Graph partitioning benchmark ' || g FROM generate_series(1, 500) g
        """,
    )
    _execute(
        dsn,
        """
        INSERT INTO paper_chunks (arxiv_id, chunk_index, chunk_text)
        SELECT 'filler-' || g, 0, 'Sorting networks and caches ' || g FROM generate_series(1, 500) g
        """,
    )
    _execute(dsn, "ANALYZE")

    sql, params = build_lexical_candidates_query("diffusion transformer", limit=5)
    plan = _explain(dsn, sql, params)
    assert "idx_paper_chunks_chunk_vector" in plan
    assert "idx_papers_title_abstract_vector" in plan


def test_chunk_windows_match_single_window_queries(dsn):
    repository = DsnPaperRepository(dsn)
    repository.ensure_schema()
    _seed(repository)

    centers = [("2601.00001", 1), ("2601.00002", 0), ("2601.00001", 0), ("missing", 0)]
    windows = repository.list_chunk_windows(centers, window=1)
    assert windows == [repository.list_chunk_window(arxiv_id, index, window=1) for arxiv_id, index in centers]
    assert [len(window) for window in windows] == [3, 2, 2, 0]

    retriever = PaperRetriever(repository=repository, embedding_client=object(), vector_repository=object())
    contexts = retriever._build_contexts(
        [{"arxiv_id": "2601.00001", "chunk_index": 1, "chunk_id": 1}], adjacency_window=1
    )
    assert [chunk["chunk_index"] for chunk in contexts[0]["context_chunks"]] == [0, 1, 2]
    assert contexts[0]["context_chunks"][2]["content_role"] == "toc"


def _seed_embeddings(dsn: str) -> tuple[dict[tuple[str, int], int], list[float]]:
    repository = DsnPaperRepository(dsn)
    repository.ensure_schema()
    _seed(repository)
    ids = _chunk_ids(dsn)
    vectors = {
        ("2601.00001", 0): _basis_vector((0, 1.0), (1, 0.2)),
        ("2601.00001", 1): _basis_vector((0, 1.0), (1, 1.0)),
        ("2601.00001", 2): _basis_vector((0, 1.0)),
        ("2601.00002", 0): _basis_vector((1, 1.0), (2, 0.1)),
        ("2601.00002", 1): _basis_vector((5, 1.0)),
    }
    rows = [
        {"chunk_id": ids[key], "embedding": vector, "model_name": EMBEDDING_MODEL} for key, vector in vectors.items()
    ]
    rows.append({"chunk_id": ids[("2601.00002", 1)], "embedding": _basis_vector((0, 1.0)), "model_name": "old-model"})
    DsnVectorRepository(dsn).upsert_paper_embeddings(rows)
    return ids, _basis_vector((0, 1.0))


def test_vector_search_two_stage_ranks_by_similarity_then_adjusts(dsn):
    ids, query = _seed_embeddings(dsn)
    stored_models = dict(_execute(dsn, "SELECT chunk_id, model_name FROM paper_embeddings"))
    assert stored_models[ids[("2601.00002", 1)]] == "old-model"

    results = DsnVectorRepository(dsn, openai_embedding_model="old-model").search_paper_chunks(query, limit=5)
    assert [row["chunk_id"] for row in results] == [ids[("2601.00002", 1)]]
    assert results[0]["raw_similarity_score"] == pytest.approx(1.0)

    results = DsnVectorRepository(dsn).search_paper_chunks(query, limit=5)
    assert [row["chunk_id"] for row in results] == [
        ids[("2601.00001", 0)],
        ids[("2601.00001", 1)],
        ids[("2601.00002", 0)],
    ]
    top = results[0]
    assert top["raw_similarity_score"] == pytest.approx(1 / math.sqrt(1.04), abs=1e-6)
    assert top["score"] == pytest.approx(top["raw_similarity_score"] + 0.12, abs=1e-6)
    assert top["score_breakdown"]["section_boost"] == pytest.approx(0.12)
    assert top["paper_title"] == "Diffusion Transformers for Video Generation"
    assert top["paper_abstract"].startswith("We scale")
    assert top["retrieval_method"] == "vector"
    assert results[1]["score_breakdown"]["section_boost"] == pytest.approx(0.03)

    wide = DsnVectorRepository(dsn).search_paper_chunks(query, limit=30)
    assert [row["chunk_id"] for row in wide] == [row["chunk_id"] for row in results]

    filtered = DsnVectorRepository(dsn, vector_min_similarity=0.5).search_paper_chunks(query, limit=5)
    assert [row["chunk_id"] for row in filtered] == [ids[("2601.00001", 0)], ids[("2601.00001", 1)]]

    scoped = DsnVectorRepository(dsn).search_paper_chunks(query, limit=5, arxiv_id="2601.00002")
    assert [row["chunk_id"] for row in scoped] == [ids[("2601.00002", 0)]]


def test_vector_candidate_query_can_use_hnsw_index(dsn):
    _, query = _seed_embeddings(dsn)
    _execute(dsn, "ANALYZE")

    sql, params = build_vector_search_query(query, limit=5, model_name=EMBEDDING_MODEL)
    assert "paper_embeddings_embedding_hnsw" in _explain(dsn, sql, params)

    scoped_sql, scoped_params = build_vector_search_query(
        query, limit=5, arxiv_id="2601.00001", model_name=EMBEDDING_MODEL
    )
    assert "paper_embeddings_embedding_hnsw" not in _explain(dsn, scoped_sql, scoped_params)


def test_pool_reuses_connections_and_rolls_back_on_error(dsn):
    params = {"dsn": dsn}
    with db.get_connection(params) as connection, connection.cursor() as cursor:
        cursor.execute("CREATE TABLE pool_probe (value INTEGER)")
        cursor.execute("SELECT pg_backend_pid()")
        first_pid = cursor.fetchone()[0]

    try:
        with db.get_connection(params) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT pg_backend_pid()")
            assert cursor.fetchone()[0] == first_pid
            cursor.execute("INSERT INTO pool_probe VALUES (1)")
            raise RuntimeError("abort")
    except RuntimeError:
        pass

    try:
        with db.get_connection(params) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM pool_probe")
            assert cursor.fetchone()[0] == 0
            cursor.execute("SELECT pg_backend_pid()")
            assert cursor.fetchone()[0] == first_pid
        assert db.get_pool(params).idle_count == 1

        with pytest.raises(psycopg2.OperationalError):
            with db.get_connection(params) as connection, connection.cursor() as cursor:
                cursor.execute("SELECT pg_terminate_backend(pg_backend_pid())")
        assert db.get_pool(params).idle_count == 0

        with db.get_connection(params) as connection, connection.cursor() as cursor:
            cursor.execute("SELECT pg_backend_pid()")
            assert cursor.fetchone()[0] != first_pid
    finally:
        _execute(dsn, "DROP TABLE IF EXISTS pool_probe")
