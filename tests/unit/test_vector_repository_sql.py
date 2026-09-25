from __future__ import annotations

import re
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from src.integrations import vector_repository as vector_repository_module
from src.integrations.vector_repository import (
    VectorRepository,
    build_vector_search_query,
    resolve_candidate_limit,
    resolve_hnsw_ef_search,
)


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


class RecordingCursor:
    def __init__(self, fetchall_results: list[list[Any]] | None = None) -> None:
        self.executed: list[tuple[str, Any]] = []
        self._fetchall_results = list(fetchall_results or [])

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self._fetchall_results.pop(0) if self._fetchall_results else []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RecordingConnection:
    def __init__(self, cursor: RecordingCursor) -> None:
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _repository(cursor: RecordingCursor, settings: Any = None) -> VectorRepository:
    repository = VectorRepository(settings=settings if settings is not None else SimpleNamespace())

    @contextmanager
    def fake_connection():
        yield RecordingConnection(cursor)

    repository._connection = fake_connection  # type: ignore[method-assign]
    return repository


def _named_params(sql: str) -> set[str]:
    return set(re.findall(r"%\((\w+)\)s", sql))


QUERY = [0.5, -0.25, 1.0]


@pytest.mark.parametrize("limit,expected", [(1, 40), (5, 40), (10, 40), (11, 44), (30, 120), (0, 40)])
def test_candidate_limit(limit, expected):
    assert resolve_candidate_limit(limit) == expected


@pytest.mark.parametrize("candidates,expected", [(40, None), (44, 44), (1200, 1000)])
def test_hnsw_ef_search(candidates, expected):
    assert resolve_hnsw_ef_search(candidates) == expected


def test_global_query_orders_candidates_by_raw_distance_for_hnsw():
    sql, params = build_vector_search_query(QUERY, limit=5, model_name="text-embedding-3-large")
    normalized = _normalize_sql(sql)

    inner = normalized.split("ranked AS")[0]
    assert (
        "FROM paper_embeddings e WHERE e.model_name = %(model_name)s "
        "ORDER BY e.embedding <=> %(query)s::vector LIMIT %(candidate_limit)s"
    ) in inner
    assert "paper_chunks" not in inner
    assert "MATERIALIZED" not in normalized
    assert (
        "FROM candidates k JOIN paper_chunks c ON c.id = k.chunk_id JOIN papers p ON p.arxiv_id = c.arxiv_id"
        in normalized
    )
    assert "1 - k.distance AS raw_similarity_score" in normalized
    assert normalized.endswith(
        "ORDER BY (raw_similarity_score + content_role_adjustment + section_boost) DESC, id DESC LIMIT %(limit)s"
    )
    assert "WHERE content_role <> 'toc'" in normalized
    assert "min_similarity" not in normalized
    assert params == {
        "query": "[0.500000000000,-0.250000000000,1.000000000000]",
        "candidate_limit": 40,
        "limit": 5,
        "model_name": "text-embedding-3-large",
    }
    assert _named_params(sql) == set(params)


def test_query_keeps_existing_penalties():
    normalized = _normalize_sql(build_vector_search_query(QUERY, limit=5)[0])
    for fragment in (
        "WHEN COALESCE(c.metadata->>'content_role', '') = 'references' THEN -0.22",
        "WHEN COALESCE(c.metadata->>'content_role', '') = 'toc' THEN -0.28",
        "WHEN COALESCE(c.metadata->>'content_role', '') = 'appendix' THEN -0.1",
        "WHEN c.section_title ILIKE 'Abstract' THEN 0.12",
        "WHEN c.section_title ILIKE '%%Introduction%%' THEN 0.09",
        "WHEN c.section_title ILIKE '%%Implementation Details%%' THEN -0.08",
    ):
        assert fragment in normalized


def test_query_without_model_or_scope_has_no_filters():
    sql, params = build_vector_search_query(QUERY, limit=5)
    normalized = _normalize_sql(sql)
    assert "model_name" not in normalized
    assert "FROM paper_embeddings e ORDER BY e.embedding <=> %(query)s::vector LIMIT %(candidate_limit)s" in normalized
    assert _named_params(sql) == set(params) == {"query", "candidate_limit", "limit"}


def test_paper_scoped_query_is_exact_and_materialized():
    sql, params = build_vector_search_query(QUERY, limit=5, arxiv_id="2401.00001", model_name="m")
    normalized = _normalize_sql(sql)

    assert (
        "scoped AS MATERIALIZED ( SELECT e.chunk_id, e.embedding <=> %(query)s::vector AS distance "
        "FROM paper_chunks sc JOIN paper_embeddings e ON e.chunk_id = sc.id "
        "WHERE sc.arxiv_id = %(arxiv_id)s AND e.model_name = %(model_name)s )"
    ) in normalized
    assert "FROM scoped ORDER BY distance ASC, chunk_id DESC LIMIT %(candidate_limit)s" in normalized
    assert "ORDER BY e.embedding <=>" not in normalized
    assert params["arxiv_id"] == "2401.00001"
    assert _named_params(sql) == set(params)


def test_min_similarity_filters_on_raw_similarity():
    sql, params = build_vector_search_query(QUERY, limit=5, min_similarity=0.3)
    assert "AND raw_similarity_score >= %(min_similarity)s" in _normalize_sql(sql)
    assert params["min_similarity"] == 0.3

    sql, params = build_vector_search_query(QUERY, limit=5, min_similarity=0)
    assert "min_similarity" not in sql


def _row(chunk_id: int = 7) -> tuple:
    return (chunk_id, "2401.00001", "Title", None, "text", 3, "1 Introduction", 0.95, 0.86, "body", 0.0, 0.09)


def test_search_maps_rows_to_the_public_shape():
    cursor = RecordingCursor(fetchall_results=[[_row()]])
    results = _repository(cursor).search_paper_chunks(QUERY, limit=5)

    assert results == [
        {
            "chunk_id": 7,
            "arxiv_id": "2401.00001",
            "paper_title": "Title",
            "paper_abstract": "",
            "chunk_text": "text",
            "chunk_index": 3,
            "section_title": "1 Introduction",
            "score": 0.95,
            "similarity_score": 0.95,
            "raw_similarity_score": 0.86,
            "content_role": "body",
            "retrieval_method": "vector",
            "score_breakdown": {
                "raw_similarity_score": 0.86,
                "content_role_adjustment": 0.0,
                "section_boost": 0.09,
            },
        }
    ]
    assert len(cursor.executed) == 1


def test_search_uses_settings_model_and_min_similarity():
    cursor = RecordingCursor()
    settings = SimpleNamespace(openai_embedding_model="text-embedding-3-large", vector_min_similarity="0.25")
    _repository(cursor, settings).search_paper_chunks(QUERY, limit=5)

    _, params = cursor.executed[-1]
    assert params["model_name"] == "text-embedding-3-large"
    assert params["min_similarity"] == 0.25


def test_search_tolerates_settings_without_optional_fields():
    cursor = RecordingCursor()
    _repository(cursor, SimpleNamespace(vector_min_similarity="bad")).search_paper_chunks(QUERY, limit=5)
    _, params = cursor.executed[-1]
    assert "model_name" not in params
    assert "min_similarity" not in params


def test_search_raises_ef_search_only_for_large_global_candidate_sets():
    cursor = RecordingCursor()
    repository = _repository(cursor)

    repository.search_paper_chunks(QUERY, limit=30)
    assert cursor.executed[0] == ("SELECT set_config('hnsw.ef_search', %s, true)", ("120",))
    assert cursor.executed[1][1]["candidate_limit"] == 120

    cursor.executed.clear()
    repository.search_paper_chunks(QUERY, limit=5)
    assert len(cursor.executed) == 1

    cursor.executed.clear()
    repository.search_paper_chunks(QUERY, limit=30, arxiv_id="2401.00001")
    assert len(cursor.executed) == 1


def test_upsert_embeddings_uses_execute_values_and_deduplicates(monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_execute_values(cursor, sql, argslist, template=None, page_size=100, fetch=False):
        calls.append({"sql": sql, "args": list(argslist), "template": template, "page_size": page_size})

    monkeypatch.setattr(vector_repository_module, "execute_values", fake_execute_values)
    cursor = RecordingCursor()
    _repository(cursor).upsert_paper_embeddings(
        [
            {"chunk_id": 1, "embedding": [0.1, 0.2], "model_name": "m"},
            {"chunk_id": "2", "embedding": [0.3, 0.4], "model_name": "m"},
            {"chunk_id": 1, "embedding": [0.5, 0.6], "model_name": "m2"},
        ]
    )

    assert len(calls) == 1
    call = calls[0]
    normalized = _normalize_sql(call["sql"])
    assert "INSERT INTO paper_embeddings (chunk_id, embedding, model_name, updated_at) VALUES %s" in normalized
    assert "ON CONFLICT (chunk_id) DO UPDATE SET" in normalized
    assert call["template"] == "(%s, %s::vector, %s, NOW())"
    assert call["args"] == [
        (1, "[0.500000000000,0.600000000000]", "m2"),
        (2, "[0.300000000000,0.400000000000]", "m"),
    ]


def test_upsert_embeddings_skips_empty_input(monkeypatch):
    monkeypatch.setattr(vector_repository_module, "execute_values", pytest.fail)
    repository = VectorRepository(settings=SimpleNamespace())
    repository._connection = pytest.fail  # type: ignore[method-assign]
    repository.upsert_paper_embeddings([])


def test_constructor_opens_no_connection(monkeypatch):
    monkeypatch.setattr(vector_repository_module, "get_connection", pytest.fail)
    VectorRepository(settings=SimpleNamespace())
