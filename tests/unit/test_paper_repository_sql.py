from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from typing import Any

import psycopg2
import pytest

from src.integrations import paper_repository as paper_repository_module
from src.integrations.paper_repository import (
    PER_PAPER_CANDIDATE_CAP,
    REFERENCES_SECTION_TITLE_SQL_REGEX,
    STRICT_MATCH_BONUS,
    PaperRepository,
    build_lexical_candidates_query,
    escape_like,
)
from src.integrations.paper_retriever import PaperRetriever
from src.integrations.pdf_parser.section_roles import is_references_section_title


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


def _named_params(sql: str) -> set[str]:
    return set(re.findall(r"%\((\w+)\)s", sql))


class RecordingCursor:
    def __init__(self, *, fetchone_results: list[Any] | None = None, fetchall_results: list[list[Any]] | None = None):
        self.executed: list[tuple[str, Any]] = []
        self._fetchone_results = list(fetchone_results or [])
        self._fetchall_results = list(fetchall_results or [])

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self._fetchone_results.pop(0) if self._fetchone_results else None

    def fetchall(self):
        return self._fetchall_results.pop(0) if self._fetchall_results else []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RecordingConnection:
    def __init__(self, cursor: RecordingCursor) -> None:
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _repository_with_cursor(cursor: RecordingCursor) -> PaperRepository:
    repository = PaperRepository(settings=object())

    @contextmanager
    def fake_connection():
        yield RecordingConnection(cursor)

    repository._connection = fake_connection  # type: ignore[method-assign]
    return repository


def test_constructor_runs_no_ddl(monkeypatch):
    def fail_connect(**kwargs):
        raise AssertionError("PaperRepository() must not open a DB connection")

    monkeypatch.setattr(paper_repository_module.psycopg2, "connect", fail_connect)
    PaperRepository(settings=object())
    assert not hasattr(PaperRepository, "_ensure_schema")


def test_ensure_schema_is_explicit_and_never_drops_tables():
    cursor = RecordingCursor()
    _repository_with_cursor(cursor).ensure_schema()

    statements = [_normalize_sql(sql).upper() for sql, _ in cursor.executed]
    assert any("CREATE TABLE IF NOT EXISTS PAPERS" in statement for statement in statements)
    assert any("CREATE TABLE IF NOT EXISTS PAPER_EMBEDDINGS" in statement for statement in statements)
    assert not any("DROP TABLE" in statement for statement in statements)
    assert [statement for statement in statements if "DROP " in statement] == [
        "DROP INDEX IF EXISTS IDX_PAPER_CHUNKS_FTS"
    ]


def test_ensure_schema_adds_generated_search_vectors_and_indexes():
    cursor = RecordingCursor()
    _repository_with_cursor(cursor).ensure_schema()
    statements = [_normalize_sql(sql) for sql, _ in cursor.executed]

    assert (
        "ALTER TABLE papers ADD COLUMN IF NOT EXISTS title_abstract_vector tsvector GENERATED ALWAYS AS ("
        "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
        "setweight(to_tsvector('english', coalesce(abstract, '')), 'B')) STORED"
    ) in statements
    assert (
        "ALTER TABLE paper_chunks ADD COLUMN IF NOT EXISTS chunk_vector tsvector GENERATED ALWAYS AS ("
        "setweight(to_tsvector('english', coalesce(chunk_text, '')), 'C')) STORED"
    ) in statements
    assert (
        "CREATE INDEX IF NOT EXISTS idx_papers_title_abstract_vector ON papers USING GIN (title_abstract_vector)"
        in statements
    )
    assert (
        "CREATE INDEX IF NOT EXISTS idx_paper_chunks_chunk_vector ON paper_chunks USING GIN (chunk_vector)"
        in statements
    )
    assert "ALTER TABLE paper_fulltexts ADD COLUMN IF NOT EXISTS content_hash TEXT NULL;" in statements

    hnsw = (
        "CREATE INDEX IF NOT EXISTS paper_embeddings_embedding_hnsw "
        "ON paper_embeddings USING hnsw (embedding vector_cosine_ops)"
    )
    position = statements.index(hnsw)
    assert statements[position - 1] == "SAVEPOINT paper_embeddings_hnsw"
    assert statements[position + 1] == "RELEASE SAVEPOINT paper_embeddings_hnsw"
    assert statements.index(hnsw) > statements.index(
        next(statement for statement in statements if "CREATE TABLE IF NOT EXISTS paper_embeddings" in statement)
    )


class HnswUnsupportedCursor(RecordingCursor):
    def execute(self, sql, params=None):
        super().execute(sql, params)
        if "USING hnsw" in sql:
            raise psycopg2.errors.UndefinedObject('access method "hnsw" does not exist')


def test_ensure_schema_continues_without_hnsw_support(caplog):
    cursor = HnswUnsupportedCursor()
    with caplog.at_level("WARNING", logger=paper_repository_module.__name__):
        _repository_with_cursor(cursor).ensure_schema()

    statements = [_normalize_sql(sql) for sql, _ in cursor.executed]
    position = next(index for index, statement in enumerate(statements) if "USING hnsw" in statement)
    assert statements[position + 1] == "ROLLBACK TO SAVEPOINT paper_embeddings_hnsw"
    assert "RELEASE SAVEPOINT paper_embeddings_hnsw" not in statements
    assert any(
        "CREATE TABLE IF NOT EXISTS paper_ai_detailed_summaries" in statement for statement in statements[position:]
    )
    assert "HNSW" in caplog.text


def test_save_paper_preserves_enrichment_on_conflict():
    cursor = RecordingCursor()
    _repository_with_cursor(cursor).save_paper(
        {
            "arxiv_id": "2604.00001",
            "title": "T",
            "abstract": "A",
            "primary_category": None,
            "categories": [],
            "pdf_url": "https://arxiv.org/pdf/2604.00001.pdf",
            "upvotes": 3,
            "source": "hf_daily_papers_raw",
        }
    )

    sql, params = cursor.executed[0]
    normalized = _normalize_sql(sql)
    assert "primary_category = COALESCE(EXCLUDED.primary_category, papers.primary_category)" in normalized
    assert (
        "categories = CASE WHEN EXCLUDED.categories = '[]'::jsonb THEN papers.categories ELSE EXCLUDED.categories END"
        in normalized
    )
    for column in ("published_at", "updated_at", "github_url", "github_stars", "citation_count"):
        assert f"{column} = COALESCE(EXCLUDED.{column}, papers.{column})" in normalized
    assert "upvotes = COALESCE(%(upvotes)s::integer, papers.upvotes)" in normalized
    assert "COALESCE(%(upvotes)s::integer, 0)" in normalized
    for column in ("title", "abstract", "pdf_url", "authors"):
        assert f"{column} = EXCLUDED.{column}," in normalized
    assert params["primary_category"] is None
    assert params["categories"].adapted == []
    assert params["upvotes"] == 3


def test_save_paper_passes_null_upvotes_when_missing():
    cursor = RecordingCursor()
    _repository_with_cursor(cursor).save_paper({"arxiv_id": "2604.00001", "title": "T"})
    _, params = cursor.executed[0]
    assert params["upvotes"] is None
    assert params["published_at"] is None


def test_get_paper_fulltext_source():
    cursor = RecordingCursor(fetchone_results=[("layout_pdf",)])
    repository = _repository_with_cursor(cursor)
    assert repository.get_paper_fulltext_source("2604.00001") == "layout_pdf"
    sql, params = cursor.executed[0]
    assert "SELECT source FROM paper_fulltexts WHERE arxiv_id = %s" in _normalize_sql(sql)
    assert params == ("2604.00001",)

    assert _repository_with_cursor(RecordingCursor()).get_paper_fulltext_source("missing") is None


def test_lexical_query_uses_full_title_references_rule():
    cursor = RecordingCursor(fetchall_results=[[]])
    _repository_with_cursor(cursor).list_chunk_candidates_by_query("direct preference optimization", limit=3)

    sql, params = cursor.executed[0]
    assert f"WHEN section_title ~* '{REFERENCES_SECTION_TITLE_SQL_REGEX}' THEN -0.08" in sql
    assert "ILIKE '%%References%%'" not in sql
    assert "%" not in REFERENCES_SECTION_TITLE_SQL_REGEX
    assert "'" not in REFERENCES_SECTION_TITLE_SQL_REGEX
    assert "%s" not in sql
    assert set(re.findall(r"%\((\w+)\)s", sql)) == set(params)


def test_lexical_candidates_come_from_indexed_generated_columns():
    sql, params = build_lexical_candidates_query("retrieval-augmented generation", limit=7)
    normalized = _normalize_sql(sql)
    matched = normalized.split("ranked AS")[0]

    assert "WHERE c.chunk_vector @@ (SELECT any_term FROM query_terms)" in matched
    assert "WHERE p.title_abstract_vector @@ (SELECT any_term FROM query_terms)" in matched
    assert " UNION " in matched
    assert "ILIKE" not in matched
    assert "to_tsvector('english', coalesce(" not in normalized
    assert ("tsvector_to_array(to_tsvector('english', %(query)s) || to_tsvector('english', %(fts_query)s))") in matched
    assert params == {
        "query": "retrieval-augmented generation",
        "fts_query": "retrieval augmented generation",
        "like_pattern": "%retrieval-augmented generation%",
        "limit": 7,
        "partial_pool": 500,
        "per_paper_cap": 3,
    }


def test_lexical_scoring_is_two_tier_strict_then_coverage():
    normalized = _normalize_sql(build_lexical_candidates_query("dpo", limit=5)[0])

    assert (
        "CROSS JOIN LATERAL ( SELECT p.title_abstract_vector || c.chunk_vector AS search_vector OFFSET 0 ) v"
    ) in normalized
    assert (
        "( v.search_vector @@ websearch_to_tsquery('english', %(query)s) "
        "OR v.search_vector @@ plainto_tsquery('english', %(fts_query)s) ) AS strict_match"
    ) in normalized
    assert (
        "(length(v.search_vector) - length(ts_delete(v.search_vector, q.lexemes)))::double precision "
        "/ GREATEST(q.lexeme_count, 1) AS coverage"
    ) in normalized
    assert (
        "CASE WHEN strict_match THEN ts_rank_cd(search_vector, websearch_to_tsquery('english', %(query)s)) "
        "+ 0.65 * ts_rank_cd(search_vector, plainto_tsquery('english', %(fts_query)s)) + 1.0 "
        "ELSE coverage * ts_rank_cd(search_vector, any_term, 32) END AS fts_score"
    ) in normalized
    assert "@@ websearch_to_tsquery('english', %(query)s) OR" not in normalized.split("AS strict_match")[1]
    assert "WHERE (p.title_abstract_vector || c.chunk_vector) @@" not in normalized
    assert "array_agg(lexeme) AS lexemes, count(*) AS lexeme_count" in normalized
    assert STRICT_MATCH_BONUS == 1.0


def test_lexical_scores_only_strict_rows_and_a_bounded_partial_pool():
    sql, params = build_lexical_candidates_query("token level credit assignment", limit=50)
    normalized = _normalize_sql(sql)

    assert params["partial_pool"] == 1000
    assert build_lexical_candidates_query("x", limit=5)[1]["partial_pool"] == 500
    screened = normalized.split("screened AS MATERIALIZED (", 1)[1].split("pool AS (", 1)[0]
    assert "ts_rank_cd" not in screened
    assert "ILIKE" not in screened
    assert (
        "pool AS ( SELECT chunk_id, strict_match, coverage FROM screened WHERE strict_match UNION ALL "
        "( SELECT chunk_id, strict_match, coverage FROM screened WHERE NOT strict_match "
        "ORDER BY coverage DESC, chunk_id DESC LIMIT %(partial_pool)s ) )"
    ) in normalized
    assert "FROM pool k JOIN paper_chunks c ON c.id = k.chunk_id" in normalized


def test_lexical_scoring_keeps_bonuses_and_breakdown():
    normalized = _normalize_sql(build_lexical_candidates_query("dpo", limit=5)[0])

    assert (
        "WHEN NOT strict_match THEN 0 WHEN paper_title ILIKE %(like_pattern)s THEN 0.45 "
        "WHEN paper_abstract ILIKE %(like_pattern)s THEN 0.2 WHEN chunk_text ILIKE %(like_pattern)s THEN 0.15"
    ) in normalized
    assert (
        "(fts_score + ilike_bonus + content_role_adjustment + section_boost + structural_adjustment) AS score"
        in normalized
    )
    assert "WHERE score > 0.01" in normalized
    assert "WHERE coalesce(c.metadata->>'content_role', '') <> 'toc'" in normalized
    assert (
        "t.fts_score, t.ilike_bonus, t.content_role_adjustment, t.section_boost, t.structural_adjustment, "
        "t.score, t.strict_match, t.coverage FROM top_hits t"
    ) in normalized
    assert normalized.endswith("ORDER BY t.score DESC, t.chunk_id DESC")


def test_lexical_query_caps_chunks_per_paper():
    sql, params = build_lexical_candidates_query("quantization", limit=30)
    normalized = _normalize_sql(sql)

    assert params["per_paper_cap"] == PER_PAPER_CANDIDATE_CAP == 3
    assert (
        "row_number() OVER (PARTITION BY arxiv_id ORDER BY score DESC, chunk_id DESC) AS paper_rank "
        "FROM scored WHERE score > 0.01"
    ) in normalized
    assert (
        "FROM kept WHERE paper_rank <= %(per_paper_cap)s ORDER BY score DESC, chunk_id DESC LIMIT %(limit)s"
    ) in normalized
    assert "scored AS MATERIALIZED" in normalized
    assert _named_params(sql) == set(params)


@pytest.mark.parametrize(
    "kwargs",
    [{"arxiv_id": "2401.00001"}, {"per_paper_cap": None}, {"per_paper_cap": 0}],
    ids=["scoped", "disabled", "zero"],
)
def test_lexical_per_paper_cap_is_skipped(kwargs):
    sql, params = build_lexical_candidates_query("quantization", limit=10, **kwargs)
    normalized = _normalize_sql(sql)

    assert "per_paper_cap" not in params
    assert "paper_rank <=" not in normalized
    assert "FROM kept ORDER BY score DESC, chunk_id DESC LIMIT %(limit)s" in normalized
    assert _named_params(sql) == set(params)


def test_lexical_query_scopes_both_candidate_branches():
    sql, params = build_lexical_candidates_query("loss", limit=5, arxiv_id="2401.00001")
    normalized = _normalize_sql(sql)
    assert "WHERE c.chunk_vector @@ (SELECT any_term FROM query_terms) AND c.arxiv_id = %(arxiv_id)s" in normalized
    assert (
        "WHERE p.title_abstract_vector @@ (SELECT any_term FROM query_terms) AND p.arxiv_id = %(arxiv_id)s"
        in normalized
    )
    assert params["arxiv_id"] == "2401.00001"


@pytest.mark.parametrize(
    "raw,escaped",
    [("plain text", "plain text"), ("50%", "50\\%"), ("snake_case", "snake\\_case"), ("a\\b", "a\\\\b")],
)
def test_like_pattern_is_escaped(raw, escaped):
    assert escape_like(raw) == escaped
    assert build_lexical_candidates_query(raw, limit=1)[1]["like_pattern"] == f"%{escaped}%"


def test_blank_lexical_query_skips_the_database():
    repository = PaperRepository(settings=object())
    repository._connection = pytest.fail  # type: ignore[method-assign]
    assert repository.list_chunk_candidates_by_query("   ", limit=3) == []
    assert build_lexical_candidates_query("", limit=3) is None


def test_lexical_rows_keep_the_public_shape():
    row = (
        "c1",
        "2401.00001",
        "Title",
        "Abstract",
        "chunk",
        4,
        "Method",
        "body",
        0.3,
        0.15,
        0.0,
        0.04,
        0.0,
        0.49,
        True,
        1.0,
    )
    cursor = RecordingCursor(fetchall_results=[[row]])
    result = _repository_with_cursor(cursor).list_chunk_candidates_by_query("chunk", limit=3)[0]

    assert {
        key: result[key] for key in ("chunk_id", "arxiv_id", "chunk_text", "section_title", "content_role", "score")
    } == {
        "chunk_id": "c1",
        "arxiv_id": "2401.00001",
        "chunk_text": "chunk",
        "section_title": "Method",
        "content_role": "body",
        "score": 0.49,
    }
    assert result["score_breakdown"] == {
        "fts_score": 0.3,
        "ilike_bonus": 0.15,
        "content_role_adjustment": 0.0,
        "section_boost": 0.04,
        "structural_adjustment": 0.0,
        "coverage": 1.0,
        "strict_match": True,
    }
    assert result["retrieval_method"] == "lexical"


def test_lexical_rows_report_partial_matches():
    row = ("c2", "2401.00002", "T", "A", "chunk", 1, "Method", "body", 0.42, 0, 0, 0.04, 0, 0.46, False, 0.6)
    cursor = RecordingCursor(fetchall_results=[[row]])
    result = _repository_with_cursor(cursor).list_chunk_candidates_by_query("a b c d e", limit=3)[0]

    assert result["score_breakdown"]["strict_match"] is False
    assert result["score_breakdown"]["coverage"] == 0.6
    assert result["score"] == 0.46


def test_repository_passes_per_paper_cap_to_the_query():
    cursor = RecordingCursor(fetchall_results=[[], []])
    repository = _repository_with_cursor(cursor)

    repository.list_chunk_candidates_by_query("loss", limit=30)
    repository.list_chunk_candidates_by_query("loss", limit=30, per_paper_cap=None)

    assert cursor.executed[0][1]["per_paper_cap"] == 3
    assert "per_paper_cap" not in cursor.executed[1][1]


def test_save_paper_chunks_uses_one_bulk_insert(monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_execute_values(cursor, sql, argslist, template=None, page_size=100, fetch=False):
        calls.append({"sql": sql, "args": list(argslist), "template": template})

    monkeypatch.setattr(paper_repository_module, "execute_values", fake_execute_values)
    cursor = RecordingCursor(fetchall_results=[[(0, _md5("a")), (1, _md5("old"))]])
    replaced = _repository_with_cursor(cursor).save_paper_chunks(
        "2401.00001",
        [
            {
                "chunk_index": 0,
                "chunk_text": "a",
                "section_title": "Abstract",
                "token_count": 1,
                "metadata": {"content_role": "body"},
            },
            {"chunk_index": "1", "chunk_text": "b\ud800"},
        ],
    )

    assert replaced is True
    assert cursor.executed == [
        (
            "SELECT chunk_index, md5(chunk_text) FROM paper_chunks WHERE arxiv_id = %s ORDER BY chunk_index ASC",
            ("2401.00001",),
        ),
        ("DELETE FROM paper_chunks WHERE arxiv_id = %s", ("2401.00001",)),
    ]
    assert len(calls) == 1
    assert (
        "INSERT INTO paper_chunks (arxiv_id, chunk_index, chunk_text, section_title, token_count, metadata, updated_at) VALUES %s"
        in _normalize_sql(calls[0]["sql"])
    )
    assert calls[0]["template"] == "(%s, %s, %s, %s, %s, %s, NOW())"
    first, second = calls[0]["args"]
    assert first[:5] == ("2401.00001", 0, "a", "Abstract", 1)
    assert first[5].adapted == {"content_role": "body"}
    assert second[:5] == ("2401.00001", 1, "b", None, 0)


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def test_save_paper_chunks_keeps_rows_when_texts_are_unchanged(monkeypatch):
    calls: list[dict[str, Any]] = []

    def fake_execute_values(cursor, sql, argslist, template=None, page_size=100, fetch=False):
        calls.append({"sql": sql, "args": list(argslist), "template": template})

    monkeypatch.setattr(paper_repository_module, "execute_values", fake_execute_values)
    cursor = RecordingCursor(fetchall_results=[[(0, _md5("a")), (1, _md5("b"))]])
    replaced = _repository_with_cursor(cursor).save_paper_chunks(
        "2401.00001",
        [
            {"chunk_index": 1, "chunk_text": "b", "section_title": "Method", "metadata": {"content_role": "body"}},
            {"chunk_index": 0, "chunk_text": "a", "section_title": "Abstract", "token_count": 1},
        ],
    )

    assert replaced is False
    assert len(cursor.executed) == 1
    assert not any("DELETE" in sql for sql, _ in cursor.executed)
    assert len(calls) == 1
    normalized = _normalize_sql(calls[0]["sql"])
    assert normalized.startswith("UPDATE paper_chunks AS c SET section_title = v.section_title")
    assert "INSERT" not in normalized
    assert "c.metadata <> v.metadata" in normalized
    assert [args[:4] for args in calls[0]["args"]] == [("2401.00001", 0, "Abstract", 1), ("2401.00001", 1, "Method", 0)]


@pytest.mark.parametrize(
    "existing",
    [
        [],
        [(0, "x")],
        [(0, None), (1, None), (2, None)],
    ],
)
def test_save_paper_chunks_replaces_when_texts_or_count_differ(monkeypatch, existing):
    monkeypatch.setattr(paper_repository_module, "execute_values", lambda *args, **kwargs: None)
    existing = [(index, _md5("a") if digest is None else digest) for index, digest in existing]
    cursor = RecordingCursor(fetchall_results=[existing])
    replaced = _repository_with_cursor(cursor).save_paper_chunks("2401.00001", [{"chunk_index": 0, "chunk_text": "a"}])

    assert replaced is True
    assert cursor.executed[-1] == ("DELETE FROM paper_chunks WHERE arxiv_id = %s", ("2401.00001",))


def test_get_paper_fulltext_state_includes_chunk_count():
    cursor = RecordingCursor(fetchone_results=[("layout_pdf", "abc", 3)])
    assert _repository_with_cursor(cursor).get_paper_fulltext_state("2604.00001") == {
        "source": "layout_pdf",
        "content_hash": "abc",
        "chunk_count": 3,
    }
    sql, params = cursor.executed[0]
    normalized = _normalize_sql(sql)
    assert "SELECT f.source, f.content_hash" in normalized
    assert "(SELECT COUNT(*) FROM paper_chunks c WHERE c.arxiv_id = f.arxiv_id)" in normalized
    assert params == ("2604.00001",)
    assert _repository_with_cursor(RecordingCursor()).get_paper_fulltext_state("missing") is None


def test_update_paper_fulltext_content_hash():
    cursor = RecordingCursor()
    _repository_with_cursor(cursor).update_paper_fulltext_content_hash("2604.00001", "abc")
    assert cursor.executed == [
        ("UPDATE paper_fulltexts SET content_hash = %s WHERE arxiv_id = %s", ("abc", "2604.00001"))
    ]


def test_save_paper_fulltext_stores_content_hash():
    cursor = RecordingCursor()
    _repository_with_cursor(cursor).save_paper_fulltext("2604.00001", text="t", source="pdf", content_hash="abc")
    sql, params = cursor.executed[0]
    assert "content_hash = EXCLUDED.content_hash" in _normalize_sql(sql)
    assert params[-1] == "abc"


def test_list_chunk_windows_fetches_all_windows_in_one_query():
    rows = [
        (1, 11, "A", 0, "a0", "Abstract", 3, {"content_role": "body"}, None),
        (1, 12, "A", 1, "a1", "Intro", 4, None, None),
        (3, 12, "A", 1, "a1", "Intro", 4, None, None),
    ]
    cursor = RecordingCursor(fetchall_results=[rows])
    windows = _repository_with_cursor(cursor).list_chunk_windows([("A", 0), ("B", 5), ("A", 2)], window=1)

    assert len(cursor.executed) == 1
    sql, params = cursor.executed[0]
    normalized = _normalize_sql(sql)
    assert (
        "FROM unnest(%(arxiv_ids)s::text[], %(center_indexes)s::integer[]) WITH ORDINALITY AS w(arxiv_id, center_index, ord)"
        in normalized
    )
    assert (
        "c.chunk_index BETWEEN GREATEST(0, w.center_index - %(window)s) AND w.center_index + %(window)s"
    ) in normalized
    assert normalized.endswith("ORDER BY w.ord ASC, c.chunk_index ASC")
    assert params == {"arxiv_ids": ["A", "B", "A"], "center_indexes": [0, 5, 2], "window": 1}
    assert [[chunk["chunk_id"] for chunk in window] for window in windows] == [[11, 12], [], [12]]
    assert windows[0][0] == {
        "chunk_id": 11,
        "arxiv_id": "A",
        "chunk_index": 0,
        "chunk_text": "a0",
        "section_title": "Abstract",
        "token_count": 3,
        "metadata": {"content_role": "body"},
        "updated_at": None,
    }
    assert windows[0][1]["metadata"] == {}


def test_list_chunk_windows_without_centers_skips_the_database():
    repository = PaperRepository(settings=object())
    repository._connection = pytest.fail  # type: ignore[method-assign]
    assert repository.list_chunk_windows([], window=1) == []


class WindowRepository:
    def __init__(self) -> None:
        self.bulk_calls: list[tuple[list[tuple[str, int]], int]] = []

    def list_chunk_windows(self, centers, *, window):
        self.bulk_calls.append((list(centers), window))
        return [
            [{"chunk_text": f"{arxiv_id}:{index}", "metadata": {"content_role": "body"}}] for arxiv_id, index in centers
        ]

    def list_chunk_window(self, arxiv_id, center_chunk_index, *, window=1):
        raise AssertionError("per-hit window query must not be used")


class SingleWindowRepository:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def list_chunk_window(self, arxiv_id, center_chunk_index, *, window=1):
        self.calls.append((arxiv_id, center_chunk_index, window))
        return [{"chunk_text": f"{arxiv_id}:{center_chunk_index}", "metadata": {"content_role": "body"}}]


def _hits() -> list[dict[str, Any]]:
    return [
        {"chunk_id": 1, "arxiv_id": "A", "chunk_index": "3", "chunk_text": "x"},
        {"chunk_id": 2, "arxiv_id": "B", "chunk_index": 0, "chunk_text": "y"},
    ]


def test_build_contexts_uses_a_single_window_query():
    repository = WindowRepository()
    retriever = PaperRetriever(repository=repository, embedding_client=object(), vector_repository=object())
    contexts = retriever._build_contexts(_hits(), adjacency_window=2)

    assert repository.bulk_calls == [([("A", 3), ("B", 0)], 2)]
    assert [context["context_text"] for context in contexts] == ["A:3", "B:0"]
    assert contexts[0]["context_chunks"] == [
        {"chunk_text": "A:3", "metadata": {"content_role": "body"}, "section_title": "", "content_role": "body"}
    ]
    assert retriever._build_contexts([], adjacency_window=1) == []


def test_build_contexts_output_is_identical_with_single_window_repositories():
    bulk = PaperRetriever(repository=WindowRepository(), embedding_client=object(), vector_repository=object())
    single_repository = SingleWindowRepository()
    single = PaperRetriever(repository=single_repository, embedding_client=object(), vector_repository=object())

    assert bulk._build_contexts(_hits(), adjacency_window=1) == single._build_contexts(_hits(), adjacency_window=1)
    assert single_repository.calls == [("A", 3, 1), ("B", 0, 1)]


AGREEING_TITLES = [
    "References",
    "REFERENCES",
    "7 References",
    "7. References",
    "7.1 References",
    "VII. References",
    "VII References",
    "vii. references",
    "Bibliography",
    "Works Cited",
    "Literature Cited",
    "References and Notes",
    "References:",
    "  References  ",
    "Reference Model",
    "References Model Selection",
    "Direct Preference Optimization",
    "Preference Data Collection",
    "A References",
    "Appendix A References",
    "Reference-free evaluation",
    "Referenced works",
    "Bibliography of Prior Work",
    "Introduction",
]


@pytest.mark.parametrize("title", AGREEING_TITLES)
def test_sql_regex_agrees_with_parser_rule(title):
    sql_match = re.search(REFERENCES_SECTION_TITLE_SQL_REGEX, title, re.IGNORECASE) is not None
    assert sql_match == is_references_section_title(title)


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Reference Model", False),
        ("Direct Preference Optimization", False),
        ("7 References", True),
        ("Bibliography", True),
    ],
)
def test_sql_regex_expected_examples(title, expected):
    assert (re.search(REFERENCES_SECTION_TITLE_SQL_REGEX, title, re.IGNORECASE) is not None) is expected


from scripts import backfill_content_roles as backfill  # noqa: E402


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Direct Preference Optimization", True),
        ("Reference Model", True),
        ("3.2 Reference-free Reward", True),
        ("Bibliography of Prior Work", True),
        ("References", False),
        ("7 References", False),
        ("Bibliography", False),
        ("Introduction", False),
        (None, False),
        ("Full Text", False),
    ],
)
def test_backfill_should_reclassify(title, expected):
    assert backfill.should_reclassify(title) is expected


def test_backfill_candidate_sql_text():
    normalized = _normalize_sql(backfill.CANDIDATE_SQL)
    assert "c.metadata->>'content_role' = 'references'" in normalized
    assert "(c.section_title ILIKE %s OR c.section_title ILIKE %s)" in normalized
    assert "LEFT JOIN paper_embeddings e ON e.chunk_id = c.id" in normalized
    assert backfill.CANDIDATE_PARAMS == ("%reference%", "bibliography%")
    assert "DELETE" not in backfill.CANDIDATE_SQL.upper()


def test_backfill_update_sql_only_touches_metadata():
    normalized = _normalize_sql(backfill.UPDATE_SQL)
    assert normalized.startswith("UPDATE paper_chunks SET")
    assert "jsonb_build_object('content_role', 'body', 'content_role_backfilled_from', 'references')" in normalized
    assert "WHERE id = ANY(%s) AND metadata->>'content_role' = 'references'" in normalized
    for sql in (backfill.UPDATE_SQL, backfill.NEEDS_EMBEDDING_SQL, backfill.CANDIDATE_SQL):
        assert "DELETE" not in sql.upper()
        assert "paper_embeddings" not in sql.split("FROM")[0]


def test_backfill_select_and_summarize_targets():
    rows = [
        (1, "p1", "Direct Preference Optimization", False),
        (2, "p1", "Direct Preference Optimization", True),
        (3, "p1", "References", False),
        (4, "p2", "Reference Model", False),
    ]
    targets = backfill.select_targets(rows)
    assert [target["chunk_id"] for target in targets] == [1, 2, 4]
    summary = backfill.summarize_targets(targets)
    assert summary["papers"] == 2
    assert summary["chunks"] == 3
    assert summary["embeddings"] == 1


def test_backfill_apply_reclassification_executes_update_then_count():
    cursor = RecordingCursor(fetchall_results=[[(1,), (4,)]], fetchone_results=[(2,)])
    applied = backfill.apply_reclassification(cursor, [1, 4])
    assert applied == {"updated_chunks": 2, "needs_embedding": 2}
    assert cursor.executed[0] == (backfill.UPDATE_SQL, ([1, 4],))
    assert cursor.executed[1] == (backfill.NEEDS_EMBEDDING_SQL, ([1, 4],))


def test_backfill_apply_reclassification_noop_for_empty_ids():
    cursor = RecordingCursor()
    assert backfill.apply_reclassification(cursor, []) == {"updated_chunks": 0, "needs_embedding": 0}
    assert cursor.executed == []


def _connection_factory(cursor: RecordingCursor, holder: dict[str, Any]):
    @contextmanager
    def factory():
        connection = RecordingConnection(cursor)
        holder["connection"] = connection
        yield connection

    return factory


def test_backfill_main_dry_run_does_not_write(capsys):
    cursor = RecordingCursor(fetchall_results=[[(1, "p1", "Direct Preference Optimization", False)]])
    holder: dict[str, Any] = {}
    assert backfill.main([], connection_factory=_connection_factory(cursor, holder)) == 0

    assert len(cursor.executed) == 1
    assert cursor.executed[0][0] == backfill.CANDIDATE_SQL
    assert holder["connection"].committed is False
    assert holder["connection"].rolled_back is True
    output = capsys.readouterr().out
    assert "[DRY-RUN]" in output
    assert "논문 1편, 청크 1개, 기존 임베딩 0개" in output


def test_backfill_main_apply_updates_and_commits(capsys):
    cursor = RecordingCursor(
        fetchall_results=[[(1, "p1", "Direct Preference Optimization", False), (3, "p1", "References", False)], [(1,)]],
        fetchone_results=[(1,)],
    )
    holder: dict[str, Any] = {}
    assert backfill.main(["--apply"], connection_factory=_connection_factory(cursor, holder)) == 0

    assert cursor.executed[1] == (backfill.UPDATE_SQL, ([1],))
    assert holder["connection"].committed is True
    assert "임베딩이 필요한 청크 1개" in capsys.readouterr().out
