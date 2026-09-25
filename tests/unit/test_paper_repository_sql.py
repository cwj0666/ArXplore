from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Any

import pytest

from src.integrations import paper_repository as paper_repository_module
from src.integrations.paper_repository import REFERENCES_SECTION_TITLE_SQL_REGEX, PaperRepository
from src.integrations.pdf_parser.section_roles import is_references_section_title


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


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
    assert not any("DROP " in statement for statement in statements)


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
    assert f"WHEN c.section_title ~* '{REFERENCES_SECTION_TITLE_SQL_REGEX}' THEN -0.08" in sql
    assert "ILIKE '%%References%%'" not in sql
    assert "%" not in REFERENCES_SECTION_TITLE_SQL_REGEX
    assert "'" not in REFERENCES_SECTION_TITLE_SQL_REGEX
    assert sql.count("%s") == len(params)


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
    # PostgreSQL ~* 는 대소문자 무시이며, 이 패턴에 쓰인 문법은 Python re와 의미가 같다.
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


# ---- scripts/backfill_content_roles.py ----

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
