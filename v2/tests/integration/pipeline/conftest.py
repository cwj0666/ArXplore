import importlib.util
import sys
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row

from arxplore.db.migrate import MigrationState, fetch_applied, load_migrations, migrate, migration_status
from arxplore.db.pool import Pool, create_pool, open_pool
from arxplore.domain.jobs import QueuePolicy
from arxplore.parsing.models import FulltextParseResult, Section
from arxplore.parsing.quality import fulltext_quality_metrics

PIPELINE_TABLES = (
    "papers",
    "paper_fulltexts",
    "paper_chunks",
    "paper_embeddings",
    "raw_daily_papers",
    "pipeline_state",
    "prepare_jobs",
)

CLEAR_TABLES_SQL = """
DELETE FROM papers;
DELETE FROM raw_daily_papers;
DELETE FROM pipeline_state;
DELETE FROM prepare_jobs;
ALTER SEQUENCE paper_chunks_id_seq RESTART;
ALTER SEQUENCE prepare_jobs_id_seq RESTART;
"""

FAKE_SERVER_RELATIVE_PATH = Path("tests/fakes/openai_server.py")


async def _schema_is_current(connection: psycopg.AsyncConnection[Any]) -> bool:
    statuses = migration_status(load_migrations(), await fetch_applied(connection))
    if not statuses or any(status.state is not MigrationState.APPLIED for status in statuses):
        return False
    cursor = await connection.execute(
        "SELECT count(*) AS present FROM unnest(%s::text[]) AS name WHERE to_regclass(name) IS NOT NULL",
        (list(PIPELINE_TABLES),),
    )
    row = await cursor.fetchone()
    return row is not None and row["present"] == len(PIPELINE_TABLES)


@pytest.fixture
async def pipeline_database(database_url: str) -> str:
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True, row_factory=dict_row) as connection:
        if await _schema_is_current(connection):
            await connection.execute(CLEAR_TABLES_SQL)
        else:
            await connection.execute("DROP SCHEMA IF EXISTS public CASCADE")
            await connection.execute("CREATE SCHEMA public")
            await migrate(connection)
    return database_url


@pytest.fixture
async def pool(pipeline_database: str) -> AsyncIterator[Pool]:
    test_pool = create_pool(pipeline_database, role="test-pipeline", max_size=8, timeout=10, connect_timeout=5)
    async with open_pool(test_pool):
        yield test_pool


class Probe:
    def __init__(self, pool: Pool) -> None:
        self._pool = pool

    async def execute(self, query: str, params: Sequence[Any] = ()) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(query, params)

    async def rows(self, query: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        async with self._pool.connection() as connection:
            cursor = await connection.execute(query, params)
            return list(await cursor.fetchall())

    async def row(self, query: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = await self.rows(query, params)
        return rows[0] if rows else None

    async def scalar(self, query: str, params: Sequence[Any] = ()) -> Any:
        row = await self.row(query, params)
        assert row is not None
        return next(iter(row.values()))


@pytest.fixture
def probe(pool: Pool) -> Probe:
    return Probe(pool)


@pytest.fixture
def policy() -> QueuePolicy:
    return QueuePolicy(max_attempts=3, stale_seconds=900)


def hf_item(arxiv_id: str, *, title: str | None = None, summary: str | None = None, upvotes: int = 3) -> dict[str, Any]:
    return {
        "paper": {
            "id": arxiv_id,
            "title": title or f"Paper {arxiv_id}",
            "summary": summary or f"We study problem {arxiv_id}.",
            "authors": [{"name": "Ada Fake"}, {"name": " Ben Stub "}],
            "publishedAt": "2026-09-14T20:00:00.000Z",
            "upvotes": upvotes,
            "githubRepo": f"https://github.com/example/{arxiv_id}",
            "githubStars": 7,
        },
        "title": title or f"Paper {arxiv_id}",
    }


def body_sections(arxiv_id: str, *, extra: str = "") -> list[Section]:
    return [
        {"title": "Abstract", "text": f"We study problem {arxiv_id} with a new method."},
        {
            "title": "1 Introduction",
            "text": f"Large models are useful for {arxiv_id}. We describe the method in detail.{extra}",
        },
        {"title": "2 Method", "text": "The method trains a transformer. It uses a contrastive objective."},
    ]


def parse_result(source: str, sections: list[Section], *, text: str | None = None) -> FulltextParseResult:
    joined = text if text is not None else "\n\n".join(f"{s['title']}\n{s['text']}" for s in sections)
    return FulltextParseResult(
        text=joined,
        sections=sections,
        source=source,
        quality_metrics=fulltext_quality_metrics(text=joined, sections=sections, source=source),
        parser_metadata={"provider": "huridocs"} if source == "layout_pdf" else {},
    )


@dataclass
class FakePdfSource:
    missing: set[str] = field(default_factory=set)
    requested: list[str] = field(default_factory=list)

    async def fetch(self, url: str) -> bytes | None:
        self.requested.append(url)
        if any(arxiv_id in url for arxiv_id in self.missing):
            return None
        return f"%PDF {url}".encode()


@dataclass
class FakeParser:
    source: str = "layout_pdf"
    sections_by_id: dict[str, list[Section]] = field(default_factory=dict)
    failing_ids: set[str] = field(default_factory=set)
    parsed: list[bytes | None] = field(default_factory=list)

    async def parse(self, pdf_content: bytes | None, *, fallback_text: str) -> FulltextParseResult:
        self.parsed.append(pdf_content)
        if pdf_content is None:
            sections: list[Section] = [{"title": "Abstract", "text": fallback_text}] if fallback_text else []
            return parse_result("fallback_abstract", sections, text=fallback_text)
        arxiv_id = pdf_content.decode().rsplit("/", 1)[-1].removesuffix(".pdf")
        if arxiv_id in self.failing_ids:
            raise RuntimeError(f"parser exploded on {arxiv_id}")
        return parse_result(self.source, self.sections_by_id.get(arxiv_id, body_sections(arxiv_id)))

    def chunk(self, text: str, sections: Sequence[Section]) -> list[Any]:
        from arxplore.parsing.chunker import build_chunks

        return build_chunks(text, sections)


def load_fake_openai_module() -> ModuleType:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / FAKE_SERVER_RELATIVE_PATH
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("arxplore_fake_openai_server", candidate)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    pytest.skip(f"{FAKE_SERVER_RELATIVE_PATH} was not found above {Path(__file__).parent}")


class Kit:
    hf_item = staticmethod(hf_item)
    body_sections = staticmethod(body_sections)
    parse_result = staticmethod(parse_result)
    FakePdfSource = FakePdfSource
    FakeParser = FakeParser


@pytest.fixture
def kit() -> type[Kit]:
    return Kit


@pytest.fixture(scope="session")
def fake_openai_module() -> ModuleType:
    return load_fake_openai_module()


@pytest.fixture
def fake_openai(fake_openai_module: ModuleType) -> Iterator[Any]:
    server = fake_openai_module.FakeOpenAIServer().start()
    try:
        yield server
    finally:
        server.stop()
