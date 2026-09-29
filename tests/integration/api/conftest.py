import importlib.util
import json
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import psycopg
import pytest

from arxplore.config import AppSettings
from arxplore.db.migrate import migrate

FAKE_SERVER_PATH = Path(__file__).resolve().parents[3] / "tests" / "fakes" / "openai_server.py"
FAKE_SERVER_MODULE = "arxplore_tests_fake_openai_server"

PAPERS = (
    {
        "arxiv_id": "2401.00001",
        "title": "Direct Preference Optimization Revisited",
        "abstract": "We revisit direct preference optimization for alignment.",
        "categories": ["cs.CL", "cs.LG"],
        "primary_category": "cs.CL",
        "upvotes": 5,
        "published_at": "2024-01-02T00:00:00+00:00",
        "sections": [
            {"title": "1 Introduction", "text": "Preference optimization aligns language models."},
            {"title": "3 Method", "text": "The loss uses a reference policy margin."},
        ],
        "chunks": [
            ("1 Introduction", "Preference optimization aligns language models with human feedback."),
            ("3 Method", "The contrastive loss uses a reference policy margin for each preference pair."),
        ],
    },
    {
        "arxiv_id": "2401.00002",
        "title": "Human Feedback for Preference Optimization",
        "abstract": "Human feedback improves preference optimization.",
        "categories": ["cs.LG"],
        "primary_category": "cs.LG",
        "upvotes": 9,
        "published_at": "2024-01-01T00:00:00+00:00",
        "sections": [],
        "chunks": [],
    },
)


def load_fake_server_module() -> ModuleType:
    if FAKE_SERVER_MODULE in sys.modules:
        return sys.modules[FAKE_SERVER_MODULE]
    if not FAKE_SERVER_PATH.is_file():
        pytest.skip(f"fake OpenAI server not found at {FAKE_SERVER_PATH}")
    spec = importlib.util.spec_from_file_location(FAKE_SERVER_MODULE, FAKE_SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[FAKE_SERVER_MODULE] = module
    spec.loader.exec_module(module)
    return module


async def seed(connection: psycopg.AsyncConnection[Any]) -> None:
    for paper in PAPERS:
        await connection.execute(
            """
            INSERT INTO papers (arxiv_id, title, authors, abstract, primary_category, categories, pdf_url,
                                published_at, upvotes, source)
            VALUES (%s, %s, %s::jsonb, %s, %s, %s::jsonb, NULL, %s, %s, 'hf_daily_papers')
            """,
            (
                paper["arxiv_id"],
                paper["title"],
                json.dumps(["Ada Lovelace"]),
                paper["abstract"],
                paper["primary_category"],
                json.dumps(paper["categories"]),
                paper["published_at"],
                paper["upvotes"],
            ),
        )
        if paper["sections"]:
            await connection.execute(
                "INSERT INTO paper_fulltexts (arxiv_id, text, sections, source) VALUES (%s, '', %s::jsonb, 'pdf')",
                (paper["arxiv_id"], json.dumps(paper["sections"])),
            )
        for index, (section, text) in enumerate(paper["chunks"]):
            await connection.execute(
                """
                INSERT INTO paper_chunks (arxiv_id, chunk_index, chunk_text, section_title, token_count, metadata)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                """,
                (paper["arxiv_id"], index, text, section, len(text.split()), json.dumps({"content_role": "body"})),
            )


@pytest.fixture
async def api_database(empty_database: str) -> str:
    async with await psycopg.AsyncConnection.connect(empty_database, autocommit=True) as connection:
        await migrate(connection)
        await seed(connection)
    return empty_database


@pytest.fixture(scope="session")
def fakes() -> ModuleType:
    return load_fake_server_module()


@pytest.fixture
def fake_openai(fakes: ModuleType) -> Iterator[Any]:
    with fakes.FakeOpenAIServer() as server:
        yield server


@pytest.fixture
def api_settings(database_settings: Callable[..., AppSettings], fake_openai: Any) -> Callable[..., AppSettings]:
    def build(**overrides: str) -> AppSettings:
        values = {
            "OPENAI_API_KEY": fake_openai.expected_api_key,
            "OPENAI_BASE_URL": fake_openai.base_url,
            "ARXIV_API_BASE_URL": fake_openai.arxiv_url,
            "RETRIEVAL_MODE": "lexical",
            "RATE_LIMIT_ENABLED": "false",
            "LANGSMITH_TRACING": "false",
            "POSTGRES_POOL_MAX": "4",
        }
        values.update(overrides)
        return database_settings(**values)

    return build


@pytest.fixture
async def open_connection_to(api_database: str) -> AsyncIterator[psycopg.AsyncConnection[Any]]:
    async with await psycopg.AsyncConnection.connect(api_database, autocommit=True) as connection:
        yield connection
