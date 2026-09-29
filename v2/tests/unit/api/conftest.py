import importlib.util
import json
import sys
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from arxplore.api.app import build_services, create_app
from arxplore.api.dependencies import ApiServices
from arxplore.api.ratelimit import MemoryCounterStore, RateLimiter
from arxplore.api.services.arxiv_cache import CachedArxivSearch
from arxplore.config import AppSettings, RetrievalMode
from arxplore.domain.chunks import StoredChunk
from arxplore.domain.papers import AiOverview, DetailedSummary, Paper, PaperFulltext
from arxplore.domain.retrieval import RetrievalContext, RetrievalHit, RetrievalResult
from arxplore.llm.detail import ANALYSIS_PROMPT_VERSION
from arxplore.llm.models import ChatModelFactory
from arxplore.llm.summary_graph import SUMMARY_PROMPT_VERSION
from arxplore.llm.tracing import Tracer
from arxplore.sources.arxiv import ArxivPaper

FAKE_SERVER_PATH = Path(__file__).resolve().parents[4] / "tests" / "fakes" / "openai_server.py"
FAKE_SERVER_MODULE = "arxplore_tests_fake_openai_server"
MAIN_ID = "2401.00001"
OTHER_ID = "2401.00002"
CACHED_ID = "2401.00003"
CLIENT_IP = "198.51.100.1"
BASE_URL = "http://localhost"


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


def make_paper(arxiv_id: str, **overrides: Any) -> Paper:
    values: dict[str, Any] = {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "authors": ["Ada Lovelace"],
        "abstract": f"Abstract of {arxiv_id}.",
        "primary_category": "cs.CL",
        "categories": ["cs.CL"],
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        "published_at": datetime(2024, 1, 1, tzinfo=UTC),
        "updated_at": None,
        "upvotes": 0,
        "github_url": None,
        "github_stars": None,
        "citation_count": None,
    }
    values.update(overrides)
    return Paper(**values)


def make_chunk(arxiv_id: str, chunk_id: int, text: str, section: str | None = "1 Introduction") -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        arxiv_id=arxiv_id,
        chunk_index=chunk_id,
        chunk_text=text,
        section_title=section,
        token_count=len(text.split()),
        metadata={},
        updated_at=None,
    )


def make_context(arxiv_id: str, chunk_id: int, text: str, section: str = "3 Method") -> RetrievalContext:
    hit = RetrievalHit(
        chunk_id=chunk_id,
        arxiv_id=arxiv_id,
        chunk_index=chunk_id,
        paper_title=f"Paper {arxiv_id}",
        paper_abstract="",
        chunk_text=text,
        section_title=section,
        content_role="body",
        method="hybrid",
        score=1.0,
        snippet="",
    )
    return RetrievalContext(hit=hit, chunks=(), text=text)


def parse_sse(body: str) -> list[Any]:
    events: list[Any] = []
    for block in body.split("\n\n"):
        if not block:
            continue
        assert block.startswith("data: ")
        data = block[len("data: ") :]
        events.append(data if data == "[DONE]" else json.loads(data))
    return events


@dataclass
class MemoryStore:
    papers: list[Paper] = field(default_factory=list)
    fulltexts: dict[str, PaperFulltext] = field(default_factory=dict)
    chunks: dict[str, list[StoredChunk]] = field(default_factory=dict)
    overviews: dict[tuple[str, int], AiOverview] = field(default_factory=dict)
    summaries: dict[tuple[str, str, int], DetailedSummary] = field(default_factory=dict)
    error: Exception | None = None
    recent_limits: list[int] = field(default_factory=list)

    def _check(self) -> None:
        if self.error is not None:
            raise self.error

    async def get_paper(self, arxiv_id: str) -> Paper | None:
        self._check()
        return next((paper for paper in self.papers if paper.arxiv_id == arxiv_id), None)

    async def list_recent_papers(self, *, limit: int) -> Sequence[Paper]:
        self._check()
        self.recent_limits.append(limit)
        return self.papers[:limit]

    def _matching(self, tokens: Sequence[str]) -> list[Paper]:
        self._check()
        lowered = [token.lower() for token in tokens]
        return [
            paper
            for paper in self.papers
            if all(token in paper.title.lower() or token in paper.abstract.lower() for token in lowered)
        ]

    async def count_papers(self, tokens: Sequence[str]) -> int:
        return len(self._matching(tokens))

    async def list_papers_page(self, tokens: Sequence[str], *, sort: str, limit: int, offset: int) -> Sequence[Paper]:
        papers = self._matching(tokens)
        if sort == "upvotes":
            papers = sorted(papers, key=lambda paper: paper.upvotes, reverse=True)
        return papers[offset : offset + limit]

    async def get_fulltext(self, arxiv_id: str) -> PaperFulltext | None:
        return self.fulltexts.get(arxiv_id)

    async def list_chunks(self, arxiv_id: str, *, limit: int) -> Sequence[StoredChunk]:
        return self.chunks.get(arxiv_id, [])[:limit]

    async def get_overview(self, arxiv_id: str, *, prompt_version: int) -> AiOverview | None:
        return self.overviews.get((arxiv_id, prompt_version))

    async def save_overview(
        self, arxiv_id: str, *, overview: str, key_findings: list[Any], model: str, prompt_version: int
    ) -> None:
        self.overviews[(arxiv_id, prompt_version)] = AiOverview(
            overview=overview, key_findings=key_findings, model=model, generated_at=None
        )

    async def get_detailed_summary(self, arxiv_id: str, model: str, *, prompt_version: int) -> DetailedSummary | None:
        return self.summaries.get((arxiv_id, model, prompt_version))

    async def save_detailed_summary(self, arxiv_id: str, *, summary: str, model: str, prompt_version: int) -> None:
        self.summaries[(arxiv_id, model, prompt_version)] = DetailedSummary(
            summary=summary, model=model, generated_at=None
        )


@dataclass
class FakeRetriever:
    contexts: dict[str | None, list[RetrievalContext]] = field(default_factory=dict)
    mode: RetrievalMode = "hybrid"
    error: Exception | None = None
    calls: list[tuple[str, int, str | None]] = field(default_factory=list)

    async def retrieve_contexts(self, query: str, *, limit: int = 5, arxiv_id: str | None = None) -> RetrievalResult:
        self.calls.append((query, limit, arxiv_id))
        if self.error is not None:
            raise self.error
        return RetrievalResult(contexts=tuple(self.contexts.get(arxiv_id, [])), mode=self.mode)


@dataclass
class FakeArxiv:
    papers: list[ArxivPaper] = field(default_factory=list)
    error: Exception | None = None
    queries: list[tuple[str, int]] = field(default_factory=list)

    async def search_papers(self, query: str, *, max_results: int = 5) -> list[ArxivPaper]:
        self.queries.append((query, max_results))
        if self.error is not None:
            raise self.error
        return list(self.papers)


@dataclass
class Clock:
    now: float = 1_000_020.0

    def __call__(self) -> float:
        return self.now


def seeded_store() -> MemoryStore:
    main = make_paper(
        MAIN_ID,
        title="Direct Preference Optimization Revisited",
        abstract="We revisit direct preference optimization.",
        categories=["cs.CL", "cs.LG"],
        upvotes=5,
    )
    other = make_paper(
        OTHER_ID,
        title="Preference Optimization with Human Feedback",
        abstract="Human feedback for preference optimization.",
        categories=["cs.LG"],
        primary_category="cs.LG",
        upvotes=9,
    )
    cached = make_paper(CACHED_ID, title="Cached Paper", categories=["math.CO"], primary_category="math.CO")
    return MemoryStore(
        papers=[main, other, cached],
        fulltexts={
            MAIN_ID: PaperFulltext(
                arxiv_id=MAIN_ID,
                text="",
                sections=[{"title": "1 Introduction", "text": "Intro body."}, {"title": "3 Method", "text": "Method."}],
                source="pdf",
                quality_metrics={},
                artifacts={},
                parser_metadata={},
                updated_at=None,
            )
        },
        chunks={MAIN_ID: [make_chunk(MAIN_ID, 1, "first chunk"), make_chunk(MAIN_ID, 2, "second chunk", None)]},
        overviews={
            (CACHED_ID, ANALYSIS_PROMPT_VERSION): AiOverview(
                overview="cached overview", key_findings=["k1"], model="gpt-5-mini", generated_at=None
            )
        },
        summaries={
            (CACHED_ID, "gpt-5-mini", SUMMARY_PROMPT_VERSION): DetailedSummary(
                summary="cached summary", model="gpt-5-mini", generated_at=None
            )
        },
    )


@dataclass
class Harness:
    fakes: ModuleType
    fake_openai: Any
    store: MemoryStore = field(default_factory=seeded_store)
    retriever: FakeRetriever = field(
        default_factory=lambda: FakeRetriever(contexts={MAIN_ID: [make_context(MAIN_ID, 10, "method context")]})
    )
    arxiv: FakeArxiv = field(default_factory=FakeArxiv)
    clock: Clock = field(default_factory=Clock)
    settings_values: dict[str, str] = field(default_factory=dict)
    frontend_dist: Path | None = None

    def settings(self) -> AppSettings:
        return AppSettings.model_validate(
            {
                "OPENAI_API_KEY": self.fake_openai.expected_api_key,
                "OPENAI_BASE_URL": self.fake_openai.base_url,
                "RATE_LIMIT_ENABLED": "false",
                "ALLOWED_HOSTS": "localhost",
                **self.settings_values,
            }
        )

    @asynccontextmanager
    async def services(self, settings: AppSettings) -> AsyncIterator[ApiServices]:
        rate_limiter = RateLimiter.from_settings(settings, store=MemoryCounterStore())
        rate_limiter.clock = self.clock
        yield build_services(
            settings,
            store=self.store,
            retriever=self.retriever,
            arxiv=CachedArxivSearch(self.arxiv),
            rate_limiter=rate_limiter,
            chat_models=ChatModelFactory.from_settings(settings),
            tracer=Tracer.disabled(),
        )

    @asynccontextmanager
    async def app(self, **settings_values: str) -> AsyncIterator[FastAPI]:
        self.settings_values.update(settings_values)
        app = create_app(self.settings(), services_factory=self.services, frontend_dist=self.frontend_dist)
        async with app.router.lifespan_context(app):
            yield app

    @asynccontextmanager
    async def client(self, **settings_values: str) -> AsyncIterator[httpx.AsyncClient]:
        async with self.app(**settings_values) as app:
            transport = httpx.ASGITransport(app=app, client=(CLIENT_IP, 50000))
            async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
                yield client

    @staticmethod
    async def csrf_token(client: httpx.AsyncClient) -> str:
        response = await client.get("/bootstrap.json")
        token = response.cookies.get("csrftoken")
        client.cookies.clear()
        assert token
        return token

    async def post(
        self,
        client: httpx.AsyncClient,
        path: str,
        payload: Any = None,
        *,
        raw: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        token = await self.csrf_token(client)
        if raw is not None:
            content: bytes | None = raw.encode()
        else:
            content = None if payload is None else json.dumps(payload, ensure_ascii=False).encode()
        request_headers = {"cookie": f"csrftoken={token}", "x-csrftoken": token, **(headers or {})}
        response = await client.post(path, content=content, headers=request_headers)
        client.cookies.clear()
        return response

    def llm_requests(self) -> list[Any]:
        return [request.body for request in self.fake_openai.requests_for("/v1/chat/completions")]


@pytest.fixture(scope="session")
def fakes() -> ModuleType:
    return load_fake_server_module()


@pytest.fixture
def fake_openai(fakes: ModuleType) -> Iterator[Any]:
    with fakes.FakeOpenAIServer() as server:
        yield server


@pytest.fixture
def harness(fakes: ModuleType, fake_openai: Any) -> Harness:
    return Harness(fakes=fakes, fake_openai=fake_openai)


@pytest.fixture
def ids() -> dict[str, str]:
    return {"main": MAIN_ID, "other": OTHER_ID, "cached": CACHED_ID}


@pytest.fixture
def helpers() -> ModuleType:
    return sys.modules[__name__]
