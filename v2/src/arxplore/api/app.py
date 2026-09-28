import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from pydantic.json_schema import models_json_schema
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response

from arxplore.api.dependencies import ApiServices
from arxplore.api.errors import (
    CSRF_FAILURE_MESSAGE,
    INVALID_REQUEST_MESSAGE,
    ROUTE_NOT_FOUND_MESSAGE,
    SERVER_ERROR_MESSAGE,
    RateLimitExceededError,
)
from arxplore.api.openapi_docs import COMPONENT_REF
from arxplore.api.ratelimit import RateLimiter
from arxplore.api.responses import empty_response, error_response
from arxplore.api.routes import bootstrap, chat, generation, papers
from arxplore.api.schemas import EVENT_MODELS, REQUEST_MODELS
from arxplore.api.security import (
    CsrfRejectedError,
    SecurityHeadersMiddleware,
    TrustedHostMiddleware,
    TrustedOrigins,
)
from arxplore.api.services.arxiv_cache import CachedArxivSearch
from arxplore.api.services.chat import ChatService
from arxplore.api.services.detail import ArxivSearch
from arxplore.api.services.generation import GenerationService
from arxplore.api.services.store import PaperStore, PostgresPaperStore
from arxplore.config import AppSettings, get_settings
from arxplore.db.migrate import verify_schema
from arxplore.db.pool import create_pool_from_settings
from arxplore.embedding import EmbeddingClient
from arxplore.llm.agent.graph import build_agent_graph
from arxplore.llm.agent.tools import build_agent_tools
from arxplore.llm.detail import PaperDetailWriter
from arxplore.llm.models import ChatModelFactory
from arxplore.llm.paper_chat import ContextRetriever
from arxplore.llm.summary_graph import SummaryWriter
from arxplore.llm.tracing import Tracer
from arxplore.retrieval.service import RetrievalConfig, Retriever
from arxplore.sources.arxiv import ArxivClient

logger = logging.getLogger(__name__)

API_TITLE = "ArXplore API"
POOL_ROLE = "api"
DEFAULT_FRONTEND_DIST = Path("frontend/dist")
SPA_INDEX = "index.html"
SPA_METHODS = frozenset({"GET", "HEAD"})
MISSING_BUILD_MESSAGE = "React build output이 없습니다. frontend에서 npm run build를 먼저 실행하세요."
OPENAPI_URL = "/openapi.json"
DEBUG_DOCS_URL = "/docs"

type ServicesFactory = Callable[[AppSettings], AbstractAsyncContextManager[ApiServices]]


def api_version() -> str:
    try:
        return version("arxplore")
    except PackageNotFoundError:
        return "0"


def build_services(
    settings: AppSettings,
    *,
    store: PaperStore,
    retriever: ContextRetriever,
    arxiv: ArxivSearch,
    rate_limiter: RateLimiter,
    chat_models: ChatModelFactory | None = None,
    tracer: Tracer | None = None,
) -> ApiServices:
    models = chat_models or ChatModelFactory.from_settings(settings)
    active_tracer = tracer or Tracer.from_settings(settings)
    agent_graph = build_agent_graph(build_agent_tools(retriever, store), models)
    return ApiServices(
        settings=settings,
        store=store,
        arxiv=arxiv,
        generation=GenerationService(
            store,
            models,
            PaperDetailWriter(models, active_tracer),
            SummaryWriter(models, active_tracer),
        ),
        chat=ChatService(
            settings=settings,
            store=store,
            retriever=retriever,
            models=models,
            tracer=active_tracer,
            agent_graph=agent_graph,
        ),
        rate_limiter=rate_limiter,
        trusted_origins=TrustedOrigins.parse(settings.csrf_trusted_origins),
    )


@asynccontextmanager
async def open_services(settings: AppSettings) -> AsyncIterator[ApiServices]:
    pool = create_pool_from_settings(settings, role=POOL_ROLE)
    await pool.open(wait=True, timeout=pool.timeout)
    try:
        async with pool.connection() as connection:
            await verify_schema(connection, embedding_dimensions=settings.openai_embedding_dimensions)
        embedder = EmbeddingClient.from_settings(settings)
        arxiv = ArxivClient.from_settings(settings)
        rate_limiter = RateLimiter.from_settings(settings)
        try:
            yield build_services(
                settings,
                store=PostgresPaperStore(pool),
                retriever=Retriever(pool, embedder, RetrievalConfig.from_settings(settings)),
                arxiv=CachedArxivSearch(arxiv),
                rate_limiter=rate_limiter,
            )
        finally:
            await rate_limiter.aclose()
            await arxiv.aclose()
            await embedder.aclose()
    finally:
        await pool.close()


def spa_response(request: Request, dist: Path) -> Response:
    root = dist.resolve()
    relative = request.url.path.lstrip("/")
    if relative:
        candidate = (root / relative).resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            return FileResponse(candidate)
    index = root / SPA_INDEX
    if not index.is_file():
        return PlainTextResponse(MISSING_BUILD_MESSAGE, status_code=503)
    return HTMLResponse(index.read_bytes())


class ArxploreApi(FastAPI):
    def openapi(self) -> dict[str, Any]:
        return build_openapi_schema(self)


def build_openapi_schema(app: FastAPI) -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    _, definitions = models_json_schema(
        [(model, "validation") for model in (*REQUEST_MODELS, *EVENT_MODELS)],
        ref_template=COMPONENT_REF,
    )
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    for name, definition in definitions.get("$defs", {}).items():
        components.setdefault(name, definition)
    app.openapi_schema = schema
    return schema


def install_exception_handlers(app: FastAPI, *, frontend_dist: Path | None) -> None:
    async def csrf_rejected(request: Request, exc: Exception) -> Response:
        if isinstance(exc, CsrfRejectedError):
            logger.info("CSRF 검증 실패(%s): %s", exc.failure.value, request.url.path)
        return error_response(403, CSRF_FAILURE_MESSAGE)

    async def rate_limited(request: Request, exc: Exception) -> Response:
        if not isinstance(exc, RateLimitExceededError):
            return error_response(500, SERVER_ERROR_MESSAGE)
        return JSONResponse(
            {"error": exc.message, "retry_after": exc.retry_after},
            status_code=429,
            headers={"Retry-After": str(exc.retry_after)},
        )

    async def http_error(request: Request, exc: Exception) -> Response:
        status_code = exc.status_code if isinstance(exc, StarletteHTTPException) else 500
        headers = dict(exc.headers or {}) if isinstance(exc, StarletteHTTPException) else None
        if status_code == 405:
            return empty_response(405, headers=headers)
        if status_code == 404:
            if frontend_dist is not None and request.method in SPA_METHODS:
                return spa_response(request, frontend_dist)
            return error_response(404, ROUTE_NOT_FOUND_MESSAGE)
        return error_response(status_code, INVALID_REQUEST_MESSAGE if status_code < 500 else SERVER_ERROR_MESSAGE)

    async def invalid_request(request: Request, exc: Exception) -> Response:
        return error_response(400, INVALID_REQUEST_MESSAGE)

    async def server_error(request: Request, exc: Exception) -> Response:
        logger.exception("처리하지 못한 오류", exc_info=exc)
        return error_response(500, SERVER_ERROR_MESSAGE)

    app.add_exception_handler(CsrfRejectedError, csrf_rejected)
    app.add_exception_handler(RateLimitExceededError, rate_limited)
    app.add_exception_handler(StarletteHTTPException, http_error)
    app.add_exception_handler(RequestValidationError, invalid_request)
    app.add_exception_handler(Exception, server_error)


def create_app(
    settings: AppSettings | None = None,
    *,
    services_factory: ServicesFactory | None = None,
    frontend_dist: Path | None = None,
) -> FastAPI:
    active_settings = settings or get_settings()
    factory = services_factory or open_services

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with factory(active_settings) as services:
            app.state.services = services
            yield

    app = ArxploreApi(
        title=API_TITLE,
        version=api_version(),
        lifespan=lifespan,
        openapi_url=OPENAPI_URL,
        docs_url=DEBUG_DOCS_URL if active_settings.debug else None,
        redoc_url=None,
        redirect_slashes=False,
    )
    app.include_router(bootstrap.router)
    app.include_router(chat.router)
    app.include_router(papers.router)
    app.include_router(generation.router)
    serve_dist = (frontend_dist or DEFAULT_FRONTEND_DIST) if active_settings.serve_frontend_dist else None
    install_exception_handlers(app, frontend_dist=serve_dist)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=active_settings.allowed_hosts,
        rejection=lambda: error_response(400, INVALID_REQUEST_MESSAGE),
    )
    app.add_middleware(SecurityHeadersMiddleware)
    return app
