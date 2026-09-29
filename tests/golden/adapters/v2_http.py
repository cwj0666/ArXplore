import asyncio
from collections.abc import Awaitable, Iterator
from contextlib import AbstractAsyncContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import fastapi
import httpx
from arxplore.api.app import create_app
from arxplore.api.dependencies import ApiServices
from arxplore.config import AppSettings, load_settings
from arxplore.db.migrate import migrate
from arxplore.db.pool import connect_from_settings

from tests.golden.adapters import HttpResult

T = TypeVar("T")

CLIENT_ADDRESS = ("127.0.0.1", 50000)
BASE_URL = "http://localhost"
MIGRATION_ROLE = "golden-migrate"


@dataclass
class _RunningApp:
    app: fastapi.FastAPI
    lifespan: AbstractAsyncContextManager[None]
    client: httpx.AsyncClient

    @property
    def services(self) -> ApiServices:
        services: ApiServices = self.app.state.services
        return services


class V2HttpAdapter:
    def __init__(self, *, frontend_dist: Path) -> None:
        settings = load_settings().model_copy(update={"serve_frontend_dist": True})
        if not settings.openai_base_url:
            raise RuntimeError("the v2 HTTP golden needs OPENAI_BASE_URL pointing at the fake OpenAI server")
        self._runner = asyncio.Runner()
        self._run(self._migrate(settings))
        self._apps = {
            True: self._start(settings, frontend_dist),
            False: self._start(settings.model_copy(update={"openai_api_key": None}), frontend_dist),
        }
        self._active = self._apps[True]

    def _run(self, awaitable: Awaitable[T]) -> T:
        async def wrapper() -> T:
            return await awaitable

        return self._runner.run(wrapper())

    @staticmethod
    async def _migrate(settings: AppSettings) -> None:
        connection = await connect_from_settings(settings, role=MIGRATION_ROLE, autocommit=True)
        async with connection:
            await migrate(connection)

    def _start(self, settings: AppSettings, frontend_dist: Path) -> _RunningApp:
        app = create_app(settings, frontend_dist=frontend_dist)
        lifespan = app.router.lifespan_context(app)
        self._run(lifespan.__aenter__())
        transport = httpx.ASGITransport(app=app, client=CLIENT_ADDRESS)
        return _RunningApp(app=app, lifespan=lifespan, client=httpx.AsyncClient(transport=transport, base_url=BASE_URL))

    def describe(self) -> dict[str, str]:
        return {"implementation": "v2", "framework": f"fastapi {fastapi.__version__}"}

    def request(self, method: str, path: str, *, headers: list[tuple[str, str]], body: bytes | None) -> HttpResult:
        client = self._active.client
        response = self._run(client.request(method, path, headers=headers, content=body))
        client.cookies.clear()
        return HttpResult(
            response.status_code,
            [(name.lower(), value) for name, value in response.headers.multi_items()],
            response.content,
        )

    def reset_state(self) -> None:
        for running in self._apps.values():
            self._run(running.services.reset_state())

    @contextmanager
    def openai_key(self, configured: bool) -> Iterator[None]:
        previous = self._active
        self._active = self._apps[configured]
        try:
            yield
        finally:
            self._active = previous

    def close(self) -> None:
        try:
            for running in self._apps.values():
                self._run(running.client.aclose())
                self._run(running.lifespan.__aexit__(None, None, None))
        finally:
            self._runner.close()


def http_adapter(*, frontend_dist: Path, **options: Any) -> V2HttpAdapter:
    return V2HttpAdapter(frontend_dist=frontend_dist)
