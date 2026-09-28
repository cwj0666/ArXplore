from dataclasses import dataclass
from typing import cast

from fastapi import Request

from arxplore.api.errors import RateLimitExceededError
from arxplore.api.ratelimit import DETAIL_SCOPE, LLM_SCOPE, RateLimiter
from arxplore.api.security import TrustedOrigins, check_csrf, client_ip
from arxplore.api.services.arxiv_cache import CachedArxivSearch
from arxplore.api.services.chat import ChatService
from arxplore.api.services.detail import ArxivSearch
from arxplore.api.services.generation import GenerationService
from arxplore.api.services.store import PaperStore
from arxplore.config import AppSettings


@dataclass
class ApiServices:
    settings: AppSettings
    store: PaperStore
    arxiv: ArxivSearch
    generation: GenerationService
    chat: ChatService
    rate_limiter: RateLimiter
    trusted_origins: TrustedOrigins

    async def reset_state(self) -> None:
        await self.rate_limiter.reset()
        if isinstance(self.arxiv, CachedArxivSearch):
            self.arxiv.clear()


def get_services(request: Request) -> ApiServices:
    return cast(ApiServices, request.app.state.services)


async def enforce_csrf(request: Request) -> None:
    services = get_services(request)
    check_csrf(
        request,
        trusted_origins=services.trusted_origins,
        trust_forwarded_proto=services.settings.secure_cookies,
    )


async def _enforce_rate_limit(request: Request, scope: str) -> None:
    services = get_services(request)
    retry_after = await services.rate_limiter.hit(scope, client_ip(request, services.settings.rate_limit_ip_header))
    if retry_after is not None:
        raise RateLimitExceededError(retry_after)


async def enforce_llm_rate_limit(request: Request) -> None:
    await _enforce_rate_limit(request, LLM_SCOPE)


async def enforce_detail_rate_limit(request: Request) -> None:
    await _enforce_rate_limit(request, DETAIL_SCOPE)
