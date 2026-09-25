"""Django cache 기반 고정 윈도 rate limiter.

카운터는 `default` 캐시에 저장한다. LocMemCache는 프로세스마다 따로라서 gunicorn worker가 N개면
실제 허용량이 N배가 된다. 여러 프로세스가 한도를 공유하려면 REDIS_URL로 Redis 캐시를 지정한다.
"""

from __future__ import annotations

import functools
import math
import time
from collections.abc import Callable

from django.conf import settings
from django.core.cache import cache
from django.http import HttpRequest, JsonResponse

WINDOW_SECONDS = 60
RATE_LIMITED_MESSAGE = "요청이 너무 많습니다. {retry_after}초 후 다시 시도하세요."


def rate_limit(scope: str, *, limit_setting: str, per: str) -> Callable:
    """`per`는 "ip"(클라이언트 IP) 또는 "user"(로그인 사용자, 비로그인은 IP)."""
    if per not in {"ip", "user"}:
        raise ValueError(f"unsupported rate limit identity: {per}")

    def decorator(view: Callable) -> Callable:
        @functools.wraps(view)
        def wrapped(request: HttpRequest, *args, **kwargs):
            if getattr(settings, "RATE_LIMIT_ENABLED", True):
                retry_after = register_hit(
                    scope,
                    _identity(request, per),
                    limit=int(getattr(settings, limit_setting)),
                )
                if retry_after is not None:
                    return JsonResponse(
                        {"error": RATE_LIMITED_MESSAGE.format(retry_after=retry_after), "retry_after": retry_after},
                        status=429,
                        headers={"Retry-After": str(retry_after)},
                    )
            return view(request, *args, **kwargs)

        return wrapped

    return decorator


def register_hit(
    scope: str, identity: str, *, limit: int, window: int = WINDOW_SECONDS, now: float | None = None
) -> int | None:
    """요청 1회를 기록하고, 한도를 넘었으면 윈도가 끝날 때까지 남은 초를 돌려준다."""
    current = time.time() if now is None else now
    window_start = int(current // window) * window
    cache_key = f"ratelimit:{scope}:{identity}:{window_start}"
    timeout = window + 5
    cache.add(cache_key, 0, timeout=timeout)
    try:
        count = cache.incr(cache_key)
    except ValueError:
        cache.set(cache_key, 1, timeout=timeout)
        count = 1
    if count <= limit:
        return None
    return max(1, math.ceil(window_start + window - current))


def client_ip(request: HttpRequest) -> str:
    header = (getattr(settings, "RATE_LIMIT_IP_HEADER", "") or "").strip()
    if header:
        meta_key = "HTTP_" + header.upper().replace("-", "_")
        forwarded = (request.META.get(meta_key) or "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return request.META.get("REMOTE_ADDR") or "unknown"


def _identity(request: HttpRequest, per: str) -> str:
    user = getattr(request, "user", None)
    if per == "user" and getattr(user, "is_authenticated", False):
        return f"user:{getattr(user, 'pk', None) or user.get_username()}"
    return f"ip:{client_ip(request)}"
