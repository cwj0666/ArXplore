import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol, Self

from redis.asyncio import Redis

from arxplore.config import AppSettings

WINDOW_SECONDS = 60
KEY_TTL_SECONDS = WINDOW_SECONDS + 5
LLM_SCOPE = "llm"
DETAIL_SCOPE = "detail"
KEY_PREFIX = "ratelimit:"


class CounterStore(Protocol):
    async def increment(self, key: str, *, ttl_seconds: int) -> int: ...

    async def clear(self) -> None: ...

    async def aclose(self) -> None: ...


@dataclass
class MemoryCounterStore:
    clock: Callable[[], float] = time.monotonic
    counters: dict[str, tuple[int, float]] = field(default_factory=dict)

    async def increment(self, key: str, *, ttl_seconds: int) -> int:
        now = self.clock()
        self._evict(now)
        count, expires_at = self.counters.get(key, (0, now + ttl_seconds))
        self.counters[key] = (count + 1, expires_at)
        return count + 1

    def _evict(self, now: float) -> None:
        expired = [key for key, (_, expires_at) in self.counters.items() if expires_at <= now]
        for key in expired:
            del self.counters[key]

    async def clear(self) -> None:
        self.counters.clear()

    async def aclose(self) -> None:
        self.counters.clear()


class RedisCounterStore:
    def __init__(self, client: Redis) -> None:
        self.client = client

    @classmethod
    def from_url(cls, url: str) -> Self:
        return cls(Redis.from_url(url))

    async def increment(self, key: str, *, ttl_seconds: int) -> int:
        async with self.client.pipeline(transaction=True) as pipeline:
            pipeline.set(key, 0, ex=ttl_seconds, nx=True)
            pipeline.incr(key)
            _, count = await pipeline.execute()
        return int(count)

    async def clear(self) -> None:
        async for key in self.client.scan_iter(match=f"{KEY_PREFIX}*"):
            await self.client.delete(key)

    async def aclose(self) -> None:
        await self.client.aclose()


def window_start(now: float, window: int = WINDOW_SECONDS) -> int:
    return int(now // window) * window


def rate_limit_key(scope: str, identity: str, start: int) -> str:
    return f"{KEY_PREFIX}{scope}:{identity}:{start}"


def seconds_until_window_end(now: float, start: int, window: int = WINDOW_SECONDS) -> int:
    return max(1, math.ceil(start + window - now))


@dataclass
class RateLimiter:
    store: CounterStore
    limits: Mapping[str, int]
    enabled: bool = True
    clock: Callable[[], float] = time.time

    @classmethod
    def from_settings(cls, settings: AppSettings, *, store: CounterStore | None = None) -> Self:
        if store is None:
            store = RedisCounterStore.from_url(settings.redis_url) if settings.redis_url else MemoryCounterStore()
        return cls(
            store=store,
            limits={
                LLM_SCOPE: settings.rate_limit_llm_per_minute,
                DETAIL_SCOPE: settings.rate_limit_detail_per_minute,
            },
            enabled=settings.rate_limit_enabled,
        )

    async def hit(self, scope: str, client_ip: str) -> int | None:
        if not self.enabled:
            return None
        now = self.clock()
        start = window_start(now)
        count = await self.store.increment(rate_limit_key(scope, f"ip:{client_ip}", start), ttl_seconds=KEY_TTL_SECONDS)
        if count <= self.limits[scope]:
            return None
        return seconds_until_window_end(now, start)

    async def reset(self) -> None:
        await self.store.clear()

    async def aclose(self) -> None:
        await self.store.aclose()
