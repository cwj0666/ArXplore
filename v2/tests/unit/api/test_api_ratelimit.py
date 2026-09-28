from dataclasses import dataclass

import pytest

from arxplore.api.ratelimit import (
    DETAIL_SCOPE,
    KEY_TTL_SECONDS,
    LLM_SCOPE,
    MemoryCounterStore,
    RateLimiter,
    RedisCounterStore,
    rate_limit_key,
    seconds_until_window_end,
    window_start,
)
from arxplore.config import AppSettings


@dataclass
class Clock:
    now: float

    def __call__(self) -> float:
        return self.now


def limiter(clock: Clock, *, llm: int = 3, detail: int = 1, enabled: bool = True) -> RateLimiter:
    return RateLimiter(
        store=MemoryCounterStore(clock=clock),
        limits={LLM_SCOPE: llm, DETAIL_SCOPE: detail},
        enabled=enabled,
        clock=clock,
    )


def test_windows_are_aligned_to_the_wall_clock_minute() -> None:
    assert window_start(1_000_040.0) == 1_000_020
    assert window_start(1_000_019.9) == 999_960
    assert seconds_until_window_end(1_000_040.0, 1_000_020) == 40
    assert seconds_until_window_end(1_000_079.5, 1_000_020) == 1
    assert rate_limit_key("llm", "ip:1.2.3.4", 1_000_020) == "ratelimit:llm:ip:1.2.3.4:1000020"
    assert KEY_TTL_SECONDS == 65


async def test_allows_up_to_the_limit_then_returns_the_seconds_left() -> None:
    clock = Clock(1_000_040.0)
    rate = limiter(clock)

    results = [await rate.hit(LLM_SCOPE, "1.1.1.1") for _ in range(4)]

    assert results == [None, None, None, 40]


async def test_new_window_resets_the_counter() -> None:
    clock = Clock(1_000_000.0)
    rate = limiter(clock, llm=2)
    for _ in range(3):
        await rate.hit(LLM_SCOPE, "1.1.1.1")

    clock.now = 1_000_021.0
    assert await rate.hit(LLM_SCOPE, "1.1.1.1") is None


async def test_scopes_and_addresses_are_counted_separately() -> None:
    clock = Clock(1_000_020.0)
    rate = limiter(clock, llm=1, detail=1)
    await rate.hit(LLM_SCOPE, "1.1.1.1")

    assert await rate.hit(LLM_SCOPE, "2.2.2.2") is None
    assert await rate.hit(DETAIL_SCOPE, "1.1.1.1") is None
    assert await rate.hit(LLM_SCOPE, "1.1.1.1") == 60


async def test_disabled_limiter_never_counts() -> None:
    rate = limiter(Clock(1_000_020.0), llm=1, enabled=False)

    assert [await rate.hit(LLM_SCOPE, "1.1.1.1") for _ in range(5)] == [None] * 5
    store = rate.store
    assert isinstance(store, MemoryCounterStore)
    assert store.counters == {}


async def test_memory_counters_expire_after_their_ttl_and_reset_clears_them() -> None:
    clock = Clock(100.0)
    store = MemoryCounterStore(clock=clock)

    assert await store.increment("k", ttl_seconds=65) == 1
    assert await store.increment("k", ttl_seconds=65) == 2
    clock.now = 165.0
    assert await store.increment("k", ttl_seconds=65) == 1
    await store.clear()
    assert store.counters == {}


async def test_redis_url_selects_the_redis_store() -> None:
    redis = RateLimiter.from_settings(AppSettings.model_validate({"REDIS_URL": "redis://cache:6379/0"}))
    memory = RateLimiter.from_settings(AppSettings.model_validate({}))

    assert isinstance(redis.store, RedisCounterStore)
    assert isinstance(memory.store, MemoryCounterStore)
    assert memory.limits == {LLM_SCOPE: 30, DETAIL_SCOPE: 60}
    await redis.aclose()


@pytest.mark.parametrize(("flag", "expected"), [("true", True), ("false", False)])
def test_enabled_flag_follows_the_settings(flag: str, expected: bool) -> None:
    assert RateLimiter.from_settings(AppSettings.model_validate({"RATE_LIMIT_ENABLED": flag})).enabled is expected
