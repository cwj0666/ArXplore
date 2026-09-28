import asyncio
from collections.abc import Sequence
from contextvars import ContextVar

import pytest
from openai import OpenAIError

from arxplore.context import override_openai_runtime, request_openai_api_key
from arxplore.retrieval.service import resolve_retrieval_mode, run_hybrid_channels

WAIT_SECONDS = 2.0

_REQUEST_SCOPED: ContextVar[str] = ContextVar("request_scoped", default="unset")


class StubEmbedder:
    def __init__(self, *, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


async def _value[T](value: T) -> T:
    return value


async def _fail(error: BaseException) -> None:
    raise error


class TestRunHybridChannels:
    async def test_returns_both_channel_results(self) -> None:
        assert await run_hybrid_channels(_value("lexical"), _value("vector")) == ("lexical", "vector")

    async def test_channels_overlap_in_time(self) -> None:
        lexical_started = asyncio.Event()
        vector_started = asyncio.Event()

        async def lexical() -> str:
            lexical_started.set()
            await asyncio.wait_for(vector_started.wait(), WAIT_SECONDS)
            return "lexical"

        async def vector() -> str:
            vector_started.set()
            await asyncio.wait_for(lexical_started.wait(), WAIT_SECONDS)
            return "vector"

        assert await run_hybrid_channels(lexical(), vector()) == ("lexical", "vector")

    async def test_vector_error_is_raised_after_lexical_succeeds(self) -> None:
        error = OpenAIError("embedding failed")
        finished: list[str] = []

        async def lexical() -> str:
            await asyncio.sleep(0.01)
            finished.append("lexical")
            return "lexical"

        with pytest.raises(OpenAIError) as caught:
            await run_hybrid_channels(lexical(), _fail(error))

        assert caught.value is error
        assert finished == ["lexical"]

    async def test_lexical_error_wins_when_both_fail(self) -> None:
        lexical_error = RuntimeError("canceling statement due to statement timeout")
        vector_error = OpenAIError("embedding failed")

        with pytest.raises(RuntimeError) as caught:
            await run_hybrid_channels(_fail(lexical_error), _fail(vector_error))

        assert caught.value is lexical_error

    async def test_lexical_failure_does_not_wait_for_the_vector_channel(self) -> None:
        release = asyncio.Event()
        vector_cancelled = asyncio.Event()

        async def vector() -> str:
            try:
                await release.wait()
            except asyncio.CancelledError:
                vector_cancelled.set()
                raise
            return "vector"

        with pytest.raises(RuntimeError, match="lexical"):
            await asyncio.wait_for(run_hybrid_channels(_fail(RuntimeError("lexical")), vector()), WAIT_SECONDS)

        assert vector_cancelled.is_set()

    async def test_both_channels_see_the_request_context(self) -> None:
        seen: list[tuple[str, str | None]] = []

        async def channel(name: str) -> str:
            seen.append((_REQUEST_SCOPED.get(), request_openai_api_key()))
            return name

        token = _REQUEST_SCOPED.set("request-value")
        try:
            with override_openai_runtime(api_key="sk-request"):
                await run_hybrid_channels(channel("lexical"), channel("vector"))
        finally:
            _REQUEST_SCOPED.reset(token)

        assert seen == [("request-value", "sk-request"), ("request-value", "sk-request")]

    async def test_outer_cancellation_cancels_both_channels(self) -> None:
        cancelled: list[str] = []
        started = asyncio.Event()

        async def channel(name: str) -> str:
            try:
                started.set()
                await asyncio.sleep(WAIT_SECONDS)
            except asyncio.CancelledError:
                cancelled.append(name)
                raise
            return name

        task = asyncio.ensure_future(run_hybrid_channels(channel("lexical"), channel("vector")))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert sorted(cancelled) == ["lexical", "vector"]


class TestResolveRetrievalMode:
    def test_hybrid_needs_an_available_embedder(self) -> None:
        assert resolve_retrieval_mode("hybrid", StubEmbedder(available=True)) == "hybrid"
        assert resolve_retrieval_mode("hybrid", StubEmbedder(available=False)) == "lexical"
        assert resolve_retrieval_mode("hybrid", None) == "lexical"

    def test_lexical_setting_never_embeds(self) -> None:
        assert resolve_retrieval_mode("lexical", StubEmbedder(available=True)) == "lexical"
