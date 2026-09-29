import asyncio

from arxplore.context import (
    RetrievalObservation,
    observe_retrieval,
    openai_runtime,
    override_openai_runtime,
    record_retrieval,
    request_openai_api_key,
    request_openai_model,
)
from arxplore.domain.retrieval import RetrievalResult


def _observation(query: str) -> RetrievalObservation:
    return RetrievalObservation(
        query=query, arxiv_id=None, limit=5, result=RetrievalResult(contexts=(), mode="lexical")
    )


class TestOverrideOpenAIRuntime:
    def test_nothing_is_overridden_by_default(self) -> None:
        assert request_openai_api_key() is None
        assert request_openai_model() is None

    def test_override_is_scoped_to_the_block(self) -> None:
        with override_openai_runtime(api_key=" sk-request ", model="gpt-4o-mini"):
            assert request_openai_api_key() == "sk-request"
            assert request_openai_model() == "gpt-4o-mini"
        assert request_openai_api_key() is None
        assert request_openai_model() is None

    def test_blank_key_counts_as_no_override_but_stays_visible_raw(self) -> None:
        with override_openai_runtime(api_key="  "):
            assert request_openai_api_key() is None
            assert openai_runtime().api_key == "  "

    def test_nested_overrides_keep_the_other_field(self) -> None:
        with override_openai_runtime(api_key="sk-a"), override_openai_runtime(model="gpt-4o"):
            assert (request_openai_api_key(), request_openai_model()) == ("sk-a", "gpt-4o")
            with override_openai_runtime(api_key="sk-b"):
                assert (request_openai_api_key(), request_openai_model()) == ("sk-b", "gpt-4o")
            assert request_openai_api_key() == "sk-a"

    async def test_concurrent_requests_do_not_share_keys(self) -> None:
        async def request(key: str) -> str | None:
            with override_openai_runtime(api_key=key):
                await asyncio.sleep(0)
                return request_openai_api_key()

        assert await asyncio.gather(request("sk-a"), request("sk-b")) == ["sk-a", "sk-b"]


class TestObserveRetrieval:
    def test_records_only_inside_the_block(self) -> None:
        record_retrieval(_observation("before"))
        with observe_retrieval() as observations:
            record_retrieval(_observation("inside"))
        record_retrieval(_observation("after"))

        assert [observation.query for observation in observations] == ["inside"]

    def test_nested_observers_both_receive_records(self) -> None:
        with observe_retrieval() as outer:
            record_retrieval(_observation("first"))
            with observe_retrieval() as inner:
                record_retrieval(_observation("second"))

        assert [observation.query for observation in outer] == ["first", "second"]
        assert [observation.query for observation in inner] == ["second"]

    async def test_records_from_child_tasks_reach_the_observer(self) -> None:
        async def child() -> None:
            record_retrieval(_observation("child"))

        with observe_retrieval() as observations:
            await asyncio.gather(child(), child())

        assert [observation.query for observation in observations] == ["child", "child"]
