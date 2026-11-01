import json
from types import ModuleType
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from arxplore.context import observe_retrieval
from arxplore.llm.agent import tools as agent_tools
from arxplore.llm.agent.stream import EMPTY_ANSWER_MESSAGE
from arxplore.llm.agent.tools import format_search_results
from arxplore.llm.prompts.agent import AGENT_SYSTEM_PROMPT
from arxplore_eval.agent_plan import (
    PLAN_KIND_DIRECT,
    PLAN_KIND_TOOLS,
    AgentPlan,
    PlanError,
    PlannedSearch,
    PlanReplayRetriever,
    answer_from_plan,
    build_capture_graph,
    capture_plan,
    plan_from_payload,
    plan_payload,
    text_sha256,
    usage_of,
)
from arxplore_eval.fusion_variants import FUSION_CONVEX, FUSION_LEGACY_RULES, fuse_inputs
from tests.unit.eval.conftest import build_context
from tests.unit.eval.plan_support import FakeRetriever, sample_channels, sample_inputs
from tests.unit.eval.test_eval_paper_chat_ab import _service

QUERY = "policy loss"
LIMIT = agent_tools.SEARCH_RESULT_LIMIT


def _call(fakes: ModuleType, query: str = QUERY, call_id: str = "call_a") -> Any:
    return fakes.ToolCall(name=agent_tools.SEARCH_TOOL_NAME, arguments=json.dumps({"query": query}), id=call_id)


def _chat_requests(fake_openai: Any) -> list[Any]:
    return fake_openai.requests_for("/v1/chat/completions")


def _retriever(**kwargs: Any) -> FakeRetriever:
    return FakeRetriever({QUERY: sample_channels(), "dpo": sample_channels()}, **kwargs)


async def _capture(
    fake_openai: Any, retriever: FakeRetriever, *, history: list[tuple[str, str]] | None = None
) -> AgentPlan:
    service = _service(fake_openai, retriever)
    return await capture_plan(
        service,
        retriever,
        build_capture_graph(service),
        query_id="q-1",
        question="질문입니다",
        history=history or [],
    )


def _plan(*queries: str, tool_names: tuple[str, ...] | None = None) -> AgentPlan:
    searches = tuple(
        PlannedSearch(tool_call_id=f"call_{index}", query=query, inputs=sample_inputs(query))
        for index, query in enumerate(queries)
    )
    calls = tuple(
        {"name": agent_tools.SEARCH_TOOL_NAME, "args": {"query": search.query}, "id": search.tool_call_id}
        for search in searches
    )
    if tool_names is not None:
        calls = tuple({"name": name, "args": {}, "id": f"call_t{index}"} for index, name in enumerate(tool_names))
    return AgentPlan(
        query_id="q-1",
        content="검색하겠습니다",
        tool_calls=calls,
        searches=searches,
        search_limit=LIMIT,
        usage={"input_tokens": 7},
    )


class TestUsage:
    def test_reads_input_and_output_tokens(self) -> None:
        message = AIMessage(content="x", usage_metadata={"input_tokens": 12, "output_tokens": 3, "total_tokens": 15})
        assert usage_of(message) == {"input_tokens": 12, "output_tokens": 3}

    def test_missing_metadata_gives_empty_usage(self) -> None:
        assert usage_of(AIMessage(content="x")) == {}

    def test_partial_metadata(self) -> None:
        message = AIMessage(content="x", usage_metadata={"input_tokens": 4, "total_tokens": 4, "output_tokens": 0})
        assert usage_of(message) == {"input_tokens": 4, "output_tokens": 0}


class TestCapturePlan:
    async def test_search_call_runs_the_graph_only_to_the_first_model_turn(
        self, fakes: ModuleType, fake_openai: Any
    ) -> None:
        fake_openai.chat_handler = lambda request: _call(fakes)
        retriever = _retriever()
        plan = await _capture(fake_openai, retriever)
        assert len(_chat_requests(fake_openai)) == 1
        assert plan.kind == PLAN_KIND_TOOLS
        assert plan.tool_calls == ({"name": agent_tools.SEARCH_TOOL_NAME, "args": {"query": QUERY}, "id": "call_a"},)
        assert plan.search_limit == LIMIT
        assert retriever.hybrid_calls == [(QUERY, LIMIT, None)]
        (search,) = plan.searches
        assert (search.tool_call_id, search.query) == ("call_a", QUERY)
        assert search.inputs == await retriever.hybrid_inputs(QUERY, limit=LIMIT)
        assert plan.usage.keys() <= {"input_tokens", "output_tokens"}
        assert plan.query_id == "q-1"

    async def test_request_carries_the_system_prompt_history_and_question(
        self, fakes: ModuleType, fake_openai: Any
    ) -> None:
        fake_openai.chat_handler = lambda request: _call(fakes)
        await _capture(fake_openai, _retriever(), history=[("user", "이전"), ("assistant", "답")])
        (request,) = _chat_requests(fake_openai)
        assert [(m["role"], m["content"]) for m in request.body["messages"]] == [
            ("system", AGENT_SYSTEM_PROMPT),
            ("user", "이전"),
            ("assistant", "답"),
            ("user", "질문입니다"),
        ]

    async def test_trending_calls_are_kept_but_not_searched(self, fakes: ModuleType, fake_openai: Any) -> None:
        fake_openai.chat_handler = lambda request: [
            _call(fakes, QUERY, "call_a"),
            fakes.ToolCall(name=agent_tools.TRENDING_TOOL_NAME, arguments="{}", id="call_b"),
            _call(fakes, "dpo", "call_c"),
        ]
        retriever = _retriever()
        plan = await _capture(fake_openai, retriever)
        assert len(_chat_requests(fake_openai)) == 1
        assert [call["name"] for call in plan.tool_calls] == [
            agent_tools.SEARCH_TOOL_NAME,
            agent_tools.TRENDING_TOOL_NAME,
            agent_tools.SEARCH_TOOL_NAME,
        ]
        assert [call["id"] for call in plan.tool_calls] == ["call_a", "call_b", "call_c"]
        assert plan.tool_calls[1]["args"] == {}
        assert [(s.tool_call_id, s.query) for s in plan.searches] == [("call_a", QUERY), ("call_c", "dpo")]
        assert [call[0] for call in retriever.hybrid_calls] == [QUERY, "dpo"]

    async def test_trending_only_has_no_searches_and_needs_no_hybrid_retriever(
        self, fakes: ModuleType, fake_openai: Any
    ) -> None:
        fake_openai.chat_handler = lambda request: fakes.ToolCall(name=agent_tools.TRENDING_TOOL_NAME, id="call_t")
        retriever = _retriever(mode="lexical")
        plan = await _capture(fake_openai, retriever)
        assert plan.kind == PLAN_KIND_TOOLS
        assert plan.searches == ()
        assert retriever.hybrid_calls == []

    async def test_no_tool_call_is_a_direct_plan(self, fake_openai: Any) -> None:
        fake_openai.chat_handler = lambda request: "바로 답합니다."
        retriever = _retriever()
        plan = await _capture(fake_openai, retriever)
        assert plan.kind == PLAN_KIND_DIRECT
        assert plan.tool_calls == () and plan.searches == ()
        assert plan.content == "바로 답합니다."
        assert retriever.hybrid_calls == []
        assert len(_chat_requests(fake_openai)) == 1

    async def test_preamble_text_is_kept(self, fakes: ModuleType, fake_openai: Any) -> None:
        fake_openai.chat_handler = lambda request: fakes.ChatReply(
            content="먼저 검색합니다.", tool_calls=(_call(fakes),)
        )
        plan = await _capture(fake_openai, _retriever())
        assert plan.content == "먼저 검색합니다."

    async def test_unknown_tool_name_is_a_plan_error(self, fakes: ModuleType, fake_openai: Any) -> None:
        fake_openai.chat_handler = lambda request: [_call(fakes), fakes.ToolCall(name="bogus_tool", id="call_x")]
        retriever = _retriever()
        with pytest.raises(PlanError, match="bogus_tool"):
            await _capture(fake_openai, retriever)
        assert retriever.hybrid_calls == []

    async def test_search_without_hybrid_mode_is_a_plan_error(self, fakes: ModuleType, fake_openai: Any) -> None:
        fake_openai.chat_handler = lambda request: _call(fakes)
        retriever = _retriever(mode="lexical")
        with pytest.raises(PlanError, match="hybrid"):
            await _capture(fake_openai, retriever)
        assert retriever.hybrid_calls == []

    async def test_search_arguments_without_a_query_search_the_empty_string(
        self, fakes: ModuleType, fake_openai: Any
    ) -> None:
        fake_openai.chat_handler = lambda request: fakes.ToolCall(
            name=agent_tools.SEARCH_TOOL_NAME, arguments=json.dumps({"query": ""}), id="call_e"
        )
        plan = await _capture(fake_openai, _retriever())
        (search,) = plan.searches
        assert search.query == "" and search.inputs.query == ""


class TestPlanPayload:
    def test_round_trips_through_json(self) -> None:
        plan = _plan(QUERY, "dpo")
        payload = json.loads(json.dumps(plan_payload(plan), ensure_ascii=False))
        assert plan_from_payload("q-1", payload) == plan

    def test_direct_plan_round_trips(self) -> None:
        plan = AgentPlan(query_id="q-2", content="바로", tool_calls=(), searches=(), search_limit=5, usage={})
        payload = plan_payload(plan)
        assert payload["kind"] == PLAN_KIND_DIRECT
        assert plan_from_payload("q-2", json.loads(json.dumps(payload))) == plan

    def test_payload_kind_and_shape(self) -> None:
        payload = plan_payload(_plan(QUERY))
        assert payload["kind"] == PLAN_KIND_TOOLS
        assert payload["search_limit"] == LIMIT
        assert payload["usage"] == {"input_tokens": 7}
        assert payload["searches"][0]["tool_call_id"] == "call_0"

    def test_missing_usage_and_content_default(self) -> None:
        payload = plan_payload(_plan(QUERY))
        del payload["usage"]
        del payload["content"]
        restored = plan_from_payload("q-1", payload)
        assert restored.usage == {} and restored.content == ""

    @pytest.mark.parametrize("payload", [None, [], "plan", 3])
    def test_non_object_payload(self, payload: Any) -> None:
        with pytest.raises(PlanError, match="must be an object"):
            plan_from_payload("q-1", payload)

    @pytest.mark.parametrize("key", ["tool_calls", "searches", "search_limit"])
    def test_missing_required_key(self, key: str) -> None:
        payload = plan_payload(_plan(QUERY))
        del payload[key]
        with pytest.raises(PlanError, match="계획 형식 오류"):
            plan_from_payload("q-1", payload)

    def test_bad_search_limit(self) -> None:
        payload = plan_payload(_plan(QUERY))
        payload["search_limit"] = "many"
        with pytest.raises(PlanError, match="계획 형식 오류"):
            plan_from_payload("q-1", payload)

    def test_search_entry_missing_inputs(self) -> None:
        payload = plan_payload(_plan(QUERY))
        del payload["searches"][0]["inputs"]
        with pytest.raises(PlanError, match="계획 형식 오류"):
            plan_from_payload("q-1", payload)

    def test_tool_call_without_a_name(self) -> None:
        payload = plan_payload(_plan(QUERY))
        payload["tool_calls"] = [{"args": {}, "id": "x"}]
        with pytest.raises(PlanError, match="계획 형식 오류"):
            plan_from_payload("q-1", payload)

    def test_non_numeric_usage(self) -> None:
        payload = plan_payload(_plan(QUERY))
        payload["usage"] = {"input_tokens": "lots"}
        with pytest.raises(PlanError, match="계획 형식 오류"):
            plan_from_payload("q-1", payload)

    def test_tool_calls_not_iterable(self) -> None:
        payload = plan_payload(_plan(QUERY))
        payload["tool_calls"] = 5
        with pytest.raises(PlanError, match="계획 형식 오류"):
            plan_from_payload("q-1", payload)


class TestPlanReplayRetriever:
    def test_search_inputs_map_by_query(self) -> None:
        plan = _plan(QUERY, "dpo")
        assert list(plan.search_inputs()) == [QUERY, "dpo"]

    async def test_unknown_query_is_a_plan_error(self) -> None:
        replay = PlanReplayRetriever(FakeRetriever(), _plan(QUERY), FUSION_CONVEX)
        with pytest.raises(PlanError, match="계획에 없는 검색 질의"):
            await replay.retrieve_contexts("other", limit=LIMIT)

    async def test_limit_other_than_the_plan_limit_is_a_plan_error(self) -> None:
        replay = PlanReplayRetriever(FakeRetriever(), _plan(QUERY), FUSION_CONVEX)
        with pytest.raises(PlanError, match="계획"):
            await replay.retrieve_contexts(QUERY, limit=LIMIT + 1)

    @pytest.mark.parametrize("fusion", [FUSION_CONVEX, FUSION_LEGACY_RULES])
    async def test_fuses_the_stored_inputs_and_builds_contexts(self, fusion: str) -> None:
        retriever = FakeRetriever()
        plan = _plan(QUERY)
        with observe_retrieval() as observations:
            result = await PlanReplayRetriever(retriever, plan, fusion).retrieve_contexts(QUERY, limit=LIMIT)
        expected = fuse_inputs(plan.searches[0].inputs, fusion, limit=LIMIT)
        assert retriever.built == [[hit.chunk_id for hit in expected]]
        assert [context.hit for context in result.contexts] == expected
        assert result.mode == "hybrid"
        (observation,) = observations
        assert (observation.query, observation.limit, observation.result) == (QUERY, LIMIT, result)

    async def test_never_touches_the_search_channels(self) -> None:
        retriever = FakeRetriever(embed_error=True)
        result = await PlanReplayRetriever(retriever, _plan(QUERY), FUSION_CONVEX).retrieve_contexts(QUERY, limit=LIMIT)
        assert result.contexts
        assert retriever.hybrid_calls == []


def _tool_contexts(plan: AgentPlan, fusion: str, query: str = QUERY) -> str:
    inputs = plan.search_inputs()[query]
    hits = fuse_inputs(inputs, fusion, limit=LIMIT)
    return format_search_results([build_context(hit) for hit in hits])


class TestAnswerFromPlan:
    async def _answer(self, fake_openai: Any, plan: AgentPlan, fusion: str, **kwargs: Any):
        retriever = FakeRetriever()
        service = _service(fake_openai, retriever)
        return await answer_from_plan(
            service, retriever, plan, fusion, question="질문입니다", history=kwargs.get("history", [])
        )

    async def test_sends_system_history_question_planned_call_and_real_tool_output(self, fake_openai: Any) -> None:
        fake_openai.chat_handler = lambda request: "최종 답변"
        plan = _plan(QUERY)
        result = await self._answer(fake_openai, plan, FUSION_CONVEX, history=[("user", "이전"), ("assistant", "답")])
        (request,) = _chat_requests(fake_openai)
        messages = request.body["messages"]
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "tool"]
        assert messages[0]["content"] == AGENT_SYSTEM_PROMPT
        assert [m["content"] for m in messages[1:4]] == ["이전", "답", "질문입니다"]
        (planned,) = messages[4]["tool_calls"]
        assert planned["id"] == "call_0"
        assert planned["function"]["name"] == agent_tools.SEARCH_TOOL_NAME
        assert json.loads(planned["function"]["arguments"]) == {"query": QUERY}
        assert messages[5]["tool_call_id"] == "call_0"
        assert messages[5]["content"] == _tool_contexts(plan, FUSION_CONVEX)
        assert result.tool_outputs == [messages[5]["content"]]
        assert result.answer == "최종 답변"

    async def test_final_call_is_bound_with_tools_and_tool_choice_none(self, fake_openai: Any) -> None:
        fake_openai.chat_handler = lambda request: "답"
        await self._answer(fake_openai, _plan(QUERY), FUSION_CONVEX)
        (request,) = _chat_requests(fake_openai)
        assert request.body["tool_choice"] == "none"
        assert [tool["function"]["name"] for tool in request.body["tools"]] == [
            agent_tools.SEARCH_TOOL_NAME,
            agent_tools.TRENDING_TOOL_NAME,
        ]
        assert request.body["temperature"] == 0.0

    async def test_direct_plan_sends_only_system_history_and_question(self, fake_openai: Any) -> None:
        fake_openai.chat_handler = lambda request: "바로 답"
        plan = AgentPlan(query_id="q-1", content="바로 답", tool_calls=(), searches=(), search_limit=LIMIT)
        result = await self._answer(fake_openai, plan, FUSION_CONVEX, history=[("user", "이전"), ("assistant", "답")])
        (request,) = _chat_requests(fake_openai)
        assert [(m["role"], m["content"]) for m in request.body["messages"]] == [
            ("system", AGENT_SYSTEM_PROMPT),
            ("user", "이전"),
            ("assistant", "답"),
            ("user", "질문입니다"),
        ]
        assert result.tool_outputs == [] and result.observations == []
        assert result.citations == []

    async def test_trending_calls_run_the_real_trending_tool(self, fake_openai: Any) -> None:
        fake_openai.chat_handler = lambda request: "답"
        plan = _plan(tool_names=(agent_tools.TRENDING_TOOL_NAME,))
        result = await self._answer(fake_openai, plan, FUSION_CONVEX)
        (output,) = result.tool_outputs
        assert "Paper Title" in output and "추천수" in output
        assert result.observations == []

    async def test_mixed_calls_keep_their_order(self, fake_openai: Any) -> None:
        fake_openai.chat_handler = lambda request: "답"
        plan = _plan(QUERY)
        plan = AgentPlan(
            query_id=plan.query_id,
            content="",
            tool_calls=(
                {"name": agent_tools.TRENDING_TOOL_NAME, "args": {}, "id": "call_t"},
                *plan.tool_calls,
            ),
            searches=plan.searches,
            search_limit=plan.search_limit,
        )
        await self._answer(fake_openai, plan, FUSION_CONVEX)
        (request,) = _chat_requests(fake_openai)
        roles = [(m["role"], m.get("tool_call_id")) for m in request.body["messages"][-3:]]
        assert roles == [("assistant", None), ("tool", "call_t"), ("tool", "call_0")]

    async def test_empty_model_text_becomes_the_empty_answer_message(self, fake_openai: Any) -> None:
        fake_openai.chat_handler = lambda request: ""
        result = await self._answer(fake_openai, _plan(QUERY), FUSION_CONVEX)
        assert result.answer == EMPTY_ANSWER_MESSAGE

    async def test_citations_come_from_the_tool_hits(self, fake_openai: Any) -> None:
        plan = _plan(QUERY)
        hits = fuse_inputs(plan.searches[0].inputs, FUSION_CONVEX, limit=LIMIT)
        cited = hits[0]
        fake_openai.chat_handler = lambda request: f"- [Paper](https://arxiv.org/abs/{cited.arxiv_id}) 요약"
        result = await self._answer(fake_openai, plan, FUSION_CONVEX)
        by_id = {citation.arxiv_id: citation for citation in result.citations}
        assert by_id[cited.arxiv_id].in_answer is True
        assert set(by_id) == {cited.arxiv_id}
        assert by_id[cited.arxiv_id].chunk_id in {hit.chunk_id for hit in hits if hit.arxiv_id == cited.arxiv_id}

    async def test_answer_without_links_cites_every_hit_as_not_in_answer(self, fake_openai: Any) -> None:
        fake_openai.chat_handler = lambda request: "링크 없는 답"
        result = await self._answer(fake_openai, _plan(QUERY), FUSION_CONVEX)
        assert result.citations
        assert all(not citation.in_answer for citation in result.citations)

    async def test_observations_and_usage(self, fake_openai: Any) -> None:
        fake_openai.chat_handler = lambda request: "답"
        result = await self._answer(fake_openai, _plan(QUERY), FUSION_CONVEX)
        (observation,) = result.observations
        assert (observation.query, observation.limit, observation.result.mode) == (QUERY, LIMIT, "hybrid")
        assert result.usage["input_tokens"] > 0 and result.usage["output_tokens"] > 0

    @pytest.mark.parametrize("fusion", [FUSION_CONVEX, FUSION_LEGACY_RULES])
    async def test_same_plan_and_fusion_give_identical_tool_outputs(self, fake_openai: Any, fusion: str) -> None:
        fake_openai.chat_handler = lambda request: "답"
        plan = _plan(QUERY)
        first = await self._answer(fake_openai, plan, fusion)
        second = await self._answer(fake_openai, plan, fusion)
        assert first.tool_outputs == second.tool_outputs
        assert [text_sha256(output) for output in first.tool_outputs] == [
            text_sha256(output) for output in second.tool_outputs
        ]
        assert first.tool_outputs == [_tool_contexts(plan, fusion)]

    async def test_a_query_outside_the_plan_fails_the_replay(self, fake_openai: Any) -> None:
        fake_openai.chat_handler = lambda request: "답"
        plan = _plan(QUERY)
        broken = AgentPlan(
            query_id=plan.query_id,
            content="",
            tool_calls=({"name": agent_tools.SEARCH_TOOL_NAME, "args": {"query": "other"}, "id": "call_z"},),
            searches=plan.searches,
            search_limit=plan.search_limit,
        )
        with pytest.raises(Exception, match="other"):
            await self._answer(fake_openai, broken, FUSION_CONVEX)

    async def test_unknown_fusion_is_rejected(self, fake_openai: Any) -> None:
        fake_openai.chat_handler = lambda request: "답"
        with pytest.raises(ValueError, match="알 수 없는 fusion"):
            await self._answer(fake_openai, _plan(QUERY), "nope")


def test_text_sha256_is_hex_of_utf8() -> None:
    import hashlib

    assert text_sha256("한글") == hashlib.sha256("한글".encode()).hexdigest()
