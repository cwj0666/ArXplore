import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from arxplore.llm.agent import tools as agent_tools
from arxplore_eval.agent_plan import AgentPlan, PlannedSearch, plan_payload, text_sha256
from arxplore_eval.answers import (
    AGENT_ORIGIN_LIVE,
    AGENT_ORIGIN_PLAN_ONLY,
    AGENT_ORIGIN_PLAN_REPLAY,
    AgentOptions,
    AgentTrace,
    AnswersError,
    GenerationResult,
    append_answer_record,
    build_answer_record,
    build_generators,
    collect_answers,
    load_agent_plans,
    load_answer_records,
)
from arxplore_eval.dataset import EvalQuery
from tests.unit.eval.plan_support import FakeRetriever, sample_channels, sample_inputs
from tests.unit.eval.test_eval_paper_chat_ab import _service

QUERY_TEXT = "policy loss"
NEW_KEYS = ("fusion", "agent_origin", "plan_sha256", "searches", "usage", "plan_only", "plan", "tool_output_sha256")


def _q(query_id: str = "a-en", **kwargs: Any) -> EvalQuery:
    base: dict[str, Any] = {
        "query": "What is policy loss?",
        "lang": "en",
        "relevant_arxiv_ids": ("2401.00001",),
        "source": "manual",
    }
    base.update(kwargs)
    return EvalQuery(id=query_id, **base)


def _plan(query_id: str = "a-en") -> AgentPlan:
    search = PlannedSearch(tool_call_id="call_0", query=QUERY_TEXT, inputs=sample_inputs(QUERY_TEXT))
    call = {"name": agent_tools.SEARCH_TOOL_NAME, "args": {"query": QUERY_TEXT}, "id": "call_0"}
    return AgentPlan(
        query_id=query_id,
        content="",
        tool_calls=(call,),
        searches=(search,),
        search_limit=5,
        usage={"input_tokens": 5, "output_tokens": 2},
    )


class TestAgentOptionsValidation:
    def test_defaults_are_live_without_fusion(self) -> None:
        options = AgentOptions()
        assert options.origin == AGENT_ORIGIN_LIVE and options.fusion is None

    def test_fusion_alone_is_live(self) -> None:
        assert AgentOptions(fusion="convex").origin == AGENT_ORIGIN_LIVE

    def test_unknown_fusion(self) -> None:
        with pytest.raises(ValueError, match="알 수 없는 fusion"):
            AgentOptions(fusion="rrf")

    def test_plan_only_with_replay(self) -> None:
        with pytest.raises(ValueError, match="함께 쓸 수 없습니다"):
            AgentOptions(plan_only=True, replay={}, retriever=FakeRetriever())

    def test_plan_only_with_fusion(self) -> None:
        with pytest.raises(ValueError, match="fusion을 받지 않습니다"):
            AgentOptions(plan_only=True, fusion="convex", retriever=FakeRetriever())

    def test_replay_without_fusion(self) -> None:
        with pytest.raises(ValueError, match="fusion이 필요합니다"):
            AgentOptions(replay={}, retriever=FakeRetriever())

    @pytest.mark.parametrize("kwargs", [{"plan_only": True}, {"replay": {}, "fusion": "convex"}])
    def test_missing_retriever(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="Retriever가 필요합니다"):
            AgentOptions(**kwargs)

    def test_origins(self) -> None:
        retriever = FakeRetriever()
        assert AgentOptions(plan_only=True, retriever=retriever).origin == AGENT_ORIGIN_PLAN_ONLY
        assert AgentOptions(replay={}, fusion="convex", retriever=retriever).origin == AGENT_ORIGIN_PLAN_REPLAY

    def test_require_retriever(self) -> None:
        retriever = FakeRetriever()
        assert AgentOptions(plan_only=True, retriever=retriever).require_retriever() is retriever
        with pytest.raises(ValueError, match="Retriever가 없습니다"):
            AgentOptions().require_retriever()


class TestAgentTraceRecordFields:
    def test_live_trace(self) -> None:
        fields = AgentTrace(fusion="convex", origin=AGENT_ORIGIN_LIVE, searches=[{"query": "q"}]).record_fields()
        assert fields == {
            "fusion": "convex",
            "agent_origin": "live",
            "plan_sha256": None,
            "searches": [{"query": "q"}],
            "usage": {},
        }

    def test_plan_only_adds_plan_only_and_plan(self) -> None:
        payload = plan_payload(_plan())
        fields = AgentTrace(fusion=None, origin=AGENT_ORIGIN_PLAN_ONLY, plan=payload).record_fields()
        assert fields["plan_only"] is True
        assert fields["plan"] == payload
        assert "tool_output_sha256" not in fields

    def test_plan_replay_adds_tool_output_hashes(self) -> None:
        trace = AgentTrace(
            fusion="legacy_rules",
            origin=AGENT_ORIGIN_PLAN_REPLAY,
            plan_sha256="abc",
            tool_output_sha256=["h1", "h2"],
            usage={"input_tokens": 3},
        )
        fields = trace.record_fields()
        assert fields["tool_output_sha256"] == ["h1", "h2"]
        assert fields["plan_sha256"] == "abc" and fields["usage"] == {"input_tokens": 3}
        assert "plan_only" not in fields and "plan" not in fields

    def test_record_fields_copy_the_inputs(self) -> None:
        searches = [{"query": "q"}]
        fields = AgentTrace(fusion=None, origin=AGENT_ORIGIN_LIVE, searches=searches).record_fields()
        fields["searches"][0]["query"] = "changed"
        assert searches == [{"query": "q"}]


def _retriever() -> FakeRetriever:
    return FakeRetriever({QUERY_TEXT: sample_channels()})


async def _collect(
    fake_openai: Any, agent: AgentOptions | None, retriever: FakeRetriever, query: EvalQuery | None = None
) -> list[dict[str, Any]]:
    service = _service(fake_openai, retriever)
    queries = [query or _q()]
    return await collect_answers(queries, ["agent"], build_generators(service, ["agent"], agent=agent))


def _search_then_answer(fakes: ModuleType, answer: str = "최종 답변") -> Any:
    def handler(request: Any) -> Any:
        if request["messages"][-1]["role"] == "tool":
            return answer
        return fakes.ToolCall(
            name=agent_tools.SEARCH_TOOL_NAME, arguments=json.dumps({"query": QUERY_TEXT}), id="call_live"
        )

    return handler


class TestLiveAgentRecords:
    async def test_default_live_path_adds_no_new_keys(self, fakes: ModuleType, fake_openai: Any) -> None:
        fake_openai.chat_handler = _search_then_answer(fakes)
        (record,) = await _collect(fake_openai, None, _retriever())
        assert record["answer"] == "최종 답변"
        assert record["retrieval_mode"] == "hybrid"
        assert record["contexts"]
        for key in NEW_KEYS:
            assert key not in record, key

    async def test_default_options_object_adds_no_new_keys(self, fakes: ModuleType, fake_openai: Any) -> None:
        fake_openai.chat_handler = _search_then_answer(fakes)
        (record,) = await _collect(fake_openai, AgentOptions(), _retriever())
        for key in NEW_KEYS:
            assert key not in record, key

    async def test_live_fusion_adds_trace_fields(self, fakes: ModuleType, fake_openai: Any) -> None:
        fake_openai.chat_handler = _search_then_answer(fakes)
        (record,) = await _collect(fake_openai, AgentOptions(fusion="convex"), _retriever())
        assert record["fusion"] == "convex"
        assert record["agent_origin"] == "live"
        assert record["plan_sha256"] is None
        assert [search["query"] for search in record["searches"]] == [QUERY_TEXT]
        assert record["searches"][0]["retrieval_mode"] == "hybrid"
        assert record["searches"][0]["chunk_ids"]
        assert "plan_only" not in record

    async def test_plan_only_records_the_plan_without_an_answer(self, fakes: ModuleType, fake_openai: Any) -> None:
        fake_openai.chat_handler = _search_then_answer(fakes)
        retriever = _retriever()
        options = AgentOptions(plan_only=True, retriever=retriever)
        (record,) = await _collect(fake_openai, options, retriever)
        assert record["plan_only"] is True and record["agent_origin"] == "plan_only"
        assert record["answer"] == "" and record["contexts"] == []
        assert record["fusion"] is None
        assert record["plan"]["kind"] == "tools"
        assert record["searches"] == [{"query": QUERY_TEXT, "tool_call_id": "call_live"}]
        assert len(fake_openai.requests_for("/v1/chat/completions")) == 1

    async def test_plan_replay_records_hashes_and_the_plan_digest(self, fakes: ModuleType, fake_openai: Any) -> None:
        fake_openai.chat_handler = _search_then_answer(fakes)
        plan = _plan()
        retriever = FakeRetriever()
        options = AgentOptions(fusion="convex", replay={"a-en": plan}, replay_sha256="d" * 64, retriever=retriever)
        (record,) = await _collect(fake_openai, options, retriever)
        assert record["agent_origin"] == "plan_replay" and record["fusion"] == "convex"
        assert record["plan_sha256"] == "d" * 64
        assert len(record["tool_output_sha256"]) == 1 and len(record["tool_output_sha256"][0]) == 64
        assert record["answer"] == "최종 답변"
        assert record["searches"][0]["query"] == QUERY_TEXT
        assert record["usage"]["input_tokens"] > 0
        assert "plan_only" not in record

    async def test_plan_replay_hashes_match_the_tool_output_text(self, fakes: ModuleType, fake_openai: Any) -> None:
        fake_openai.chat_handler = _search_then_answer(fakes)
        retriever = FakeRetriever()
        options = AgentOptions(fusion="convex", replay={"a-en": _plan()}, retriever=retriever)
        (record,) = await _collect(fake_openai, options, retriever)
        (request,) = fake_openai.requests_for("/v1/chat/completions")
        tool_message = request.body["messages"][-1]
        assert record["tool_output_sha256"] == [text_sha256(tool_message["content"])]

    async def test_replay_without_a_plan_for_the_query_is_a_lookup_error(
        self, fakes: ModuleType, fake_openai: Any
    ) -> None:
        retriever = FakeRetriever()
        options = AgentOptions(fusion="convex", replay={}, retriever=retriever)
        with pytest.raises(LookupError, match="재생할 에이전트 계획이 없습니다"):
            await _collect(fake_openai, options, retriever)


def _plan_record(query: EvalQuery, agent_plan: AgentPlan | None = None, **extra: Any) -> dict[str, Any]:
    trace = AgentTrace(fusion=None, origin=AGENT_ORIGIN_PLAN_ONLY, plan=plan_payload(agent_plan or _plan(query.id)))
    record = build_answer_record(query, "agent", result=GenerationResult(answer="", contexts=[], agent=trace))
    return {**record, **extra}


class TestLoadAgentPlans:
    def test_loads_plans_and_questions(self, tmp_path: Path) -> None:
        path = tmp_path / "plans.jsonl"
        append_answer_record(path, _plan_record(_q("a-en")))
        append_answer_record(path, _plan_record(_q("b-en", query="Other?"), _plan("b-en")))
        plans, questions = load_agent_plans(path)
        assert plans == {"a-en": _plan("a-en"), "b-en": _plan("b-en")}
        assert questions == {"a-en": "What is policy loss?", "b-en": "Other?"}

    def test_non_plan_record_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "answers.jsonl"
        append_answer_record(path, build_answer_record(_q(), "agent", result=GenerationResult(answer="a", contexts=[])))
        with pytest.raises(AnswersError, match=r"a-en.*--plan-only"):
            load_agent_plans(path)

    def test_error_and_rejected_records_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "plans.jsonl"
        append_answer_record(path, build_answer_record(_q("e-en"), "agent", error="Boom: x"))
        append_answer_record(path, build_answer_record(_q("r-en"), "agent", rejection="too long"))
        append_answer_record(path, _plan_record(_q("a-en")))
        plans, questions = load_agent_plans(path)
        assert list(plans) == ["a-en"] and list(questions) == ["a-en"]

    def test_paper_chat_records_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "plans.jsonl"
        append_answer_record(
            path, build_answer_record(_q("p-en"), "paper_chat", result=GenerationResult(answer="a", contexts=[]))
        )
        append_answer_record(path, _plan_record(_q("a-en")))
        plans, _ = load_agent_plans(path)
        assert list(plans) == ["a-en"]

    def test_malformed_plan_is_an_answers_error(self, tmp_path: Path) -> None:
        path = tmp_path / "plans.jsonl"
        append_answer_record(path, _plan_record(_q("a-en"), plan={"kind": "tools"}))
        with pytest.raises(AnswersError, match="계획 형식 오류"):
            load_agent_plans(path)

    def test_missing_plan_value_is_an_answers_error(self, tmp_path: Path) -> None:
        path = tmp_path / "plans.jsonl"
        append_answer_record(path, _plan_record(_q("a-en"), plan=None))
        with pytest.raises(AnswersError, match="must be an object"):
            load_agent_plans(path)

    def test_later_record_for_the_same_id_wins(self, tmp_path: Path) -> None:
        path = tmp_path / "plans.jsonl"
        append_answer_record(path, _plan_record(_q("a-en"), plan=None))
        append_answer_record(path, _plan_record(_q("a-en")))
        plans, _ = load_agent_plans(path)
        assert list(plans) == ["a-en"]
        assert len(load_answer_records(path)) == 1
