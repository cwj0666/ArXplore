import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from arxplore.api.services.chat import CHAT_HISTORY_MAX_MESSAGES, build_chat_service
from arxplore.config import AppSettings
from arxplore.context import RetrievalObservation, record_retrieval
from arxplore.domain.chunks import StoredChunk
from arxplore.domain.papers import Paper
from arxplore.domain.retrieval import RetrievalContext, RetrievalResult
from arxplore.llm.agent import tools as agent_tools
from arxplore.llm.models import ChatModelFactory
from arxplore.llm.tracing import Tracer
from arxplore_eval.answers import (
    AnswersError,
    CapturedContexts,
    GenerationResult,
    agent_search_limit,
    append_answer_record,
    build_answer_record,
    build_generators,
    collect_answers,
    completed_keys,
    generate_agent_answer,
    generate_paper_chat_answer,
    load_answer_records,
    load_reference_answers,
    resolve_modes,
)
from arxplore_eval.behavior import OUTCOME_REFUSED, OUTCOME_REJECTED
from arxplore_eval.dataset import EvalQuery

QUERIES = [
    EvalQuery(
        id="ki-2401.00001-ko",
        query="선호 최적화를 다시 본 논문",
        lang="ko",
        relevant_arxiv_ids=("2401.00001",),
        source="known_item",
    ),
    EvalQuery(
        id="cs-11",
        query="What learning rate was used?",
        lang="en",
        relevant_arxiv_ids=("2401.00002",),
        source="llm_synth",
        relevant_chunk_ids=(11,),
    ),
]


class FakeClock:
    def __init__(self, step: float = 0.25) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def _result(answer: str = "answer", contexts: list[str] | None = None) -> GenerationResult:
    return GenerationResult(
        answer=answer,
        contexts=contexts if contexts is not None else ["ctx a", "ctx b"],
        citations=[{"arxiv_id": "2401.00001", "in_answer": True}],
        retrieval_mode="hybrid",
    )


def _q(query_id: str, **kwargs: Any) -> EvalQuery:
    base: dict[str, Any] = {
        "query": "question",
        "lang": "en",
        "relevant_arxiv_ids": ("2401.00001",),
        "source": "manual",
    }
    base.update(kwargs)
    return EvalQuery(id=query_id, **base)


def _returning(result: GenerationResult):
    async def generate(query: EvalQuery) -> GenerationResult:
        return result

    return generate


class TestModes:
    def test_both_expands_and_dedupes(self) -> None:
        assert resolve_modes(["both", "agent"]) == ["agent", "paper_chat"]
        assert resolve_modes([" paper_chat "]) == ["paper_chat"]

    @pytest.mark.parametrize("values", [["chat"], [""]])
    def test_invalid_modes(self, values: list[str]) -> None:
        with pytest.raises(ValueError):
            resolve_modes(values)


class TestCollectAnswers:
    async def test_builds_records_from_generators(self) -> None:
        sunk: list[dict[str, Any]] = []

        async def agent(query: EvalQuery) -> GenerationResult:
            return _result(answer=f"agent:{query.id}")

        async def paper_chat(query: EvalQuery) -> GenerationResult:
            return GenerationResult(
                answer="chat",
                contexts=["abstract"],
                retrieval_mode="first_chunks",
                arxiv_id=query.relevant_arxiv_ids[0],
            )

        records = await collect_answers(
            QUERIES,
            ["agent", "paper_chat"],
            {"agent": agent, "paper_chat": paper_chat},
            reference_answers={"cs-11": "3e-4"},
            model="gpt-4o",
            sink=sunk.append,
            clock=FakeClock(0.5),
        )

        assert records == sunk
        assert [(record["id"], record["mode"]) for record in records] == [
            ("ki-2401.00001-ko", "agent"),
            ("ki-2401.00001-ko", "paper_chat"),
            ("cs-11", "agent"),
            ("cs-11", "paper_chat"),
        ]
        first = records[0]
        assert first["answer"] == "agent:ki-2401.00001-ko"
        assert first["contexts"] == ["ctx a", "ctx b"]
        assert first["latency_ms"] == 500.0
        assert (first["model"], first["reference_answer"], first["error"]) == ("gpt-4o", None, None)
        assert records[3]["reference_answer"] == "3e-4"
        assert records[3]["relevant_chunk_ids"] == [11]
        assert (records[3]["arxiv_id"], records[3]["retrieval_mode"]) == ("2401.00002", "first_chunks")

    async def test_done_keys_are_skipped(self) -> None:
        calls: list[str] = []

        async def generate(query: EvalQuery) -> GenerationResult:
            calls.append(query.id)
            return _result()

        records = await collect_answers(QUERIES, ["agent"], {"agent": generate}, done={("ki-2401.00001-ko", "agent")})
        assert calls == ["cs-11"]
        assert len(records) == 1

    async def test_keep_going_records_error_and_otherwise_raises_after_sinking(self) -> None:
        async def generate(query: EvalQuery) -> GenerationResult:
            if query.id == "cs-11":
                raise RuntimeError("boom")
            return _result()

        records = await collect_answers(QUERIES, ["agent"], {"agent": generate}, keep_going=True)
        assert records[0]["error"] is None
        assert records[1]["error"] == "RuntimeError: boom"
        assert (records[1]["answer"], records[1]["contexts"], records[1]["latency_ms"]) == ("", [], None)

        sunk: list[dict[str, Any]] = []
        with pytest.raises(RuntimeError):
            await collect_answers(QUERIES, ["agent"], {"agent": generate}, sink=sunk.append)
        assert [record["id"] for record in sunk] == ["ki-2401.00001-ko"]

    async def test_rejected_inputs_skip_generation_and_mode_hints_filter(self) -> None:
        calls: list[tuple[str, str, str]] = []

        def generator(mode: str):
            async def generate(query: EvalQuery) -> GenerationResult:
                calls.append((mode, query.id, query.query))
                return GenerationResult(answer="찾을 수 없습니다.", contexts=[], hit_arxiv_ids=["2401.00001"])

            return generate

        queries = [
            _q("ok", query="  padded  ", category="safety", mode="agent"),
            _q("empty", query=" ", expect_rejected=True, expected_behavior="clarify", category="safety"),
            _q("chat-only", mode="paper_chat", expected_behavior="refuse"),
        ]
        records = await collect_answers(
            queries, ["agent", "paper_chat"], {m: generator(m) for m in ("agent", "paper_chat")}
        )

        assert [(record["id"], record["mode"]) for record in records] == [
            ("ok", "agent"),
            ("empty", "agent"),
            ("empty", "paper_chat"),
            ("chat-only", "paper_chat"),
        ]
        assert calls == [("agent", "ok", "padded"), ("paper_chat", "chat-only", "question")]
        empty = records[1]
        assert (empty["outcome"], empty["rejection"]) == (OUTCOME_REJECTED, "메시지를 입력하세요.")
        assert empty["error"] is None and empty["answer"] == "" and empty["latency_ms"] is None
        assert empty["expect_rejected"] is True
        assert records[0]["outcome"] == OUTCOME_REFUSED
        assert records[0]["hit_arxiv_ids"] == ["2401.00001"]
        assert records[0]["history_turns"] == 0

    async def test_generators_receive_truncated_history(self) -> None:
        seen: list[int] = []
        history = tuple(("user" if index % 2 == 0 else "assistant", f"t{index}") for index in range(26))

        async def generate(query: EvalQuery) -> GenerationResult:
            seen.append(len(query.history))
            return GenerationResult(answer="ok", contexts=["c"])

        records = await collect_answers([_q("h", history=history, mode="agent")], ["agent"], {"agent": generate})
        assert seen == [CHAT_HISTORY_MAX_MESSAGES]
        assert records[0]["history_turns"] == CHAT_HISTORY_MAX_MESSAGES

    async def test_custom_refusal_phrases(self) -> None:
        records = await collect_answers(
            [_q("x", mode="agent")],
            ["agent"],
            {"agent": _returning(GenerationResult(answer="NOPE", contexts=[]))},
            refusal_phrases=["nope"],
        )
        assert records[0]["outcome"] == OUTCOME_REFUSED


class TestAnswerFiles:
    def test_append_load_last_wins_and_completed_keys(self, tmp_path) -> None:
        path = tmp_path / "nested" / "answers.jsonl"
        failed = build_answer_record(QUERIES[0], "agent", error="RuntimeError: x")
        ok = build_answer_record(QUERIES[0], "agent", result=_result())
        other = build_answer_record(QUERIES[1], "paper_chat", error="LookupError: y")
        for record in (failed, other, ok):
            append_answer_record(path, record)

        records = load_answer_records(path)

        assert [(record["id"], record["mode"], record["error"]) for record in records] == [
            ("cs-11", "paper_chat", "LookupError: y"),
            ("ki-2401.00001-ko", "agent", None),
        ]
        assert completed_keys(records) == {("ki-2401.00001-ko", "agent")}

    @pytest.mark.parametrize(
        "line",
        [
            "not json",
            json.dumps(["list"]),
            json.dumps({"id": "a", "query": "q", "mode": "agent", "answer": ""}),
            json.dumps({"id": "a", "query": "q", "mode": "web", "answer": "", "contexts": []}),
            json.dumps({"id": "a", "query": "q", "mode": "agent", "answer": "", "contexts": [1]}),
        ],
    )
    def test_malformed_lines(self, tmp_path, line: str) -> None:
        path = tmp_path / "answers.jsonl"
        path.write_text(line + "\n", encoding="utf-8")
        with pytest.raises(AnswersError):
            load_answer_records(path)

    def test_reference_answers_are_optional(self, tmp_path) -> None:
        path = tmp_path / "queries.jsonl"
        lines = [
            {**QUERIES[0].to_dict(), "reference_answer": "  DPO를 다시 검토한다. "},
            {**QUERIES[1].to_dict(), "reference_answer": ""},
        ]
        path.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n", encoding="utf-8")
        assert load_reference_answers(path) == {"ki-2401.00001-ko": "DPO를 다시 검토한다."}


def _paper(arxiv_id: str = "2401.00002", abstract: str = "We study X.") -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title="Paper",
        authors=[],
        abstract=abstract,
        primary_category=None,
        categories=[],
        pdf_url=None,
        published_at=None,
        updated_at=None,
        upvotes=3,
        github_url=None,
        github_stars=None,
        citation_count=None,
    )


@dataclass
class FakeStore:
    papers: dict[str, Paper] = field(default_factory=dict)

    async def get_paper(self, arxiv_id: str) -> Paper | None:
        return self.papers.get(arxiv_id)

    async def list_recent_papers(self, *, limit: int) -> Sequence[Paper]:
        return list(self.papers.values())[:limit]

    async def list_chunks(self, arxiv_id: str, *, limit: int) -> Sequence[StoredChunk]:
        return []


@dataclass
class ObservedRetriever:
    contexts: Sequence[RetrievalContext]
    mode: str = "hybrid"
    calls: list[tuple[str, int, str | None]] = field(default_factory=list)

    async def retrieve_contexts(self, query: str, *, limit: int = 5, arxiv_id: str | None = None) -> RetrievalResult:
        self.calls.append((query, limit, arxiv_id))
        result = RetrievalResult(contexts=tuple(self.contexts), mode=self.mode)
        record_retrieval(RetrievalObservation(query=query, arxiv_id=arxiv_id, limit=limit, result=result))
        return result


def _service(fake_openai: Any, retriever: ObservedRetriever, store: FakeStore) -> Any:
    models = ChatModelFactory(
        server_api_key=fake_openai.expected_api_key, default_model="gpt-4o", base_url=fake_openai.base_url
    )
    return build_chat_service(
        AppSettings.model_validate({}), store=store, retriever=retriever, models=models, tracer=Tracer.disabled()
    )


def _after_tool(request: dict[str, Any]) -> bool:
    return request["messages"][-1]["role"] == "tool"


class TestCapturedContexts:
    def test_observations_flatten_in_call_order(self, make_hit, make_context) -> None:
        first = RetrievalResult(contexts=(make_context(make_hit(1, arxiv_id="A")),), mode="hybrid")
        second = RetrievalResult(
            contexts=(make_context(make_hit(2, arxiv_id="B"), text=""), make_context(make_hit(3, arxiv_id="A"))),
            mode="lexical",
        )
        observations = [
            RetrievalObservation(query="q1", arxiv_id=None, limit=5, result=first),
            RetrievalObservation(query="q2", arxiv_id=None, limit=5, result=second),
        ]

        captured = CapturedContexts.from_observations(observations)

        assert captured.texts == ["context 1", "body", "context 3"]
        assert captured.retrieval_modes == ["hybrid", "lexical"]
        assert captured.arxiv_ids == ["A", "B", "A"]


class TestProductGenerators:
    async def test_agent_answer_captures_the_tool_contexts_through_the_observer(
        self, fakes, fake_openai, make_hit, make_context
    ) -> None:
        contexts = [
            make_context(make_hit(1, arxiv_id="2401.00001"), text="window one"),
            make_context(make_hit(2, arxiv_id="2401.00001"), text="window two"),
        ]
        retriever = ObservedRetriever(contexts)
        fake_openai.chat_handler = lambda request: (
            "요약: [Paper](https://arxiv.org/abs/2401.00001)"
            if _after_tool(request)
            else fakes.ToolCall(name="search_paper_chunks_tool", arguments=json.dumps({"query": "loss"}))
        )
        query = _q("a", query="DPO 손실은?", history=(("user", "earlier"), ("assistant", "reply")))

        result = await generate_agent_answer(_service(fake_openai, retriever, FakeStore()), query)

        assert retriever.calls == [("loss", 5, None)]
        assert result.answer == "요약: [Paper](https://arxiv.org/abs/2401.00001)"
        assert result.contexts == ["window one", "window two"]
        assert result.hit_arxiv_ids == ["2401.00001"]
        assert result.retrieval_mode == "hybrid"
        assert [citation["arxiv_id"] for citation in result.citations] == ["2401.00001", "2401.00001"]
        assert all(citation["in_answer"] for citation in result.citations)
        first_request = fake_openai.requests_for("/v1/chat/completions")[0].body
        assert [message["role"] for message in first_request["messages"]] == ["system", "user", "assistant", "user"]

    async def test_agent_answer_without_tool_calls_has_no_contexts(self, fake_openai) -> None:
        fake_openai.chat_handler = lambda request: "바로 답합니다."

        result = await generate_agent_answer(_service(fake_openai, ObservedRetriever([]), FakeStore()), _q("b"))

        assert (result.answer, result.contexts, result.retrieval_mode, result.hit_arxiv_ids) == (
            "바로 답합니다.",
            [],
            None,
            [],
        )

    async def test_paper_chat_answer_uses_the_numbered_sources(self, fake_openai, make_hit, make_context) -> None:
        store = FakeStore({"2401.00002": _paper()})
        retriever = ObservedRetriever([make_context(make_hit(11, arxiv_id="2401.00002"), text="lr is 3e-4")])
        fake_openai.chat_handler = lambda request: "The rate is 3e-4 [2]."
        query = _q("cs-11", query="What learning rate was used?", relevant_arxiv_ids=("2401.00002",))

        result = await generate_paper_chat_answer(_service(fake_openai, retriever, store), query)

        assert retriever.calls == [("What learning rate was used?", 5, "2401.00002")]
        assert result.answer == "The rate is 3e-4 [2]."
        assert result.contexts == ["We study X.", "lr is 3e-4"]
        assert (result.retrieval_mode, result.arxiv_id, result.hit_arxiv_ids) == (
            "hybrid",
            "2401.00002",
            ["2401.00002"],
        )
        assert [citation["chunk_id"] for citation in result.citations] == [11]

    async def test_paper_chat_answer_needs_a_known_paper(self, fake_openai) -> None:
        service = _service(fake_openai, ObservedRetriever([]), FakeStore())
        with pytest.raises(LookupError, match="papers에"):
            await generate_paper_chat_answer(service, _q("x", open_arxiv_id="2401.00009"))
        with pytest.raises(LookupError, match="open_arxiv_id"):
            await generate_paper_chat_answer(service, _q("y", relevant_arxiv_ids=(), mode="agent"))

    def test_build_generators_follow_the_modes(self, fake_openai) -> None:
        service = _service(fake_openai, ObservedRetriever([]), FakeStore())
        assert set(build_generators(service, ["agent"])) == {"agent"}
        assert set(build_generators(service, ["agent", "paper_chat"])) == {"agent", "paper_chat"}


class TestSearchLimit:
    async def test_collection_overrides_the_agent_tool_limit_and_restores_it(
        self, fakes, fake_openai, make_hit, make_context
    ) -> None:
        retriever = ObservedRetriever([make_context(make_hit(1, arxiv_id="2401.00001"), text="window")])
        fake_openai.chat_handler = lambda request: (
            "[Paper](https://arxiv.org/abs/2401.00001)"
            if _after_tool(request)
            else fakes.ToolCall(name="search_paper_chunks_tool", arguments=json.dumps({"query": "loss"}))
        )
        service = _service(fake_openai, retriever, FakeStore({"2401.00001": _paper("2401.00001")}))
        seen: list[int] = []

        def progress(*_: Any) -> None:
            seen.append(agent_tools.SEARCH_RESULT_LIMIT)

        records = await collect_answers(
            [_q("a", query="DPO 손실은?")],
            ["agent", "paper_chat"],
            build_generators(service, ["agent", "paper_chat"]),
            progress=progress,
            search_limit=3,
        )

        assert [call[1] for call in retriever.calls] == [3, 5]
        assert seen == [3, 3]
        assert agent_tools.SEARCH_RESULT_LIMIT == 5
        assert records[0]["search_limit"] == 3
        assert "search_limit" not in records[1]

    async def test_without_a_limit_records_and_tool_stay_as_before(self, fakes, fake_openai) -> None:
        retriever = ObservedRetriever([])
        fake_openai.chat_handler = lambda request: (
            "없습니다."
            if _after_tool(request)
            else fakes.ToolCall(name="search_paper_chunks_tool", arguments=json.dumps({"query": "loss"}))
        )
        service = _service(fake_openai, retriever, FakeStore())

        records = await collect_answers([_q("a", mode="agent")], ["agent"], build_generators(service, ["agent"]))

        assert retriever.calls == [("loss", agent_tools.SEARCH_RESULT_LIMIT, None)]
        assert "search_limit" not in records[0]
        assert list(records[0]) == list(build_answer_record(_q("a"), "agent"))

    async def test_the_limit_is_restored_when_generation_raises(self) -> None:
        async def generate(query: EvalQuery) -> GenerationResult:
            assert agent_tools.SEARCH_RESULT_LIMIT == 10
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await collect_answers([_q("a")], ["agent"], {"agent": generate}, search_limit=10)
        assert agent_tools.SEARCH_RESULT_LIMIT == 5

    def test_invalid_limit_is_refused_without_touching_the_tool(self) -> None:
        with pytest.raises(ValueError), agent_search_limit(0):
            pass
        assert agent_tools.SEARCH_RESULT_LIMIT == 5

    def test_answer_files_keep_the_limit_and_reject_bad_values(self, tmp_path) -> None:
        path = tmp_path / "answers.jsonl"
        append_answer_record(path, build_answer_record(QUERIES[0], "agent", result=_result(), search_limit=10))
        assert load_answer_records(path)[0]["search_limit"] == 10

        for bad in (0, "3", True):
            broken = tmp_path / f"bad_{bad}.jsonl"
            record = {**build_answer_record(QUERIES[0], "agent", result=_result()), "search_limit": bad}
            append_answer_record(broken, record)
            with pytest.raises(AnswersError, match="search_limit"):
                load_answer_records(broken)
