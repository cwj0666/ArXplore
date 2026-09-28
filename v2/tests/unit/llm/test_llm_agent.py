import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.errors import GraphRecursionError

from arxplore.config import RetrievalMode
from arxplore.context import override_openai_runtime, request_openai_api_key
from arxplore.domain.papers import Paper
from arxplore.domain.retrieval import RetrievalContext, RetrievalHit, RetrievalResult
from arxplore.llm.agent.graph import STEP_LIMIT_SENTINEL, AgentGraph, build_agent_graph
from arxplore.llm.agent.stream import EMPTY_ANSWER_MESSAGE, STEP_LIMIT_MESSAGE, stream_agent_answer
from arxplore.llm.agent.tools import (
    NO_SEARCH_RESULTS_MESSAGE,
    NO_TRENDING_PAPERS_MESSAGE,
    SEARCH_TOOL_DESCRIPTION,
    TRENDING_TOOL_DESCRIPTION,
    build_agent_tools,
    format_search_results,
    format_trending_papers,
)
from arxplore.llm.events import AnswerChunk, AnswerCitations, AnswerEvent
from arxplore.llm.models import ChatModelFactory
from arxplore.llm.prompts.agent import AGENT_SYSTEM_PROMPT
from arxplore.llm.tracing import Tracer

QUESTION = "DPO 관련 논문 알려줘"
LINK_ANSWER = "- **[Direct Preference Optimization Revisited](https://arxiv.org/abs/2401.00001)** — 요약입니다."


def hit_context(chunk_id: int = 42, *, section: str = "3 Method", text: str = "hit with neighbors") -> RetrievalContext:
    hit = RetrievalHit(
        chunk_id=chunk_id,
        arxiv_id="2401.00001",
        chunk_index=chunk_id,
        paper_title="Direct Preference Optimization Revisited",
        paper_abstract="",
        chunk_text="hit",
        section_title=section,
        content_role="body",
        method="hybrid",
        score=0.4,
        snippet="",
    )
    return RetrievalContext(hit=hit, chunks=(), text=text)


def paper(arxiv_id: str, title: str, upvotes: int, pdf_url: str | None = None) -> Paper:
    return Paper(
        arxiv_id=arxiv_id,
        title=title,
        authors=[],
        abstract="",
        primary_category=None,
        categories=[],
        pdf_url=pdf_url,
        published_at=None,
        updated_at=None,
        upvotes=upvotes,
        github_url=None,
        github_stars=None,
        citation_count=None,
    )


@dataclass
class FakeRetriever:
    contexts: Sequence[RetrievalContext] = ()
    mode: RetrievalMode = "hybrid"
    calls: list[tuple[str, int, str | None]] = field(default_factory=list)
    keys: list[str | None] = field(default_factory=list)

    async def retrieve_contexts(self, query: str, *, limit: int = 5, arxiv_id: str | None = None) -> RetrievalResult:
        self.calls.append((query, limit, arxiv_id))
        self.keys.append(request_openai_api_key())
        return RetrievalResult(contexts=tuple(self.contexts), mode=self.mode)


@dataclass
class FakePapers:
    papers: Sequence[Paper] = ()

    async def list_recent_papers(self, *, limit: int) -> Sequence[Paper]:
        return list(self.papers)[:limit]


def search_call(fakes: ModuleType, query: str = "preference optimization") -> Any:
    return fakes.ToolCall(name="search_paper_chunks_tool", arguments=json.dumps({"query": query}))


def after_tool(request: dict[str, Any]) -> bool:
    return request["messages"][-1]["role"] == "tool"


async def run_agent(
    graph: AgentGraph, question: str = QUESTION, *, recursion_limit: int = 12, buffer_chars: int = 120, **kwargs: Any
) -> list[AnswerEvent]:
    return [
        event
        async for event in stream_agent_answer(
            graph,
            question,
            recursion_limit=recursion_limit,
            buffer_chars=buffer_chars,
            tracer=Tracer.disabled(),
            **kwargs,
        )
    ]


def answer_text(events: list[AnswerEvent]) -> str:
    return "".join(event.text for event in events if isinstance(event, AnswerChunk))


def citations(events: list[AnswerEvent]) -> list[dict[str, Any]]:
    last = events[-1]
    assert isinstance(last, AnswerCitations)
    assert sum(isinstance(event, AnswerCitations) for event in events) == 1
    return [citation.to_payload() for citation in last.citations]


def graph_for(
    models: ChatModelFactory, retriever: FakeRetriever | None = None, papers: FakePapers | None = None
) -> AgentGraph:
    return build_agent_graph(build_agent_tools(retriever or FakeRetriever(), papers or FakePapers()), models)


async def test_search_then_link_streams_chunks_and_one_citations_event(
    fakes: ModuleType, fake_openai: Any, models: ChatModelFactory
) -> None:
    fake_openai.chat_handler = lambda request: LINK_ANSWER if after_tool(request) else search_call(fakes)
    retriever = FakeRetriever(contexts=[hit_context()])

    events = await run_agent(graph_for(models, retriever))

    assert answer_text(events) == LINK_ANSWER
    assert len(events) > 2
    assert citations(events) == [
        {
            "arxiv_id": "2401.00001",
            "title": "Direct Preference Optimization Revisited",
            "url": "https://arxiv.org/abs/2401.00001",
            "section_title": "3 Method",
            "chunk_id": 42,
            "in_answer": True,
        }
    ]
    assert retriever.calls == [("preference optimization", 5, None)]


async def test_requests_carry_the_system_prompt_history_and_both_tools(
    fakes: ModuleType, fake_openai: Any, models: ChatModelFactory
) -> None:
    fake_openai.chat_handler = lambda request: "바로 답합니다."

    await run_agent(graph_for(models), history=[("user", "이전"), ("assistant", "답")])

    (request,) = fake_openai.requests_for("/v1/chat/completions")
    body = request.body
    assert [(message["role"], message["content"]) for message in body["messages"]] == [
        ("system", AGENT_SYSTEM_PROMPT),
        ("user", "이전"),
        ("assistant", "답"),
        ("user", QUESTION),
    ]
    assert (body["model"], body["temperature"], body["stream"]) == ("gpt-4o", 0.0, True)
    assert [tool["function"] for tool in body["tools"]] == [
        {
            "name": "search_paper_chunks_tool",
            "description": SEARCH_TOOL_DESCRIPTION,
            "parameters": {"properties": {"query": {"type": "string"}}, "required": ["query"], "type": "object"},
        },
        {
            "name": "get_trending_papers_tool",
            "description": TRENDING_TOOL_DESCRIPTION,
            "parameters": {"properties": {}, "type": "object"},
        },
    ]


async def test_short_preamble_before_a_tool_call_is_dropped(
    fakes: ModuleType, fake_openai: Any, models: ChatModelFactory
) -> None:
    fake_openai.chat_handler = lambda request: (
        "최종 답변입니다."
        if after_tool(request)
        else fakes.ChatReply(content="먼저 검색하겠습니다.", tool_calls=(search_call(fakes),))
    )

    events = await run_agent(graph_for(models, FakeRetriever(contexts=[hit_context()])))

    assert answer_text(events) == "최종 답변입니다."


async def test_preamble_longer_than_the_buffer_leaks(
    fakes: ModuleType, fake_openai: Any, models: ChatModelFactory
) -> None:
    preamble = "먼저 관련 논문을 찾아보겠습니다. " * 8
    fake_openai.chat_handler = lambda request: (
        LINK_ANSWER if after_tool(request) else fakes.ChatReply(content=preamble, tool_calls=(search_call(fakes),))
    )

    events = await run_agent(graph_for(models, FakeRetriever(contexts=[hit_context()])))

    assert answer_text(events) == preamble + LINK_ANSWER


async def test_zero_buffer_streams_everything_live(
    fakes: ModuleType, fake_openai: Any, models: ChatModelFactory
) -> None:
    fake_openai.chat_handler = lambda request: (
        LINK_ANSWER if after_tool(request) else fakes.ChatReply(content="검색", tool_calls=(search_call(fakes),))
    )

    events = await run_agent(graph_for(models, FakeRetriever(contexts=[hit_context()])), buffer_chars=0)

    assert answer_text(events) == "검색" + LINK_ANSWER


async def test_step_limit_ends_with_the_notice_and_skips_the_last_tool_call(
    fakes: ModuleType, fake_openai: Any, models: ChatModelFactory
) -> None:
    fake_openai.chat_handler = lambda request: search_call(fakes)
    retriever = FakeRetriever(contexts=[hit_context()])

    events = await run_agent(graph_for(models, retriever), recursion_limit=4)

    assert answer_text(events) == STEP_LIMIT_MESSAGE
    assert STEP_LIMIT_SENTINEL not in answer_text(events)
    assert len(fake_openai.requests_for("/v1/chat/completions")) == 2
    assert len(retriever.calls) == 1
    assert [citation["in_answer"] for citation in citations(events)] == [False]


async def test_default_recursion_limit_allows_six_model_calls_and_five_searches(
    fakes: ModuleType, fake_openai: Any, models: ChatModelFactory
) -> None:
    fake_openai.chat_handler = lambda request: search_call(fakes)
    retriever = FakeRetriever(contexts=[hit_context()])

    events = await run_agent(graph_for(models, retriever))

    assert answer_text(events) == STEP_LIMIT_MESSAGE
    assert len(fake_openai.requests_for("/v1/chat/completions")) == 6
    assert len(retriever.calls) == 5


async def test_answer_before_the_step_limit_gets_the_notice_appended(
    fakes: ModuleType, fake_openai: Any, models: ChatModelFactory
) -> None:
    fake_openai.chat_handler = lambda request: fakes.ChatReply(content="x" * 130, tool_calls=(search_call(fakes),))

    events = await run_agent(graph_for(models, FakeRetriever(contexts=[hit_context()])), recursion_limit=4)

    assert answer_text(events) == "x" * 130 + "x" * 130 + f"\n\n{STEP_LIMIT_MESSAGE}"


async def test_empty_answer_gets_a_fallback_message(
    fakes: ModuleType, fake_openai: Any, models: ChatModelFactory
) -> None:
    fake_openai.chat_handler = lambda request: fakes.ChatReply(content="")

    events = await run_agent(graph_for(models))

    assert answer_text(events) == EMPTY_ANSWER_MESSAGE
    assert citations(events) == []


async def test_empty_search_results_tell_the_model_not_to_invent_papers(
    fakes: ModuleType, fake_openai: Any, models: ChatModelFactory
) -> None:
    fake_openai.chat_handler = lambda request: "검색 결과가 없습니다." if after_tool(request) else search_call(fakes)

    events = await run_agent(graph_for(models, FakeRetriever()))

    tool_message = fake_openai.requests_for("/v1/chat/completions")[-1].body["messages"][-1]
    assert tool_message["content"] == NO_SEARCH_RESULTS_MESSAGE
    assert "지어내지 말고" in NO_SEARCH_RESULTS_MESSAGE
    assert answer_text(events) == "검색 결과가 없습니다."
    assert citations(events) == []


async def test_trending_tool_ranks_recent_papers_and_records_hits(
    fakes: ModuleType, fake_openai: Any, models: ChatModelFactory
) -> None:
    papers = FakePapers(
        papers=[
            paper("2401.00001", "Low Votes", 1),
            paper("2401.00002", "High Votes", 10, pdf_url="https://arxiv.org/pdf/2401.00002"),
        ]
    )
    fake_openai.chat_handler = lambda request: (
        "[High Votes](https://arxiv.org/pdf/2401.00002)"
        if after_tool(request)
        else fakes.ToolCall(name="get_trending_papers_tool", arguments="{}")
    )

    events = await run_agent(graph_for(models, papers=papers))

    tool_output = fake_openai.requests_for("/v1/chat/completions")[-1].body["messages"][-1]["content"]
    assert tool_output.split("\n") == [
        "[1] High Votes | URL: https://arxiv.org/pdf/2401.00002 (추천수: 10)",
        "[2] Low Votes | URL: https://arxiv.org/abs/2401.00001 (추천수: 1)",
    ]
    assert citations(events) == [
        {
            "arxiv_id": "2401.00002",
            "title": "High Votes",
            "url": "https://arxiv.org/pdf/2401.00002",
            "section_title": None,
            "chunk_id": None,
            "in_answer": True,
        }
    ]


async def test_hits_from_every_search_are_deduplicated_and_unverified_without_links(
    fakes: ModuleType, fake_openai: Any, models: ChatModelFactory
) -> None:
    replies = iter([search_call(fakes, "first"), search_call(fakes, "second"), "관련 논문을 찾았습니다."])
    fake_openai.chat_handler = lambda request: next(replies)
    retriever = FakeRetriever(contexts=[hit_context(), hit_context(43, section="4 Results")])

    events = await run_agent(graph_for(models, retriever))

    assert [(citation["chunk_id"], citation["in_answer"]) for citation in citations(events)] == [
        (42, False),
        (43, False),
    ]


async def test_request_scoped_key_reaches_the_model_and_the_tools(fakes: ModuleType, models: ChatModelFactory) -> None:
    with fakes.FakeOpenAIServer(expected_api_key="sk-request") as server:
        server.chat_handler = lambda request: "끝" if after_tool(request) else search_call(fakes)
        scoped = ChatModelFactory(server_api_key="sk-server", default_model="gpt-4o", base_url=server.base_url)
        retriever = FakeRetriever(contexts=[hit_context()])

        with override_openai_runtime(api_key="sk-request", model="gpt-4o-mini"):
            events = await run_agent(graph_for(scoped, retriever))

        requests = server.requests_for("/v1/chat/completions")
        assert [request.authorized for request in requests] == [True, True]
        assert {request.body["model"] for request in requests} == {"gpt-4o-mini"}
        assert retriever.keys == ["sk-request"]
        assert answer_text(events) == "끝"


def test_search_results_format() -> None:
    formatted = format_search_results([hit_context(3, text="previous chunk\n\nhit chunk only\n\nnext chunk")])

    assert formatted == (
        "[1] 제목: Direct Preference Optimization Revisited | arxiv_id: 2401.00001 | 섹션: 3 Method | chunk_id: 3\n"
        "출처(URL): https://arxiv.org/abs/2401.00001\n"
        "내용: previous chunk\n\nhit chunk only\n\nnext chunk\n"
    )
    assert "섹션: - | chunk_id: 3" in format_search_results([hit_context(3, section="", text="")])
    assert "내용: hit\n" in format_search_results([hit_context(3, text="")])
    assert format_search_results([]) == NO_SEARCH_RESULTS_MESSAGE


def test_trending_format_joins_lines_with_real_newlines() -> None:
    assert format_trending_papers([paper("a", "A", 3)]) == "[1] A | URL: https://arxiv.org/abs/a (추천수: 3)"
    assert "데이터가 없다고" in NO_TRENDING_PAPERS_MESSAGE


@dataclass
class ScriptedGraph:
    events: Sequence[tuple[str, Any]] = ()
    error: BaseException | None = None
    trace: list[str] = field(default_factory=list)
    configs: list[dict[str, Any]] = field(default_factory=list)

    async def astream(self, inputs: Any, *, config: dict[str, Any], stream_mode: Any) -> AsyncIterator[tuple[str, Any]]:
        self.configs.append(config)
        for index, event in enumerate(self.events):
            self.trace.append(f"source:{index}")
            yield event
        if self.error is not None:
            raise self.error


def chunk(text: str, message_id: str, *, tool_call: bool = False, node: str = "model") -> tuple[str, Any]:
    tool_call_chunks: list[Any] = (
        [{"name": "search_paper_chunks_tool", "args": "{}", "id": "c1", "index": 0, "type": "tool_call_chunk"}]
        if tool_call
        else []
    )
    return (
        "messages",
        (AIMessageChunk(content=text, id=message_id, tool_call_chunks=tool_call_chunks), {"langgraph_node": node}),
    )


def model_update(message_id: str, *, tool_calls: bool, content: str = "") -> tuple[str, Any]:
    calls: list[Any] = (
        [{"name": "search_paper_chunks_tool", "args": {}, "id": "c1", "type": "tool_call"}] if tool_calls else []
    )
    return ("updates", {"model": {"messages": [AIMessage(content=content, id=message_id, tool_calls=calls)]}})


async def run_scripted(graph: ScriptedGraph, **kwargs: Any) -> list[AnswerEvent]:
    events: list[AnswerEvent] = []
    async for event in stream_agent_answer(
        cast(AgentGraph, graph),
        "질문",
        recursion_limit=kwargs.pop("recursion_limit", 12),
        buffer_chars=kwargs.pop("buffer_chars", 120),
        tracer=Tracer.disabled(runtime="production"),
    ):
        graph.trace.append(f"yield:{event.text if isinstance(event, AnswerChunk) else 'citations'}")
        events.append(event)
    return events


async def test_each_message_is_buffered_until_tool_calls_or_completion_are_known() -> None:
    graph = ScriptedGraph(
        events=[
            chunk("Let me", "m1"),
            chunk(" search.", "m1"),
            chunk("", "m1", tool_call=True),
            chunk(" leaked?", "m1"),
            model_update("m1", tool_calls=True),
            ("messages", (ToolMessage(content="result", tool_call_id="c1"), {"langgraph_node": "tools"})),
            chunk("Final", "m2"),
            chunk(" answer", "m2"),
            model_update("m2", tool_calls=False),
        ]
    )

    events = await run_scripted(graph)

    assert [event.text for event in events if isinstance(event, AnswerChunk)] == ["Final", " answer"]
    assert graph.trace.index("yield:Final") > graph.trace.index("source:8")


async def test_long_answers_go_live_after_the_buffer_threshold() -> None:
    head = "x" * 119
    graph = ScriptedGraph(
        events=[
            chunk(head, "m1"),
            chunk("y", "m1"),
            chunk(" tail1", "m1"),
            chunk(" tail2", "m1"),
            model_update("m1", tool_calls=False),
        ]
    )

    events = await run_scripted(graph)

    assert [event.text for event in events if isinstance(event, AnswerChunk)] == [head, "y", " tail1", " tail2"]
    assert graph.trace.index(f"yield:{head}") < graph.trace.index("source:2")
    assert graph.trace.index("yield: tail1") < graph.trace.index("source:3")


async def test_newline_does_not_release_the_buffer_and_preambles_never_leak() -> None:
    newline = ScriptedGraph(events=[chunk("Short line", "m1"), chunk("\n", "m1"), chunk("more", "m1")])
    preamble = ScriptedGraph(
        events=[
            chunk("Searching now.", "m1"),
            chunk("\n", "m1"),
            model_update("m1", tool_calls=True),
            chunk("Answer", "m2"),
            model_update("m2", tool_calls=False),
        ]
    )

    newline_events = await run_scripted(newline)
    preamble_events = await run_scripted(preamble)

    assert answer_text(newline_events) == "Short line\nmore"
    assert newline.trace.index("yield:Short line") > newline.trace.index("source:2")
    assert answer_text(preamble_events) == "Answer"


async def test_messages_from_other_nodes_are_ignored() -> None:
    graph = ScriptedGraph(events=[chunk("tool text", "t1", node="tools"), chunk("Answer", "m1", node="model")])

    assert answer_text(await run_scripted(graph)) == "Answer"


async def test_step_limit_sentinel_in_an_update_ends_with_the_notice() -> None:
    graph = ScriptedGraph(
        events=[chunk("", "m1", tool_call=True), model_update("m1", tool_calls=False, content=STEP_LIMIT_SENTINEL)]
    )

    events = await run_scripted(graph)

    assert answer_text(events) == STEP_LIMIT_MESSAGE


async def test_graph_recursion_error_is_turned_into_the_notice() -> None:
    graph = ScriptedGraph(error=GraphRecursionError("limit"))

    events = await run_scripted(graph)

    assert answer_text(events) == STEP_LIMIT_MESSAGE
    assert citations(events) == []


async def test_other_graph_errors_propagate() -> None:
    graph = ScriptedGraph(events=[chunk("partial", "m1")], error=RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        await run_scripted(graph)


async def test_recursion_limit_and_agent_chat_trace_config_reach_the_graph() -> None:
    graph = ScriptedGraph()

    events = await run_scripted(graph, recursion_limit=7)

    (config,) = graph.configs
    assert config["recursion_limit"] == 7
    assert config["run_name"] == "agent_chat"
    assert config["metadata"]["stage"] == "agent_chat"
    assert config["metadata"]["runtime"] == "production"
    assert config["metadata"]["recursion_limit"] == 7
    assert answer_text(events) == EMPTY_ANSWER_MESSAGE
