import hashlib
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage, convert_to_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from arxplore.api.services.chat import ChatService
from arxplore.context import RetrievalObservation, observe_retrieval, record_retrieval
from arxplore.domain.retrieval import HybridInputs, RetrievalResult
from arxplore.llm.agent import tools as agent_tools
from arxplore.llm.agent.graph import AGENT_TEMPERATURE, AgentGraph, build_agent_graph
from arxplore.llm.agent.stream import EMPTY_ANSWER_MESSAGE, build_agent_messages, finished_model_messages, message_text
from arxplore.llm.citations import Citation, collect_tool_hits, verify_agent_citations
from arxplore.llm.prompts.agent import AGENT_SYSTEM_PROMPT
from arxplore.retrieval.service import Retriever
from arxplore_eval.fusion_variants import fuse_inputs, inputs_from_payload, inputs_payload

PLAN_KIND_TOOLS = "tools"
PLAN_KIND_DIRECT = "direct"
FINAL_TOOL_CHOICE = "none"
QUERY_ARG = "query"
TOOL_CALL_TYPE = "tool_call"

type Usage = dict[str, int]


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class PlannedSearch:
    tool_call_id: str
    query: str
    inputs: HybridInputs


@dataclass(frozen=True)
class AgentPlan:
    query_id: str
    content: str
    tool_calls: tuple[dict[str, Any], ...]
    searches: tuple[PlannedSearch, ...]
    search_limit: int
    usage: Usage = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return PLAN_KIND_TOOLS if self.tool_calls else PLAN_KIND_DIRECT

    def search_inputs(self) -> dict[str, HybridInputs]:
        return {search.query: search.inputs for search in self.searches}


class EmptyRetriever:
    async def retrieve_contexts(
        self, query: str, *, limit: int = agent_tools.SEARCH_RESULT_LIMIT, arxiv_id: str | None = None
    ) -> RetrievalResult:
        return RetrievalResult(contexts=(), mode="lexical")


def build_capture_graph(service: ChatService) -> AgentGraph:
    return build_agent_graph(agent_tools.build_agent_tools(EmptyRetriever(), service.store), service.models)


def usage_of(message: BaseMessage) -> Usage:
    metadata = getattr(message, "usage_metadata", None) or {}
    return {key: int(metadata[key]) for key in ("input_tokens", "output_tokens") if key in metadata}


def tool_call_payload(call: Mapping[str, Any]) -> dict[str, Any]:
    return {"name": str(call["name"]), "args": dict(call.get("args") or {}), "id": str(call.get("id") or "")}


async def capture_first_turn(
    graph: AgentGraph, question: str, history: Sequence[tuple[str, str]], *, recursion_limit: int
) -> AIMessage:
    config: RunnableConfig = {"recursion_limit": recursion_limit}
    inputs = {"messages": build_agent_messages(question, history)}
    stream = cast(AsyncGenerator[Any], graph.astream(inputs, config=config, stream_mode="updates"))
    async with aclosing(stream) as updates:
        async for update in updates:
            finished = finished_model_messages(update)
            if finished:
                return finished[-1]
    raise PlanError("에이전트 첫 모델 턴이 나오지 않았습니다.")


async def capture_plan(
    service: ChatService,
    retriever: Retriever,
    graph: AgentGraph,
    *,
    query_id: str,
    question: str,
    history: Sequence[tuple[str, str]],
) -> AgentPlan:
    message = await capture_first_turn(graph, question, history, recursion_limit=service.settings.agent_recursion_limit)
    calls = tuple(tool_call_payload(call) for call in message.tool_calls)
    known = {agent_tools.SEARCH_TOOL_NAME, agent_tools.TRENDING_TOOL_NAME}
    unknown = sorted({call["name"] for call in calls} - known)
    if unknown:
        raise PlanError(f"{query_id}: 알 수 없는 도구 호출 {unknown}")
    limit = agent_tools.SEARCH_RESULT_LIMIT
    searches: list[PlannedSearch] = []
    for call in calls:
        if call["name"] != agent_tools.SEARCH_TOOL_NAME:
            continue
        if retriever.resolve_mode() != "hybrid":
            raise PlanError(f"{query_id}: 계획 캡처는 hybrid 검색(질의 임베딩)이 필요합니다.")
        text = str(call["args"].get(QUERY_ARG) or "")
        inputs = await retriever.hybrid_inputs(text, limit=limit)
        searches.append(PlannedSearch(tool_call_id=call["id"], query=text, inputs=inputs))
    return AgentPlan(
        query_id=query_id,
        content=message_text(message),
        tool_calls=calls,
        searches=tuple(searches),
        search_limit=limit,
        usage=usage_of(message),
    )


def plan_payload(plan: AgentPlan) -> dict[str, Any]:
    return {
        "kind": plan.kind,
        "content": plan.content,
        "tool_calls": [dict(call) for call in plan.tool_calls],
        "search_limit": plan.search_limit,
        "searches": [
            {"tool_call_id": search.tool_call_id, "query": search.query, "inputs": inputs_payload(search.inputs)}
            for search in plan.searches
        ],
        "usage": dict(plan.usage),
    }


def plan_from_payload(query_id: str, payload: Any) -> AgentPlan:
    if not isinstance(payload, Mapping):
        raise PlanError(f"{query_id}: 'plan' must be an object")
    try:
        return AgentPlan(
            query_id=query_id,
            content=str(payload.get("content") or ""),
            tool_calls=tuple(tool_call_payload(call) for call in payload["tool_calls"]),
            searches=tuple(
                PlannedSearch(
                    tool_call_id=str(item["tool_call_id"]),
                    query=str(item["query"]),
                    inputs=inputs_from_payload(item["inputs"]),
                )
                for item in payload["searches"]
            ),
            search_limit=int(payload["search_limit"]),
            usage={key: int(value) for key, value in (payload.get("usage") or {}).items()},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanError(f"{query_id}: 계획 형식 오류: {type(exc).__name__}: {exc}") from exc


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PlanReplayRetriever:
    def __init__(self, retriever: Retriever, plan: AgentPlan, fusion: str) -> None:
        self.retriever = retriever
        self.plan = plan
        self.fusion = fusion
        self.inputs = plan.search_inputs()

    async def retrieve_contexts(
        self, query: str, *, limit: int = agent_tools.SEARCH_RESULT_LIMIT, arxiv_id: str | None = None
    ) -> RetrievalResult:
        if query not in self.inputs:
            raise PlanError(f"{self.plan.query_id}: 계획에 없는 검색 질의 {query!r}")
        if limit != self.plan.search_limit:
            raise PlanError(f"{self.plan.query_id}: 검색 결과 수 {limit}이 계획({self.plan.search_limit})과 다릅니다.")
        hits = fuse_inputs(self.inputs[query], self.fusion, limit=limit, arxiv_id=arxiv_id)
        contexts = await self.retriever.build_contexts(hits)
        result = RetrievalResult(contexts=tuple(contexts), mode="hybrid")
        record_retrieval(RetrievalObservation(query=query, arxiv_id=arxiv_id, limit=limit, result=result))
        return result


@dataclass(frozen=True)
class PlanAnswer:
    answer: str
    citations: list[Citation]
    observations: list[RetrievalObservation]
    tool_outputs: list[str]
    usage: Usage


def planned_ai_message(plan: AgentPlan) -> AIMessage:
    return AIMessage(
        content=plan.content,
        tool_calls=[{**call, "type": TOOL_CALL_TYPE} for call in plan.tool_calls],
    )


async def run_planned_tools(plan: AgentPlan, tools: Mapping[str, BaseTool]) -> list[ToolMessage]:
    messages: list[ToolMessage] = []
    for call in plan.tool_calls:
        tool = tools.get(call["name"])
        if tool is None:
            raise PlanError(f"{plan.query_id}: 도구 {call['name']}가 없습니다.")
        message = await tool.ainvoke({**call, "type": TOOL_CALL_TYPE})
        if not isinstance(message, ToolMessage):
            raise PlanError(f"{plan.query_id}: 도구 {call['name']}가 ToolMessage를 돌려주지 않았습니다.")
        messages.append(message)
    return messages


def conversation_messages(question: str, history: Sequence[tuple[str, str]]) -> list[BaseMessage]:
    return [SystemMessage(AGENT_SYSTEM_PROMPT), *convert_to_messages(build_agent_messages(question, history))]


async def answer_from_plan(
    service: ChatService,
    retriever: Retriever,
    plan: AgentPlan,
    fusion: str,
    *,
    question: str,
    history: Sequence[tuple[str, str]],
) -> PlanAnswer:
    replay = PlanReplayRetriever(retriever, plan, fusion)
    tool_list = agent_tools.build_agent_tools(replay, service.store)
    tools = {tool.name: tool for tool in tool_list}
    messages = conversation_messages(question, history)
    with observe_retrieval() as observations, collect_tool_hits() as hits:
        tool_messages: list[ToolMessage] = []
        if plan.tool_calls:
            tool_messages = await run_planned_tools(plan, tools)
            messages = [*messages, planned_ai_message(plan), *tool_messages]
        model = service.models.create(temperature=AGENT_TEMPERATURE).bind_tools(
            tool_list, tool_choice=FINAL_TOOL_CHOICE
        )
        response = cast(BaseMessage, await model.ainvoke(messages))
        answer = message_text(response) or EMPTY_ANSWER_MESSAGE
        citations = verify_agent_citations(answer, list(hits))
    return PlanAnswer(
        answer=answer,
        citations=citations,
        observations=list(observations),
        tool_outputs=[str(message.content) for message in tool_messages],
        usage=usage_of(response),
    )
