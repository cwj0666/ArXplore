import logging
from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.types import StreamMode

from arxplore.llm.agent.graph import MODEL_NODE, STEP_LIMIT_SENTINEL, AgentGraph, is_step_limit_message
from arxplore.llm.citations import collect_tool_hits, verify_agent_citations
from arxplore.llm.events import AnswerChunk, AnswerCitations, AnswerEvent
from arxplore.llm.tracing import Tracer

logger = logging.getLogger(__name__)

STEP_LIMIT_MESSAGE = (
    "검색 단계가 허용 횟수를 넘어 답변을 끝까지 만들지 못했습니다. 질문을 더 구체적으로 좁혀 다시 시도해 주세요."
)
EMPTY_ANSWER_MESSAGE = "답변을 생성하지 못했습니다. 질문을 바꿔 다시 시도해 주세요."
STREAM_MODES: list[StreamMode] = ["messages", "updates"]
USER_ROLE = "user"

type ChatHistory = Sequence[tuple[str, str]]


def build_agent_messages(question: str, history: ChatHistory = ()) -> list[tuple[str, str]]:
    return [*((role, content) for role, content in history), (USER_ROLE, question)]


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if not isinstance(block, dict) or block.get("type") in (None, "text")
        )
    return ""


def has_tool_calls(message: AIMessage) -> bool:
    return bool(message.tool_calls or getattr(message, "tool_call_chunks", None))


def is_model_message(message: Any, metadata: Mapping[str, Any]) -> bool:
    return isinstance(message, AIMessage) and metadata.get("langgraph_node") in (None, MODEL_NODE)


def finished_model_messages(update: Any) -> list[AIMessage]:
    if not isinstance(update, Mapping):
        return []
    node_output = update.get(MODEL_NODE)
    if not isinstance(node_output, Mapping):
        return []
    return [message for message in node_output.get("messages") or [] if isinstance(message, AIMessage)]


@dataclass
class AnswerBuffer:
    buffer_chars: int
    parts: list[str] = field(default_factory=list)
    pending: dict[str, list[str]] = field(default_factory=dict)
    live_ids: set[str] = field(default_factory=set)
    tool_call_ids: set[str] = field(default_factory=set)
    step_limit_hit: bool = False

    def emit(self, texts: Sequence[str]) -> list[str]:
        self.parts.extend(texts)
        return list(texts)

    def on_update(self, update: Any) -> list[str]:
        released: list[str] = []
        for finished in finished_model_messages(update):
            if is_step_limit_message(finished):
                self.step_limit_hit = True
            message_id = finished.id or ""
            texts = self.pending.pop(message_id, [])
            if has_tool_calls(finished):
                self.tool_call_ids.add(message_id)
                continue
            released.extend(self.emit(texts))
        return released

    def on_message(self, message: Any, metadata: Mapping[str, Any]) -> list[str]:
        if not is_model_message(message, metadata):
            return []
        message_id = message.id or ""
        if message_id in self.tool_call_ids:
            return []
        if has_tool_calls(message):
            self.tool_call_ids.add(message_id)
            self.pending.pop(message_id, None)
            return []
        text = message_text(message)
        if not text or text == STEP_LIMIT_SENTINEL:
            return []
        if message_id in self.live_ids:
            return self.emit([text])
        buffered = self.pending.setdefault(message_id, [])
        buffered.append(text)
        if sum(len(piece) for piece in buffered) < self.buffer_chars:
            return []
        self.live_ids.add(message_id)
        return self.emit(self.pending.pop(message_id))

    def drain(self) -> list[str]:
        released: list[str] = []
        for texts in self.pending.values():
            released.extend(self.emit(texts))
        self.pending.clear()
        return released

    def closing(self) -> str | None:
        if self.step_limit_hit:
            return f"\n\n{STEP_LIMIT_MESSAGE}" if self.parts else STEP_LIMIT_MESSAGE
        if not self.parts:
            return EMPTY_ANSWER_MESSAGE
        return None

    @property
    def answer(self) -> str:
        return "".join(self.parts)


async def stream_agent_answer(
    graph: AgentGraph,
    question: str,
    *,
    history: ChatHistory = (),
    recursion_limit: int,
    buffer_chars: int,
    tracer: Tracer,
) -> AsyncGenerator[AnswerEvent]:
    config: RunnableConfig = {
        **tracer.config("agent_chat", metadata={"recursion_limit": recursion_limit}),
        "recursion_limit": recursion_limit,
    }
    buffer = AnswerBuffer(buffer_chars=max(0, buffer_chars))
    inputs = {"messages": build_agent_messages(question, history)}
    with tracer.scope(), collect_tool_hits() as hits:
        try:
            stream = cast(
                AsyncGenerator[tuple[str, Any]],
                graph.astream(inputs, config=config, stream_mode=STREAM_MODES),
            )
            async with aclosing(stream) as events:
                async for mode, payload in events:
                    if mode == "updates":
                        released = buffer.on_update(payload)
                    else:
                        message, metadata = payload
                        released = buffer.on_message(message, metadata or {})
                    for text in released:
                        yield AnswerChunk(text)
        except GraphRecursionError:
            buffer.step_limit_hit = True
        for text in buffer.drain():
            yield AnswerChunk(text)
        closing = buffer.closing()
        if buffer.step_limit_hit:
            logger.warning("에이전트가 recursion_limit=%s에 도달했습니다.", recursion_limit)
        if closing is not None:
            buffer.emit([closing])
            yield AnswerChunk(closing)
        yield AnswerCitations.of(verify_agent_citations(buffer.answer, list(hits)))
