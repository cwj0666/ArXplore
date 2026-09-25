"""논문 탐색 에이전트(LangGraph ReAct)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from src.core.prompts.agent import AGENT_SYSTEM_PROMPT
from src.core.tracing import build_agent_chat_trace_config
from src.shared import get_settings

from .citations import collect_tool_hits, verify_agent_citations
from .llm import build_chat_llm
from .tools import get_trending_papers_tool, search_paper_chunks_tool

logger = logging.getLogger(__name__)

AGENT_TOOLS = (search_paper_chunks_tool, get_trending_papers_tool)
AGENT_NODE = "agent"
# create_react_agent가 remaining_steps 부족으로 도구 호출을 끊을 때 넣는 문구
_LANGGRAPH_STEP_LIMIT_TEXT = "Sorry, need more steps to process this request."
STEP_LIMIT_MESSAGE = (
    "검색 단계가 허용 횟수를 넘어 답변을 끝까지 만들지 못했습니다. "
    "질문을 더 구체적으로 좁혀 다시 시도해 주세요."
)
EMPTY_ANSWER_MESSAGE = "답변을 생성하지 못했습니다. 질문을 바꿔 다시 시도해 주세요."


def _build_agent_llm() -> BaseChatModel:
    return build_chat_llm(temperature=0.0)


def _select_agent_model(state: Any, runtime: Any):
    # 호출마다 요청 범위 키로 모델을 만든다. 컴파일된 그래프는 공유하지만 OpenAI 클라이언트는 공유하지 않는다.
    return _build_agent_llm().bind_tools(list(AGENT_TOOLS))


@lru_cache(maxsize=1)
def get_agent_graph():
    return create_react_agent(_select_agent_model, list(AGENT_TOOLS), prompt=AGENT_SYSTEM_PROMPT)


def build_agent_messages(question: str, chat_history: list[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
    messages = [(role, content) for role, content in (chat_history or [])]
    messages.append(("user", question))
    return messages


def _message_text(message: BaseMessage) -> str:
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


def _is_answer_message(message: Any, metadata: dict[str, Any]) -> bool:
    if not isinstance(message, AIMessage):
        return False
    if metadata.get("langgraph_node") not in (None, AGENT_NODE):
        return False
    return not (message.tool_calls or getattr(message, "tool_call_chunks", None))


def _hit_step_limit(update: Any) -> bool:
    if not isinstance(update, dict):
        return False
    node_output = update.get(AGENT_NODE)
    if not isinstance(node_output, dict):
        return False
    return any(
        isinstance(message, AIMessage) and _message_text(message) == _LANGGRAPH_STEP_LIMIT_TEXT
        for message in node_output.get("messages") or []
    )


def stream_agent_search(
    question: str,
    *,
    chat_history: list[tuple[str, str]] | None = None,
    runtime: str | None = None,
    user: str | None = None,
    recursion_limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """`{"chunk": str}`를 순서대로 내보내고 마지막에 `{"citations": [...]}`를 한 번 내보낸다."""
    limit = recursion_limit or get_settings().agent_recursion_limit
    config = {
        **build_agent_chat_trace_config(runtime=runtime, user=user, extra_metadata={"recursion_limit": limit}),
        "recursion_limit": limit,
    }
    parts: list[str] = []
    step_limit_hit = False

    with collect_tool_hits() as hits:
        try:
            for mode, payload in get_agent_graph().stream(
                {"messages": build_agent_messages(question, chat_history)},
                config=config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "updates":
                    step_limit_hit = step_limit_hit or _hit_step_limit(payload)
                    continue
                message, metadata = payload
                if not _is_answer_message(message, metadata or {}):
                    continue
                text = _message_text(message)
                if not text or text == _LANGGRAPH_STEP_LIMIT_TEXT:
                    continue
                parts.append(text)
                yield {"chunk": text}
        except GraphRecursionError:
            step_limit_hit = True

        if step_limit_hit:
            logger.warning("에이전트가 recursion_limit=%s에 도달했습니다.", limit)
            closing = f"\n\n{STEP_LIMIT_MESSAGE}" if parts else STEP_LIMIT_MESSAGE
            parts.append(closing)
            yield {"chunk": closing}
        elif not parts:
            parts.append(EMPTY_ANSWER_MESSAGE)
            yield {"chunk": EMPTY_ANSWER_MESSAGE}

        yield {"citations": verify_agent_citations("".join(parts), list(hits))}


def agent_search(
    question: str,
    *,
    chat_history: list[tuple[str, str]] | None = None,
    runtime: str | None = None,
    user: str | None = None,
    recursion_limit: int | None = None,
) -> dict[str, Any]:
    parts: list[str] = []
    citations: list[dict[str, Any]] = []
    for event in stream_agent_search(
        question,
        chat_history=chat_history,
        runtime=runtime,
        user=user,
        recursion_limit=recursion_limit,
    ):
        if "chunk" in event:
            parts.append(event["chunk"])
        elif "citations" in event:
            citations = event["citations"]
    return {"answer": "".join(parts), "citations": citations}
