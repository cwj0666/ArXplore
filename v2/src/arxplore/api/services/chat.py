from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from arxplore.api.errors import EMPTY_MESSAGE_MESSAGE, INVALID_REQUEST_MESSAGE, AiUnavailableError, InvalidRequestError
from arxplore.api.services.detail import find_paper
from arxplore.api.services.store import PaperStore
from arxplore.config import AppSettings
from arxplore.domain.papers import Paper
from arxplore.llm.agent.graph import AgentGraph
from arxplore.llm.agent.stream import stream_agent_answer
from arxplore.llm.events import AnswerEvent, CollectedAnswer, collect_answer
from arxplore.llm.models import ChatModelFactory
from arxplore.llm.paper_chat import (
    ContextRetriever,
    PaperChatRetrieval,
    retrieve_paper_chat_sources,
    stream_paper_chat_answer,
)
from arxplore.llm.tracing import Tracer
from arxplore.text import strip_control_characters

CHAT_HISTORY_MAX_MESSAGES = 20
CHAT_MESSAGE_MAX_CHARS = 4000
MESSAGE_TOO_LONG_MESSAGE = f"메시지는 {CHAT_MESSAGE_MAX_CHARS:,}자 이하로 입력하세요."
EMPTY_AGENT_ANSWER = "답변을 생성할 수 없습니다."
CHAT_ROLES = ("user", "assistant")
PAPER_CHAT_TEMPERATURE = 0.0

type ChatHistory = list[tuple[str, str]]


@dataclass(frozen=True, slots=True)
class ChatInput:
    message: str
    history: ChatHistory


@dataclass(frozen=True, slots=True)
class PreparedPaperChat:
    message: str
    history: ChatHistory
    paper: Paper


@dataclass(frozen=True, slots=True)
class PaperChatAnswer:
    answer: CollectedAnswer
    retrieval_mode: str


def normalize_history(history: Sequence[Any], *, current_message: str | None = None) -> ChatHistory:
    messages = [
        (message["role"], message["content"][:CHAT_MESSAGE_MAX_CHARS])
        for message in history
        if isinstance(message, Mapping)
        and message.get("role") in CHAT_ROLES
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ]
    if (
        current_message
        and messages
        and messages[-1][0] == "user"
        and messages[-1][1].strip() == current_message.strip()
    ):
        messages.pop()
    return messages[-CHAT_HISTORY_MAX_MESSAGES:]


def validate_chat_input(message: str, history: Any) -> ChatInput:
    cleaned = strip_control_characters(message).strip()
    if not cleaned:
        raise InvalidRequestError(EMPTY_MESSAGE_MESSAGE)
    if len(cleaned) > CHAT_MESSAGE_MAX_CHARS:
        raise InvalidRequestError(MESSAGE_TOO_LONG_MESSAGE)
    if not isinstance(history, list):
        raise InvalidRequestError(INVALID_REQUEST_MESSAGE)
    return ChatInput(message=cleaned, history=normalize_history(history, current_message=cleaned))


def chat_fields(body: Mapping[str, Any]) -> tuple[str, Any]:
    return str(body.get("message", "")), body.get("history", [])


class ChatService:
    def __init__(
        self,
        *,
        settings: AppSettings,
        store: PaperStore,
        retriever: ContextRetriever,
        models: ChatModelFactory,
        tracer: Tracer,
        agent_graph: AgentGraph,
    ) -> None:
        self.settings = settings
        self.store = store
        self.retriever = retriever
        self.models = models
        self.tracer = tracer
        self.agent_graph = agent_graph

    def require_api_key(self) -> None:
        if not self.models.is_available():
            raise AiUnavailableError()

    async def prepare_paper_chat(self, arxiv_id: str, message: str, history: Any) -> PreparedPaperChat:
        chat = validate_chat_input(message, history)
        paper = await find_paper(self.store, arxiv_id)
        self.require_api_key()
        return PreparedPaperChat(message=chat.message, history=chat.history, paper=paper)

    def prepare_agent_chat(self, message: str, history: Any) -> ChatInput:
        chat = validate_chat_input(message, history)
        self.require_api_key()
        return chat

    async def retrieve_paper_sources(self, prepared: PreparedPaperChat) -> PaperChatRetrieval:
        return await retrieve_paper_chat_sources(
            prepared.paper, prepared.message, retriever=self.retriever, chunks=self.store
        )

    def answer_paper_sources(
        self, prepared: PreparedPaperChat, retrieval: PaperChatRetrieval
    ) -> AsyncGenerator[AnswerEvent]:
        return stream_paper_chat_answer(
            prepared.message,
            paper=prepared.paper,
            retrieval=retrieval,
            history=prepared.history,
            model=self.models.create(temperature=PAPER_CHAT_TEMPERATURE),
            tracer=self.tracer,
        )

    async def stream_paper_chat(self, prepared: PreparedPaperChat) -> AsyncGenerator[AnswerEvent]:
        retrieval = await self.retrieve_paper_sources(prepared)
        async for event in self.answer_paper_sources(prepared, retrieval):
            yield event

    async def answer_paper_chat(self, prepared: PreparedPaperChat) -> PaperChatAnswer:
        retrieval = await self.retrieve_paper_sources(prepared)
        answer = await collect_answer(self.answer_paper_sources(prepared, retrieval))
        return PaperChatAnswer(answer=answer, retrieval_mode=retrieval.retrieval_mode)

    def stream_agent_chat(self, chat: ChatInput) -> AsyncGenerator[AnswerEvent]:
        return stream_agent_answer(
            self.agent_graph,
            chat.message,
            history=chat.history,
            recursion_limit=self.settings.agent_recursion_limit,
            buffer_chars=self.settings.agent_stream_buffer_chars,
            tracer=self.tracer,
        )

    async def answer_agent_chat(self, chat: ChatInput) -> CollectedAnswer:
        answer = await collect_answer(self.stream_agent_chat(chat))
        return CollectedAnswer(answer=answer.answer or EMPTY_AGENT_ANSWER, citations=answer.citations)
