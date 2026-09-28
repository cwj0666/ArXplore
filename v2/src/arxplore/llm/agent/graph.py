from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed.is_last_step import RemainingStepsManager

from arxplore.llm.models import ChatModelFactory
from arxplore.llm.prompts.agent import AGENT_SYSTEM_PROMPT

AGENT_TEMPERATURE = 0.0
MODEL_NODE = "model"
STEP_LIMIT_SENTINEL = "Sorry, need more steps to process this request."
MIN_STEPS_FOR_TOOL_CALLS = 2
UNBOUND_MODEL_MESSAGE = "the agent model is chosen per request by RequestModelMiddleware"

type AgentGraph = CompiledStateGraph[Any, Any, Any, Any]


class UnboundChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "arxplore-unbound"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError(UNBOUND_MODEL_MESSAGE)


class RequestModelMiddleware(AgentMiddleware):
    def __init__(self, models: ChatModelFactory, *, temperature: float = AGENT_TEMPERATURE) -> None:
        super().__init__()
        self.models = models
        self.temperature = temperature

    def _bind(self, request: ModelRequest) -> ModelRequest:
        return request.override(model=self.models.create(temperature=self.temperature))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._bind(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._bind(request))


class StepLimitState(AgentState):
    remaining_steps: NotRequired[Annotated[int, PrivateStateAttr, RemainingStepsManager]]


def is_step_limit_message(message: BaseMessage) -> bool:
    return isinstance(message, AIMessage) and message.content == STEP_LIMIT_SENTINEL


def _guard_step_limit(request: ModelRequest, response: ModelResponse) -> ModelResponse:
    remaining = request.state.get("remaining_steps")
    if not isinstance(remaining, int) or remaining >= MIN_STEPS_FOR_TOOL_CALLS:
        return response
    messages: list[BaseMessage] = []
    for message in response.result:
        if isinstance(message, AIMessage) and message.tool_calls:
            messages.append(AIMessage(id=message.id, content=STEP_LIMIT_SENTINEL))
        else:
            messages.append(message)
    return ModelResponse(result=messages, structured_response=response.structured_response)


class StepLimitMiddleware(AgentMiddleware[StepLimitState]):
    state_schema = StepLimitState

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return _guard_step_limit(request, handler(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return _guard_step_limit(request, await handler(request))


def build_agent_graph(tools: Sequence[BaseTool], models: ChatModelFactory) -> AgentGraph:
    middleware: list[AgentMiddleware[Any, Any]] = [RequestModelMiddleware(models), StepLimitMiddleware()]
    return create_agent(UnboundChatModel(), list(tools), system_prompt=AGENT_SYSTEM_PROMPT, middleware=middleware)
