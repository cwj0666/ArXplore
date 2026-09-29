from typing import Any

from pydantic import BaseModel

from arxplore.api.schemas import ErrorResponse, RateLimitedResponse

COMPONENT_REF = "#/components/schemas/{model}"
EVENT_STREAM_DESCRIPTION = (
    "Server-sent events. Each event is `data: <json>` followed by a blank line: StreamChunkEvent zero or more "
    "times, then StreamCitationsEvent exactly once, then `data: [DONE]`. On failure StreamErrorEvent replaces "
    "the citations event before `data: [DONE]`."
)

ERROR_RESPONSE = {"model": ErrorResponse}
LLM_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: ERROR_RESPONSE,
    403: {"description": "CSRF check failed"},
    404: ERROR_RESPONSE,
    429: {"model": RateLimitedResponse},
    500: ERROR_RESPONSE,
    503: ERROR_RESPONSE,
}
AGENT_ERROR_RESPONSES = {status: spec for status, spec in LLM_ERROR_RESPONSES.items() if status != 404}


def component_ref(model: type[BaseModel]) -> dict[str, str]:
    return {"$ref": COMPONENT_REF.format(model=model.__name__)}


def request_body(model: type[BaseModel]) -> dict[str, Any]:
    return {"requestBody": {"required": True, "content": {"application/json": {"schema": component_ref(model)}}}}


def event_stream_response() -> dict[str, Any]:
    return {
        "description": EVENT_STREAM_DESCRIPTION,
        "content": {"text/event-stream": {"schema": {"type": "string"}}},
    }
