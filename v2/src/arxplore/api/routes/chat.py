import logging

from fastapi import APIRouter, Depends, Request
from starlette.responses import JSONResponse, Response

from arxplore.api.dependencies import enforce_csrf, enforce_llm_rate_limit, get_services
from arxplore.api.disconnect import CLIENT_CLOSED_REQUEST, ClientDisconnectedError, until_disconnected
from arxplore.api.errors import STREAM_ERROR_MESSAGE, ApiError
from arxplore.api.openapi_docs import AGENT_ERROR_RESPONSES, LLM_ERROR_RESPONSES, event_stream_response, request_body
from arxplore.api.responses import api_error_response, citation_items, empty_response, error_response, json_object
from arxplore.api.schemas import AgentChatResponse, ChatRequest, PaperChatResponse
from arxplore.api.services.chat import chat_fields
from arxplore.api.sse import event_stream_response as sse_response

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(enforce_csrf), Depends(enforce_llm_rate_limit)])


@router.post(
    "/papers/assistant/chat/",
    response_model=AgentChatResponse,
    responses=AGENT_ERROR_RESPONSES,
    openapi_extra=request_body(ChatRequest),
    operation_id="chatWithAssistant",
)
async def assistant_chat(request: Request) -> Response:
    services = get_services(request)
    try:
        chat = services.chat.prepare_agent_chat(*chat_fields(await json_object(request)))
        answer = await until_disconnected(request, services.chat.answer_agent_chat(chat))
    except ApiError as exc:
        return api_error_response(exc)
    except ClientDisconnectedError:
        return empty_response(CLIENT_CLOSED_REQUEST)
    except Exception:
        logger.exception("에이전트 응답 생성 실패")
        return error_response(500, STREAM_ERROR_MESSAGE)
    return JSONResponse({"answer": answer.answer, "citations": citation_items(answer.citations)})


@router.post(
    "/papers/assistant/stream/",
    responses={200: event_stream_response(), **AGENT_ERROR_RESPONSES},
    openapi_extra=request_body(ChatRequest),
    operation_id="streamAssistantChat",
)
async def assistant_stream(request: Request) -> Response:
    services = get_services(request)
    try:
        chat = services.chat.prepare_agent_chat(*chat_fields(await json_object(request)))
    except ApiError as exc:
        return api_error_response(exc)
    except Exception:
        logger.exception("에이전트 스트림 준비 실패")
        return error_response(500, STREAM_ERROR_MESSAGE)
    return sse_response(services.chat.stream_agent_chat(chat))


@router.post(
    "/papers/{arxiv_id}/chat/",
    response_model=PaperChatResponse,
    responses=LLM_ERROR_RESPONSES,
    openapi_extra=request_body(ChatRequest),
    operation_id="chatWithPaper",
)
async def paper_chat(arxiv_id: str, request: Request) -> Response:
    services = get_services(request)
    try:
        prepared = await services.chat.prepare_paper_chat(arxiv_id, *chat_fields(await json_object(request)))
        result = await until_disconnected(request, services.chat.answer_paper_chat(prepared))
    except ApiError as exc:
        return api_error_response(exc)
    except ClientDisconnectedError:
        return empty_response(CLIENT_CLOSED_REQUEST)
    except Exception:
        logger.exception("상세 챗 응답 생성 실패")
        return error_response(500, STREAM_ERROR_MESSAGE)
    return JSONResponse(
        {
            "answer": result.answer.answer,
            "citations": citation_items(result.answer.citations),
            "retrieval_mode": result.retrieval_mode,
        }
    )


@router.post(
    "/papers/{arxiv_id}/chat/stream/",
    responses={200: event_stream_response(), **LLM_ERROR_RESPONSES},
    openapi_extra=request_body(ChatRequest),
    operation_id="streamPaperChat",
)
async def paper_chat_stream(arxiv_id: str, request: Request) -> Response:
    services = get_services(request)
    try:
        prepared = await services.chat.prepare_paper_chat(arxiv_id, *chat_fields(await json_object(request)))
    except ApiError as exc:
        return api_error_response(exc)
    except Exception:
        logger.exception("상세 챗 스트림 준비 실패")
        return error_response(500, STREAM_ERROR_MESSAGE)
    return sse_response(services.chat.stream_paper_chat(prepared))
