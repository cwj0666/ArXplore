import logging

from fastapi import APIRouter, Depends, Request
from starlette.responses import JSONResponse, Response

from arxplore.api.dependencies import enforce_csrf, enforce_llm_rate_limit, get_services
from arxplore.api.disconnect import CLIENT_CLOSED_REQUEST, ClientDisconnectedError, until_disconnected
from arxplore.api.errors import GENERATION_ERROR_MESSAGE, ApiError
from arxplore.api.openapi_docs import LLM_ERROR_RESPONSES, request_body
from arxplore.api.responses import api_error_response, empty_response, error_response, json_object
from arxplore.api.schemas import AnalysisResponse, SummaryRequest, SummaryResponse

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(enforce_csrf), Depends(enforce_llm_rate_limit)])


@router.post(
    "/papers/{arxiv_id}/analyze/",
    response_model=AnalysisResponse,
    responses=LLM_ERROR_RESPONSES,
    operation_id="analyzePaper",
)
async def paper_analyze(arxiv_id: str, request: Request) -> Response:
    services = get_services(request)
    try:
        payload = await until_disconnected(request, services.generation.analysis(arxiv_id))
    except ApiError as exc:
        return api_error_response(exc)
    except ClientDisconnectedError:
        return empty_response(CLIENT_CLOSED_REQUEST)
    except Exception:
        logger.exception("overview 생성 실패")
        return error_response(500, GENERATION_ERROR_MESSAGE)
    return JSONResponse({"overview": payload.overview, "key_findings": payload.key_findings, "cached": payload.cached})


@router.post(
    "/papers/{arxiv_id}/summary/",
    response_model=SummaryResponse,
    responses=LLM_ERROR_RESPONSES,
    openapi_extra=request_body(SummaryRequest),
    operation_id="summarizePaper",
)
async def paper_summary(arxiv_id: str, request: Request) -> Response:
    services = get_services(request)
    try:
        body = await json_object(request)
        payload = await until_disconnected(
            request, services.generation.summary(arxiv_id, model=str(body.get("model", "")))
        )
    except ApiError as exc:
        return api_error_response(exc)
    except ClientDisconnectedError:
        return empty_response(CLIENT_CLOSED_REQUEST)
    except Exception:
        logger.exception("상세 요약 생성 실패")
        return error_response(500, GENERATION_ERROR_MESSAGE)
    return JSONResponse({"summary": payload.summary, "cached": payload.cached, "model": payload.model})
