import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request
from starlette.responses import JSONResponse, Response

from arxplore.api.dependencies import enforce_detail_rate_limit, get_services
from arxplore.api.errors import DATA_LOAD_ERROR_MESSAGE, ApiError
from arxplore.api.responses import api_error_response, empty_response, error_response, paper_item, related_item
from arxplore.api.schemas import ErrorResponse, PaperDetailResponse, PaperListResponse, RateLimitedResponse
from arxplore.api.services.detail import find_paper, find_related_papers
from arxplore.api.services.listing import SEARCH_MODE, SORT_LATEST, list_papers

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/papers/list.json",
    response_model=PaperListResponse,
    responses={500: {"model": ErrorResponse}},
    operation_id="listPapers",
)
async def paper_list(
    request: Request,
    q: Annotated[str, Query()] = "",
    sort: Annotated[str, Query()] = SORT_LATEST,
    mode: Annotated[str, Query()] = SEARCH_MODE,
    page: Annotated[str, Query()] = "1",
) -> Response:
    services = get_services(request)
    try:
        result = await list_papers(services.store, query=q, sort=sort, mode=mode, page=page)
    except Exception:
        logger.exception("논문 목록 조회 실패")
        return error_response(500, DATA_LOAD_ERROR_MESSAGE)
    return JSONResponse(
        {
            "items": [paper_item(paper) for paper in result.papers],
            "page": result.page,
            "page_size": result.page_size,
            "total_items": result.total_items,
            "total_pages": result.total_pages,
            "query": result.query,
            "sort": result.sort,
            "mode": result.mode,
        }
    )


@router.get(
    "/papers/{arxiv_id}/detail.json",
    response_model=PaperDetailResponse,
    responses={404: {"model": ErrorResponse}, 429: {"model": RateLimitedResponse}, 500: {"model": ErrorResponse}},
    dependencies=[Depends(enforce_detail_rate_limit)],
    operation_id="getPaperDetail",
)
async def paper_detail(arxiv_id: str, request: Request) -> Response:
    services = get_services(request)
    try:
        paper = await find_paper(services.store, arxiv_id)
        related = await find_related_papers(services.store, services.arxiv, paper)
    except ApiError as exc:
        return api_error_response(exc)
    except Exception:
        logger.exception("논문 상세 조회 실패")
        return error_response(500, DATA_LOAD_ERROR_MESSAGE)
    return JSONResponse({"paper": {**paper_item(paper), "related_papers": [related_item(item) for item in related]}})


@router.get("/papers/", include_in_schema=False)
async def paper_list_alias(request: Request) -> Response:
    query = request.url.query
    return empty_response(302, headers={"location": f"/?{query}" if query else "/"})
