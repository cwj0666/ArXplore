from fastapi import APIRouter, Request
from starlette.responses import JSONResponse, Response

from arxplore.api.dependencies import get_services
from arxplore.api.schemas import BootstrapResponse
from arxplore.api.security import csrf_cookie_token, set_csrf_cookie
from arxplore.api.services.generation import AVAILABLE_SUMMARY_MODELS, DEFAULT_SUMMARY_MODEL

router = APIRouter()


@router.get("/bootstrap.json", response_model=BootstrapResponse, operation_id="bootstrap")
async def bootstrap(request: Request) -> Response:
    response = JSONResponse(
        {
            "default_summary_model": DEFAULT_SUMMARY_MODEL,
            "available_summary_models": list(AVAILABLE_SUMMARY_MODELS),
        }
    )
    set_csrf_cookie(response, csrf_cookie_token(request), secure=get_services(request).settings.secure_cookies)
    return response
