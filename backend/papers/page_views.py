from __future__ import annotations

import logging
from pathlib import Path

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.views.decorators.http import require_GET

from .ratelimit import rate_limit
from .services import PaperNotFoundError, build_paper_detail_payload, build_paper_list_payload

logger = logging.getLogger(__name__)

DATA_LOAD_ERROR_MESSAGE = "논문 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."

detail_rate_limit = rate_limit("detail", limit_setting="RATE_LIMIT_DETAIL_PER_MINUTE")


def paper_list(request: HttpRequest) -> HttpResponse:
    return _render_react_shell()


def paper_list_alias(request: HttpRequest) -> HttpResponse:
    redirect_to = "/"
    if request.META.get("QUERY_STRING"):
        redirect_to = f"{redirect_to}?{request.META['QUERY_STRING']}"
    return redirect(redirect_to)


@require_GET
def paper_list_data(request: HttpRequest) -> JsonResponse:
    try:
        payload = build_paper_list_payload(
            query=request.GET.get("q", ""),
            sort=request.GET.get("sort", "latest"),
            mode=request.GET.get("mode", "search"),
            page=request.GET.get("page", 1),
        )
    except Exception:
        logger.exception("논문 목록 조회 실패")
        return JsonResponse({"error": DATA_LOAD_ERROR_MESSAGE}, status=500)

    return JsonResponse(payload)


def paper_detail(request: HttpRequest, arxiv_id: str) -> HttpResponse:
    return _render_react_shell()


@require_GET
@detail_rate_limit
def paper_detail_data(request: HttpRequest, arxiv_id: str) -> JsonResponse:
    try:
        return JsonResponse(build_paper_detail_payload(arxiv_id))
    except PaperNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception:
        logger.exception("논문 상세 조회 실패")
        return JsonResponse({"error": DATA_LOAD_ERROR_MESSAGE}, status=500)


def paper_agent(request: HttpRequest) -> HttpResponse:
    return _render_react_shell()


def _render_react_shell() -> HttpResponse:
    entry_path = Path(settings.FRONTEND_DIST_DIR) / "index.html"
    if not entry_path.exists():
        return HttpResponse(
            "React build output이 없습니다. frontend에서 npm run build를 먼저 실행하세요.",
            status=503,
            content_type="text/plain; charset=utf-8",
        )

    return HttpResponse(
        entry_path.read_text(encoding="utf-8"),
        content_type="text/html; charset=utf-8",
    )
