from __future__ import annotations

import logging
from pathlib import Path

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect
from django.views.decorators.http import require_GET

from .ratelimit import rate_limit
from .services import (
    AuthenticationRequiredError,
    PaperNotFoundError,
    build_paper_detail_payload,
    build_paper_list_payload,
    demo_mode_enabled,
)

logger = logging.getLogger(__name__)

DATA_LOAD_ERROR_MESSAGE = "논문 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요."

# 익명 데모 트래픽도 관련 논문 합성 과정에서 arXiv 검색을 호출하므로 IP당으로 제한한다.
detail_rate_limit = rate_limit("detail", limit_setting="RATE_LIMIT_DETAIL_PER_MINUTE", per="ip")


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
            user=request.user,
        )
    except Exception:
        logger.exception("논문 목록 조회 실패")
        return JsonResponse({"error": DATA_LOAD_ERROR_MESSAGE}, status=500)

    return JsonResponse(payload)


def paper_detail(request: HttpRequest, arxiv_id: str) -> HttpResponse:
    if not demo_mode_enabled() and not getattr(request.user, "is_authenticated", False):
        return redirect(f"/login/?next=/papers/{arxiv_id}/")
    return _render_react_shell()


@require_GET
@detail_rate_limit
def paper_detail_data(request: HttpRequest, arxiv_id: str) -> JsonResponse:
    try:
        return JsonResponse(build_paper_detail_payload(arxiv_id, user=request.user))
    except AuthenticationRequiredError as exc:
        return JsonResponse({"error": str(exc), "login_required": True}, status=401)
    except PaperNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception:
        logger.exception("논문 상세 조회 실패")
        return JsonResponse({"error": DATA_LOAD_ERROR_MESSAGE}, status=500)


def paper_agent(request: HttpRequest) -> HttpResponse:
    return _render_react_shell()


def login_page(request: HttpRequest) -> HttpResponse:
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
