import json
import logging
from collections.abc import Iterable, Iterator
from typing import Any

from django.http import HttpRequest, JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from openai import AuthenticationError

from .services import (
    AuthenticationRequiredError,
    InvalidSummaryModelError,
    MissingApiKeyError,
    PaperNotFoundError,
    answer_agent_chat,
    answer_paper_chat,
    build_auth_payload,
    build_bootstrap_payload,
    build_favorites_payload,
    build_settings_payload,
    clear_personal_api_key,
    get_paper_analysis,
    get_paper_summary,
    get_session_api_key,
    login_user,
    logout_user,
    prepare_agent_chat,
    prepare_paper_chat,
    register_user,
    save_personal_api_key,
    stream_agent_chat,
    stream_paper_chat,
    toggle_favorite_paper,
    update_user_settings,
)

logger = logging.getLogger(__name__)

STREAM_ERROR_MESSAGE = "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
INVALID_API_KEY_MESSAGE = "OpenAI API 키가 유효하지 않습니다. 설정에서 키를 확인하세요."


def _json_body(request: HttpRequest) -> dict:
    try:
        payload = json.loads(request.body)
    except Exception as exc:
        raise ValueError("잘못된 요청입니다.") from exc
    if not isinstance(payload, dict):
        raise ValueError("잘못된 요청입니다.")
    return payload


@require_GET
@ensure_csrf_cookie
def bootstrap(request: HttpRequest):
    return JsonResponse(build_bootstrap_payload(request))


@require_POST
def auth_signup(request: HttpRequest):
    try:
        body = _json_body(request)
        payload = register_user(
            username=str(body.get("username", "")),
            password=str(body.get("password", "")),
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    return JsonResponse(payload)


@require_POST
def auth_login(request: HttpRequest):
    try:
        body = _json_body(request)
        payload = login_user(
            request,
            username=str(body.get("username", "")),
            password=str(body.get("password", "")),
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    return JsonResponse(payload)


@require_POST
def auth_logout(request: HttpRequest):
    return JsonResponse(logout_user(request))


@require_GET
def auth_me(request: HttpRequest):
    return JsonResponse(build_auth_payload(request.user))


@require_http_methods(["GET", "POST"])
def settings_detail(request: HttpRequest):
    if request.method == "GET":
        try:
            return JsonResponse(build_settings_payload(request.user))
        except AuthenticationRequiredError as exc:
            return JsonResponse({"error": str(exc)}, status=401)

    try:
        body = _json_body(request)
        payload = update_user_settings(
            user=request.user,
            preferred_summary_model=str(body.get("preferred_summary_model", "")),
        )
    except AuthenticationRequiredError as exc:
        return JsonResponse({"error": str(exc)}, status=401)
    except (InvalidSummaryModelError, ValueError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    return JsonResponse(payload)


@require_http_methods(["POST", "DELETE"])
def settings_api_key_detail(request: HttpRequest):
    if request.method == "DELETE":
        try:
            payload = clear_personal_api_key(request)
        except AuthenticationRequiredError as exc:
            return JsonResponse({"error": str(exc)}, status=401)
        return JsonResponse(payload)

    try:
        body = _json_body(request)
        payload = save_personal_api_key(request, str(body.get("api_key", "")))
    except AuthenticationRequiredError as exc:
        return JsonResponse({"error": str(exc)}, status=401)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    return JsonResponse(payload)


@require_GET
def favorites_list(request: HttpRequest):
    try:
        return JsonResponse(build_favorites_payload(request.user))
    except AuthenticationRequiredError as exc:
        return JsonResponse({"error": str(exc)}, status=401)


@require_POST
def favorites_toggle(request: HttpRequest):
    try:
        body = _json_body(request)
        payload = toggle_favorite_paper(
            user=request.user,
            arxiv_id=str(body.get("arxiv_id", "")),
        )
    except AuthenticationRequiredError as exc:
        return JsonResponse({"error": str(exc)}, status=401)
    except PaperNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    return JsonResponse(payload)


@require_POST
def paper_analyze(request: HttpRequest, arxiv_id: str):
    try:
        return JsonResponse(
            get_paper_analysis(
                arxiv_id,
                user=request.user,
                session_api_key=get_session_api_key(request),
            )
        )
    except AuthenticationRequiredError as exc:
        return JsonResponse({"error": str(exc)}, status=401)
    except MissingApiKeyError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except PaperNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)


@require_POST
def paper_summary(request: HttpRequest, arxiv_id: str):
    try:
        body = _json_body(request)
        return JsonResponse(
            get_paper_summary(
                arxiv_id,
                model=str(body.get("model", "")),
                user=request.user,
                session_api_key=get_session_api_key(request),
            )
        )
    except AuthenticationRequiredError as exc:
        return JsonResponse({"error": str(exc)}, status=401)
    except (MissingApiKeyError, InvalidSummaryModelError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except PaperNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)


@require_POST
def paper_chat(request: HttpRequest, arxiv_id: str):
    try:
        body = _json_body(request)
        payload = answer_paper_chat(
            arxiv_id,
            user_message=str(body.get("message", "")),
            chat_history=body.get("history", []),
            user=request.user,
            session_api_key=get_session_api_key(request),
        )
    except AuthenticationRequiredError as exc:
        return JsonResponse({"error": str(exc)}, status=401)
    except MissingApiKeyError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except PaperNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        logger.exception("상세 챗 응답 생성 실패")
        return JsonResponse({"error": _stream_error_message(exc)}, status=500)

    return JsonResponse(payload)


@require_POST
def paper_chat_stream(request: HttpRequest, arxiv_id: str):
    try:
        body = _json_body(request)
        prepared = prepare_paper_chat(
            arxiv_id,
            user_message=str(body.get("message", "")),
            chat_history=body.get("history", []),
            user=request.user,
            session_api_key=get_session_api_key(request),
        )
    except AuthenticationRequiredError as exc:
        return JsonResponse({"error": str(exc)}, status=401)
    except MissingApiKeyError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except PaperNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception:
        logger.exception("상세 챗 스트림 준비 실패")
        return JsonResponse({"error": STREAM_ERROR_MESSAGE}, status=500)

    return _sse_response(stream_paper_chat(prepared))


@require_POST
def paper_agent_chat(request: HttpRequest):
    try:
        body = _json_body(request)
        payload = answer_agent_chat(
            user_message=str(body.get("message", "")),
            chat_history=body.get("history", []),
            user=request.user,
            session_api_key=get_session_api_key(request),
        )
    except AuthenticationRequiredError as exc:
        return JsonResponse({"error": str(exc)}, status=401)
    except MissingApiKeyError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("에이전트 응답 생성 실패")
        return JsonResponse({"error": _stream_error_message(exc)}, status=500)

    return JsonResponse(payload)


@require_POST
def paper_agent_stream(request: HttpRequest):
    try:
        body = _json_body(request)
        prepared = prepare_agent_chat(
            user_message=str(body.get("message", "")),
            chat_history=body.get("history", []),
            user=request.user,
            session_api_key=get_session_api_key(request),
        )
    except AuthenticationRequiredError as exc:
        return JsonResponse({"error": str(exc)}, status=401)
    except MissingApiKeyError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception:
        logger.exception("에이전트 스트림 준비 실패")
        return JsonResponse({"error": STREAM_ERROR_MESSAGE}, status=500)

    return _sse_response(stream_agent_chat(prepared))


def _sse(payload: Any) -> str:
    if payload == "[DONE]":
        return "data: [DONE]\n\n"
    return f"data: {json.dumps(payload)}\n\n"


def _stream_error_message(exc: Exception) -> str:
    if isinstance(exc, AuthenticationError):
        return INVALID_API_KEY_MESSAGE
    return STREAM_ERROR_MESSAGE


def _sse_events(events: Iterable[Any]) -> Iterator[str]:
    """이벤트 순서: chunk* → citations(정확히 1회) → [DONE]. 오류가 나면 error → [DONE]."""
    citations_sent = False
    try:
        for event in events:
            if isinstance(event, str):
                event = {"chunk": event}
            if "citations" in event:
                if citations_sent:
                    continue
                citations_sent = True
                yield _sse({"citations": event["citations"]})
            elif "chunk" in event:
                if event["chunk"]:
                    yield _sse({"chunk": event["chunk"]})
        if not citations_sent:
            yield _sse({"citations": []})
    except Exception as exc:
        logger.exception("SSE 스트림 중 오류")
        yield _sse({"error": _stream_error_message(exc)})
    yield _sse("[DONE]")


def _sse_response(events: Iterable[Any]) -> StreamingHttpResponse:
    return StreamingHttpResponse(
        _sse_events(events),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
