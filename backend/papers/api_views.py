import json
import logging
from collections.abc import Iterable, Iterator
from typing import Any

from django.http import HttpRequest, JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from .ratelimit import rate_limit
from .services import (
    InvalidRequestError,
    MissingApiKeyError,
    PaperNotFoundError,
    answer_agent_chat,
    answer_paper_chat,
    build_bootstrap_payload,
    get_paper_analysis,
    get_paper_summary,
    prepare_agent_chat,
    prepare_paper_chat,
    stream_agent_chat,
    stream_paper_chat,
)

logger = logging.getLogger(__name__)

STREAM_ERROR_MESSAGE = "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
GENERATION_ERROR_MESSAGE = "AI 결과 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

llm_rate_limit = rate_limit("llm", limit_setting="RATE_LIMIT_LLM_PER_MINUTE")


def _json_body(request: HttpRequest) -> dict:
    try:
        payload = json.loads(request.body)
    except Exception as exc:
        raise InvalidRequestError("잘못된 요청입니다.") from exc
    if not isinstance(payload, dict):
        raise InvalidRequestError("잘못된 요청입니다.")
    return payload


def _ai_unavailable(exc: MissingApiKeyError) -> JsonResponse:
    return JsonResponse({"error": str(exc)}, status=503)


@require_GET
@ensure_csrf_cookie
def bootstrap(request: HttpRequest):
    return JsonResponse(build_bootstrap_payload())


@require_POST
@llm_rate_limit
def paper_analyze(request: HttpRequest, arxiv_id: str):
    try:
        return JsonResponse(get_paper_analysis(arxiv_id))
    except MissingApiKeyError as exc:
        return _ai_unavailable(exc)
    except PaperNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception:
        logger.exception("overview 생성 실패")
        return JsonResponse({"error": GENERATION_ERROR_MESSAGE}, status=500)


@require_POST
@llm_rate_limit
def paper_summary(request: HttpRequest, arxiv_id: str):
    try:
        body = _json_body(request)
        return JsonResponse(get_paper_summary(arxiv_id, model=str(body.get("model", ""))))
    except MissingApiKeyError as exc:
        return _ai_unavailable(exc)
    except InvalidRequestError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except PaperNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception:
        logger.exception("상세 요약 생성 실패")
        return JsonResponse({"error": GENERATION_ERROR_MESSAGE}, status=500)


@require_POST
@llm_rate_limit
def paper_chat(request: HttpRequest, arxiv_id: str):
    try:
        body = _json_body(request)
        payload = answer_paper_chat(arxiv_id, str(body.get("message", "")), body.get("history", []))
    except MissingApiKeyError as exc:
        return _ai_unavailable(exc)
    except InvalidRequestError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except PaperNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception:
        logger.exception("상세 챗 응답 생성 실패")
        return JsonResponse({"error": STREAM_ERROR_MESSAGE}, status=500)

    return JsonResponse(payload)


@require_POST
@llm_rate_limit
def paper_chat_stream(request: HttpRequest, arxiv_id: str):
    try:
        body = _json_body(request)
        prepared = prepare_paper_chat(arxiv_id, str(body.get("message", "")), body.get("history", []))
    except MissingApiKeyError as exc:
        return _ai_unavailable(exc)
    except InvalidRequestError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except PaperNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception:
        logger.exception("상세 챗 스트림 준비 실패")
        return JsonResponse({"error": STREAM_ERROR_MESSAGE}, status=500)

    return _sse_response(stream_paper_chat(prepared))


@require_POST
@llm_rate_limit
def paper_agent_chat(request: HttpRequest):
    try:
        body = _json_body(request)
        payload = answer_agent_chat(str(body.get("message", "")), body.get("history", []))
    except MissingApiKeyError as exc:
        return _ai_unavailable(exc)
    except InvalidRequestError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception:
        logger.exception("에이전트 응답 생성 실패")
        return JsonResponse({"error": STREAM_ERROR_MESSAGE}, status=500)

    return JsonResponse(payload)


@require_POST
@llm_rate_limit
def paper_agent_stream(request: HttpRequest):
    try:
        body = _json_body(request)
        prepared = prepare_agent_chat(str(body.get("message", "")), body.get("history", []))
    except MissingApiKeyError as exc:
        return _ai_unavailable(exc)
    except InvalidRequestError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception:
        logger.exception("에이전트 스트림 준비 실패")
        return JsonResponse({"error": STREAM_ERROR_MESSAGE}, status=500)

    return _sse_response(stream_agent_chat(prepared))


def _sse(payload: Any) -> str:
    if payload == "[DONE]":
        return "data: [DONE]\n\n"
    return f"data: {json.dumps(payload)}\n\n"


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
    except Exception:
        logger.exception("SSE 스트림 중 오류")
        yield _sse({"error": STREAM_ERROR_MESSAGE})
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
