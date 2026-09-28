import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from typing import Any

from starlette.responses import StreamingResponse

from arxplore.api.errors import STREAM_ERROR_MESSAGE
from arxplore.llm.events import AnswerChunk, AnswerEvent

logger = logging.getLogger(__name__)

EVENT_STREAM_MEDIA_TYPE = "text/event-stream"
EVENT_STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
DONE_MESSAGE = b"data: [DONE]\n\n"


def sse_message(payload: Any) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=True)}\n\n".encode("ascii")


async def sse_events(events: AsyncGenerator[AnswerEvent]) -> AsyncIterator[bytes]:
    citations_sent = False
    try:
        async with aclosing(events) as stream:
            async for event in stream:
                if isinstance(event, AnswerChunk):
                    if event.text:
                        yield sse_message({"chunk": event.text})
                    continue
                if citations_sent:
                    continue
                citations_sent = True
                yield sse_message({"citations": [citation.to_payload() for citation in event.citations]})
        if not citations_sent:
            yield sse_message({"citations": []})
    except Exception:
        logger.exception("SSE 스트림 중 오류")
        yield sse_message({"error": STREAM_ERROR_MESSAGE})
    yield DONE_MESSAGE


def event_stream_response(events: AsyncGenerator[AnswerEvent]) -> StreamingResponse:
    return StreamingResponse(sse_events(events), media_type=EVENT_STREAM_MEDIA_TYPE, headers=EVENT_STREAM_HEADERS)
