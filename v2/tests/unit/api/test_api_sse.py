from collections.abc import AsyncGenerator

from arxplore.api.errors import STREAM_ERROR_MESSAGE
from arxplore.api.sse import EVENT_STREAM_HEADERS, event_stream_response, sse_events, sse_message
from arxplore.llm.citations import Citation
from arxplore.llm.events import AnswerChunk, AnswerCitations, AnswerEvent

CITATION = Citation(
    arxiv_id="2401.00001",
    title="T",
    url="https://arxiv.org/abs/2401.00001",
    section_title="3 Method",
    chunk_id=10,
    in_answer=True,
)


async def body(events: AsyncGenerator[AnswerEvent]) -> str:
    return b"".join([chunk async for chunk in sse_events(events)]).decode("ascii")


async def scripted(*events: AnswerEvent, error: Exception | None = None) -> AsyncGenerator[AnswerEvent]:
    for event in events:
        yield event
    if error is not None:
        raise error


def test_messages_are_ascii_json_with_default_separators() -> None:
    assert sse_message({"chunk": "답"}) == b'data: {"chunk": "\\ub2f5"}\n\n'


async def test_chunks_then_empty_citations_then_done() -> None:
    assert await body(scripted(AnswerChunk("a"), AnswerChunk(""), AnswerChunk("b"))) == (
        'data: {"chunk": "a"}\n\ndata: {"chunk": "b"}\n\ndata: {"citations": []}\n\ndata: [DONE]\n\n'
    )


async def test_only_the_first_citations_event_is_sent() -> None:
    other = Citation(arxiv_id="x", title="X", url="u", section_title=None, chunk_id=None, in_answer=False)

    text = await body(scripted(AnswerChunk("a"), AnswerCitations.of([CITATION]), AnswerCitations.of([other])))

    assert text == (
        'data: {"chunk": "a"}\n\n'
        'data: {"citations": [{"arxiv_id": "2401.00001", "title": "T", "url": "https://arxiv.org/abs/2401.00001", '
        '"section_title": "3 Method", "chunk_id": 10, "in_answer": true}]}\n\n'
        "data: [DONE]\n\n"
    )


async def test_errors_end_with_an_error_event_and_done_without_citations() -> None:
    text = await body(scripted(AnswerChunk("a"), error=RuntimeError("db password leaked in message")))

    assert text == (
        'data: {"chunk": "a"}\n\n'
        f"data: {sse_message({'error': STREAM_ERROR_MESSAGE}).decode('ascii')[6:-2]}\n\n"
        "data: [DONE]\n\n"
    )
    assert "password" not in text


def test_response_headers() -> None:
    response = event_stream_response(scripted())

    assert response.media_type == "text/event-stream"
    assert EVENT_STREAM_HEADERS == {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
