import asyncio
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from starlette.types import ASGIApp, Message

TOKEN = "a" * 32
GATE_SECONDS = 10


@dataclass
class SlowModel:
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    calls: int = 0

    def __call__(self, request: Any) -> str:
        self.calls += 1
        self.entered.set()
        self.release.wait(GATE_SECONDS)
        return "늦은 답변입니다."


@pytest.fixture
def slow_model(harness: Any) -> Iterator[SlowModel]:
    model = SlowModel()
    harness.fake_openai.chat_handler = model
    yield model
    model.release.set()


async def call_and_disconnect(app: ASGIApp, path: str, payload: Any, model: SlowModel) -> list[Message]:
    disconnect = asyncio.Event()
    body_sent = False

    async def receive() -> Message:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    async def disconnect_once_the_model_is_busy() -> None:
        assert await asyncio.to_thread(model.entered.wait, GATE_SECONDS)
        disconnect.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"localhost"),
            (b"content-type", b"application/json"),
            (b"cookie", f"csrftoken={TOKEN}".encode()),
            (b"x-csrftoken", TOKEN.encode()),
        ],
        "client": ("198.51.100.1", 50000),
        "server": ("localhost", 80),
    }
    watcher = asyncio.create_task(disconnect_once_the_model_is_busy())
    await asyncio.wait_for(app(scope, receive, send), timeout=GATE_SECONDS / 2)
    await watcher
    return sent


def statuses(sent: list[Message]) -> list[int]:
    return [message["status"] for message in sent if message["type"] == "http.response.start"]


def body(sent: list[Message]) -> bytes:
    return b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")


async def test_leaving_during_an_overview_cancels_it_and_caches_nothing(
    harness: Any, ids: dict[str, str], slow_model: SlowModel
) -> None:
    async with harness.app() as app:
        sent = await call_and_disconnect(app, f"/papers/{ids['main']}/analyze/", {}, slow_model)

    assert statuses(sent) == [499]
    assert slow_model.calls == 1
    assert all(arxiv_id != ids["main"] for arxiv_id, _ in harness.store.overviews)


async def test_leaving_during_a_summary_cancels_the_graph(
    harness: Any, ids: dict[str, str], slow_model: SlowModel
) -> None:
    async with harness.app() as app:
        sent = await call_and_disconnect(app, f"/papers/{ids['main']}/summary/", {"model": "gpt-5"}, slow_model)

    assert statuses(sent) == [499]
    assert all(arxiv_id != ids["main"] for arxiv_id, _, _ in harness.store.summaries)


@pytest.mark.parametrize("path", ["/papers/{main}/chat/", "/papers/assistant/chat/"])
async def test_leaving_a_non_streaming_chat_cancels_the_answer(
    harness: Any, ids: dict[str, str], slow_model: SlowModel, path: str
) -> None:
    async with harness.app() as app:
        sent = await call_and_disconnect(app, path.format(**ids), {"message": "질문"}, slow_model)

    assert statuses(sent) == [499]
    assert slow_model.calls == 1


@pytest.mark.parametrize("path", ["/papers/{main}/chat/stream/", "/papers/assistant/stream/"])
async def test_leaving_a_stream_stops_it_without_citations_or_done(
    harness: Any, ids: dict[str, str], slow_model: SlowModel, path: str
) -> None:
    async with harness.app() as app:
        sent = await call_and_disconnect(app, path.format(**ids), {"message": "질문"}, slow_model)

    assert statuses(sent) == [200]
    assert b"citations" not in body(sent)
    assert b"[DONE]" not in body(sent)
    assert slow_model.calls == 1
