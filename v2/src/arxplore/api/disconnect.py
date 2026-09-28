import asyncio
import logging
from collections.abc import Awaitable
from contextlib import suppress

from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)

CLIENT_CLOSED_REQUEST = 499


class ClientDisconnectedError(Exception):
    pass


async def wait_for_disconnect(receive: Receive) -> None:
    while (await receive())["type"] != "http.disconnect":
        continue


async def until_disconnected[T](request: Request, work: Awaitable[T]) -> T:
    task = asyncio.ensure_future(work)
    watcher = asyncio.ensure_future(wait_for_disconnect(request.receive))
    try:
        await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            return task.result()
        logger.info("클라이언트 연결이 끊겨 생성을 중단합니다: %s", request.url.path)
        task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task
        raise ClientDisconnectedError(request.url.path)
    finally:
        for pending in (task, watcher):
            if not pending.done():
                pending.cancel()
        with suppress(asyncio.CancelledError):
            await watcher


class DisconnectAwareStreamingResponse(StreamingResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        stream = asyncio.ensure_future(self.stream_response(send))
        listener = asyncio.ensure_future(self.listen_for_disconnect(receive))
        try:
            await asyncio.wait({stream, listener}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in (stream, listener):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stream, listener, return_exceptions=True)
        error = None if stream.cancelled() else stream.exception()
        if error is not None:
            raise error
        if self.background is not None:
            await self.background()
