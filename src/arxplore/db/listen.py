from types import TracebackType
from typing import Any, Self

from psycopg import AsyncConnection, OperationalError, sql

from arxplore.db.pool import connect


class ChannelListener:
    def __init__(self, conninfo: str, channel: str, *, role: str, connect_timeout: int) -> None:
        self._conninfo = conninfo
        self._channel = channel
        self._role = role
        self._connect_timeout = connect_timeout
        self._connection: AsyncConnection[Any] | None = None

    @property
    def channel(self) -> str:
        return self._channel

    @property
    def connected(self) -> bool:
        return self._connection is not None and not self._connection.closed and not self._connection.broken

    @property
    def backend_pid(self) -> int | None:
        return self._connection.info.backend_pid if self.connected and self._connection is not None else None

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def open(self) -> None:
        if self.connected:
            return
        await self._discard()
        connection = await connect(
            self._conninfo,
            role=self._role,
            connect_timeout=self._connect_timeout,
            autocommit=True,
        )
        try:
            await connection.execute(sql.SQL("LISTEN {}").format(sql.Identifier(self._channel)))
        except BaseException:
            await connection.close()
            raise
        self._connection = connection

    async def close(self) -> None:
        await self._discard()

    async def wait(self, timeout: float) -> list[str]:
        if not self.connected:
            reconnecting = self._connection is not None
            await self.open()
            if reconnecting:
                return []
        connection = self._connection
        if connection is None:
            return []
        try:
            payloads = [notify.payload async for notify in connection.notifies(timeout=max(0.0, timeout), stop_after=1)]
            if payloads:
                payloads.extend([notify.payload async for notify in connection.notifies(timeout=0.0)])
        except OperationalError:
            await self._discard()
            await self.open()
            return []
        return payloads

    async def _discard(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None and not connection.closed:
            await connection.close()
