import asyncio
import time
from collections.abc import AsyncIterator
from datetime import date

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from arxplore.config import AppSettings
from arxplore.db.listen import ChannelListener
from arxplore.db.repositories import jobs
from arxplore.db.repositories.jobs import JOB_CHANNEL, job_listener

pytestmark = pytest.mark.integration


@pytest.fixture
async def listener(repo_database: str) -> AsyncIterator[ChannelListener]:
    channel_listener = ChannelListener(repo_database, JOB_CHANNEL, role="test-listener", connect_timeout=5)
    async with channel_listener:
        yield channel_listener


async def _notify(probe, payload: str) -> None:
    await probe.execute("SELECT pg_notify(%s, %s)", (JOB_CHANNEL, payload))


async def test_wait_times_out_with_no_notifications(listener) -> None:
    started = time.monotonic()

    assert await listener.wait(0.3) == []

    assert 0.25 <= time.monotonic() - started < 3.0


async def test_notifications_queued_while_busy_wake_one_wait_and_are_drained_together(listener, probe) -> None:
    for day in ("2026-04-07", "2026-04-08", "2026-04-09"):
        await _notify(probe, f"auto:{day}")
    await asyncio.sleep(0.2)

    started = time.monotonic()
    received = await listener.wait(5.0)

    assert received == ["auto:2026-04-07", "auto:2026-04-08", "auto:2026-04-09"]
    assert time.monotonic() - started < 1.0
    assert await listener.wait(0.2) == []


async def test_one_connection_keeps_listening_between_waits(listener, pool) -> None:
    pid = listener.backend_pid

    await jobs.enqueue(pool, target_date=date(2026, 4, 7))
    assert await listener.wait(5.0) == ["auto:2026-04-07"]
    await jobs.enqueue(pool, target_date=date(2026, 4, 8))
    assert await listener.wait(5.0) == ["auto:2026-04-08"]

    assert listener.backend_pid == pid


async def test_wait_wakes_up_when_a_notification_arrives(listener, probe) -> None:
    async def notify_later() -> None:
        await asyncio.sleep(0.3)
        await _notify(probe, "auto:2026-04-07")

    sender = asyncio.create_task(notify_later())
    started = time.monotonic()
    received = await listener.wait(10.0)
    await sender

    assert received == ["auto:2026-04-07"]
    assert time.monotonic() - started < 5.0


async def test_other_channels_do_not_wake_the_listener(listener, probe) -> None:
    await probe.execute("SELECT pg_notify('another_channel', 'x')")

    assert await listener.wait(0.3) == []


async def test_wait_reconnects_after_the_connection_is_lost(listener, probe, repo_database) -> None:
    lost_pid = listener.backend_pid
    async with await psycopg.AsyncConnection.connect(repo_database, autocommit=True) as admin:
        await admin.execute("SELECT pg_terminate_backend(%s)", (lost_pid,))

    started = time.monotonic()
    assert await listener.wait(10.0) == []
    assert time.monotonic() - started < 5.0
    assert listener.connected
    assert listener.backend_pid != lost_pid

    await _notify(probe, "auto:2026-04-07")
    assert await listener.wait(5.0) == ["auto:2026-04-07"]


async def test_listener_opens_on_first_wait_and_closes(repo_database, probe) -> None:
    listener = ChannelListener(repo_database, JOB_CHANNEL, role="test-listener", connect_timeout=5)
    assert not listener.connected

    assert await listener.wait(0.1) == []
    assert listener.connected
    await _notify(probe, "auto:2026-04-07")
    assert await listener.wait(5.0) == ["auto:2026-04-07"]

    await listener.close()
    assert not listener.connected
    assert listener.backend_pid is None


def _settings_for(url: str) -> AppSettings:
    params = conninfo_to_dict(url)
    return AppSettings.model_validate(
        {
            "POSTGRES_HOST": f"{params.get('host', 'localhost')}:{params.get('port', 5432)}",
            "APP_POSTGRES_DB": str(params["dbname"]),
            "POSTGRES_USER": str(params.get("user", "")),
            "POSTGRES_PASSWORD": str(params.get("password") or "unused-by-trust-auth"),
            "POSTGRES_CONNECT_TIMEOUT": "4",
        }
    )


async def test_job_listener_uses_the_settings_and_role(repo_database, probe) -> None:
    settings = _settings_for(repo_database)

    async with job_listener(settings) as listener:
        assert listener.channel == "arxplore_prepare_jobs"
        application_name = await probe.scalar(
            "SELECT application_name FROM pg_stat_activity WHERE pid = %s", (listener.backend_pid,)
        )
        await _notify(probe, "auto:2026-04-07")
        received = await listener.wait(5.0)

    assert application_name == "arxplore-worker-listener"
    assert received == ["auto:2026-04-07"]
