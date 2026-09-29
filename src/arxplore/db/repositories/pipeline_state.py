from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg.rows import dict_row

from arxplore.db.pool import ConnectionSource, use_connection
from arxplore.db.repositories.raw import raw_jsonb
from arxplore.text import strip_nul_and_surrogates_deep

DEFAULT_STATE_NAME = "default"
RESERVED_STATE_KEYS = frozenset({"pipeline", "name", "updated_at"})

LOAD_PIPELINE_STATE_SQL = "SELECT value, updated_at FROM pipeline_state WHERE key = %s"

SAVE_PIPELINE_STATE_SQL = """
INSERT INTO pipeline_state (key, value, updated_at)
VALUES (%s, %s, NOW())
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class PipelineState:
    pipeline: str
    name: str
    values: dict[str, Any]
    updated_at: datetime


def pipeline_state_key(pipeline: str, name: str = DEFAULT_STATE_NAME) -> str:
    return f"{pipeline}:{name}"


async def load_pipeline_state(
    db: ConnectionSource,
    *,
    pipeline: str,
    name: str = DEFAULT_STATE_NAME,
) -> PipelineState | None:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(LOAD_PIPELINE_STATE_SQL, (pipeline_state_key(pipeline, name),))
        row = await cursor.fetchone()
    if row is None:
        return None
    values = row["value"] if isinstance(row["value"], dict) else {}
    return PipelineState(pipeline=pipeline, name=name, values=values, updated_at=row["updated_at"])


async def save_pipeline_state(
    db: ConnectionSource,
    *,
    pipeline: str,
    state: Mapping[str, Any],
    name: str = DEFAULT_STATE_NAME,
) -> None:
    values = {key: item for key, item in state.items() if key not in RESERVED_STATE_KEYS}
    async with use_connection(db) as connection, connection.cursor() as cursor:
        await cursor.execute(
            SAVE_PIPELINE_STATE_SQL,
            (pipeline_state_key(pipeline, name), raw_jsonb(strip_nul_and_surrogates_deep(values))),
        )
