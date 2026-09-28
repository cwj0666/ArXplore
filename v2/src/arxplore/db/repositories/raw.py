import hashlib
import json
from dataclasses import dataclass
from datetime import date
from functools import partial
from typing import Any

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from arxplore.db.pool import ConnectionSource, use_connection
from arxplore.text import strip_nul_and_surrogates_deep

DAILY_PAPERS_SOURCE = "hf_daily_papers"

SAVE_DAILY_PAPERS_SQL = """
WITH previous AS (
    SELECT payload_hash
    FROM raw_daily_papers
    WHERE source = %(source)s AND date = %(date)s
),
upserted AS (
    INSERT INTO raw_daily_papers (source, date, payload, payload_hash, revision, fetched_count, collected_at)
    VALUES (%(source)s, %(date)s, %(payload)s, %(payload_hash)s, 1, %(fetched_count)s, NOW())
    ON CONFLICT (source, date) DO UPDATE SET
        payload = CASE
            WHEN raw_daily_papers.payload_hash = EXCLUDED.payload_hash THEN raw_daily_papers.payload
            ELSE EXCLUDED.payload
        END,
        payload_hash = EXCLUDED.payload_hash,
        fetched_count = EXCLUDED.fetched_count,
        revision = CASE
            WHEN raw_daily_papers.payload_hash = EXCLUDED.payload_hash THEN raw_daily_papers.revision
            ELSE raw_daily_papers.revision + 1
        END,
        collected_at = NOW()
    RETURNING revision, payload_hash
)
SELECT
    upserted.revision,
    (SELECT previous.payload_hash FROM previous) IS DISTINCT FROM upserted.payload_hash AS changed
FROM upserted
"""

LOAD_DAILY_PAPERS_SQL = "SELECT payload FROM raw_daily_papers WHERE source = %s AND date = %s"

HAS_DAILY_PAPERS_SQL = "SELECT EXISTS (SELECT 1 FROM raw_daily_papers WHERE source = %s AND date = %s) AS present"

DATE_FILTER_OPERATORS = (("date_gt", ">"), ("date_gte", ">="), ("date_lte", "<="))

raw_json_dumps = partial(json.dumps, ensure_ascii=False, default=str)


class RawSaveError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class RawSaveResult:
    record_id: str
    revision: int
    changed: bool


def compute_payload_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def raw_jsonb(value: Any) -> Jsonb:
    return Jsonb(value, dumps=raw_json_dumps)


def record_id(target_date: date) -> str:
    return f"{DAILY_PAPERS_SOURCE}:{target_date.isoformat()}"


async def save_daily_papers(db: ConnectionSource, *, target_date: date, payload: Any) -> RawSaveResult:
    sanitized = strip_nul_and_surrogates_deep(payload)
    params = {
        "source": DAILY_PAPERS_SOURCE,
        "date": target_date,
        "payload": raw_jsonb(sanitized),
        "payload_hash": compute_payload_hash(sanitized),
        "fetched_count": len(sanitized) if isinstance(sanitized, list) else 1,
    }
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(SAVE_DAILY_PAPERS_SQL, params)
        row = await cursor.fetchone()
    if row is None:
        raise RawSaveError("raw_daily_papers upsert가 행을 반환하지 않았습니다.")
    return RawSaveResult(record_id=record_id(target_date), revision=int(row["revision"]), changed=bool(row["changed"]))


async def load_daily_papers(db: ConnectionSource, *, target_date: date) -> list[Any]:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(LOAD_DAILY_PAPERS_SQL, (DAILY_PAPERS_SOURCE, target_date))
        row = await cursor.fetchone()
    if row is None:
        return []
    payload = row["payload"]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    return []


async def has_daily_papers(db: ConnectionSource, *, target_date: date) -> bool:
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(HAS_DAILY_PAPERS_SQL, (DAILY_PAPERS_SOURCE, target_date))
        row = await cursor.fetchone()
    return bool(row and row["present"])


async def list_daily_papers_dates(
    db: ConnectionSource,
    *,
    date_gt: date | None = None,
    date_gte: date | None = None,
    date_lte: date | None = None,
    limit: int | None = None,
    ascending: bool = True,
) -> list[date]:
    bounds = {"date_gt": date_gt, "date_gte": date_gte, "date_lte": date_lte}
    conditions: list[sql.Composable] = [sql.SQL("source = %s")]
    params: list[Any] = [DAILY_PAPERS_SOURCE]
    for name, operator in DATE_FILTER_OPERATORS:
        value = bounds[name]
        if value:
            conditions.append(sql.SQL("date {} %s").format(sql.SQL(operator)))
            params.append(value)
    query = sql.SQL("SELECT date FROM raw_daily_papers WHERE {} ORDER BY date {}").format(
        sql.SQL(" AND ").join(conditions),
        sql.SQL("ASC" if ascending else "DESC"),
    )
    if isinstance(limit, int) and limit > 0:
        query = sql.Composed([query, sql.SQL(" LIMIT %s")])
        params.append(limit)
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(query, params)
        rows = await cursor.fetchall()
    return [row["date"] for row in rows if row["date"] is not None]
