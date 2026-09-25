from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from psycopg2.extras import Json

from src.integrations.db import get_connection
from src.shared import AppSettings, build_postgres_connection_params, get_settings

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


def compute_payload_hash(payload: Any) -> str:
    """키 순서와 무관한 payload sha256 해시."""
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sanitize_payload(value: Any) -> Any:
    """JSONB에 넣을 수 없는 NUL 문자와 짝 없는 서로게이트를 문자열·키에서 제거한 사본을 돌려준다."""
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, (list, tuple)):
        return [sanitize_payload(item) for item in value]
    if isinstance(value, dict):
        return {_sanitize_text(str(key)): sanitize_payload(item) for key, item in value.items()}
    return value


def pipeline_state_key(pipeline: str, name: str = "default") -> str:
    """`pipeline_state.key` 값. 파이프라인 이름과 상태 이름을 `:`로 잇는다."""
    return f"{pipeline}:{name}"


def _sanitize_text(value: str) -> str:
    return "".join(char for char in value if char != "\x00" and not 0xD800 <= ord(char) <= 0xDFFF)


def _to_json(value: Any) -> Json:
    return Json(value, dumps=lambda obj: json.dumps(obj, ensure_ascii=False, default=str))


class RawPaperStore:
    """HF Daily Papers 원본 응답과 파이프라인 진행 상태를 PostgreSQL JSONB로 저장하는 진입점.

    생성자는 DDL을 실행하지 않는다. `raw_daily_papers`·`pipeline_state` 테이블은 `ensure_schema()`나
    `scripts/migrate_schema.py`로 만든다.
    `connection`을 받는 메서드는 그 연결의 트랜잭션 안에서 실행하고 commit하지 않는다. 받지 않으면 풀에서 연결을
    빌려 메서드 단위로 commit한다.
    """

    def __init__(self, *, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def transaction(self):
        """풀에서 연결 하나를 빌려 준다. 블록이 정상 종료하면 commit, 예외가 나면 rollback한다."""
        return self._connection()

    def save_daily_papers_response(
        self,
        *,
        date: str,
        payload: list[dict[str, Any]] | dict[str, Any],
        connection: Any | None = None,
    ) -> dict[str, Any]:
        """원본 응답을 (source, date)당 1행으로 upsert한다.

        payload_hash가 저장된 값과 다를 때만 revision을 1 올린다(첫 저장은 1). 같은 payload면 collected_at만 갱신한다.
        revision 계산은 한 번의 `INSERT ... ON CONFLICT` 안에서 끝난다. changed는 문장 시작 시점의 hash와 저장된
        hash를 비교한 값이므로, 같은 날짜를 동시에 저장하는 경합에서는 True로 기울 수 있다.
        반환값은 record_id(`source:date`), revision, changed(이번 저장으로 payload가 바뀌었는지)다.
        """
        sanitized = sanitize_payload(payload)
        params = {
            "source": DAILY_PAPERS_SOURCE,
            "date": date,
            "payload": _to_json(sanitized),
            "payload_hash": compute_payload_hash(sanitized),
            "fetched_count": len(sanitized) if isinstance(sanitized, list) else 1,
        }
        with self._cursor(connection) as cursor:
            cursor.execute(SAVE_DAILY_PAPERS_SQL, params)
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError("raw_daily_papers upsert가 행을 반환하지 않았습니다.")
        return {
            "record_id": f"{DAILY_PAPERS_SOURCE}:{date}",
            "revision": int(row[0]),
            "changed": bool(row[1]),
        }

    def load_daily_papers_response(self, *, date: str) -> list[dict[str, Any]]:
        """수집 날짜의 원본 payload를 리스트로 조회한다. 없으면 빈 리스트, dict payload는 1원소 리스트로 감싼다."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT payload FROM raw_daily_papers WHERE source = %s AND date = %s",
                (DAILY_PAPERS_SOURCE, date),
            )
            row = cursor.fetchone()
        if row is None:
            return []
        payload = row[0]
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        return []

    def has_daily_papers_response(self, *, date: str) -> bool:
        """특정 날짜의 HF Daily Papers 원본이 이미 저장돼 있는지 확인한다."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM raw_daily_papers WHERE source = %s AND date = %s)",
                (DAILY_PAPERS_SOURCE, date),
            )
            row = cursor.fetchone()
        return bool(row and row[0])

    def list_daily_papers_dates(
        self,
        *,
        date_gt: str | None = None,
        date_gte: str | None = None,
        date_lte: str | None = None,
        limit: int | None = None,
        ascending: bool = True,
    ) -> list[str]:
        """조건에 맞는 HF Daily Papers raw 날짜 목록을 `YYYY-MM-DD` 문자열로 정렬해 조회한다."""
        conditions = ["source = %s"]
        params: list[Any] = [DAILY_PAPERS_SOURCE]
        for operator, value in ((">", date_gt), (">=", date_gte), ("<=", date_lte)):
            if value:
                conditions.append(f"date {operator} %s")
                params.append(value)
        sql = (
            f"SELECT date FROM raw_daily_papers WHERE {' AND '.join(conditions)} "
            f"ORDER BY date {'ASC' if ascending else 'DESC'}"
        )
        if isinstance(limit, int) and limit > 0:
            sql += " LIMIT %s"
            params.append(limit)
        with self._cursor() as cursor:
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
        return [_format_date(row[0]) for row in rows if row[0] is not None]

    def load_pipeline_state(self, *, pipeline: str, name: str = "default") -> dict[str, Any] | None:
        """파이프라인 진행 상태를 읽는다. 저장된 상태 필드에 pipeline, name, updated_at을 더해 돌려준다."""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT value, updated_at FROM pipeline_state WHERE key = %s",
                (pipeline_state_key(pipeline, name),),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        value = row[0] if isinstance(row[0], dict) else {}
        return {**value, "pipeline": pipeline, "name": name, "updated_at": row[1]}

    def save_pipeline_state(
        self,
        *,
        pipeline: str,
        state: dict[str, Any],
        name: str = "default",
        connection: Any | None = None,
    ) -> None:
        """파이프라인 진행 상태를 통째로 덮어쓴다."""
        value = {key: item for key, item in state.items() if key not in {"pipeline", "name", "updated_at"}}
        with self._cursor(connection) as cursor:
            cursor.execute(
                """
                INSERT INTO pipeline_state (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                """,
                (pipeline_state_key(pipeline, name), _to_json(sanitize_payload(value))),
            )

    def ensure_schema(self) -> None:
        """raw_daily_papers·pipeline_state 테이블을 멱등하게 생성한다. 마이그레이션/프로세스 시작 시 1회만 호출한다."""
        with self._cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS raw_daily_papers (
                    source TEXT NOT NULL,
                    date DATE NOT NULL,
                    payload JSONB NOT NULL,
                    payload_hash TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1,
                    fetched_count INTEGER,
                    collected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (source, date)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_state (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

    @contextmanager
    def _cursor(self, connection: Any | None = None) -> Iterator[Any]:
        if connection is not None:
            with connection.cursor() as cursor:
                yield cursor
            return
        with self._connection() as owned, owned.cursor() as cursor:
            yield cursor

    def _connection(self):
        return get_connection(build_postgres_connection_params(self.settings), settings=self.settings)


def _format_date(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)
