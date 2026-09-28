import asyncio
import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row

MIGRATION_FILE_PATTERN = re.compile(r"^(?P<version>\d{4})_(?P<slug>[a-z0-9_]+)\.sql$")
MIGRATION_LOCK_KEY = 0x41525850_4D494752
LOCK_POLL_SECONDS = 0.2
SEARCH_INDEXES = (
    "idx_papers_title_abstract_vector",
    "idx_paper_chunks_chunk_vector",
    "paper_embeddings_embedding_hnsw",
)

CREATE_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    checksum TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


class MigrationError(Exception):
    pass


class ChecksumMismatchError(MigrationError):
    pass


class UnknownMigrationError(MigrationError):
    pass


class MigrationOrderError(MigrationError):
    pass


class MigrationLockTimeoutError(MigrationError):
    pass


class MigrationState(StrEnum):
    APPLIED = "applied"
    PENDING = "pending"
    CHANGED = "changed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str
    checksum: str


@dataclass(frozen=True, slots=True)
class AppliedMigration:
    version: int
    checksum: str
    applied_at: datetime


@dataclass(frozen=True, slots=True)
class MigrationStatus:
    version: int
    name: str | None
    state: MigrationState
    applied_at: datetime | None

    @property
    def label(self) -> str:
        return self.name or format_version(self.version)


def format_version(version: int) -> str:
    return f"{version:04d}"


def compute_checksum(sql: str) -> str:
    normalized = sql.replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def default_migrations_directory() -> Traversable:
    return files("arxplore.db") / "migrations"


def load_migrations(directory: Traversable | None = None) -> list[Migration]:
    source = directory if directory is not None else default_migrations_directory()
    migrations: dict[int, Migration] = {}
    for entry in source.iterdir():
        if not entry.is_file() or not entry.name.endswith(".sql"):
            continue
        match = MIGRATION_FILE_PATTERN.match(entry.name)
        if match is None:
            raise MigrationError(f"migration file name must look like 0001_name.sql: {entry.name}")
        version = int(match["version"])
        if version == 0:
            raise MigrationError(f"migration versions start at 0001: {entry.name}")
        if version in migrations:
            raise MigrationError(
                f"duplicate migration version {format_version(version)}: {migrations[version].name}, {entry.name}"
            )
        sql = entry.read_text(encoding="utf-8")
        migrations[version] = Migration(
            version=version,
            name=entry.name.removesuffix(".sql"),
            sql=sql,
            checksum=compute_checksum(sql),
        )
    return [migrations[version] for version in sorted(migrations)]


def migration_status(
    migrations: Sequence[Migration],
    applied: Sequence[AppliedMigration],
) -> list[MigrationStatus]:
    known = {migration.version: migration for migration in migrations}
    recorded = {row.version: row for row in applied}
    statuses: list[MigrationStatus] = []
    for version in sorted(known.keys() | recorded.keys()):
        migration = known.get(version)
        row = recorded.get(version)
        if migration is None:
            state = MigrationState.UNKNOWN
        elif row is None:
            state = MigrationState.PENDING
        elif row.checksum != migration.checksum:
            state = MigrationState.CHANGED
        else:
            state = MigrationState.APPLIED
        statuses.append(
            MigrationStatus(
                version=version,
                name=migration.name if migration else None,
                state=state,
                applied_at=row.applied_at if row else None,
            )
        )
    return statuses


def _labels(statuses: Iterable[MigrationStatus]) -> str:
    return ", ".join(status.label for status in statuses)


def plan_migrations(
    migrations: Sequence[Migration],
    applied: Sequence[AppliedMigration],
    target: int | None = None,
) -> list[Migration]:
    statuses = migration_status(migrations, applied)
    changed = [status for status in statuses if status.state is MigrationState.CHANGED]
    if changed:
        raise ChecksumMismatchError(f"applied migrations were modified after they ran: {_labels(changed)}")
    unknown = [status for status in statuses if status.state is MigrationState.UNKNOWN]
    if unknown:
        raise UnknownMigrationError(f"database has migrations this build does not know: {_labels(unknown)}")
    if target is not None and target not in {migration.version for migration in migrations}:
        raise MigrationError(f"unknown target version: {format_version(target)}")

    applied_versions = {status.version for status in statuses if status.state is MigrationState.APPLIED}
    latest_applied = max(applied_versions, default=0)
    pending = [migration for migration in migrations if migration.version not in applied_versions]
    out_of_order = [migration.name for migration in pending if migration.version < latest_applied]
    if out_of_order:
        raise MigrationOrderError(
            f"pending migrations are older than applied {format_version(latest_applied)}: {', '.join(out_of_order)}"
        )
    return [migration for migration in pending if target is None or migration.version <= target]


async def migrations_table_exists(connection: AsyncConnection[Any]) -> bool:
    async with connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute("SELECT to_regclass('schema_migrations') IS NOT NULL AS present")
        row = await cursor.fetchone()
    return bool(row and row["present"])


async def fetch_applied(connection: AsyncConnection[Any]) -> list[AppliedMigration]:
    if not await migrations_table_exists(connection):
        return []
    async with connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute("SELECT version, checksum, applied_at FROM schema_migrations ORDER BY version")
        rows = await cursor.fetchall()
    return [
        AppliedMigration(version=int(row["version"]), checksum=str(row["checksum"]), applied_at=row["applied_at"])
        for row in rows
    ]


async def acquire_migration_lock(connection: AsyncConnection[Any], timeout: float) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, timeout)
    while True:
        async with connection.cursor(row_factory=dict_row) as cursor:
            await cursor.execute("SELECT pg_try_advisory_lock(%s) AS locked", (MIGRATION_LOCK_KEY,))
            row = await cursor.fetchone()
        if row and row["locked"]:
            return
        if loop.time() >= deadline:
            raise MigrationLockTimeoutError(f"another process holds the migration lock; gave up after {timeout:g}s")
        await asyncio.sleep(LOCK_POLL_SECONDS)


async def release_migration_lock(connection: AsyncConnection[Any]) -> None:
    if connection.closed or connection.broken:
        return
    await connection.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_KEY,))


async def apply_migration(connection: AsyncConnection[Any], migration: Migration) -> None:
    async with connection.transaction():
        await connection.execute(migration.sql.encode(connection.info.encoding))
        await connection.execute(
            "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
            (migration.version, migration.checksum),
        )


async def migrate(
    connection: AsyncConnection[Any],
    *,
    migrations: Sequence[Migration] | None = None,
    target: int | None = None,
    dry_run: bool = False,
    lock_timeout: float = 60.0,
) -> list[Migration]:
    available = list(migrations) if migrations is not None else load_migrations()
    if dry_run:
        return plan_migrations(available, await fetch_applied(connection), target)
    if not connection.autocommit:
        raise MigrationError("migrations need a connection in autocommit mode")

    await acquire_migration_lock(connection, lock_timeout)
    try:
        await connection.execute(CREATE_MIGRATIONS_TABLE_SQL)
        pending = plan_migrations(available, await fetch_applied(connection), target)
        for migration in pending:
            await apply_migration(connection, migration)
        return pending
    finally:
        await release_migration_lock(connection)


async def fetch_search_indexes(connection: AsyncConnection[Any]) -> dict[str, bool]:
    async with connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = current_schema() AND indexname = ANY(%s)",
            (list(SEARCH_INDEXES),),
        )
        present = {str(row["indexname"]) for row in await cursor.fetchall()}
    return {name: name in present for name in SEARCH_INDEXES}
