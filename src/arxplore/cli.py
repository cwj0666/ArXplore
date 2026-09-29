import asyncio
import importlib
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

import typer
from psycopg import AsyncConnection
from psycopg import Error as DatabaseError
from psycopg.errors import Diagnostic

from arxplore.config import AppSettings, build_postgres_connection_params, get_settings
from arxplore.db import migrate as migrations
from arxplore.db.pool import DictConnection, connect_from_settings

app = typer.Typer(no_args_is_help=True, add_completion=False, help="ArXplore command line")


def _echo_warning(diagnostic: Diagnostic) -> None:
    if diagnostic.severity_nonlocalized in {"WARNING", "ERROR", "FATAL", "PANIC"}:
        typer.echo(f"{diagnostic.severity}: {diagnostic.message_primary}", err=True)


def _describe_target(settings: AppSettings) -> str:
    params = build_postgres_connection_params(settings)
    return f"{params['user']}@{params['host']}:{params['port']}/{params['dbname']}"


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(code=1)


def _run(operation: Callable[[AppSettings, DictConnection], Awaitable[None]], *, role: str) -> None:
    try:
        settings = get_settings()
        typer.echo(f"database {_describe_target(settings)}")
    except ValueError as exc:
        raise _fail(str(exc)) from exc

    async def session() -> None:
        connection = await connect_from_settings(settings, role=role, autocommit=True)
        connection.add_notice_handler(_echo_warning)
        async with connection:
            await operation(settings, connection)

    try:
        asyncio.run(session())
    except (migrations.MigrationError, DatabaseError) as exc:
        raise _fail(str(exc).strip()) from exc


async def _echo_search_indexes(connection: AsyncConnection[Any]) -> None:
    for name, present in (await migrations.fetch_search_indexes(connection)).items():
        typer.echo(f"index {name}: {'present' if present else 'MISSING'}")


@app.command("migrate", help="Apply pending SQL migrations in order, one transaction per file.")
def migrate_command(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would run without changing the database.")
    ] = False,
    target: Annotated[
        int | None,
        typer.Option("--target", help="Apply migrations up to and including this version, e.g. 1 for 0001."),
    ] = None,
    lock_timeout: Annotated[
        float,
        typer.Option("--lock-timeout", help="Seconds to wait for another migration run to finish."),
    ] = 60.0,
) -> None:
    async def operation(settings: AppSettings, connection: DictConnection) -> None:
        planned = await migrations.migrate(connection, target=target, dry_run=dry_run, lock_timeout=lock_timeout)
        verb = "would apply" if dry_run else "applied"
        for migration in planned:
            typer.echo(f"{verb} {migration.name}")
        if not planned:
            typer.echo("nothing to apply")
        if not dry_run:
            await _echo_search_indexes(connection)

    _run(operation, role="migrate")


@app.command("db-status", help="Show migration state, embedding width and search indexes.")
def db_status_command() -> None:
    async def operation(settings: AppSettings, connection: DictConnection) -> None:
        statuses = migrations.migration_status(
            migrations.load_migrations(),
            await migrations.fetch_applied(connection),
        )
        width = max((len(status.label) for status in statuses), default=0)
        for status in statuses:
            applied_at = status.applied_at.isoformat(timespec="seconds") if status.applied_at else ""
            typer.echo(f"{status.label:<{width}}  {status.state.value:<8} {applied_at}".rstrip())
        dimensions = await migrations.fetch_embedding_dimensions(connection)
        typer.echo(
            f"embedding dimensions: {dimensions if dimensions is not None else 'missing'} "
            f"(OPENAI_EMBEDDING_DIMENSIONS={settings.openai_embedding_dimensions})"
        )
        await _echo_search_indexes(connection)
        problems = migrations.readiness_problems(
            statuses,
            embedding_dimensions=dimensions,
            expected_embedding_dimensions=settings.openai_embedding_dimensions,
        )
        typer.echo("ready: yes" if not problems else f"ready: no ({'; '.join(problems)})")

    _run(operation, role="cli")


COMMAND_MODULES = (
    "arxplore.api.commands",
    "arxplore.pipeline.commands",
    "arxplore.scheduler.commands",
)


def register_command_modules(target: typer.Typer, module_names: tuple[str, ...] = COMMAND_MODULES) -> list[str]:
    registered = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or (exc.name and module_name.startswith(f"{exc.name}.")):
                continue
            raise
        module.register(target)
        registered.append(module_name)
    return registered


register_command_modules(app)


if __name__ == "__main__":
    app()
