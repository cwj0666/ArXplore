import argparse
import difflib
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

LEGACY_DB = "v2_legacy_schema"
NEW_DB = "v2_new_schema"
EXISTING_DB = "v2_existing"
SCRATCH_PREFIX = "v2_"
UNUSED_PASSWORD = "unused-by-trust-auth"
DUMP_OPTIONS = ("--schema-only", "--no-owner", "--no-privileges", "--exclude-table=schema_migrations")
RESTRICT_LINE = re.compile(r"^\\(un)?restrict\s")
DROPPED_TABLES = frozenset({"topics", "topic_papers", "topic_documents"})
CONVERTED_COLUMN = ("prepare_jobs", "target_date")
ADDED_CONSTRAINT = ("prepare_jobs", "ck_prepare_jobs_status")

LEGACY_SEED = """
import json
import sys

sys.path.insert(0, sys.argv[1])
from src.integrations.db import close_all_pools
from src.integrations.prepare_job_repository import PrepareJobRepository

repository = PrepareJobRepository()
try:
    for day in ("2026-04-06", "2026-04-07", "2026-04-08", "2026-04-09"):
        repository.enqueue_prepare_job(target_date=day, mode="auto", raw_revision=1)
    repository.enqueue_prepare_job(target_date="2026-04-07", mode="backfill", source="backfill")
    done = repository.claim_prepare_job(mode="auto", worker_id="seed")
    repository.complete_prepare_job(
        job_id=done["job_id"], worker_id="seed", claim_generation=done["claim_generation"], result={"seeded": True}
    )
    failed = repository.claim_prepare_job(mode="auto", worker_id="seed")
    repository.fail_prepare_job(
        job_id=failed["job_id"], worker_id="seed", claim_generation=failed["claim_generation"], error="seeded"
    )
    repository.requeue_failed_prepare_jobs(mode="auto", dry_run=True)
    repository.claim_prepare_job(mode="auto", worker_id="seed")
    print(json.dumps({"seeded": True}))
finally:
    close_all_pools()
"""

LEGACY_AFTER_0003 = """
import json
import sys

sys.path.insert(0, sys.argv[1])
from src.integrations.db import close_all_pools
from src.integrations.prepare_job_repository import PrepareJobRepository

repository = PrepareJobRepository()
try:
    queued = repository.enqueue_prepare_job(target_date="2026-04-10", mode="auto", raw_revision=1)
    claimed = repository.claim_prepare_job(mode="auto", worker_id="after")
    completed = repository.complete_prepare_job(
        job_id=claimed["job_id"], worker_id="after", claim_generation=claimed["claim_generation"], result={}
    )
    failed = repository.requeue_failed_prepare_jobs(mode="auto", since_date="2026-04-01", dry_run=True)
    print(json.dumps({
        "enqueued": queued["status"],
        "claimed_date": str(claimed["date"]),
        "claimed_date_type": type(claimed["date"]).__name__,
        "completed": completed,
        "failed_since": failed,
    }))
finally:
    close_all_pools()
"""

LEGACY_TARGET_PROBE = """
import json
import sys

sys.path.insert(0, sys.argv[1])
from src.shared.settings import AppSettings, build_postgres_connection_params

params = build_postgres_connection_params(AppSettings())
params["password"] = "***"
print(json.dumps(params))
"""


@dataclass(frozen=True)
class Target:
    container: str | None
    host: str
    port: int
    user: str


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def section(title: str) -> None:
    print(f"\n=== {title}")


def run(
    command: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None, stdin: str | None = None
) -> str:
    result = subprocess.run(command, env=env, cwd=cwd, input=stdin, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        fail(f"{' '.join(command[:4])} ... exited {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def postgres_tool(target: Target, tool: str, *arguments: str, interactive: bool = False) -> list[str]:
    connection = ["-h", target.host, "-p", str(target.port), "-U", target.user]
    if target.container:
        prefix = ["docker", "exec", "-i"] if interactive else ["docker", "exec"]
        return [*prefix, target.container, tool, "-U", target.user, *arguments]
    return [tool, *connection, *arguments]


def admin_url(target: Target) -> str:
    return f"postgresql://{target.user}@{target.host}:{target.port}/postgres"


def database_url(target: Target, name: str) -> str:
    return f"postgresql://{target.user}:{UNUSED_PASSWORD}@{target.host}:{target.port}/{name}"


def recreate_database(target: Target, name: str) -> None:
    if not name.startswith(SCRATCH_PREFIX):
        fail(f"refusing to touch database {name}")
    with psycopg.connect(admin_url(target), autocommit=True) as connection:
        connection.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def drop_database(target: Target, name: str) -> None:
    if not name.startswith(SCRATCH_PREFIX):
        fail(f"refusing to touch database {name}")
    with psycopg.connect(admin_url(target), autocommit=True) as connection:
        connection.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


def database_env(target: Target, name: str, **extra: str) -> dict[str, str]:
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG") if key in os.environ}
    env.update(
        {
            "POSTGRES_HOST": f"{target.host}:{target.port}",
            "POSTGRES_DB": name,
            "APP_POSTGRES_DB": name,
            "POSTGRES_USER": target.user,
            "POSTGRES_PASSWORD": UNUSED_PASSWORD,
        }
    )
    env.update(extra)
    return env


def dump_schema(target: Target, name: str, output: Path) -> str:
    raw = run(postgres_tool(target, "pg_dump", "-d", name, *DUMP_OPTIONS))
    text = "\n".join(line for line in raw.splitlines() if not RESTRICT_LINE.match(line)) + "\n"
    output.write_text(text, encoding="utf-8")
    return text


def unified_diff(before: str, after: str, before_name: str, after_name: str) -> list[str]:
    return list(
        difflib.unified_diff(
            before.splitlines(), after.splitlines(), fromfile=before_name, tofile=after_name, lineterm="", n=2
        )
    )


def catalog(target: Target, name: str) -> dict[str, dict[Any, Any]]:
    with psycopg.connect(database_url(target, name), row_factory=dict_row) as connection:
        tables = connection.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name <> 'schema_migrations'"
        ).fetchall()
        columns = connection.execute(
            "SELECT table_name, column_name, data_type, is_nullable, column_default, generation_expression "
            "FROM information_schema.columns WHERE table_schema = 'public' AND table_name <> 'schema_migrations'"
        ).fetchall()
        indexes = connection.execute(
            "SELECT tablename, indexname, indexdef FROM pg_indexes "
            "WHERE schemaname = 'public' AND tablename <> 'schema_migrations'"
        ).fetchall()
        constraints = connection.execute(
            "SELECT conrelid::regclass::text AS table_name, conname, pg_get_constraintdef(oid) AS definition "
            "FROM pg_constraint WHERE connamespace = 'public'::regnamespace "
            "AND conrelid::regclass::text <> 'schema_migrations'"
        ).fetchall()
    return {
        "table": {row["table_name"]: row["table_name"] for row in tables},
        "column": {
            (row["table_name"], row["column_name"]): (
                row["data_type"],
                row["is_nullable"],
                row["column_default"],
                row["generation_expression"],
            )
            for row in columns
        },
        "index": {(row["tablename"], row["indexname"]): row["indexdef"] for row in indexes},
        "constraint": {(row["table_name"], row["conname"]): row["definition"] for row in constraints},
    }


def catalog_differences(
    before: dict[str, dict[Any, Any]], after: dict[str, dict[Any, Any]]
) -> list[tuple[str, Any, Any, Any]]:
    differences = []
    for kind in before:
        for key in sorted(before[kind].keys() | after[kind].keys(), key=str):
            old, new = before[kind].get(key), after[kind].get(key)
            if old != new:
                differences.append((kind, key, old, new))
    return differences


def table_of(kind: str, key: Any) -> str:
    return str(key) if kind == "table" else str(key[0])


def is_f16(kind: str, key: Any, new: Any) -> bool:
    return table_of(kind, key) in DROPPED_TABLES and new is None


def is_f17(kind: str, key: Any, old: Any, new: Any) -> bool:
    if kind == "column" and key == CONVERTED_COLUMN:
        return bool(old and new and (old[0], new[0]) == ("text", "date"))
    return kind == "constraint" and key == ADDED_CONSTRAINT and old is None


def classify(differences: list[tuple[str, Any, Any, Any]]) -> dict[str, list[tuple[str, Any, Any, Any]]]:
    groups: dict[str, list[tuple[str, Any, Any, Any]]] = {"F16": [], "F17": [], "unexpected": []}
    for kind, key, old, new in differences:
        if is_f16(kind, key, new):
            groups["F16"].append((kind, key, old, new))
        elif is_f17(kind, key, old, new):
            groups["F17"].append((kind, key, old, new))
        else:
            groups["unexpected"].append((kind, key, old, new))
    return groups


def report_f16_f17(label: str, before: dict[str, dict[Any, Any]], after: dict[str, dict[Any, Any]]) -> bool:
    groups = classify(catalog_differences(before, after))
    section(f"{label}: catalog differences")
    for group, differences in groups.items():
        for kind, key, old, new in differences:
            print(f"{group} {kind} {key}: {old!r} -> {new!r}")
    dropped = {key for kind, key, _, _ in groups["F16"] if kind == "table"}
    converted = {(kind, key) for kind, key, _, _ in groups["F17"]}
    ok = (
        not groups["unexpected"]
        and dropped == DROPPED_TABLES
        and converted == {("column", CONVERTED_COLUMN), ("constraint", ADDED_CONSTRAINT)}
    )
    print(f"{label}: {'only F16/F17 differences' if ok else 'UNEXPECTED differences'}")
    return ok


def legacy_python(legacy_root: Path, code: str, env: dict[str, str], cwd: Path) -> str:
    return run([str(legacy_root / ".venv" / "bin" / "python"), "-c", code, str(legacy_root)], env=env, cwd=cwd)


def arxplore(arguments: list[str], env: dict[str, str], cwd: Path) -> str:
    executable = Path(sys.executable).with_name("arxplore")
    return run([str(executable), *arguments], env=env, cwd=cwd)


def fetch_jobs(target: Target, name: str) -> list[tuple[Any, ...]]:
    with psycopg.connect(database_url(target, name), row_factory=dict_row) as connection:
        rows = connection.execute(
            "SELECT id, mode, target_date::text AS target_date, status, payload::text AS payload, "
            "pg_typeof(target_date)::text AS type FROM prepare_jobs ORDER BY id"
        ).fetchall()
    return [tuple(row.values()) for row in rows]


def compare_fresh(target: Target, legacy_root: Path, workdir: Path, output: Path) -> bool:
    section("fresh databases")
    recreate_database(target, LEGACY_DB)
    recreate_database(target, NEW_DB)
    legacy_env = database_env(target, LEGACY_DB)
    print("legacy settings resolve to", legacy_python(legacy_root, LEGACY_TARGET_PROBE, legacy_env, workdir).strip())
    print(
        run(
            [str(legacy_root / ".venv" / "bin" / "python"), str(legacy_root / "scripts" / "migrate_schema.py")],
            env=legacy_env,
            cwd=workdir,
        ).strip()
    )
    new_env = database_env(target, NEW_DB)
    print(arxplore(["migrate", "--target", "1"], new_env, workdir).strip())

    legacy_dump = dump_schema(target, LEGACY_DB, output / "legacy_fresh.sql")
    baseline_dump = dump_schema(target, NEW_DB, output / "new_fresh_0001.sql")
    baseline_diff = unified_diff(legacy_dump, baseline_dump, "legacy ensure_schema", "new migrate --target 1")
    section("fresh: legacy ensure_schema vs new 0001 (pg_dump)")
    print("\n".join(baseline_diff) if baseline_diff else "IDENTICAL")
    legacy_catalog = catalog(target, LEGACY_DB)
    baseline_same = not baseline_diff and catalog_differences(legacy_catalog, catalog(target, NEW_DB)) == []

    print(arxplore(["migrate"], new_env, workdir).strip())
    full_dump = dump_schema(target, NEW_DB, output / "new_fresh_all.sql")
    section("fresh: legacy ensure_schema vs new 0001-0003 (pg_dump)")
    print("\n".join(unified_diff(legacy_dump, full_dump, "legacy ensure_schema", "new migrate")))
    full_ok = report_f16_f17("fresh", legacy_catalog, catalog(target, NEW_DB))
    print(arxplore(["db-status"], new_env, workdir).strip())
    return baseline_same and full_ok


def compare_existing(target: Target, source_db: str, legacy_root: Path, workdir: Path, output: Path) -> bool:
    section(f"existing database restored from pg_dump --schema-only {source_db}")
    source_dump = run(postgres_tool(target, "pg_dump", "-d", source_db, *DUMP_OPTIONS))
    (output / f"{source_db}_schema.sql").write_text(source_dump, encoding="utf-8")
    recreate_database(target, EXISTING_DB)
    run(
        postgres_tool(target, "psql", "-d", EXISTING_DB, "-v", "ON_ERROR_STOP=1", "-q", interactive=True),
        stdin=source_dump,
    )

    env = database_env(target, EXISTING_DB)
    before_dump = dump_schema(target, EXISTING_DB, output / "existing_before.sql")
    before_catalog = catalog(target, EXISTING_DB)
    print(arxplore(["migrate", "--target", "1"], env, workdir).strip())
    after_dump = dump_schema(target, EXISTING_DB, output / "existing_after_0001.sql")
    no_op_diff = unified_diff(before_dump, after_dump, "restored", "after 0001")
    section("existing: before vs after 0001 (pg_dump)")
    print("\n".join(no_op_diff) if no_op_diff else "IDENTICAL")
    no_op = not no_op_diff and catalog_differences(before_catalog, catalog(target, EXISTING_DB)) == []

    print(arxplore(["migrate", "--target", "2"], env, workdir).strip())
    legacy_env = database_env(target, EXISTING_DB, PREPARE_JOB_MAX_ATTEMPTS="1")
    print("legacy seed", legacy_python(legacy_root, LEGACY_SEED, legacy_env, workdir).strip())
    jobs_before = fetch_jobs(target, EXISTING_DB)
    print(arxplore(["migrate"], env, workdir).strip())
    jobs_after = fetch_jobs(target, EXISTING_DB)

    section("existing: prepare_jobs rows written by legacy code, before and after 0003")
    for row in jobs_before:
        print("before", row)
    for row in jobs_after:
        print("after ", row)
    statuses = {row[3] for row in jobs_before}
    converted = [row[:5] for row in jobs_before] == [row[:5] for row in jobs_after]
    types_ok = {row[5] for row in jobs_before} == {"text"} and {row[5] for row in jobs_after} == {"date"}
    print(f"statuses covered: {sorted(statuses)}; values preserved: {converted}; types text->date: {types_ok}")
    print("legacy code after 0003", legacy_python(legacy_root, LEGACY_AFTER_0003, legacy_env, workdir).strip())

    final_dump = dump_schema(target, EXISTING_DB, output / "existing_after_all.sql")
    section("existing: restored vs after 0001-0003 (pg_dump)")
    print("\n".join(unified_diff(before_dump, final_dump, "restored", "after migrate")))
    changes_ok = report_f16_f17("existing", before_catalog, catalog(target, EXISTING_DB))
    print(arxplore(["db-status"], env, workdir).strip())
    all_statuses = statuses == {"pending", "processing", "done", "failed"}
    return no_op and converted and types_ok and all_statuses and changes_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare the legacy ensure_schema DDL with arxplore migrate.")
    parser.add_argument("--container", default="arxplore-golden-pg", help="Run pg_dump and psql in this container.")
    parser.add_argument("--no-container", action="store_true", help="Use local pg_dump and psql instead.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=55432)
    parser.add_argument("--user", default="arxplore")
    parser.add_argument("--source-db", default="arxplore", help="Database whose schema dump is restored.")
    parser.add_argument("--legacy-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--keep", action="store_true", help="Keep the v2_* comparison databases.")
    options = parser.parse_args()

    target = Target(
        container=None if options.no_container else options.container,
        host=options.host,
        port=options.port,
        user=options.user,
    )
    output = options.output_dir or Path(tempfile.mkdtemp(prefix="arxplore-schema-"))
    output.mkdir(parents=True, exist_ok=True)
    print(f"dumps are written to {output}")
    try:
        with tempfile.TemporaryDirectory(prefix="arxplore-schema-cwd-") as workdir:
            fresh_ok = compare_fresh(target, options.legacy_root, Path(workdir), output)
            existing_ok = compare_existing(target, options.source_db, options.legacy_root, Path(workdir), output)
    finally:
        if not options.keep:
            for name in (LEGACY_DB, NEW_DB, EXISTING_DB):
                drop_database(target, name)
    section("summary")
    print(f"fresh baseline equals legacy and later changes are F16/F17 only: {fresh_ok}")
    print(f"existing schema: 0001 is a no-op, 0002/0003 apply, target_date converts: {existing_ok}")
    return 0 if fresh_ok and existing_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
