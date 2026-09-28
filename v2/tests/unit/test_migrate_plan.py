import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from arxplore.db.migrate import (
    AppliedMigration,
    ChecksumMismatchError,
    Migration,
    MigrationError,
    MigrationOrderError,
    MigrationState,
    UnknownMigrationError,
    compute_checksum,
    load_migrations,
    migration_status,
    plan_migrations,
    readiness_problems,
)

APPLIED_AT = datetime(2026, 9, 29, tzinfo=UTC)


def _migration(version: int, sql: str = "SELECT 1;") -> Migration:
    return Migration(version=version, name=f"{version:04d}_step", sql=sql, checksum=compute_checksum(sql))


def _applied(migration: Migration, checksum: str | None = None) -> AppliedMigration:
    return AppliedMigration(
        version=migration.version,
        checksum=checksum or migration.checksum,
        applied_at=APPLIED_AT,
    )


class TestPackagedMigrations:
    def test_versions_and_names(self) -> None:
        migrations = load_migrations()

        assert [(migration.version, migration.name) for migration in migrations] == [
            (1, "0001_baseline"),
            (2, "0002_drop_legacy_topics"),
            (3, "0003_prepare_jobs_date_and_status"),
        ]

    def test_checksums_are_sha256_of_file_contents(self) -> None:
        for migration in load_migrations():
            assert migration.checksum == compute_checksum(migration.sql)
            assert re.fullmatch(r"[0-9a-f]{64}", migration.checksum)

    def test_files_do_not_manage_transactions(self) -> None:
        for migration in load_migrations():
            assert not re.search(r"^\s*(BEGIN|COMMIT|ROLLBACK)\s*;", migration.sql, re.IGNORECASE | re.MULTILINE)

    def test_baseline_keeps_legacy_embedding_width(self) -> None:
        baseline = load_migrations()[0]

        assert "embedding VECTOR(1536) NOT NULL" in baseline.sql


class TestLoadMigrations:
    def test_reads_sorted_sql_files_and_ignores_other_files(self, tmp_path: Path) -> None:
        (tmp_path / "0002_second.sql").write_text("SELECT 2;", encoding="utf-8")
        (tmp_path / "0001_first.sql").write_text("SELECT 1;", encoding="utf-8")
        (tmp_path / "README.txt").write_text("notes", encoding="utf-8")
        (tmp_path / "__pycache__").mkdir()

        assert [migration.name for migration in load_migrations(tmp_path)] == ["0001_first", "0002_second"]

    @pytest.mark.parametrize("name", ["1_short.sql", "0001-dash.sql", "0001_Upper.sql", "baseline.sql"])
    def test_rejects_badly_named_sql_files(self, tmp_path: Path, name: str) -> None:
        (tmp_path / name).write_text("SELECT 1;", encoding="utf-8")

        with pytest.raises(MigrationError, match="file name"):
            load_migrations(tmp_path)

    def test_rejects_version_zero(self, tmp_path: Path) -> None:
        (tmp_path / "0000_zero.sql").write_text("SELECT 1;", encoding="utf-8")

        with pytest.raises(MigrationError, match="start at 0001"):
            load_migrations(tmp_path)

    def test_rejects_duplicate_versions(self, tmp_path: Path) -> None:
        (tmp_path / "0001_first.sql").write_text("SELECT 1;", encoding="utf-8")
        (tmp_path / "0001_again.sql").write_text("SELECT 1;", encoding="utf-8")

        with pytest.raises(MigrationError, match="duplicate"):
            load_migrations(tmp_path)


class TestChecksum:
    def test_line_endings_do_not_change_the_checksum(self) -> None:
        assert compute_checksum("SELECT 1;\r\nSELECT 2;\r\n") == compute_checksum("SELECT 1;\nSELECT 2;\n")

    def test_content_changes_the_checksum(self) -> None:
        assert compute_checksum("SELECT 1;") != compute_checksum("SELECT 2;")


class TestStatus:
    def test_classifies_every_version(self) -> None:
        first, second, third = _migration(1), _migration(2), _migration(3)
        applied = [
            _applied(first),
            _applied(second, checksum="0" * 64),
            AppliedMigration(version=9, checksum="f" * 64, applied_at=APPLIED_AT),
        ]

        statuses = migration_status([first, second, third], applied)

        assert [(status.version, status.state) for status in statuses] == [
            (1, MigrationState.APPLIED),
            (2, MigrationState.CHANGED),
            (3, MigrationState.PENDING),
            (9, MigrationState.UNKNOWN),
        ]
        assert [status.label for status in statuses] == ["0001_step", "0002_step", "0003_step", "0009"]
        assert statuses[2].applied_at is None


class TestPlan:
    def test_everything_is_pending_on_an_empty_database(self) -> None:
        migrations = [_migration(1), _migration(2)]

        assert plan_migrations(migrations, []) == migrations

    def test_skips_applied_versions(self) -> None:
        first, second = _migration(1), _migration(2)

        assert plan_migrations([first, second], [_applied(first)]) == [second]

    def test_target_limits_the_plan(self) -> None:
        migrations = [_migration(1), _migration(2), _migration(3)]

        assert [migration.version for migration in plan_migrations(migrations, [], target=2)] == [1, 2]

    def test_target_below_applied_versions_plans_nothing(self) -> None:
        first, second = _migration(1), _migration(2)

        assert plan_migrations([first, second], [_applied(first), _applied(second)], target=1) == []

    def test_unknown_target_is_rejected(self) -> None:
        with pytest.raises(MigrationError, match="0004"):
            plan_migrations([_migration(1)], [], target=4)

    def test_modified_applied_migration_is_rejected(self) -> None:
        first = _migration(1)

        with pytest.raises(ChecksumMismatchError, match="0001_step"):
            plan_migrations([first], [_applied(first, checksum="0" * 64)])

    def test_unknown_applied_version_is_rejected(self) -> None:
        first = _migration(1)
        future = AppliedMigration(version=2, checksum="f" * 64, applied_at=APPLIED_AT)

        with pytest.raises(UnknownMigrationError, match="0002"):
            plan_migrations([first], [_applied(first), future])

    def test_pending_migration_older_than_applied_one_is_rejected(self) -> None:
        first, second, third = _migration(1), _migration(2), _migration(3)

        with pytest.raises(MigrationOrderError, match="0002_step"):
            plan_migrations([first, second, third], [_applied(first), _applied(third)])


class TestReadiness:
    def test_ready_when_everything_is_applied_and_dimensions_match(self) -> None:
        first = _migration(1)
        statuses = migration_status([first], [_applied(first)])

        assert readiness_problems(statuses, embedding_dimensions=1536, expected_embedding_dimensions=1536) == []

    def test_reports_pending_changed_unknown_and_dimension_problems(self) -> None:
        first, second = _migration(1), _migration(2)
        statuses = migration_status(
            [first, second],
            [_applied(first, checksum="0" * 64), AppliedMigration(version=5, checksum="f" * 64, applied_at=APPLIED_AT)],
        )

        problems = readiness_problems(statuses, embedding_dimensions=3072, expected_embedding_dimensions=1536)

        assert problems == [
            "pending migrations: 0002_step",
            "modified migrations: 0001_step",
            "unknown migrations in database: 0005",
            "paper_embeddings.embedding is vector(3072) but OPENAI_EMBEDDING_DIMENSIONS is 1536",
        ]

    def test_missing_embedding_column_is_a_problem(self) -> None:
        assert readiness_problems([], embedding_dimensions=None, expected_embedding_dimensions=1536) == [
            "paper_embeddings.embedding is missing or is not a vector(n) column"
        ]
