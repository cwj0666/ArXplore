import sys
import types
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from arxplore.cli import COMMAND_MODULES, app, register_command_modules


def _app_with_callback() -> typer.Typer:
    target = typer.Typer()

    @target.callback()
    def main() -> None:
        pass

    return target


def test_default_command_modules_cover_api_pipeline_and_scheduler() -> None:
    assert COMMAND_MODULES == (
        "arxplore.api.commands",
        "arxplore.pipeline.commands",
        "arxplore.scheduler.commands",
    )


def test_missing_modules_and_packages_are_skipped() -> None:
    target = _app_with_callback()

    registered = register_command_modules(
        target,
        ("arxplore.not_a_package.commands", "arxplore.db.not_a_module", "not_a_top_level_package.commands"),
    )

    assert registered == []


def test_present_module_registers_its_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("arxplore_cli_test_commands")

    def register(target: typer.Typer) -> None:
        @target.command("hello")
        def hello() -> None:
            typer.echo("hi")

    module.__dict__["register"] = register
    monkeypatch.setitem(sys.modules, "arxplore_cli_test_commands", module)
    target = _app_with_callback()

    assert register_command_modules(target, ("arxplore_cli_test_commands",)) == ["arxplore_cli_test_commands"]
    result = CliRunner().invoke(target, ["hello"])
    assert result.exit_code == 0
    assert result.output.strip() == "hi"


def test_missing_dependency_inside_a_command_module_is_raised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "arxplore_cli_broken_commands.py").write_text(
        "import arxplore_cli_missing_dependency\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError, match="arxplore_cli_missing_dependency"):
        register_command_modules(_app_with_callback(), ("arxplore_cli_broken_commands",))


def test_builtin_commands_stay_registered() -> None:
    names = {command.name for command in app.registered_commands}

    assert {"migrate", "db-status"} <= names
