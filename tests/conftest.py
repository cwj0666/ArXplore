import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import AliasChoices

from arxplore.config import ENV_FILE_VARIABLE, AppSettings, get_settings

REMOVED_VARIABLES = ("DJANGO_SECRET_KEY", "APP_RUNTIME_MODE", "AIRFLOW_BASE_URL")


def settings_variable_names() -> set[str]:
    names = {ENV_FILE_VARIABLE, *REMOVED_VARIABLES}
    for field in AppSettings.model_fields.values():
        alias = field.validation_alias
        if isinstance(alias, str):
            names.add(alias)
        elif isinstance(alias, AliasChoices):
            names.update(choice for choice in alias.choices if isinstance(choice, str))
    return names


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    wanted = {name.lower() for name in settings_variable_names()}
    for name in list(os.environ):
        if name.lower() in wanted:
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()
