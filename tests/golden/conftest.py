import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.golden import environment
from tests.golden.adapters import AdapterUnavailable, implementation_name, load_parser_adapter
from tests.golden.environment import loopback_only
from tests.golden.store import GOLDEN_ROOT


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.is_relative_to(GOLDEN_ROOT):
            item.add_marker(pytest.mark.golden)


@pytest.fixture(autouse=True)
def loopback_only_network() -> Iterator[None]:
    with loopback_only():
        yield


@pytest.fixture(scope="session")
def parser_adapter() -> Any:
    try:
        return load_parser_adapter()
    except AdapterUnavailable as exc:
        pytest.skip(f"{implementation_name()}: {exc}")


@pytest.fixture
def require_input() -> Callable[[Path, str], Path]:
    def check(path: Path, hint: str) -> Path:
        if not path.is_file():
            pytest.skip(f"golden input missing: {path} ({hint})")
        return path

    return check


@pytest.fixture(scope="session")
def golden_database_url() -> str:
    url = environment.database_url()
    if not environment.database_reachable(url):
        pytest.skip(f"golden database unreachable: {url} (set {environment.DATABASE_URL_ENV})")
    return url


@pytest.fixture(scope="session")
def golden_pg_options() -> str:
    options = environment.READ_ONLY_OPTION
    if os.environ.get(environment.EXACT_SEARCH_ENV, "").strip():
        options = f"{options} {environment.EXACT_SEARCH_OPTION}"
    return options
