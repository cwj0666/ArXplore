import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.golden.adapters import AdapterUnavailable, implementation_name, load_parser_adapter
from tests.golden.store import GOLDEN_ROOT

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if item.path.is_relative_to(GOLDEN_ROOT):
            item.add_marker(pytest.mark.golden)


def _reject_non_loopback(address: Any) -> None:
    if isinstance(address, tuple) and address and address[0] not in LOOPBACK_HOSTS:
        raise RuntimeError(f"golden tests may only open loopback connections, not {address!r}")


@pytest.fixture(autouse=True)
def loopback_only_network(monkeypatch: pytest.MonkeyPatch) -> None:
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def connect(sock: socket.socket, address: Any) -> None:
        _reject_non_loopback(address)
        return original_connect(sock, address)

    def connect_ex(sock: socket.socket, address: Any) -> int:
        _reject_non_loopback(address)
        return original_connect_ex(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


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
