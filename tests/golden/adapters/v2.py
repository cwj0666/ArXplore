import importlib
from typing import Any

from tests.golden.adapters import AdapterUnavailable


def _factory(kind: str) -> Any:
    module_name = f"{__name__}_{kind}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise AdapterUnavailable(f"v2 does not provide the {kind} adapter yet") from exc
        raise
    return getattr(module, f"{kind}_adapter")


def parser_adapter(**options: Any) -> Any:
    return _factory("parser")(**options)


def retrieval_adapter(**options: Any) -> Any:
    return _factory("retrieval")(**options)


def http_adapter(**options: Any) -> Any:
    return _factory("http")(**options)
