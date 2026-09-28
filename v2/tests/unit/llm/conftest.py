import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from arxplore.llm.models import ChatModelFactory
from arxplore.llm.tracing import Tracer

FAKE_SERVER_PATH = Path(__file__).resolve().parents[4] / "tests" / "fakes" / "openai_server.py"
FAKE_SERVER_MODULE = "arxplore_tests_fake_openai_server"


def load_fake_server_module() -> ModuleType:
    if FAKE_SERVER_MODULE in sys.modules:
        return sys.modules[FAKE_SERVER_MODULE]
    if not FAKE_SERVER_PATH.is_file():
        pytest.skip(f"fake OpenAI server not found at {FAKE_SERVER_PATH}")
    spec = importlib.util.spec_from_file_location(FAKE_SERVER_MODULE, FAKE_SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[FAKE_SERVER_MODULE] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def fakes() -> ModuleType:
    return load_fake_server_module()


@pytest.fixture
def fake_openai(fakes: ModuleType) -> Iterator[Any]:
    with fakes.FakeOpenAIServer() as server:
        yield server


@pytest.fixture
def models(fake_openai: Any) -> ChatModelFactory:
    return ChatModelFactory(
        server_api_key=fake_openai.expected_api_key, default_model="gpt-4o", base_url=fake_openai.base_url
    )


@pytest.fixture
def tracer() -> Tracer:
    return Tracer.disabled()
