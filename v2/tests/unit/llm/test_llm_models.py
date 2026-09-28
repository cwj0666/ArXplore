from typing import Any

import pytest

from arxplore.config import AppSettings
from arxplore.context import override_openai_runtime
from arxplore.llm.models import ChatModelFactory, OpenAIKeyMissingError, accepts_temperature


def _factory(**overrides: Any) -> ChatModelFactory:
    values: dict[str, Any] = {
        "server_api_key": "sk-server",
        "default_model": "gpt-4o",
        "base_url": "http://llm.test/v1",
    }
    values.update(overrides)
    return ChatModelFactory(**values)


@pytest.mark.parametrize(
    ("model", "expected"),
    [("gpt-4o", True), ("gpt-4o-mini", True), ("gpt-5", False), ("gpt-5-mini", False), ("gpt-5.1", False)],
)
def test_gpt5_family_takes_no_temperature(model: str, expected: bool) -> None:
    assert accepts_temperature(model) is expected


def test_created_model_uses_explicit_endpoint_key_and_usage_streaming() -> None:
    chat = _factory().create(temperature=0.2)

    assert chat.model_name == "gpt-4o"
    assert chat.temperature == 0.2
    assert chat.stream_usage is True
    assert chat.openai_api_base == "http://llm.test/v1"
    assert chat.openai_api_key is not None
    assert chat.openai_api_key.get_secret_value() == "sk-server"


def test_gpt5_models_are_created_without_temperature() -> None:
    chat = _factory().create(model="gpt-5-mini", temperature=0.2)

    assert chat.model_name == "gpt-5-mini"
    assert chat.temperature is None


def test_request_scope_overrides_key_and_default_model_but_not_an_explicit_model() -> None:
    factory = _factory()

    with override_openai_runtime(api_key="sk-request", model="gpt-4o-mini"):
        implicit = factory.create()
        explicit = factory.create(model="gpt-5")

    assert implicit.model_name == "gpt-4o-mini"
    assert explicit.model_name == "gpt-5"
    for chat in (implicit, explicit):
        assert chat.openai_api_key is not None
        assert chat.openai_api_key.get_secret_value() == "sk-request"


def test_blank_request_key_falls_back_to_the_server_key() -> None:
    with override_openai_runtime(api_key="   "):
        assert _factory().resolve_api_key() == "sk-server"


def test_missing_key_is_reported_before_building_a_client() -> None:
    factory = _factory(server_api_key="  ")

    assert factory.is_available() is False
    with pytest.raises(OpenAIKeyMissingError):
        factory.create()
    with override_openai_runtime(api_key="sk-request"):
        assert factory.is_available() is True


def test_from_settings_reads_key_model_and_base_url() -> None:
    settings = AppSettings.model_validate(
        {"OPENAI_API_KEY": "sk-settings", "OPENAI_MODEL": "gpt-4o-mini", "OPENAI_BASE_URL": "http://settings.test/v1"}
    )

    assert ChatModelFactory.from_settings(settings) == ChatModelFactory(
        server_api_key="sk-settings", default_model="gpt-4o-mini", base_url="http://settings.test/v1"
    )
    assert ChatModelFactory.from_settings(settings, base_url="http://explicit.test/v1").base_url == (
        "http://explicit.test/v1"
    )
