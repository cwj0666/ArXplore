import pytest
from langsmith import utils as langsmith_utils

from arxplore.config import AppSettings
from arxplore.llm.tracing import Tracer, resolve_trace_runtime


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("development", "dev"),
        ("production", "production"),
        ("PROD", "production"),
        (" test ", "test"),
        ("airflow", "airflow"),
        ("staging", "dev"),
        (None, "dev"),
    ],
)
def test_trace_runtime_resolution(raw: str | None, expected: str) -> None:
    assert resolve_trace_runtime(raw) == expected


def test_config_keeps_the_stage_and_tag_rules() -> None:
    tracer = Tracer(enabled=False, project="ArXplore", user="tester", runtime="production")

    config = tracer.config("paper_chat", metadata={"retrieval_mode": "hybrid"})

    assert config["run_name"] == "paper_chat"
    assert config["tags"] == [
        "project:ArXplore",
        "stage:paper_chat",
        "runtime:production",
        "user:tester",
        "llm",
        "analysis",
        "paper_chat",
    ]
    assert config["metadata"] == {
        "project": "ArXplore",
        "stage": "paper_chat",
        "runtime": "production",
        "user": "tester",
        "tracing_enabled": False,
        "retrieval_mode": "hybrid",
    }


def test_explicit_runtime_label_wins_for_one_config() -> None:
    tracer = Tracer(enabled=False, project="ArXplore", user="tester", runtime="production")

    config = tracer.config("summary", runtime="dev")

    assert "runtime:dev" in config["tags"]
    assert config["metadata"]["runtime"] == "dev"


def test_tracing_needs_both_the_flag_and_a_key() -> None:
    without_key = Tracer.from_settings(AppSettings.model_validate({"LANGSMITH_TRACING": "true"}))
    switched_off = Tracer.from_settings(
        AppSettings.model_validate({"LANGSMITH_TRACING": "false", "LANGSMITH_API_KEY": "ls-key"})
    )
    enabled = Tracer.from_settings(
        AppSettings.model_validate(
            {"LANGSMITH_API_KEY": "ls-key", "LANGSMITH_PROJECT": "P", "LANGSMITH_TRACE_USER": "u", "RUNTIME": "prod"}
        )
    )

    assert (without_key.enabled, without_key.client) == (False, None)
    assert (switched_off.enabled, switched_off.client) == (False, None)
    assert enabled.enabled is True
    assert enabled.client is not None
    assert (enabled.project, enabled.user, enabled.runtime) == ("P", "u", "production")


def test_disabled_scope_turns_off_environment_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-env-key")
    tracer = Tracer.disabled()

    with tracer.scope():
        assert not langsmith_utils.tracing_is_enabled()
