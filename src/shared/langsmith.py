import getpass
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .settings import AppSettings, get_settings


@dataclass(frozen=True)
class LangSmithTraceContext:
    project: str
    enabled: bool
    tags: list[str]
    metadata: dict[str, Any]

    def as_langchain_config(self) -> dict[str, Any]:
        return {
            "run_name": self.metadata.get("stage", "llm-task"),
            "tags": self.tags,
            "metadata": self.metadata,
        }


def is_langsmith_enabled(settings: AppSettings | None = None) -> bool:
    active_settings = settings or get_settings()
    return bool(active_settings.langsmith_tracing and active_settings.langsmith_api_key)


def apply_langsmith_environment(settings: AppSettings | None = None) -> bool:
    active_settings = settings or get_settings()
    enabled = is_langsmith_enabled(active_settings)

    if active_settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = active_settings.langsmith_api_key

    os.environ["LANGSMITH_PROJECT"] = active_settings.langsmith_project
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "true" if enabled else "false"

    if active_settings.langsmith_workspace_id:
        os.environ["LANGSMITH_WORKSPACE_ID"] = active_settings.langsmith_workspace_id

    return enabled


def build_langsmith_trace_context(
    stage: str,
    runtime: str,
    *,
    user: str | None = None,
    extra_tags: Iterable[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    settings: AppSettings | None = None,
) -> LangSmithTraceContext:
    active_settings = settings or get_settings()
    trace_user = user or active_settings.langsmith_trace_user or getpass.getuser()
    enabled = apply_langsmith_environment(active_settings)

    tags = [
        f"project:{active_settings.langsmith_project}",
        f"stage:{stage}",
        f"runtime:{runtime}",
        f"user:{trace_user}",
    ]
    if extra_tags:
        tags.extend(extra_tags)

    metadata: dict[str, Any] = {
        "project": active_settings.langsmith_project,
        "stage": stage,
        "runtime": runtime,
        "user": trace_user,
        "tracing_enabled": enabled,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return LangSmithTraceContext(
        project=active_settings.langsmith_project,
        enabled=enabled,
        tags=tags,
        metadata=metadata,
    )
