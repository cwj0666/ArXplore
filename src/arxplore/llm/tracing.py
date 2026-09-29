import getpass
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Self

from langchain_core.runnables import RunnableConfig
from langsmith import Client
from langsmith.run_helpers import tracing_context

from arxplore.config import AppSettings

type TraceStage = Literal["summary", "paper_overview", "paper_key_findings", "paper_chat", "agent_chat"]

TRACE_STAGES: frozenset[str] = frozenset(
    {"summary", "paper_overview", "paper_key_findings", "paper_chat", "agent_chat"}
)
VALID_TRACE_RUNTIMES = frozenset({"dev", "production", "airflow", "local", "test"})
TRACE_RUNTIME_ALIASES = {"development": "dev", "prod": "production"}
DEFAULT_TRACE_RUNTIME = "dev"
STAGE_BASE_TAGS = ("llm", "analysis")
UNKNOWN_TRACE_USER = "unknown"


def resolve_trace_runtime(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    normalized = TRACE_RUNTIME_ALIASES.get(normalized, normalized)
    return normalized if normalized in VALID_TRACE_RUNTIMES else DEFAULT_TRACE_RUNTIME


def system_user() -> str:
    try:
        return getpass.getuser()
    except KeyError, OSError:
        return UNKNOWN_TRACE_USER


@dataclass(frozen=True, slots=True)
class Tracer:
    enabled: bool
    project: str
    user: str
    runtime: str
    client: Client | None = None

    @classmethod
    def disabled(cls, *, project: str = "ArXplore", user: str = UNKNOWN_TRACE_USER, runtime: str = "test") -> Self:
        return cls(enabled=False, project=project, user=user, runtime=resolve_trace_runtime(runtime))

    @classmethod
    def from_settings(cls, settings: AppSettings) -> Self:
        enabled = bool(settings.langsmith_tracing and settings.langsmith_api_key)
        client = (
            Client(api_key=settings.langsmith_api_key, workspace_id=settings.langsmith_workspace_id)
            if enabled
            else None
        )
        return cls(
            enabled=enabled,
            project=settings.langsmith_project,
            user=settings.langsmith_trace_user or system_user(),
            runtime=resolve_trace_runtime(settings.runtime),
            client=client,
        )

    def config(
        self,
        stage: TraceStage,
        *,
        runtime: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RunnableConfig:
        label = runtime or self.runtime
        tags = [
            f"project:{self.project}",
            f"stage:{stage}",
            f"runtime:{label}",
            f"user:{self.user}",
            *STAGE_BASE_TAGS,
            stage,
        ]
        trace_metadata: dict[str, Any] = {
            "project": self.project,
            "stage": stage,
            "runtime": label,
            "user": self.user,
            "tracing_enabled": self.enabled,
            **(metadata or {}),
        }
        return {"run_name": stage, "tags": tags, "metadata": trace_metadata}

    @contextmanager
    def scope(self) -> Iterator[None]:
        if not self.enabled:
            with tracing_context(enabled=False):
                yield
            return
        with tracing_context(enabled=True, client=self.client, project_name=self.project):
            yield
