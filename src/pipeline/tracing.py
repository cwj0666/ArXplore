from typing import Any

from src.shared import build_langsmith_trace_context


def build_pipeline_trace_config(
    stage: str,
    *,
    runtime: str = "airflow",
    user: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = build_langsmith_trace_context(
        stage=stage,
        runtime=runtime,
        user=user,
        extra_tags=["pipeline"],
        extra_metadata=extra_metadata,
    )
    return context.as_langchain_config()
