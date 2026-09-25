from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

RETRIEVAL_MODES = frozenset({"lexical", "hybrid"})


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_runtime_mode: str = Field(default="development", alias="APP_RUNTIME_MODE")
    airflow_base_url: str | None = Field(default=None, alias="AIRFLOW_BASE_URL")

    hf_daily_papers_base_url: str = Field(default="https://huggingface.co/papers", alias="HF_DAILY_PAPERS_BASE_URL")
    hf_request_timeout_seconds: int = Field(default=20, alias="HF_REQUEST_TIMEOUT_SECONDS")

    arxiv_api_base_url: str = Field(default="https://export.arxiv.org/api/query", alias="ARXIV_API_BASE_URL")
    arxiv_request_timeout_seconds: int = Field(default=45, alias="ARXIV_REQUEST_TIMEOUT_SECONDS")
    arxiv_request_batch_size: int = Field(default=25, alias="ARXIV_REQUEST_BATCH_SIZE")
    arxiv_request_delay_seconds: float = Field(default=3.0, alias="ARXIV_REQUEST_DELAY_SECONDS")
    layout_parser_base_url: str | None = Field(default=None, alias="LAYOUT_PARSER_BASE_URL")
    layout_parser_timeout_seconds: int = Field(default=600, alias="LAYOUT_PARSER_TIMEOUT_SECONDS")
    layout_parser_fast: bool = Field(default=False, alias="LAYOUT_PARSER_FAST")
    layout_parser_parse_tables_and_math: bool = Field(
        default=True,
        alias="LAYOUT_PARSER_PARSE_TABLES_AND_MATH",
    )

    mongo_host: str | None = Field(default=None, alias="MONGO_HOST")
    server_mongo_port: int = Field(default=27017, alias="SERVER_MONGO_PORT")
    mongo_db: str = Field(default="arxplore_source", alias="MONGO_DB")
    mongo_initdb_root_username: str | None = Field(default=None, alias="MONGO_INITDB_ROOT_USERNAME")
    mongo_initdb_root_password: str | None = Field(default=None, alias="MONGO_INITDB_ROOT_PASSWORD")
    mongo_daily_papers_collection: str = Field(default="daily_papers_raw", alias="MONGO_DAILY_PAPERS_COLLECTION")
    mongo_pipeline_state_collection: str = Field(default="pipeline_state", alias="MONGO_PIPELINE_STATE_COLLECTION")

    postgres_host: str | None = Field(default=None, alias="POSTGRES_HOST")
    server_postgres_port: int = Field(default=5432, alias="SERVER_POSTGRES_PORT")
    postgres_db: str | None = Field(default=None, alias="POSTGRES_DB")
    app_postgres_db: str | None = Field(default=None, alias="APP_POSTGRES_DB")
    postgres_user: str | None = Field(default=None, alias="POSTGRES_USER")
    postgres_password: str | None = Field(default=None, alias="POSTGRES_PASSWORD")
    prepare_job_stale_seconds: int = Field(default=900, alias="PREPARE_JOB_STALE_SECONDS")
    prepare_job_max_attempts: int = Field(default=3, alias="PREPARE_JOB_MAX_ATTEMPTS")

    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="ArXplore", alias="LANGSMITH_PROJECT")
    langsmith_workspace_id: str | None = Field(default=None, alias="LANGSMITH_WORKSPACE_ID")
    langsmith_tracing: bool = Field(default=True, alias="LANGSMITH_TRACING")
    langsmith_trace_user: str | None = Field(default=None, alias="LANGSMITH_TRACE_USER")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-large", alias="OPENAI_EMBEDDING_MODEL")
    openai_embedding_dimensions: int = Field(default=1536, alias="OPENAI_EMBEDDING_DIMENSIONS")
    embedding_batch_size: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE")

    retrieval_mode: str = Field(default="hybrid", alias="RETRIEVAL_MODE")
    agent_recursion_limit: int = Field(default=12, ge=4, alias="AGENT_RECURSION_LIMIT")
    vector_min_similarity: float = Field(default=0.0, ge=0.0, le=1.0, alias="VECTOR_MIN_SIMILARITY")
    postgres_pool_max: int = Field(default=8, ge=1, alias="POSTGRES_POOL_MAX")
    postgres_pool_timeout: float = Field(default=30.0, gt=0, alias="POSTGRES_POOL_TIMEOUT")

    @field_validator("retrieval_mode", mode="before")
    @classmethod
    def _normalize_retrieval_mode(cls, value: Any) -> str:
        normalized = str(value or "hybrid").strip().lower()
        if normalized not in RETRIEVAL_MODES:
            raise ValueError(f"RETRIEVAL_MODE는 {sorted(RETRIEVAL_MODES)} 중 하나여야 합니다. 받은 값: {value}")
        return normalized

    @model_validator(mode="after")
    def _restore_empty_sensitive_values_from_env_file(self) -> "AppSettings":
        env_path = Path(".env")
        if not env_path.exists():
            return self

        env_values = dotenv_values(env_path)
        fallback_fields = {
            "openai_api_key": "OPENAI_API_KEY",
            "langsmith_api_key": "LANGSMITH_API_KEY",
        }

        for field_name, env_name in fallback_fields.items():
            current_value = getattr(self, field_name)
            if current_value is not None and str(current_value).strip():
                continue

            env_file_value = env_values.get(env_name)
            if env_file_value is None:
                continue

            normalized_value = str(env_file_value).strip()
            if normalized_value:
                setattr(self, field_name, normalized_value)

        return self


_UNSET = object()
_runtime_openai_api_key: ContextVar[str | None | object] = ContextVar("runtime_openai_api_key", default=_UNSET)
_runtime_openai_model: ContextVar[str | None | object] = ContextVar("runtime_openai_model", default=_UNSET)


def resolve_host_and_port(host: str, default_port: int) -> tuple[str, int]:
    normalized = host.strip()
    if not normalized:
        raise ValueError("호스트 값이 비어 있습니다.")

    parsed = urlsplit(f"//{normalized}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"호스트 값을 해석할 수 없습니다: {host}")

    return hostname, parsed.port or default_port


def build_postgres_connection_params(settings: AppSettings) -> dict[str, Any]:
    host = settings.postgres_host
    db_name = settings.app_postgres_db or settings.postgres_db
    user = settings.postgres_user
    password = settings.postgres_password

    if not host:
        raise ValueError("POSTGRES_HOST가 설정되지 않았습니다.")
    if not db_name:
        raise ValueError("APP_POSTGRES_DB 또는 POSTGRES_DB가 설정되지 않았습니다.")
    if not user or not password:
        raise ValueError("POSTGRES_USER 또는 POSTGRES_PASSWORD가 설정되지 않았습니다.")

    resolved_host, resolved_port = resolve_host_and_port(host, settings.server_postgres_port)
    return {
        "dbname": db_name,
        "user": user,
        "password": password,
        "host": resolved_host,
        "port": resolved_port,
    }


def build_django_postgres_database_config(settings: AppSettings) -> dict[str, Any]:
    params = build_postgres_connection_params(settings)
    return {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": params["dbname"],
        "USER": params["user"],
        "PASSWORD": params["password"],
        "HOST": params["host"],
        "PORT": params["port"],
        "CONN_MAX_AGE": 60,
        "CONN_HEALTH_CHECKS": True,
    }


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def get_runtime_openai_api_key() -> str | None:
    override = _runtime_openai_api_key.get()
    if override is not _UNSET:
        return override
    return get_settings().openai_api_key


def get_runtime_openai_api_key_override() -> str | None:
    """요청 범위에서 주입된 키만 돌려준다. 서버 키로 폴백하지 않는다."""
    override = _runtime_openai_api_key.get()
    if override is _UNSET or not override:
        return None
    normalized = str(override).strip()
    return normalized or None


def get_runtime_openai_model(default_model: str | None = None) -> str:
    override = _runtime_openai_model.get()
    if override is not _UNSET and override:
        return str(override)
    if default_model:
        return default_model
    return get_settings().openai_model


@contextmanager
def override_openai_runtime(*, api_key: str | None = None, model: str | None = None):
    api_key_token = None
    model_token = None
    if api_key is not None:
        api_key_token = _runtime_openai_api_key.set(api_key)
    if model is not None:
        model_token = _runtime_openai_model.set(model)

    try:
        yield
    finally:
        if model_token is not None:
            _runtime_openai_model.reset(model_token)
        if api_key_token is not None:
            _runtime_openai_api_key.reset(api_key_token)
