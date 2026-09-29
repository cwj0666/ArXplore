import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AliasChoices, BeforeValidator, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

RetrievalMode = Literal["lexical", "hybrid"]

RETRIEVAL_MODES = frozenset({"lexical", "hybrid"})
ENV_FILE_VARIABLE = "ARXPLORE_ENV_FILE"
DEFAULT_ENV_FILE = Path(".env")
LEGACY_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _only_exact_true(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return value


def _legacy_flag(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower() in LEGACY_TRUE_VALUES
    return value


def _comma_separated(value: Any) -> Any:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


DebugFlag = Annotated[bool, BeforeValidator(_only_exact_true)]
LegacyFlag = Annotated[bool, BeforeValidator(_legacy_flag)]
CommaSeparated = Annotated[list[str], NoDecode, BeforeValidator(_comma_separated)]


def _default_allowed_hosts(data: dict[str, Any]) -> list[str]:
    if data.get("debug"):
        return ["*"]
    return ["localhost", "127.0.0.1"]


def _default_csrf_trusted_origins(data: dict[str, Any]) -> list[str]:
    port = data.get("frontend_port", "5173")
    return [
        "http://localhost",
        "http://127.0.0.1",
        f"http://localhost:{port}",
        f"http://127.0.0.1:{port}",
    ]


class ConfigError(ValueError):
    pass


class PostgresConnectionParams(TypedDict):
    dbname: str
    user: str
    password: str
    host: str
    port: int


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=DEFAULT_ENV_FILE,
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    runtime: str = Field(default="development", validation_alias="RUNTIME")

    debug: DebugFlag = Field(default=False, validation_alias=AliasChoices("DEBUG", "DJANGO_DEBUG"))
    frontend_port: str = Field(default="5173", validation_alias="FRONTEND_PORT")
    allowed_hosts: CommaSeparated = Field(
        default_factory=_default_allowed_hosts,
        validation_alias=AliasChoices("ALLOWED_HOSTS", "DJANGO_ALLOWED_HOSTS"),
    )
    csrf_trusted_origins: CommaSeparated = Field(
        default_factory=_default_csrf_trusted_origins,
        validation_alias=AliasChoices("CSRF_TRUSTED_ORIGINS", "DJANGO_CSRF_TRUSTED_ORIGINS"),
    )
    secure_cookies: LegacyFlag = Field(
        default=False,
        validation_alias=AliasChoices("SECURE_COOKIES", "DJANGO_SECURE_COOKIES"),
    )
    serve_frontend_dist: bool = Field(default=False, validation_alias="SERVE_FRONTEND_DIST")

    redis_url: str | None = Field(default=None, validation_alias="REDIS_URL")
    rate_limit_enabled: LegacyFlag = Field(default=True, validation_alias="RATE_LIMIT_ENABLED")
    rate_limit_llm_per_minute: int = Field(default=30, validation_alias="RATE_LIMIT_LLM_PER_MINUTE")
    rate_limit_detail_per_minute: int = Field(default=60, validation_alias="RATE_LIMIT_DETAIL_PER_MINUTE")
    rate_limit_ip_header: str = Field(default="X-Real-IP", validation_alias="RATE_LIMIT_IP_HEADER")

    hf_daily_papers_base_url: str = Field(
        default="https://huggingface.co/papers",
        validation_alias="HF_DAILY_PAPERS_BASE_URL",
    )
    hf_request_timeout_seconds: int = Field(default=20, validation_alias="HF_REQUEST_TIMEOUT_SECONDS")

    arxiv_api_base_url: str = Field(
        default="https://export.arxiv.org/api/query",
        validation_alias="ARXIV_API_BASE_URL",
    )
    arxiv_request_timeout_seconds: int = Field(default=45, validation_alias="ARXIV_REQUEST_TIMEOUT_SECONDS")
    arxiv_request_batch_size: int = Field(default=25, validation_alias="ARXIV_REQUEST_BATCH_SIZE")
    arxiv_request_delay_seconds: float = Field(default=3.0, validation_alias="ARXIV_REQUEST_DELAY_SECONDS")

    layout_parser_base_url: str | None = Field(default=None, validation_alias="LAYOUT_PARSER_BASE_URL")
    layout_parser_timeout_seconds: int = Field(default=600, validation_alias="LAYOUT_PARSER_TIMEOUT_SECONDS")
    layout_parser_fast: bool = Field(default=False, validation_alias="LAYOUT_PARSER_FAST")
    layout_parser_parse_tables_and_math: bool = Field(
        default=True,
        validation_alias="LAYOUT_PARSER_PARSE_TABLES_AND_MATH",
    )

    postgres_host: str | None = Field(default=None, validation_alias="POSTGRES_HOST")
    server_postgres_port: int = Field(default=5432, validation_alias="SERVER_POSTGRES_PORT")
    postgres_db: str | None = Field(default=None, validation_alias="POSTGRES_DB")
    app_postgres_db: str | None = Field(default=None, validation_alias="APP_POSTGRES_DB")
    postgres_user: str | None = Field(default=None, validation_alias="POSTGRES_USER")
    postgres_password: str | None = Field(default=None, validation_alias="POSTGRES_PASSWORD")
    postgres_pool_max: int = Field(default=8, ge=1, validation_alias="POSTGRES_POOL_MAX")
    postgres_pool_timeout: float = Field(default=30.0, gt=0, validation_alias="POSTGRES_POOL_TIMEOUT")
    postgres_connect_timeout: int = Field(default=10, ge=0, validation_alias="POSTGRES_CONNECT_TIMEOUT")

    prepare_job_stale_seconds: int = Field(default=900, validation_alias="PREPARE_JOB_STALE_SECONDS")
    prepare_job_max_attempts: int = Field(default=3, validation_alias="PREPARE_JOB_MAX_ATTEMPTS")
    prepare_backfill_enabled: bool = Field(default=False, validation_alias="PREPARE_BACKFILL_ENABLED")

    scheduler_timezone: str = Field(default="Asia/Seoul", validation_alias="SCHEDULER_TIMEZONE")

    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="ArXplore", validation_alias="LANGSMITH_PROJECT")
    langsmith_workspace_id: str | None = Field(default=None, validation_alias="LANGSMITH_WORKSPACE_ID")
    langsmith_tracing: bool = Field(default=True, validation_alias="LANGSMITH_TRACING")
    langsmith_trace_user: str | None = Field(default=None, validation_alias="LANGSMITH_TRACE_USER")

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", validation_alias="OPENAI_MODEL")
    openai_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_BASE_URL", "OPENAI_API_BASE"),
    )
    openai_embedding_model: str = Field(default="text-embedding-3-large", validation_alias="OPENAI_EMBEDDING_MODEL")
    openai_embedding_dimensions: int = Field(default=1536, validation_alias="OPENAI_EMBEDDING_DIMENSIONS")
    embedding_batch_size: int = Field(default=64, validation_alias="EMBEDDING_BATCH_SIZE")

    retrieval_mode: RetrievalMode = Field(default="hybrid", validation_alias="RETRIEVAL_MODE")
    agent_recursion_limit: int = Field(default=12, ge=4, validation_alias="AGENT_RECURSION_LIMIT")
    agent_stream_buffer_chars: int = Field(default=120, ge=0, validation_alias="AGENT_STREAM_BUFFER_CHARS")
    vector_min_similarity: float = Field(default=0.0, ge=0.0, le=1.0, validation_alias="VECTOR_MIN_SIMILARITY")

    @model_validator(mode="before")
    @classmethod
    def _drop_blank_values(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if not (isinstance(value, str) and not value.strip())}
        return data

    @field_validator("redis_url", mode="before")
    @classmethod
    def _normalize_redis_url(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("openai_base_url", mode="before")
    @classmethod
    def _normalize_openai_base_url(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("rate_limit_ip_header", mode="before")
    @classmethod
    def _strip_ip_header(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("retrieval_mode", mode="before")
    @classmethod
    def _normalize_retrieval_mode(cls, value: Any) -> str:
        normalized = str(value or "hybrid").strip().lower()
        if normalized not in RETRIEVAL_MODES:
            raise ValueError(f"RETRIEVAL_MODE는 {sorted(RETRIEVAL_MODES)} 중 하나여야 합니다. 받은 값: {value}")
        return normalized

    @field_validator("scheduler_timezone", mode="before")
    @classmethod
    def _validate_scheduler_timezone(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        name = value.strip()
        try:
            ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"SCHEDULER_TIMEZONE을 해석할 수 없습니다: {value}") from exc
        return name


def resolve_host_and_port(host: str, default_port: int) -> tuple[str, int]:
    normalized = host.strip()
    if not normalized:
        raise ValueError("호스트 값이 비어 있습니다.")

    parsed = urlsplit(f"//{normalized}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"호스트 값을 해석할 수 없습니다: {host}")

    return hostname, parsed.port or default_port


def build_postgres_connection_params(settings: AppSettings) -> PostgresConnectionParams:
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


def resolve_env_file() -> Path:
    configured = os.environ.get(ENV_FILE_VARIABLE, "").strip()
    if not configured:
        return DEFAULT_ENV_FILE
    path = Path(configured).expanduser()
    if not path.is_file():
        raise ConfigError(f"{ENV_FILE_VARIABLE}가 가리키는 파일이 없습니다: {configured}")
    return path


def load_settings() -> AppSettings:
    return AppSettings(_env_file=resolve_env_file())


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return load_settings()
