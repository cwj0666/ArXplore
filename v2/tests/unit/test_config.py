from pathlib import Path

import pytest
from pydantic import ValidationError

from arxplore.config import (
    ENV_FILE_VARIABLE,
    AppSettings,
    ConfigError,
    build_postgres_connection_params,
    get_settings,
    load_settings,
    resolve_host_and_port,
)

pytestmark = pytest.mark.usefixtures("clean_env")

LEGACY_DEFAULTS = {
    "hf_daily_papers_base_url": "https://huggingface.co/papers",
    "hf_request_timeout_seconds": 20,
    "arxiv_api_base_url": "https://export.arxiv.org/api/query",
    "arxiv_request_timeout_seconds": 45,
    "arxiv_request_batch_size": 25,
    "arxiv_request_delay_seconds": 3.0,
    "layout_parser_base_url": None,
    "layout_parser_timeout_seconds": 600,
    "layout_parser_fast": False,
    "layout_parser_parse_tables_and_math": True,
    "postgres_host": None,
    "server_postgres_port": 5432,
    "postgres_db": None,
    "app_postgres_db": None,
    "postgres_user": None,
    "postgres_password": None,
    "postgres_pool_max": 8,
    "postgres_pool_timeout": 30.0,
    "prepare_job_stale_seconds": 900,
    "prepare_job_max_attempts": 3,
    "langsmith_api_key": None,
    "langsmith_project": "ArXplore",
    "langsmith_workspace_id": None,
    "langsmith_tracing": True,
    "langsmith_trace_user": None,
    "openai_api_key": None,
    "openai_model": "gpt-4o",
    "openai_embedding_model": "text-embedding-3-large",
    "openai_embedding_dimensions": 1536,
    "embedding_batch_size": 64,
    "retrieval_mode": "hybrid",
    "agent_recursion_limit": 12,
    "agent_stream_buffer_chars": 120,
    "vector_min_similarity": 0.0,
    "debug": False,
    "frontend_port": "5173",
    "allowed_hosts": ["localhost", "127.0.0.1"],
    "csrf_trusted_origins": [
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    "secure_cookies": False,
    "redis_url": None,
    "rate_limit_enabled": True,
    "rate_limit_llm_per_minute": 30,
    "rate_limit_detail_per_minute": 60,
    "rate_limit_ip_header": "X-Real-IP",
}

NEW_DEFAULTS = {
    "runtime": "development",
    "postgres_connect_timeout": 10,
    "prepare_backfill_enabled": False,
    "scheduler_timezone": "Asia/Seoul",
    "serve_frontend_dist": False,
}


def _load(monkeypatch: pytest.MonkeyPatch, **env: str) -> AppSettings:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return load_settings()


class TestDefaults:
    def test_legacy_and_new_defaults(self) -> None:
        settings = load_settings()

        assert {name: getattr(settings, name) for name in LEGACY_DEFAULTS} == LEGACY_DEFAULTS
        assert {name: getattr(settings, name) for name in NEW_DEFAULTS} == NEW_DEFAULTS

    def test_every_field_is_covered_by_the_default_tables(self) -> None:
        assert set(AppSettings.model_fields) == set(LEGACY_DEFAULTS) | set(NEW_DEFAULTS)


class TestLegacyNames:
    def test_app_settings_names_are_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _load(
            monkeypatch,
            HF_DAILY_PAPERS_BASE_URL="http://hf.test/papers",
            ARXIV_REQUEST_DELAY_SECONDS="0.5",
            LAYOUT_PARSER_BASE_URL="http://layout:5060",
            LAYOUT_PARSER_FAST="true",
            PREPARE_JOB_MAX_ATTEMPTS="5",
            LANGSMITH_TRACING="false",
            OPENAI_MODEL="gpt-4o-mini",
            OPENAI_EMBEDDING_DIMENSIONS="3072",
            EMBEDDING_BATCH_SIZE="16",
        )

        assert settings.hf_daily_papers_base_url == "http://hf.test/papers"
        assert settings.arxiv_request_delay_seconds == 0.5
        assert settings.layout_parser_base_url == "http://layout:5060"
        assert settings.layout_parser_fast is True
        assert settings.prepare_job_max_attempts == 5
        assert settings.langsmith_tracing is False
        assert settings.openai_model == "gpt-4o-mini"
        assert settings.openai_embedding_dimensions == 3072
        assert settings.embedding_batch_size == 16

    def test_names_are_case_insensitive_like_the_legacy_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _load(monkeypatch, openai_model="lowercase-name").openai_model == "lowercase-name"

    @pytest.mark.parametrize(
        ("legacy_name", "new_name", "raw", "field", "expected"),
        [
            ("DJANGO_DEBUG", "DEBUG", "true", "debug", True),
            ("DJANGO_ALLOWED_HOSTS", "ALLOWED_HOSTS", "a.test,b.test", "allowed_hosts", ["a.test", "b.test"]),
            (
                "DJANGO_CSRF_TRUSTED_ORIGINS",
                "CSRF_TRUSTED_ORIGINS",
                "https://a.test",
                "csrf_trusted_origins",
                ["https://a.test"],
            ),
            ("DJANGO_SECURE_COOKIES", "SECURE_COOKIES", "on", "secure_cookies", True),
        ],
    )
    def test_django_names_are_aliases(
        self,
        monkeypatch: pytest.MonkeyPatch,
        legacy_name: str,
        new_name: str,
        raw: str,
        field: str,
        expected: object,
    ) -> None:
        assert getattr(_load(monkeypatch, **{legacy_name: raw}), field) == expected
        monkeypatch.delenv(legacy_name)
        assert getattr(_load(monkeypatch, **{new_name: raw}), field) == expected

    def test_new_name_wins_over_legacy_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _load(
            monkeypatch,
            DEBUG="false",
            DJANGO_DEBUG="true",
            ALLOWED_HOSTS="new.test",
            DJANGO_ALLOWED_HOSTS="old.test",
        )

        assert settings.debug is False
        assert settings.allowed_hosts == ["new.test"]

    def test_empty_new_name_falls_back_to_legacy_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _load(monkeypatch, DEBUG="", DJANGO_DEBUG="true").debug is True

    def test_removed_variables_are_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _load(
            monkeypatch,
            DJANGO_SECRET_KEY="change-me-django-secret-key",
            APP_RUNTIME_MODE="production",
            AIRFLOW_BASE_URL="http://airflow:8080",
            AIRFLOW_FERNET_KEY="x",
        )

        assert settings.runtime == "development"
        assert not {"app_runtime_mode", "airflow_base_url", "django_secret_key"} & set(AppSettings.model_fields)

    def test_runtime_label_uses_new_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _load(monkeypatch, RUNTIME="production").runtime == "production"


class TestDjangoParsingRules:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("true", True), ("True", True), (" TRUE ", True), ("1", False), ("yes", False), ("on", False), ("x", False)],
    )
    def test_debug_only_when_exactly_true(self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
        assert _load(monkeypatch, DJANGO_DEBUG=raw).debug is expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1", True),
            ("true", True),
            ("YES", True),
            (" on ", True),
            ("0", False),
            ("false", False),
            ("off", False),
            ("t", False),
            ("maybe", False),
        ],
    )
    def test_legacy_flags_accept_one_true_yes_on(
        self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool
    ) -> None:
        settings = _load(monkeypatch, DJANGO_SECURE_COOKIES=raw, RATE_LIMIT_ENABLED=raw)

        assert settings.secure_cookies is expected
        assert settings.rate_limit_enabled is expected

    def test_allowed_hosts_default_depends_on_debug(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _load(monkeypatch, DJANGO_DEBUG="true").allowed_hosts == ["*"]

    def test_csv_values_are_stripped_and_blank_items_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _load(monkeypatch, DJANGO_ALLOWED_HOSTS=" a.test , ,b.test,")

        assert settings.allowed_hosts == ["a.test", "b.test"]

    def test_csrf_default_origins_follow_frontend_port(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _load(monkeypatch, FRONTEND_PORT="3000").csrf_trusted_origins == [
            "http://localhost",
            "http://127.0.0.1",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]

    def test_rate_limit_integers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _load(monkeypatch, RATE_LIMIT_LLM_PER_MINUTE=" 5 ", RATE_LIMIT_DETAIL_PER_MINUTE="7")

        assert (settings.rate_limit_llm_per_minute, settings.rate_limit_detail_per_minute) == (5, 7)

    def test_whitespace_only_integer_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _load(monkeypatch, RATE_LIMIT_LLM_PER_MINUTE="   ").rate_limit_llm_per_minute == 30

    def test_invalid_rate_limit_integer_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError):
            _load(monkeypatch, RATE_LIMIT_DETAIL_PER_MINUTE="many")

    def test_redis_url_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _load(monkeypatch, REDIS_URL=" redis://cache:6379/0 ").redis_url == "redis://cache:6379/0"

    def test_rate_limit_ip_header_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _load(monkeypatch, RATE_LIMIT_IP_HEADER=" X-Forwarded-For ").rate_limit_ip_header == "X-Forwarded-For"

    def test_empty_rate_limit_ip_header_keeps_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert _load(monkeypatch, RATE_LIMIT_IP_HEADER="").rate_limit_ip_header == "X-Real-IP"


class TestValidation:
    @pytest.mark.parametrize(("raw", "expected"), [("lexical", "lexical"), (" HYBRID ", "hybrid")])
    def test_retrieval_mode_is_normalized(self, monkeypatch: pytest.MonkeyPatch, raw: str, expected: str) -> None:
        assert _load(monkeypatch, RETRIEVAL_MODE=raw).retrieval_mode == expected

    def test_unknown_retrieval_mode_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="RETRIEVAL_MODE"):
            _load(monkeypatch, RETRIEVAL_MODE="vector")

    @pytest.mark.parametrize(
        ("name", "rejected", "accepted", "field", "expected"),
        [
            ("AGENT_RECURSION_LIMIT", "3", "4", "agent_recursion_limit", 4),
            ("AGENT_STREAM_BUFFER_CHARS", "-1", "0", "agent_stream_buffer_chars", 0),
            ("VECTOR_MIN_SIMILARITY", "1.01", "1", "vector_min_similarity", 1.0),
            ("VECTOR_MIN_SIMILARITY", "-0.1", "0", "vector_min_similarity", 0.0),
            ("POSTGRES_POOL_MAX", "0", "1", "postgres_pool_max", 1),
            ("POSTGRES_POOL_TIMEOUT", "0", "0.5", "postgres_pool_timeout", 0.5),
            ("POSTGRES_CONNECT_TIMEOUT", "-1", "0", "postgres_connect_timeout", 0),
        ],
    )
    def test_bounds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        rejected: str,
        accepted: str,
        field: str,
        expected: object,
    ) -> None:
        with pytest.raises(ValidationError):
            _load(monkeypatch, **{name: rejected})
        assert getattr(_load(monkeypatch, **{name: accepted}), field) == expected

    def test_new_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _load(
            monkeypatch,
            POSTGRES_CONNECT_TIMEOUT="3",
            PREPARE_BACKFILL_ENABLED="true",
            SCHEDULER_TIMEZONE="UTC",
            SERVE_FRONTEND_DIST="1",
        )

        assert settings.postgres_connect_timeout == 3
        assert settings.prepare_backfill_enabled is True
        assert settings.scheduler_timezone == "UTC"
        assert settings.serve_frontend_dist is True

    def test_unknown_scheduler_timezone_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError, match="SCHEDULER_TIMEZONE"):
            _load(monkeypatch, SCHEDULER_TIMEZONE="Mars/Olympus")


class TestEnvFile:
    def test_env_file_in_working_directory_is_read(self, clean_env: Path) -> None:
        (clean_env / ".env").write_text("OPENAI_MODEL=from-dotenv\n", encoding="utf-8")

        assert load_settings().openai_model == "from-dotenv"

    def test_environment_wins_over_env_file(self, clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (clean_env / ".env").write_text("OPENAI_MODEL=from-dotenv\n", encoding="utf-8")

        assert _load(monkeypatch, OPENAI_MODEL="from-env").openai_model == "from-env"

    @pytest.mark.parametrize("name", ["OPENAI_API_KEY", "LANGSMITH_API_KEY", "OPENAI_MODEL", "POSTGRES_HOST"])
    def test_empty_environment_value_falls_back_to_env_file(
        self, clean_env: Path, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        (clean_env / ".env").write_text(f"{name}=from-dotenv\n", encoding="utf-8")

        assert getattr(_load(monkeypatch, **{name: ""}), name.lower()) == "from-dotenv"

    def test_empty_everywhere_uses_default(self, clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (clean_env / ".env").write_text("OPENAI_API_KEY=\nOPENAI_MODEL=\n", encoding="utf-8")
        settings = _load(monkeypatch, OPENAI_API_KEY="", OPENAI_MODEL="")

        assert (settings.openai_api_key, settings.openai_model) == (None, "gpt-4o")

    def test_whitespace_only_environment_value_uses_default(
        self, clean_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (clean_env / ".env").write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

        assert _load(monkeypatch, OPENAI_API_KEY="   ").openai_api_key is None

    def test_explicit_env_file(self, clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (clean_env / ".env").write_text("OPENAI_MODEL=from-cwd\n", encoding="utf-8")
        other = clean_env / "conf" / "arxplore.env"
        other.parent.mkdir()
        other.write_text("OPENAI_MODEL=from-explicit\n", encoding="utf-8")

        assert _load(monkeypatch, **{ENV_FILE_VARIABLE: str(other)}).openai_model == "from-explicit"

    def test_missing_explicit_env_file_is_an_error(self, clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_FILE_VARIABLE, str(clean_env / "missing.env"))

        with pytest.raises(ConfigError, match=ENV_FILE_VARIABLE):
            load_settings()

    def test_blank_explicit_env_file_uses_working_directory(
        self, clean_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (clean_env / ".env").write_text("OPENAI_MODEL=from-cwd\n", encoding="utf-8")

        assert _load(monkeypatch, **{ENV_FILE_VARIABLE: "  "}).openai_model == "from-cwd"

    def test_env_file_may_contain_unrelated_variables(self, clean_env: Path) -> None:
        (clean_env / ".env").write_text("PROD_POSTGRES_HOST=x\nDJANGO_SECRET_KEY=y\nAIRFLOW_UID=1\n", encoding="utf-8")

        assert load_settings().postgres_host is None

    def test_get_settings_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first = get_settings()
        monkeypatch.setenv("OPENAI_MODEL", "changed")

        assert get_settings() is first
        get_settings.cache_clear()
        assert get_settings().openai_model == "changed"


class TestResolveHostAndPort:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("db", ("db", 5432)),
            ("  db.internal  ", ("db.internal", 5432)),
            ("db:15432", ("db", 15432)),
            ("100.64.0.1:15432", ("100.64.0.1", 15432)),
            ("[::1]", ("::1", 5432)),
            ("[::1]:6543", ("::1", 6543)),
            ("DB.Example.COM", ("db.example.com", 5432)),
        ],
    )
    def test_valid(self, raw: str, expected: tuple[str, int]) -> None:
        assert resolve_host_and_port(raw, 5432) == expected

    def test_bare_host_uses_the_default_port(self) -> None:
        assert resolve_host_and_port("server.tailnet", 15432) == ("server.tailnet", 15432)

    @pytest.mark.parametrize("raw", ["", "   ", ":5432", "::1"])
    def test_unparseable_host_raises(self, raw: str) -> None:
        with pytest.raises(ValueError):
            resolve_host_and_port(raw, 5432)

    @pytest.mark.parametrize("raw", ["db:abc", "db:99999"])
    def test_invalid_port_raises(self, raw: str) -> None:
        with pytest.raises(ValueError):
            resolve_host_and_port(raw, 5432)


def _database_settings(**overrides: str) -> AppSettings:
    values = {
        "POSTGRES_HOST": "db",
        "POSTGRES_DB": "arxplore",
        "POSTGRES_USER": "user",
        "POSTGRES_PASSWORD": "secret",
        "SERVER_POSTGRES_PORT": "5432",
    }
    values.update(overrides)
    return AppSettings.model_validate({key: value for key, value in values.items() if value is not None})


class TestBuildPostgresConnectionParams:
    def test_host_only(self) -> None:
        assert build_postgres_connection_params(_database_settings()) == {
            "dbname": "arxplore",
            "user": "user",
            "password": "secret",
            "host": "db",
            "port": 5432,
        }

    def test_host_with_port_overrides_server_port(self) -> None:
        params = build_postgres_connection_params(_database_settings(POSTGRES_HOST="db:15432"))

        assert (params["host"], params["port"]) == ("db", 15432)

    def test_server_port_is_default_for_bare_host(self) -> None:
        params = build_postgres_connection_params(_database_settings(SERVER_POSTGRES_PORT="15432"))

        assert params["port"] == 15432

    def test_app_db_takes_precedence(self) -> None:
        params = build_postgres_connection_params(_database_settings(APP_POSTGRES_DB="arxplore_app"))

        assert params["dbname"] == "arxplore_app"

    @pytest.mark.parametrize("missing", ["POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"])
    def test_missing_required_values_raise(self, missing: str) -> None:
        values = {
            "POSTGRES_HOST": "db",
            "POSTGRES_DB": "arxplore",
            "POSTGRES_USER": "user",
            "POSTGRES_PASSWORD": "secret",
        }
        del values[missing]

        with pytest.raises(ValueError, match=missing):
            build_postgres_connection_params(AppSettings.model_validate(values))

    def test_values_are_read_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _load(
            monkeypatch,
            POSTGRES_HOST="envhost:6000",
            POSTGRES_DB="envdb",
            POSTGRES_USER="envuser",
            POSTGRES_PASSWORD="envpass",
            SERVER_POSTGRES_PORT="15432",
        )

        assert build_postgres_connection_params(settings) == {
            "dbname": "envdb",
            "user": "envuser",
            "password": "envpass",
            "host": "envhost",
            "port": 6000,
        }
