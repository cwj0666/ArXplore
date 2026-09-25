from __future__ import annotations

import pytest

from src.shared.settings import (
    AppSettings,
    build_django_postgres_database_config,
    build_postgres_connection_params,
    resolve_host_and_port,
)


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
    def test_valid(self, raw, expected):
        assert resolve_host_and_port(raw, 5432) == expected

    def test_host_only_uses_default_port_even_when_server_publishes_another(self):
        # The server compose publishes PostgreSQL on 15432; a bare host still resolves to the default port.
        assert resolve_host_and_port("server.tailnet", 5432) == ("server.tailnet", 5432)

    @pytest.mark.parametrize("raw", ["", "   ", ":5432", "::1"])
    def test_unparseable_host_raises(self, raw):
        with pytest.raises(ValueError):
            resolve_host_and_port(raw, 5432)

    @pytest.mark.parametrize("raw", ["db:abc", "db:99999"])
    def test_invalid_port_raises(self, raw):
        with pytest.raises(ValueError):
            resolve_host_and_port(raw, 5432)


def _settings(**overrides) -> AppSettings:
    values = {
        "postgres_host": "db",
        "postgres_db": "arxplore",
        "app_postgres_db": None,
        "postgres_user": "user",
        "postgres_password": "secret",
        "server_postgres_port": 5432,
    }
    values.update(overrides)
    return AppSettings.model_construct(**values)


class TestBuildPostgresConnectionParams:
    def test_host_only(self):
        assert build_postgres_connection_params(_settings()) == {
            "dbname": "arxplore",
            "user": "user",
            "password": "secret",
            "host": "db",
            "port": 5432,
        }

    def test_host_with_port_overrides_server_port(self):
        params = build_postgres_connection_params(_settings(postgres_host="db:15432", server_postgres_port=5432))

        assert (params["host"], params["port"]) == ("db", 15432)

    def test_server_port_is_default_for_bare_host(self):
        params = build_postgres_connection_params(_settings(server_postgres_port=15432))

        assert params["port"] == 15432

    def test_app_db_takes_precedence(self):
        params = build_postgres_connection_params(_settings(app_postgres_db="arxplore_app"))

        assert params["dbname"] == "arxplore_app"

    @pytest.mark.parametrize(
        "overrides",
        [
            {"postgres_host": None},
            {"postgres_host": ""},
            {"postgres_db": None},
            {"postgres_user": None},
            {"postgres_password": ""},
        ],
    )
    def test_missing_required_values_raise(self, overrides):
        with pytest.raises(ValueError):
            build_postgres_connection_params(_settings(**overrides))

    def test_django_config_maps_params(self):
        config = build_django_postgres_database_config(_settings(postgres_host="db:15432"))

        assert config == {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": "arxplore",
            "USER": "user",
            "PASSWORD": "secret",
            "HOST": "db",
            "PORT": 15432,
            "CONN_MAX_AGE": 60,
            "CONN_HEALTH_CHECKS": True,
        }

    def test_values_are_read_from_environment_aliases(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("POSTGRES_HOST", "envhost:6000")
        monkeypatch.setenv("POSTGRES_DB", "envdb")
        monkeypatch.setenv("POSTGRES_USER", "envuser")
        monkeypatch.setenv("POSTGRES_PASSWORD", "envpass")
        monkeypatch.delenv("APP_POSTGRES_DB", raising=False)
        monkeypatch.setenv("SERVER_POSTGRES_PORT", "15432")

        params = build_postgres_connection_params(AppSettings())

        assert params == {"dbname": "envdb", "user": "envuser", "password": "envpass", "host": "envhost", "port": 6000}
