from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from django.core.cache import cache
from django.test import RequestFactory, override_settings
from django.urls import Resolver404, resolve

from papers import api_views, page_views, ratelimit

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def clean_cache():
    cache.clear()
    yield
    cache.clear()


@pytest.mark.usefixtures("clean_cache")
class TestRegisterHit:
    def test_allows_up_to_limit_then_returns_retry_after(self):
        now = 1_000_040.0
        results = [ratelimit.register_hit("t", "ip:1", limit=3, window=60, now=now) for _ in range(4)]

        assert results[:3] == [None, None, None]
        assert results[3] == 40

    def test_new_window_resets_counter(self):
        for _ in range(3):
            ratelimit.register_hit("t", "ip:1", limit=2, window=60, now=1_000_000.0)

        assert ratelimit.register_hit("t", "ip:1", limit=2, window=60, now=1_000_061.0) is None

    def test_identities_and_scopes_are_independent(self):
        ratelimit.register_hit("t", "ip:1", limit=1, window=60, now=1_000_020.0)

        assert ratelimit.register_hit("t", "ip:2", limit=1, window=60, now=1_000_020.0) is None
        assert ratelimit.register_hit("other", "ip:1", limit=1, window=60, now=1_000_020.0) is None
        assert ratelimit.register_hit("t", "ip:1", limit=1, window=60, now=1_000_020.0) == 60


class TestClientIp:
    def test_uses_configured_proxy_header(self):
        request = RequestFactory().get("/", HTTP_X_REAL_IP="203.0.113.9", REMOTE_ADDR="172.18.0.5")

        with override_settings(RATE_LIMIT_IP_HEADER="X-Real-IP"):
            assert ratelimit.client_ip(request) == "203.0.113.9"

    def test_takes_first_forwarded_value(self):
        request = RequestFactory().get("/", HTTP_X_FORWARDED_FOR="203.0.113.9, 10.0.0.1")

        with override_settings(RATE_LIMIT_IP_HEADER="X-Forwarded-For"):
            assert ratelimit.client_ip(request) == "203.0.113.9"

    def test_empty_header_setting_uses_remote_addr(self):
        request = RequestFactory().get("/", HTTP_X_REAL_IP="203.0.113.9", REMOTE_ADDR="172.18.0.5")

        with override_settings(RATE_LIMIT_IP_HEADER=""):
            assert ratelimit.client_ip(request) == "172.18.0.5"


@pytest.fixture
def rate_limits_on(settings):
    settings.RATE_LIMIT_ENABLED = True
    settings.RATE_LIMIT_LLM_PER_MINUTE = 30
    settings.RATE_LIMIT_DETAIL_PER_MINUTE = 60
    settings.RATE_LIMIT_IP_HEADER = "X-Real-IP"


@pytest.mark.usefixtures("clean_cache", "rate_limits_on")
class TestRateLimitedViews:
    def _analyze(self, ip="198.51.100.1"):
        request = RequestFactory().post("/papers/2401.00001/analyze/", HTTP_X_REAL_IP=ip)
        with patch.object(
            api_views, "get_paper_analysis", return_value={"overview": "o", "key_findings": [], "cached": True}
        ):
            return api_views.paper_analyze(request, "2401.00001")

    def test_llm_endpoints_are_limited_per_ip(self):
        statuses = [self._analyze().status_code for _ in range(31)]

        assert statuses[:30] == [200] * 30
        assert statuses[30] == 429
        assert self._analyze(ip="198.51.100.2").status_code == 200

    def test_429_payload_has_error_and_retry_after(self):
        for _ in range(30):
            self._analyze()

        response = self._analyze()
        payload = json.loads(response.content)

        assert response.status_code == 429
        assert payload["error"]
        assert 1 <= payload["retry_after"] <= 60
        assert response["Retry-After"] == str(payload["retry_after"])

    def test_llm_limit_is_shared_between_llm_endpoints(self):
        for _ in range(30):
            self._analyze()
        request = RequestFactory().post(
            "/papers/assistant/stream/",
            data=json.dumps({"message": "q", "history": []}),
            content_type="application/json",
            HTTP_X_REAL_IP="198.51.100.1",
        )

        with patch.object(api_views, "stream_agent_chat") as stream_mock:
            response = api_views.paper_agent_stream(request)

        assert response.status_code == 429
        stream_mock.assert_not_called()

    def test_get_is_rejected_before_counting(self):
        for _ in range(40):
            request = RequestFactory().get("/papers/2401.00001/analyze/")
            assert api_views.paper_analyze(request, "2401.00001").status_code == 405

        assert self._analyze().status_code == 200

    @pytest.mark.parametrize(
        "view",
        [
            api_views.paper_analyze,
            api_views.paper_summary,
            api_views.paper_chat,
            api_views.paper_chat_stream,
            api_views.paper_agent_chat,
            api_views.paper_agent_stream,
        ],
    )
    def test_every_llm_view_is_rate_limited(self, view):
        with patch.object(ratelimit, "register_hit", return_value=5) as hit:
            request = RequestFactory().post("/x/", data="{}", content_type="application/json")
            args = () if view in (api_views.paper_agent_chat, api_views.paper_agent_stream) else ("2401.00001",)
            response = view(request, *args)

        assert response.status_code == 429
        assert hit.call_args.args[0] == "llm"

    def _detail(self, ip: str):
        request = RequestFactory().get("/papers/2401.00001/detail.json", HTTP_X_REAL_IP=ip)
        with patch.object(page_views, "build_paper_detail_payload", return_value={"paper": {}}) as build:
            return page_views.paper_detail_data(request, "2401.00001"), build

    def test_detail_json_is_limited_per_ip(self):
        statuses = [self._detail("198.51.100.1")[0].status_code for _ in range(61)]

        assert statuses[:60] == [200] * 60
        response, build = self._detail("198.51.100.1")
        assert response.status_code == 429
        build.assert_not_called()
        assert self._detail("198.51.100.2")[0].status_code == 200

    def test_detail_json_rejects_non_get_before_counting(self):
        for _ in range(70):
            request = RequestFactory().post("/papers/2401.00001/detail.json", HTTP_X_REAL_IP="198.51.100.1")
            assert page_views.paper_detail_data(request, "2401.00001").status_code == 405

        assert self._detail("198.51.100.1")[0].status_code == 200

    def test_disabled_rate_limit_never_blocks(self):
        with override_settings(RATE_LIMIT_ENABLED=False):
            statuses = {self._analyze().status_code for _ in range(40)}

        assert statuses == {200}


class TestRemovedRoutes:
    @pytest.mark.parametrize(
        "path",
        ["/admin/", "/login/", "/auth/login/", "/auth/signup/", "/settings/", "/settings/api-key/", "/favorites/"],
    )
    def test_account_routes_do_not_exist(self, path):
        with pytest.raises(Resolver404):
            resolve(path)


class TestHttpMethods:
    @pytest.mark.parametrize(
        ("view", "args"),
        [
            (api_views.paper_analyze, ("2401.00001",)),
            (api_views.paper_summary, ("2401.00001",)),
            (api_views.paper_chat, ("2401.00001",)),
            (api_views.paper_chat_stream, ("2401.00001",)),
            (api_views.paper_agent_chat, ()),
            (api_views.paper_agent_stream, ()),
        ],
    )
    def test_llm_endpoints_reject_get(self, view, args):
        request = RequestFactory().get("/x/")

        assert view(request, *args).status_code == 405


def _load_settings_in_subprocess(env_overrides: dict[str, str]) -> dict:
    env = {**os.environ, "DJANGO_SETTINGS_MODULE": "arxplore_web.settings", **env_overrides}
    code = (
        "import json, django; django.setup(); from django.conf import settings as s; "
        "from django.core.cache import caches; "
        "print(json.dumps({k: getattr(s, k, None) for k in ["
        "'DEBUG','CSRF_COOKIE_SECURE','SECURE_PROXY_SSL_HEADER','CSRF_COOKIE_SAMESITE','X_FRAME_OPTIONS',"
        "'SECURE_CONTENT_TYPE_NOSNIFF','SECURE_REFERRER_POLICY','INSTALLED_APPS','MIDDLEWARE',"
        "'RATE_LIMIT_LLM_PER_MINUTE','RATE_LIMIT_DETAIL_PER_MINUTE']} | "
        "{'CACHE_BACKEND': type(caches['default']).__name__}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT / "backend",
        env={**env, "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{REPO_ROOT / 'backend'}"},
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestProductionSettings:
    def test_secure_defaults(self):
        values = _load_settings_in_subprocess({"DJANGO_DEBUG": "", "DJANGO_SECURE_COOKIES": "", "REDIS_URL": ""})

        assert values["DEBUG"] is False
        assert values["CSRF_COOKIE_SECURE"] is False
        assert values["SECURE_PROXY_SSL_HEADER"] is None
        assert values["CSRF_COOKIE_SAMESITE"] == "Lax"
        assert values["X_FRAME_OPTIONS"] == "DENY"
        assert values["SECURE_CONTENT_TYPE_NOSNIFF"] is True
        assert values["SECURE_REFERRER_POLICY"] == "same-origin"
        assert "django.middleware.csrf.CsrfViewMiddleware" in values["MIDDLEWARE"]
        assert not [
            app for app in values["INSTALLED_APPS"] if app.startswith(("django.contrib.auth", "django.contrib.admin"))
        ]
        assert values["RATE_LIMIT_LLM_PER_MINUTE"] == 30
        assert values["RATE_LIMIT_DETAIL_PER_MINUTE"] == 60
        assert values["CACHE_BACKEND"] == "LocMemCache"

    @pytest.mark.parametrize(("raw", "expected"), [("true", True), ("True", True), ("1", False), ("yes", False)])
    def test_debug_only_when_explicitly_true(self, raw, expected):
        assert _load_settings_in_subprocess({"DJANGO_DEBUG": raw})["DEBUG"] is expected

    def test_secure_cookies_toggle(self):
        values = _load_settings_in_subprocess({"DJANGO_SECURE_COOKIES": "true"})

        assert values["CSRF_COOKIE_SECURE"] is True
        assert values["SECURE_PROXY_SSL_HEADER"] == ["HTTP_X_FORWARDED_PROTO", "https"]

    def test_redis_url_selects_redis_cache(self):
        values = _load_settings_in_subprocess({"REDIS_URL": "redis://cache:6379/0"})

        assert values["CACHE_BACKEND"] == "RedisCache"


def _import_settings_module(settings_module: str, secret_key: str, **extra_env: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        **extra_env,
        "DJANGO_SETTINGS_MODULE": settings_module,
        "DJANGO_SECRET_KEY": secret_key,
        "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{REPO_ROOT / 'backend'}",
    }
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import django; django.setup(); from django.conf import settings; print(settings.SECRET_KEY)",
        ],
        cwd=REPO_ROOT / "backend",
        env=env,
        capture_output=True,
        text=True,
    )


class TestPlaceholderSecretKey:
    @pytest.mark.parametrize("debug", ["", "true"])
    def test_production_settings_refuse_change_me_secret(self, debug):
        result = _import_settings_module("arxplore_web.settings", "change-me-django-secret-key", DJANGO_DEBUG=debug)

        assert result.returncode != 0
        assert "ImproperlyConfigured" in result.stderr
        assert "placeholder (starts with 'change-me')" in result.stderr

    def test_test_settings_allow_change_me_secret(self):
        result = _import_settings_module("arxplore_web.test_settings", "change-me-django-secret-key")

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "change-me-django-secret-key"

    def test_real_secret_is_accepted(self):
        result = _import_settings_module("arxplore_web.settings", "a-real-secret-value")

        assert result.returncode == 0, result.stderr
