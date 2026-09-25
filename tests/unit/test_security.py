from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from django.contrib.auth.models import AnonymousUser
from django.contrib.sessions.backends.signed_cookies import SessionStore
from django.core.cache import cache
from django.test import RequestFactory, override_settings
from django.urls import Resolver404, resolve

from arxplore_web import urls as root_urls
from papers import api_views, page_views, ratelimit, secret_box, services

REPO_ROOT = Path(__file__).resolve().parents[2]


class _User:
    is_authenticated = True
    id = 7
    pk = 7

    def get_username(self) -> str:
        return "tester"


def _session_request(user=None):
    request = RequestFactory().post("/settings/api-key/")
    request.user = user or _User()
    request.session = SessionStore()
    return request


class TestSecretBox:
    def test_roundtrip(self):
        token = secret_box.encrypt_secret("dummy-personal-key-abc123")

        assert token.startswith(secret_box.TOKEN_PREFIX)
        assert "dummy-personal-key-abc123" not in token
        assert secret_box.decrypt_secret(token) == "dummy-personal-key-abc123"

    def test_nonce_makes_ciphertexts_differ(self):
        assert secret_box.encrypt_secret("same") != secret_box.encrypt_secret("same")

    def test_tampered_token_is_rejected(self):
        token = secret_box.encrypt_secret("dummy-personal-key-abc123")
        body = bytearray(token[len(secret_box.TOKEN_PREFIX) :].encode())
        body[30] = ord("A") if body[30] != ord("A") else ord("B")

        with pytest.raises(secret_box.InvalidToken):
            secret_box.decrypt_secret(secret_box.TOKEN_PREFIX + body.decode())

    @pytest.mark.parametrize("value", ["dummy-legacy-plaintext", "v1.AAAA", "v2.", "v2.!!!", "v2.한글", "", None, 123])
    def test_malformed_or_legacy_values_are_rejected(self, value):
        with pytest.raises(secret_box.InvalidToken):
            secret_box.decrypt_secret(value)

    def test_token_is_a_prefixed_fernet_token(self):
        from cryptography.fernet import Fernet

        token = secret_box.encrypt_secret("dummy-personal-key-abc123")

        assert token.startswith("v2.gAAAAA")
        assert isinstance(secret_box._fernet(), Fernet)

    def test_other_secret_key_cannot_decrypt(self):
        token = secret_box.encrypt_secret("dummy-personal-key-abc123")

        with override_settings(SECRET_KEY="another-secret"), pytest.raises(secret_box.InvalidToken):
            secret_box.decrypt_secret(token)

    def test_dedicated_encryption_key_takes_precedence_over_secret_key(self):
        with override_settings(SESSION_KEY_ENCRYPTION_KEY="dedicated-key"):
            token = secret_box.encrypt_secret("dummy-personal-key-abc123")
            with override_settings(SECRET_KEY="rotated-django-secret"):
                assert secret_box.decrypt_secret(token) == "dummy-personal-key-abc123"

        with pytest.raises(secret_box.InvalidToken):
            secret_box.decrypt_secret(token)


class TestSessionApiKey:
    def test_key_is_stored_encrypted_and_read_back(self):
        request = _session_request()

        services.save_personal_api_key(request, "  dummy-personal-key-abc123  ")

        stored = request.session[services.SESSION_API_KEY_KEY]
        assert "dummy-personal-key-abc123" not in stored
        assert services.get_session_api_key(request) == "dummy-personal-key-abc123"
        assert services.has_personal_api_key(request) is True

    def test_legacy_plaintext_value_is_cleared_and_treated_as_missing(self, caplog):
        request = _session_request()
        request.session[services.SESSION_API_KEY_KEY] = "dummy-legacy-plaintext"

        with caplog.at_level("WARNING"):
            assert services.get_session_api_key(request) is None

        assert services.SESSION_API_KEY_KEY not in request.session
        assert "dummy-legacy-plaintext" not in caplog.text

    def test_key_encrypted_with_old_secret_is_cleared(self):
        request = _session_request()
        with override_settings(SECRET_KEY="old-secret"):
            services.save_personal_api_key(request, "dummy-personal-key-abc123")

        assert services.get_session_api_key(request) is None
        assert services.SESSION_API_KEY_KEY not in request.session

    def test_clear_removes_key(self):
        request = _session_request()
        services.save_personal_api_key(request, "dummy-personal-key-abc123")

        services.clear_personal_api_key(request)

        assert services.get_session_api_key(request) is None

    def test_save_requires_login(self):
        with pytest.raises(services.AuthenticationRequiredError):
            services.save_personal_api_key(_session_request(user=AnonymousUser()), "dummy-personal-key-abc123")

    def test_save_view_response_does_not_echo_key(self):
        request = RequestFactory().post(
            "/settings/api-key/",
            data=json.dumps({"api_key": "dummy-personal-key-abc123"}),
            content_type="application/json",
        )
        request.user = _User()
        request.session = SessionStore()

        response = api_views.settings_api_key_detail(request)

        assert response.status_code == 200
        assert "dummy-personal-key-abc123" not in response.content.decode()
        assert json.loads(response.content)["has_personal_api_key"] is True


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
    settings.RATE_LIMIT_AUTH_PER_MINUTE = 10
    settings.RATE_LIMIT_LLM_PER_MINUTE = 30
    settings.RATE_LIMIT_DETAIL_PER_MINUTE = 60
    settings.RATE_LIMIT_IP_HEADER = "X-Real-IP"


@pytest.mark.usefixtures("clean_cache", "rate_limits_on")
class TestRateLimitedViews:
    def _login(self, ip: str):
        request = RequestFactory().post(
            "/auth/login/",
            data=json.dumps({"username": "u", "password": "p"}),
            content_type="application/json",
            HTTP_X_REAL_IP=ip,
        )
        request.user = AnonymousUser()
        request.session = SessionStore()
        with patch.object(api_views, "login_user", side_effect=services.InvalidRequestError("로그인에 실패했습니다.")):
            return api_views.auth_login(request)

    def test_login_is_limited_per_ip(self):
        statuses = [self._login("198.51.100.1").status_code for _ in range(11)]

        assert statuses[:10] == [400] * 10
        assert statuses[10] == 429
        assert self._login("198.51.100.2").status_code == 400

    def test_429_payload_has_error_and_retry_after(self):
        for _ in range(10):
            self._login("198.51.100.1")

        response = self._login("198.51.100.1")
        payload = json.loads(response.content)

        assert response.status_code == 429
        assert payload["error"]
        assert 1 <= payload["retry_after"] <= 60
        assert response["Retry-After"] == str(payload["retry_after"])

    def test_signup_shares_auth_limit_scope(self):
        for _ in range(10):
            self._login("198.51.100.1")
        request = RequestFactory().post(
            "/auth/signup/", data=json.dumps({}), content_type="application/json", HTTP_X_REAL_IP="198.51.100.1"
        )
        request.user = AnonymousUser()

        assert api_views.auth_signup(request).status_code == 429

    def _analyze(self, user, ip="198.51.100.1"):
        request = RequestFactory().post("/papers/2401.00001/analyze/", HTTP_X_REAL_IP=ip)
        request.user = user
        with (
            patch.object(api_views, "get_session_api_key", return_value="sk"),
            patch.object(
                api_views, "get_paper_analysis", return_value={"overview": "o", "key_findings": [], "cached": True}
            ),
        ):
            return api_views.paper_analyze(request, "2401.00001")

    def test_llm_endpoints_are_limited_per_user_across_ips(self):
        statuses = [self._analyze(_User(), ip=f"198.51.100.{i}").status_code for i in range(31)]

        assert statuses[:30] == [200] * 30
        assert statuses[30] == 429

    def test_llm_limit_is_shared_between_llm_endpoints(self):
        for _ in range(30):
            self._analyze(_User())
        request = RequestFactory().post(
            "/papers/assistant/stream/",
            data=json.dumps({"message": "q", "history": []}),
            content_type="application/json",
        )
        request.user = _User()

        with patch.object(api_views, "stream_agent_chat") as stream_mock:
            response = api_views.paper_agent_stream(request)

        assert response.status_code == 429
        stream_mock.assert_not_called()

    def test_anonymous_llm_requests_are_limited_per_ip(self):
        for _ in range(30):
            self._analyze(AnonymousUser(), ip="198.51.100.1")

        assert self._analyze(AnonymousUser(), ip="198.51.100.1").status_code == 429
        assert self._analyze(AnonymousUser(), ip="198.51.100.2").status_code == 200

    def test_get_is_rejected_before_counting(self):
        for _ in range(40):
            request = RequestFactory().get("/papers/2401.00001/analyze/")
            request.user = _User()
            assert api_views.paper_analyze(request, "2401.00001").status_code == 405

        assert self._analyze(_User()).status_code == 200

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
            request.user = _User()
            args = () if view in (api_views.paper_agent_chat, api_views.paper_agent_stream) else ("2401.00001",)
            response = view(request, *args)

        assert response.status_code == 429
        assert hit.call_args.args[0] == "llm"

    def _detail(self, ip: str):
        request = RequestFactory().get("/papers/2401.00001/detail.json", HTTP_X_REAL_IP=ip)
        request.user = AnonymousUser()
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
            request.user = AnonymousUser()
            assert page_views.paper_detail_data(request, "2401.00001").status_code == 405

        assert self._detail("198.51.100.1")[0].status_code == 200

    def test_disabled_rate_limit_never_blocks(self):
        with override_settings(RATE_LIMIT_ENABLED=False):
            statuses = {self._login("198.51.100.1").status_code for _ in range(15)}

        assert statuses == {400}


@pytest.mark.django_db
class TestSignupPasswordValidation:
    def _signup(self, username: str, password: str):
        request = RequestFactory().post(
            "/auth/signup/",
            data=json.dumps({"username": username, "password": password}),
            content_type="application/json",
        )
        request.user = AnonymousUser()
        return api_views.auth_signup(request)

    @pytest.mark.parametrize("password", ["1234", "12345678901", "password", "alice-researcher"])
    def test_weak_passwords_are_rejected_with_400(self, password):
        response = self._signup("alice-researcher", password)
        payload = json.loads(response.content)

        assert response.status_code == 400
        assert payload["error"]
        assert payload["password_errors"]

    def test_strong_password_is_accepted(self):
        response = self._signup("alice", "c0rrect-h0rse-battery")

        assert response.status_code == 200
        assert json.loads(response.content)["username"] == "alice"


class TestAdminRoute:
    def test_admin_route_is_absent_by_default(self):
        with pytest.raises(Resolver404):
            resolve("/admin/")
        assert root_urls.admin_urlpatterns() == []

    def test_admin_route_uses_configured_path_when_enabled(self):
        with override_settings(ADMIN_ENABLED=True, ADMIN_PATH="ops-console/"):
            patterns = root_urls.admin_urlpatterns()

        assert len(patterns) == 1
        assert str(patterns[0].pattern) == "ops-console/"


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
            (api_views.favorites_toggle, ()),
            (api_views.auth_login, ()),
            (api_views.auth_signup, ()),
            (api_views.auth_logout, ()),
            (api_views.settings_api_key_detail, ()),
        ],
    )
    def test_mutating_and_llm_endpoints_reject_get(self, view, args):
        request = RequestFactory().get("/x/")
        request.user = _User()
        request.session = SessionStore()

        assert view(request, *args).status_code == 405


def _load_settings_in_subprocess(env_overrides: dict[str, str]) -> dict:
    env = {**os.environ, "DJANGO_SETTINGS_MODULE": "arxplore_web.settings", **env_overrides}
    code = (
        "import json, django; django.setup(); from django.conf import settings as s; "
        "from django.core.cache import caches; "
        "print(json.dumps({k: getattr(s, k, None) for k in ["
        "'DEBUG','SESSION_COOKIE_SECURE','CSRF_COOKIE_SECURE','SECURE_PROXY_SSL_HEADER','SESSION_COOKIE_HTTPONLY',"
        "'SESSION_COOKIE_SAMESITE','X_FRAME_OPTIONS','SECURE_CONTENT_TYPE_NOSNIFF','SECURE_REFERRER_POLICY',"
        "'ADMIN_ENABLED','ADMIN_PATH','DEMO_MODE','RATE_LIMIT_AUTH_PER_MINUTE','RATE_LIMIT_LLM_PER_MINUTE',"
        "'RATE_LIMIT_DETAIL_PER_MINUTE']} | "
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
        values = _load_settings_in_subprocess(
            {"DJANGO_DEBUG": "", "DJANGO_SECURE_COOKIES": "", "DJANGO_ADMIN_ENABLED": "", "REDIS_URL": ""}
        )

        assert values["DEBUG"] is False
        assert values["SESSION_COOKIE_SECURE"] is False
        assert values["SECURE_PROXY_SSL_HEADER"] is None
        assert values["SESSION_COOKIE_HTTPONLY"] is True
        assert values["SESSION_COOKIE_SAMESITE"] == "Lax"
        assert values["X_FRAME_OPTIONS"] == "DENY"
        assert values["SECURE_CONTENT_TYPE_NOSNIFF"] is True
        assert values["SECURE_REFERRER_POLICY"] == "same-origin"
        assert values["ADMIN_ENABLED"] is False
        assert values["DEMO_MODE"] is True
        assert values["RATE_LIMIT_AUTH_PER_MINUTE"] == 10
        assert values["RATE_LIMIT_LLM_PER_MINUTE"] == 30
        assert values["RATE_LIMIT_DETAIL_PER_MINUTE"] == 60
        assert values["CACHE_BACKEND"] == "LocMemCache"

    @pytest.mark.parametrize(("raw", "expected"), [("true", True), ("True", True), ("1", False), ("yes", False)])
    def test_debug_only_when_explicitly_true(self, raw, expected):
        assert _load_settings_in_subprocess({"DJANGO_DEBUG": raw})["DEBUG"] is expected

    def test_secure_cookies_toggle(self):
        values = _load_settings_in_subprocess({"DJANGO_SECURE_COOKIES": "true"})

        assert values["SESSION_COOKIE_SECURE"] is True
        assert values["CSRF_COOKIE_SECURE"] is True
        assert values["SECURE_PROXY_SSL_HEADER"] == ["HTTP_X_FORWARDED_PROTO", "https"]

    def test_admin_path_is_normalized(self):
        values = _load_settings_in_subprocess({"DJANGO_ADMIN_ENABLED": "true", "DJANGO_ADMIN_PATH": "/ops/"})

        assert values["ADMIN_ENABLED"] is True
        assert values["ADMIN_PATH"] == "ops/"

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
