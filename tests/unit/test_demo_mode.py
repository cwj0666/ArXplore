from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from django.contrib.auth.models import AnonymousUser
from django.contrib.sessions.backends.signed_cookies import SessionStore
from django.test import RequestFactory, override_settings

from papers import api_views, page_views, services

ARXIV_ID = "2401.00001"
PAPER = {"arxiv_id": ARXIV_ID, "title": "T", "authors": ["A"], "abstract": "abs"}


class _User:
    is_authenticated = True
    id = 1
    pk = 1

    def get_username(self) -> str:
        return "tester"


def _repo(*, overview=None, summary=None, paper=PAPER) -> MagicMock:
    repo = MagicMock()
    repo.get_paper.return_value = paper
    repo.get_paper_overview.return_value = overview
    repo.get_detailed_summary.return_value = summary
    repo.get_paper_fulltext.return_value = {"text": "body", "sections": []}
    repo.list_paper_chunks.return_value = []
    repo.list_recent_papers.return_value = []
    return repo


def _post(path: str, *, user, body: dict | None = None):
    request = RequestFactory().post(path, data=json.dumps(body or {}), content_type="application/json")
    request.user = user
    request.session = SessionStore()
    return request


def _analyze(user, *, api_key=None, repo=None):
    repo = repo or _repo()
    request = _post(f"/papers/{ARXIV_ID}/analyze/", user=user)
    with patch.object(services, "get_paper_repository", return_value=repo), patch.object(
        api_views, "get_session_api_key", return_value=api_key
    ), patch("src.core.analyze_paper_detail") as llm:
        response = api_views.paper_analyze(request, ARXIV_ID)
    return response, json.loads(response.content), repo, llm


def _summary(user, *, api_key=None, repo=None, model="gpt-5-mini"):
    repo = repo or _repo()
    request = _post(f"/papers/{ARXIV_ID}/summary/", user=user, body={"model": model})
    with patch.object(services, "get_paper_repository", return_value=repo), patch.object(
        api_views, "get_session_api_key", return_value=api_key
    ), patch("src.core.translation_chains.build_summary", return_value="new summary") as llm:
        response = api_views.paper_summary(request, ARXIV_ID)
    return response, json.loads(response.content), repo, llm


CACHED_OVERVIEW = {"overview": "cached overview", "key_findings": ["k1"]}
CACHED_SUMMARY = {"summary": "cached summary"}


class TestAnalyzeDemoMode:
    @pytest.mark.parametrize("user", [AnonymousUser(), _User()], ids=["anonymous", "logged-in-no-key"])
    def test_cached_overview_is_returned_without_login_or_key(self, user):
        response, payload, repo, llm = _analyze(user, repo=_repo(overview=CACHED_OVERVIEW))

        assert response.status_code == 200
        assert payload == {"overview": "cached overview", "key_findings": ["k1"], "cached": True}
        llm.assert_not_called()
        repo.upsert_paper_overview.assert_not_called()

    def test_anonymous_without_cache_requires_login(self):
        response, payload, repo, llm = _analyze(AnonymousUser())

        assert response.status_code == 401
        assert payload["login_required"] is True
        assert payload["error"]
        llm.assert_not_called()

    def test_logged_in_without_key_and_without_cache_requires_key(self):
        response, payload, _, llm = _analyze(_User())

        assert response.status_code == 400
        assert payload["api_key_required"] is True
        llm.assert_not_called()

    def test_with_key_generates_and_caches(self):
        doc = MagicMock(overview="fresh", key_findings=["f"])
        repo = _repo()
        request = _post(f"/papers/{ARXIV_ID}/analyze/", user=_User())
        with patch.object(services, "get_paper_repository", return_value=repo), patch.object(
            api_views, "get_session_api_key", return_value="sk-user"
        ), patch("src.core.analyze_paper_detail", return_value=doc) as llm:
            response = api_views.paper_analyze(request, ARXIV_ID)

        assert response.status_code == 200
        assert json.loads(response.content) == {"overview": "fresh", "key_findings": ["f"], "cached": False}
        llm.assert_called_once()
        repo.upsert_paper_overview.assert_called_once_with(ARXIV_ID, "fresh", ["f"], services.OVERVIEW_MODEL)

    def test_unknown_paper_returns_404_for_anonymous(self):
        response, _, _, llm = _analyze(AnonymousUser(), repo=_repo(paper=None))

        assert response.status_code == 404
        llm.assert_not_called()

    @override_settings(DEMO_MODE=False)
    def test_demo_mode_off_requires_login_even_when_cached(self):
        response, payload, repo, _ = _analyze(AnonymousUser(), repo=_repo(overview=CACHED_OVERVIEW))

        assert response.status_code == 401
        assert payload["login_required"] is True
        repo.get_paper_overview.assert_not_called()

    @override_settings(DEMO_MODE=False)
    def test_demo_mode_off_requires_key_even_when_cached(self):
        response, payload, _, _ = _analyze(_User(), repo=_repo(overview=CACHED_OVERVIEW))

        assert response.status_code == 400
        assert payload["api_key_required"] is True

    def test_llm_failure_returns_generic_500(self):
        request = _post(f"/papers/{ARXIV_ID}/analyze/", user=_User())
        with patch.object(services, "get_paper_repository", return_value=_repo()), patch.object(
            api_views, "get_session_api_key", return_value="sk-user"
        ), patch("src.core.analyze_paper_detail", side_effect=RuntimeError("postgres://user:pw@host")):
            response = api_views.paper_analyze(request, ARXIV_ID)

        assert response.status_code == 500
        assert "pw@host" not in response.content.decode()


class TestSummaryDemoMode:
    @pytest.mark.parametrize("user", [AnonymousUser(), _User()], ids=["anonymous", "logged-in-no-key"])
    def test_cached_summary_is_returned_without_login_or_key(self, user):
        response, payload, repo, llm = _summary(user, repo=_repo(summary=CACHED_SUMMARY), model="gpt-5")

        assert response.status_code == 200
        assert payload == {"summary": "cached summary", "cached": True, "model": "gpt-5"}
        repo.get_detailed_summary.assert_called_once_with(ARXIV_ID, "gpt-5")
        llm.assert_not_called()

    def test_anonymous_without_cache_requires_login(self):
        response, payload, _, llm = _summary(AnonymousUser())

        assert response.status_code == 401
        assert payload["login_required"] is True
        llm.assert_not_called()

    def test_logged_in_without_key_and_without_cache_requires_key(self):
        response, payload, _, llm = _summary(_User())

        assert response.status_code == 400
        assert payload["api_key_required"] is True
        llm.assert_not_called()

    def test_with_key_generates_and_caches(self):
        response, payload, repo, llm = _summary(_User(), api_key="sk-user")

        assert response.status_code == 200
        assert payload == {"summary": "new summary", "cached": False, "model": "gpt-5-mini"}
        llm.assert_called_once()
        repo.upsert_detailed_summary.assert_called_once()

    def test_invalid_model_is_rejected_before_cache_lookup(self):
        response, _, repo, _ = _summary(AnonymousUser(), model="gpt-x")

        assert response.status_code == 400
        repo.get_detailed_summary.assert_not_called()

    def test_invalid_json_body_returns_400(self):
        request = RequestFactory().post(f"/papers/{ARXIV_ID}/summary/", data="not json", content_type="application/json")
        request.user = AnonymousUser()
        response = api_views.paper_summary(request, ARXIV_ID)

        assert response.status_code == 400

    @override_settings(DEMO_MODE=False)
    def test_demo_mode_off_requires_login_even_when_cached(self):
        response, payload, _, _ = _summary(AnonymousUser(), repo=_repo(summary=CACHED_SUMMARY))

        assert response.status_code == 401
        assert payload["login_required"] is True


class TestChatStillRequiresLoginAndKey:
    def test_anonymous_chat_stream_returns_401(self):
        request = _post(f"/papers/{ARXIV_ID}/chat/stream/", user=AnonymousUser(), body={"message": "q", "history": []})
        with patch.object(services, "get_paper_repository", return_value=_repo()):
            response = api_views.paper_chat_stream(request, ARXIV_ID)

        assert response.status_code == 401
        assert json.loads(response.content)["login_required"] is True

    def test_logged_in_agent_stream_without_key_returns_400(self):
        request = _post("/papers/assistant/stream/", user=_User(), body={"message": "q", "history": []})
        with patch.object(api_views, "get_session_api_key", return_value=None):
            response = api_views.paper_agent_stream(request)

        assert response.status_code == 400
        assert json.loads(response.content)["api_key_required"] is True


class TestDetailAndListAnonymous:
    def _detail(self, user, repo=None):
        request = RequestFactory().get(f"/papers/{ARXIV_ID}/detail.json")
        request.user = user
        with patch.object(services, "get_paper_repository", return_value=repo or _repo()), patch.object(
            services, "_search_external_related_papers", return_value=[]
        ):
            return page_views.paper_detail_data(request, ARXIV_ID)

    def test_anonymous_can_read_detail_with_favorites_false(self):
        response = self._detail(AnonymousUser())

        assert response.status_code == 200
        paper = json.loads(response.content)["paper"]
        assert paper["arxiv_id"] == ARXIV_ID
        assert paper["is_favorited"] is False
        assert paper["related_papers"] == []

    @override_settings(DEMO_MODE=False)
    def test_detail_requires_login_when_demo_mode_off(self):
        response = self._detail(AnonymousUser())

        assert response.status_code == 401
        assert json.loads(response.content)["login_required"] is True

    def test_detail_database_error_is_not_leaked(self):
        repo = _repo()
        repo.get_paper.side_effect = RuntimeError("connection to secret-host failed")
        response = self._detail(AnonymousUser(), repo=repo)

        assert response.status_code == 500
        assert "secret-host" not in response.content.decode()

    def test_anonymous_can_read_list(self):
        repo = _repo()
        repo.list_recent_papers.return_value = [PAPER]
        request = RequestFactory().get("/papers/list.json")
        request.user = AnonymousUser()
        with patch.object(services, "get_paper_repository", return_value=repo):
            response = page_views.paper_list_data(request)

        assert response.status_code == 200
        items = json.loads(response.content)["items"]
        assert [item["arxiv_id"] for item in items] == [ARXIV_ID]
        assert items[0]["is_favorited"] is False

    def test_list_database_error_is_not_leaked(self):
        repo = _repo()
        repo.list_recent_papers.side_effect = RuntimeError("password=hunter2")
        request = RequestFactory().get("/papers/list.json")
        request.user = AnonymousUser()
        with patch.object(services, "get_paper_repository", return_value=repo):
            response = page_views.paper_list_data(request)

        assert response.status_code == 500
        assert "hunter2" not in response.content.decode()

    def test_detail_page_shell_is_not_redirected_for_anonymous_in_demo_mode(self):
        request = RequestFactory().get(f"/papers/{ARXIV_ID}/")
        request.user = AnonymousUser()
        response = page_views.paper_detail(request, ARXIV_ID)

        assert response.status_code != 302

    @override_settings(DEMO_MODE=False)
    def test_detail_page_shell_redirects_anonymous_when_demo_mode_off(self):
        request = RequestFactory().get(f"/papers/{ARXIV_ID}/")
        request.user = AnonymousUser()
        response = page_views.paper_detail(request, ARXIV_ID)

        assert response.status_code == 302
        assert response["Location"].startswith("/login/")


class TestBootstrapPayload:
    def _bootstrap(self):
        request = RequestFactory().get("/bootstrap.json")
        request.user = AnonymousUser()
        request.session = SessionStore()
        return json.loads(api_views.bootstrap(request).content)

    def test_demo_mode_fields_default_on(self):
        payload = self._bootstrap()

        assert payload["demo_mode"] is True
        assert payload["login_required_for"] == ["chat", "generate"]
        assert payload["is_authenticated"] is False
        assert payload["has_personal_api_key"] is False

    @override_settings(DEMO_MODE=False)
    def test_demo_mode_off(self):
        payload = self._bootstrap()

        assert payload["demo_mode"] is False
        assert "detail" in payload["login_required_for"]
