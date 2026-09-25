from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from django.test import RequestFactory

from papers import api_views, page_views, services
from src.shared import get_runtime_openai_model

ARXIV_ID = "2401.00001"
PAPER = {"arxiv_id": ARXIV_ID, "title": "T", "authors": ["A"], "abstract": "abs"}
CACHED_OVERVIEW = {"overview": "cached overview", "key_findings": ["k1"]}
CACHED_SUMMARY = {"summary": "cached summary"}


def _repo(*, overview=None, summary=None, paper=PAPER) -> MagicMock:
    repo = MagicMock()
    repo.get_paper.return_value = paper
    repo.get_paper_overview.return_value = overview
    repo.get_detailed_summary.return_value = summary
    repo.get_paper_fulltext.return_value = {"text": "body", "sections": []}
    repo.list_paper_chunks.return_value = []
    repo.list_recent_papers.return_value = []
    return repo


def _post(path: str, body: dict | None = None):
    return RequestFactory().post(path, data=json.dumps(body or {}), content_type="application/json")


def _analyze(*, repo=None, **llm_kwargs):
    repo = repo or _repo()
    with (
        patch.object(services, "get_paper_repository", return_value=repo),
        patch("src.core.analyze_paper_detail", **llm_kwargs) as llm,
    ):
        response = api_views.paper_analyze(_post(f"/papers/{ARXIV_ID}/analyze/"), ARXIV_ID)
    return response, json.loads(response.content), repo, llm


def _summary(*, repo=None, model="gpt-5-mini", **llm_kwargs):
    repo = repo or _repo()
    llm_kwargs.setdefault("return_value", "new summary")
    with (
        patch.object(services, "get_paper_repository", return_value=repo),
        patch("src.core.translation_chains.build_summary", **llm_kwargs) as llm,
    ):
        response = api_views.paper_summary(_post(f"/papers/{ARXIV_ID}/summary/", {"model": model}), ARXIV_ID)
    return response, json.loads(response.content), repo, llm


class TestAnalyze:
    def test_cached_overview_is_returned_without_generation(self):
        response, payload, repo, llm = _analyze(repo=_repo(overview=CACHED_OVERVIEW))

        assert response.status_code == 200
        assert payload == {"overview": "cached overview", "key_findings": ["k1"], "cached": True}
        llm.assert_not_called()
        repo.upsert_paper_overview.assert_not_called()

    def test_missing_cache_generates_with_overview_model_and_caches(self):
        seen: dict[str, str] = {}

        def generate(paper):
            seen["model"] = get_runtime_openai_model()
            return MagicMock(overview="fresh", key_findings=["f"])

        response, payload, repo, llm = _analyze(side_effect=generate)

        assert response.status_code == 200
        assert payload == {"overview": "fresh", "key_findings": ["f"], "cached": False}
        llm.assert_called_once()
        assert seen["model"] == services.OVERVIEW_MODEL
        repo.upsert_paper_overview.assert_called_once_with(ARXIV_ID, "fresh", ["f"], services.OVERVIEW_MODEL)

    def test_cached_overview_is_served_without_server_key(self):
        with patch.object(services, "get_runtime_openai_api_key", return_value=None):
            response, payload, _, llm = _analyze(repo=_repo(overview=CACHED_OVERVIEW))

        assert response.status_code == 200
        assert payload["cached"] is True
        llm.assert_not_called()

    def test_missing_cache_without_server_key_returns_503(self):
        with patch.object(services, "get_runtime_openai_api_key", return_value=None):
            response, payload, repo, llm = _analyze()

        assert response.status_code == 503
        assert payload["error"]
        llm.assert_not_called()
        repo.upsert_paper_overview.assert_not_called()

    def test_unknown_paper_returns_404(self):
        response, _, _, llm = _analyze(repo=_repo(paper=None))

        assert response.status_code == 404
        llm.assert_not_called()

    def test_llm_failure_returns_generic_500(self):
        response, _, _, _ = _analyze(side_effect=RuntimeError("postgres://user:pw@host"))

        assert response.status_code == 500
        assert "pw@host" not in response.content.decode()


class TestSummary:
    def test_cached_summary_is_returned_for_requested_model(self):
        response, payload, repo, llm = _summary(repo=_repo(summary=CACHED_SUMMARY), model="gpt-5")

        assert response.status_code == 200
        assert payload == {"summary": "cached summary", "cached": True, "model": "gpt-5"}
        repo.get_detailed_summary.assert_called_once_with(ARXIV_ID, "gpt-5")
        llm.assert_not_called()

    def test_missing_cache_generates_with_requested_model_and_caches(self):
        seen: dict[str, str] = {}

        def generate(**_kwargs):
            seen["model"] = get_runtime_openai_model()
            return "new summary"

        response, payload, repo, llm = _summary(model="gpt-5", side_effect=generate)

        assert response.status_code == 200
        assert payload == {"summary": "new summary", "cached": False, "model": "gpt-5"}
        llm.assert_called_once()
        assert seen["model"] == "gpt-5"
        repo.upsert_detailed_summary.assert_called_once_with(ARXIV_ID, "new summary", "gpt-5")

    def test_missing_cache_without_server_key_returns_503(self):
        with patch.object(services, "get_runtime_openai_api_key", return_value=""):
            response, _, _, llm = _summary()

        assert response.status_code == 503
        llm.assert_not_called()

    def test_invalid_model_is_rejected_before_cache_lookup(self):
        response, _, repo, _ = _summary(model="gpt-x")

        assert response.status_code == 400
        repo.get_detailed_summary.assert_not_called()

    def test_invalid_json_body_returns_400(self):
        request = RequestFactory().post(
            f"/papers/{ARXIV_ID}/summary/", data="not json", content_type="application/json"
        )

        assert api_views.paper_summary(request, ARXIV_ID).status_code == 400


class TestDetailAndList:
    def _detail(self, repo=None):
        request = RequestFactory().get(f"/papers/{ARXIV_ID}/detail.json")
        with (
            patch.object(services, "get_paper_repository", return_value=repo or _repo()),
            patch.object(services, "_search_external_related_papers", return_value=[]),
        ):
            return page_views.paper_detail_data(request, ARXIV_ID)

    def test_detail_is_readable(self):
        response = self._detail()

        assert response.status_code == 200
        paper = json.loads(response.content)["paper"]
        assert paper["arxiv_id"] == ARXIV_ID
        assert paper["related_papers"] == []
        assert "is_favorited" not in paper

    def test_unknown_paper_detail_returns_404(self):
        assert self._detail(repo=_repo(paper=None)).status_code == 404

    def test_detail_database_error_is_not_leaked(self):
        repo = _repo()
        repo.get_paper.side_effect = RuntimeError("connection to secret-host failed")
        response = self._detail(repo=repo)

        assert response.status_code == 500
        assert "secret-host" not in response.content.decode()

    def test_list_is_readable(self):
        repo = _repo()
        repo.list_recent_papers.return_value = [PAPER]
        with patch.object(services, "get_paper_repository", return_value=repo):
            response = page_views.paper_list_data(RequestFactory().get("/papers/list.json"))

        assert response.status_code == 200
        items = json.loads(response.content)["items"]
        assert [item["arxiv_id"] for item in items] == [ARXIV_ID]
        assert "is_favorited" not in items[0]

    def test_list_database_error_is_not_leaked(self):
        repo = _repo()
        repo.list_recent_papers.side_effect = RuntimeError("password=hunter2")
        with patch.object(services, "get_paper_repository", return_value=repo):
            response = page_views.paper_list_data(RequestFactory().get("/papers/list.json"))

        assert response.status_code == 500
        assert "hunter2" not in response.content.decode()

    def test_detail_page_shell_is_not_redirected(self):
        response = page_views.paper_detail(RequestFactory().get(f"/papers/{ARXIV_ID}/"), ARXIV_ID)

        assert response.status_code != 302


class TestBootstrapPayload:
    def test_payload_lists_summary_models_and_sets_csrf_cookie(self):
        response = api_views.bootstrap(RequestFactory().get("/bootstrap.json"))
        payload = json.loads(response.content)

        assert payload == {
            "default_summary_model": services.DEFAULT_SUMMARY_MODEL,
            "available_summary_models": list(services.AVAILABLE_SUMMARY_MODELS),
        }
        assert payload["default_summary_model"] in payload["available_summary_models"]
        assert "csrftoken" in response.cookies
