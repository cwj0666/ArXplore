from __future__ import annotations

import json
from unittest.mock import patch

from django.contrib.auth.models import AnonymousUser
from django.test import RequestFactory, SimpleTestCase

from papers import api_views

STREAM_PATH = "/papers/assistant/stream/"


class _AuthenticatedUser:
    is_authenticated = True
    id = 1

    def get_username(self) -> str:
        return "tester"


class AgentStreamViewTests(SimpleTestCase):
    def setUp(self) -> None:
        self.factory = RequestFactory()

    def _post_stream(self, *, user, message: str = "hello", api_key: str | None = "sk-user"):
        request = self.factory.post(
            STREAM_PATH,
            data=json.dumps({"message": message, "history": []}),
            content_type="application/json",
        )
        request.user = user
        request._dont_enforce_csrf_checks = True
        with patch.object(api_views, "get_session_api_key", return_value=api_key), patch.object(
            api_views, "stream_agent_chat"
        ) as stream_mock:
            response = api_views.paper_agent_stream(request)
        return response, stream_mock

    def test_unauthenticated_request_returns_401(self):
        response, stream_mock = self._post_stream(user=AnonymousUser())

        self.assertEqual(response.status_code, 401)
        self.assertIn("error", json.loads(response.content))
        stream_mock.assert_not_called()

    def test_missing_api_key_returns_400(self):
        response, stream_mock = self._post_stream(user=_AuthenticatedUser(), api_key=None)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.content)["error"], "개인 API 키를 먼저 등록하세요.")
        stream_mock.assert_not_called()

    def test_empty_message_returns_400(self):
        response, stream_mock = self._post_stream(user=_AuthenticatedUser(), message="   ")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.content)["error"], "메시지를 입력하세요.")
        stream_mock.assert_not_called()

    def test_valid_request_streams_sse_chunks(self):
        request = self.factory.post(
            STREAM_PATH,
            data=json.dumps({"message": "hello", "history": [{"role": "user", "content": "hi"}]}),
            content_type="application/json",
        )
        request.user = _AuthenticatedUser()

        with patch.object(api_views, "get_session_api_key", return_value="sk-user"), patch.object(
            api_views, "stream_agent_chat", return_value=iter(["a", "b"])
        ) as stream_mock:
            response = api_views.paper_agent_stream(request)
            body = b"".join(response.streaming_content).decode()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/event-stream")
        self.assertEqual(
            body,
            'data: {"chunk": "a"}\n\ndata: {"chunk": "b"}\n\ndata: [DONE]\n\n',
        )
        prepared = stream_mock.call_args.args[0]
        self.assertEqual(prepared.message, "hello")
        self.assertEqual(prepared.api_key, "sk-user")
        self.assertEqual(prepared.history, [("user", "hi")])


class PaperAnalyzeViewTests(SimpleTestCase):
    def setUp(self) -> None:
        self.factory = RequestFactory()

    def test_get_is_not_allowed(self):
        request = self.factory.get("/papers/2401.00001/analyze/")
        request.user = _AuthenticatedUser()

        with patch.object(api_views, "get_paper_analysis") as analysis_mock:
            response = api_views.paper_analyze(request, "2401.00001")

        self.assertEqual(response.status_code, 405)
        analysis_mock.assert_not_called()

    def test_post_calls_analysis(self):
        request = self.factory.post("/papers/2401.00001/analyze/")
        request.user = _AuthenticatedUser()
        payload = {"overview": "ok", "key_findings": [], "cached": True}

        with patch.object(api_views, "get_session_api_key", return_value="sk-user"), patch.object(
            api_views, "get_paper_analysis", return_value=payload
        ) as analysis_mock:
            response = api_views.paper_analyze(request, "2401.00001")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.content), payload)
        analysis_mock.assert_called_once()
