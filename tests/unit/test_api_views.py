from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from django.test import RequestFactory, SimpleTestCase
from django.urls import resolve

from papers import api_views, services

STREAM_PATH = "/papers/assistant/stream/"
PAPER_STREAM_PATH = "/papers/2401.00001/chat/stream/"
CITATION = {
    "arxiv_id": "2401.00001",
    "title": "T",
    "url": "https://arxiv.org/abs/2401.00001",
    "section_title": "3 Method",
    "chunk_id": 10,
    "in_answer": True,
}


def _sse_payloads(body: str) -> list:
    events = []
    for block in body.split("\n\n"):
        if not block:
            continue
        assert block.startswith("data: ")
        data = block[len("data: ") :]
        events.append(data if data == "[DONE]" else json.loads(data))
    return events


class AgentStreamViewTests(SimpleTestCase):
    def setUp(self) -> None:
        self.factory = RequestFactory()

    def _post_stream(self, *, message: str = "hello"):
        request = self.factory.post(
            STREAM_PATH,
            data=json.dumps({"message": message, "history": []}),
            content_type="application/json",
        )
        with patch.object(api_views, "stream_agent_chat") as stream_mock:
            response = api_views.paper_agent_stream(request)
        return response, stream_mock

    def test_missing_server_api_key_returns_503(self):
        with patch.object(services, "get_runtime_openai_api_key", return_value=None):
            response, stream_mock = self._post_stream()

        self.assertEqual(response.status_code, 503)
        self.assertIn("error", json.loads(response.content))
        stream_mock.assert_not_called()

    def test_empty_message_returns_400(self):
        response, stream_mock = self._post_stream(message="   ")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(response.content)["error"], "메시지를 입력하세요.")
        stream_mock.assert_not_called()

    def test_valid_request_streams_sse_chunks(self):
        request = self.factory.post(
            STREAM_PATH,
            data=json.dumps({"message": "hello", "history": [{"role": "user", "content": "hi"}]}),
            content_type="application/json",
        )

        with patch.object(api_views, "stream_agent_chat", return_value=iter(["a", "b"])) as stream_mock:
            response = api_views.paper_agent_stream(request)
            body = b"".join(response.streaming_content).decode()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/event-stream")
        self.assertEqual(
            body,
            'data: {"chunk": "a"}\n\ndata: {"chunk": "b"}\n\ndata: {"citations": []}\n\ndata: [DONE]\n\n',
        )
        prepared = stream_mock.call_args.args[0]
        self.assertEqual(prepared.message, "hello")
        self.assertEqual(prepared.history, [("user", "hi")])

    def test_citations_event_is_sent_once_before_done(self):
        request = self.factory.post(
            STREAM_PATH, data=json.dumps({"message": "hello", "history": []}), content_type="application/json"
        )
        events = iter([{"chunk": "a"}, {"citations": [CITATION]}])

        with patch.object(api_views, "stream_agent_chat", return_value=events):
            response = api_views.paper_agent_stream(request)
            payloads = _sse_payloads(b"".join(response.streaming_content).decode())

        self.assertEqual(payloads, [{"chunk": "a"}, {"citations": [CITATION]}, "[DONE]"])

    def test_error_during_stream_emits_error_then_done_without_citations(self):
        request = self.factory.post(
            STREAM_PATH, data=json.dumps({"message": "hello", "history": []}), content_type="application/json"
        )

        def failing():
            yield {"chunk": "a"}
            raise RuntimeError("db password leaked in message")

        with patch.object(api_views, "stream_agent_chat", return_value=failing()):
            response = api_views.paper_agent_stream(request)
            payloads = _sse_payloads(b"".join(response.streaming_content).decode())

        self.assertEqual(payloads[0], {"chunk": "a"})
        self.assertEqual(list(payloads[1]), ["error"])
        self.assertNotIn("password", payloads[1]["error"])
        self.assertEqual(payloads[2], "[DONE]")
        self.assertEqual(len(payloads), 3)


class PaperChatStreamViewTests(SimpleTestCase):
    def setUp(self) -> None:
        self.factory = RequestFactory()

    def _post(self, *, body: dict | None = None, paper: dict | None = None, events=None):
        request = self.factory.post(
            PAPER_STREAM_PATH,
            data=json.dumps(body if body is not None else {"message": "loss?", "history": []}),
            content_type="application/json",
        )
        repo = MagicMock()
        repo.get_paper.return_value = paper
        with (
            patch.object(services, "get_paper_repository", return_value=repo),
            patch.object(api_views, "stream_paper_chat", return_value=iter(events or [])) as stream_mock,
        ):
            response = api_views.paper_chat_stream(request, "2401.00001")
            content = b"".join(response.streaming_content).decode() if response.streaming else response.content.decode()
        return response, content, stream_mock

    def test_route_resolves_to_stream_view(self):
        self.assertIs(resolve(PAPER_STREAM_PATH).func, api_views.paper_chat_stream)
        self.assertIs(resolve("/papers/2401.00001/chat/").func, api_views.paper_chat)

    def test_get_is_not_allowed(self):
        request = self.factory.get(PAPER_STREAM_PATH)

        self.assertEqual(api_views.paper_chat_stream(request, "2401.00001").status_code, 405)

    def test_missing_server_api_key_returns_503(self):
        with patch.object(services, "get_runtime_openai_api_key", return_value="  "):
            response, content, stream_mock = self._post(paper={"arxiv_id": "2401.00001"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("error", json.loads(content))
        stream_mock.assert_not_called()

    def test_invalid_body_returns_400(self):
        for body in (
            {"message": "   ", "history": []},
            {"message": "hi", "history": "nope"},
            {"message": "x" * 4001, "history": []},
        ):
            response, _, stream_mock = self._post(body=body, paper={"arxiv_id": "2401.00001"})
            self.assertEqual(response.status_code, 400, body)
            stream_mock.assert_not_called()

    def test_unknown_paper_returns_404(self):
        response, content, stream_mock = self._post(paper=None)

        self.assertEqual(response.status_code, 404)
        self.assertIn("error", json.loads(content))
        stream_mock.assert_not_called()

    def test_valid_request_streams_chunks_citations_done(self):
        paper = {"arxiv_id": "2401.00001", "title": "T", "abstract": "A"}
        events = [{"chunk": "답"}, {"chunk": "변 [2]"}, {"citations": [CITATION]}]

        response, content, stream_mock = self._post(
            body={"message": "loss?", "history": [{"role": "assistant", "content": "prev"}]},
            paper=paper,
            events=events,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/event-stream")
        self.assertEqual(
            _sse_payloads(content), [{"chunk": "답"}, {"chunk": "변 [2]"}, {"citations": [CITATION]}, "[DONE]"]
        )
        prepared = stream_mock.call_args.args[0]
        self.assertEqual(prepared.paper, paper)
        self.assertEqual(prepared.message, "loss?")
        self.assertEqual(prepared.history, [("assistant", "prev")])

    def test_current_question_echoed_in_history_is_dropped(self):
        paper = {"arxiv_id": "2401.00001", "title": "T", "abstract": "A"}
        history = [
            {"role": "user", "content": "first?"},
            {"role": "assistant", "content": "prev"},
            {"role": "user", "content": " loss? "},
        ]

        _, _, stream_mock = self._post(
            body={"message": "loss?", "history": history},
            paper=paper,
            events=[{"chunk": "ok"}],
        )

        prepared = stream_mock.call_args.args[0]
        self.assertEqual(prepared.history, [("user", "first?"), ("assistant", "prev")])


class ChatHistoryTupleTests(SimpleTestCase):
    def test_only_trailing_user_turn_matching_current_message_is_dropped(self):
        history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
        self.assertEqual(
            services._build_history_tuples([*history, {"role": "user", "content": "q"}], current_message="q"),
            [("user", "q"), ("assistant", "a")],
        )
        self.assertEqual(
            services._build_history_tuples([*history, {"role": "user", "content": "other"}], current_message="q"),
            [("user", "q"), ("assistant", "a"), ("user", "other")],
        )
        self.assertEqual(
            services._build_history_tuples(history, current_message="a"),
            [("user", "q"), ("assistant", "a")],
        )


class PaperChatViewTests(SimpleTestCase):
    def test_non_streaming_chat_returns_answer_and_citations(self):
        request = RequestFactory().post(
            "/papers/2401.00001/chat/",
            data=json.dumps({"message": "loss?", "history": []}),
            content_type="application/json",
        )
        payload = {"answer": "답변", "citations": [CITATION], "retrieval_mode": "hybrid"}

        with patch.object(api_views, "answer_paper_chat", return_value=payload):
            response = api_views.paper_chat(request, "2401.00001")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.content), payload)


class PaperAnalyzeViewTests(SimpleTestCase):
    def setUp(self) -> None:
        self.factory = RequestFactory()

    def test_get_is_not_allowed(self):
        request = self.factory.get("/papers/2401.00001/analyze/")

        with patch.object(api_views, "get_paper_analysis") as analysis_mock:
            response = api_views.paper_analyze(request, "2401.00001")

        self.assertEqual(response.status_code, 405)
        analysis_mock.assert_not_called()

    def test_post_calls_analysis(self):
        request = self.factory.post("/papers/2401.00001/analyze/")
        payload = {"overview": "ok", "key_findings": [], "cached": True}

        with patch.object(api_views, "get_paper_analysis", return_value=payload) as analysis_mock:
            response = api_views.paper_analyze(request, "2401.00001")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.content), payload)
        analysis_mock.assert_called_once_with("2401.00001")


class PaperSummaryViewTests(SimpleTestCase):
    def test_get_is_not_allowed(self):
        request = RequestFactory().get("/papers/2401.00001/summary/")

        with patch.object(api_views, "get_paper_summary") as summary_mock:
            response = api_views.paper_summary(request, "2401.00001")

        self.assertEqual(response.status_code, 405)
        summary_mock.assert_not_called()

    def test_unexpected_error_is_not_leaked(self):
        request = RequestFactory().post(
            "/papers/2401.00001/summary/", data=json.dumps({"model": "gpt-5-mini"}), content_type="application/json"
        )

        with (
            patch.object(api_views, "get_paper_summary", side_effect=ValueError("internal detail /srv/secret")),
            self.assertLogs("papers.api_views", level="ERROR"),
        ):
            response = api_views.paper_summary(request, "2401.00001")

        self.assertEqual(response.status_code, 500)
        self.assertNotIn("/srv/secret", response.content.decode())
