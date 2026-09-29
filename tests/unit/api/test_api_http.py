from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from arxplore.api.errors import (
    AI_UNAVAILABLE_MESSAGE,
    CSRF_FAILURE_MESSAGE,
    DATA_LOAD_ERROR_MESSAGE,
    GENERATION_ERROR_MESSAGE,
    INVALID_REQUEST_MESSAGE,
    PAPER_NOT_FOUND_MESSAGE,
    ROUTE_NOT_FOUND_MESSAGE,
    STREAM_ERROR_MESSAGE,
    UNSUPPORTED_MODEL_MESSAGE,
)
from arxplore.llm.detail import ANALYSIS_PROMPT_VERSION
from arxplore.llm.summary_graph import SUMMARY_PROMPT_VERSION
from arxplore.sources.arxiv import ArxivPaper

SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "same-origin",
    "cross-origin-opener-policy": "same-origin",
    "x-frame-options": "DENY",
}
LLM_POSTS = (
    "/papers/{main}/analyze/",
    "/papers/{main}/summary/",
    "/papers/{main}/chat/",
    "/papers/{main}/chat/stream/",
    "/papers/assistant/chat/",
    "/papers/assistant/stream/",
)


def assert_security_headers(response: Any) -> None:
    assert {name: response.headers.get(name) for name in SECURITY_HEADERS} == SECURITY_HEADERS


class TestBootstrapAndPages:
    async def test_bootstrap_lists_summary_models_and_sets_the_csrf_cookie(self, harness: Any) -> None:
        async with harness.client() as client:
            response = await client.get("/bootstrap.json")

        assert response.status_code == 200
        assert response.json() == {
            "default_summary_model": "gpt-5-mini",
            "available_summary_models": ["gpt-5-mini", "gpt-5"],
        }
        cookie = response.headers["set-cookie"]
        attributes = {part.strip().split("=", 1)[0].lower() for part in cookie.split(";")[1:]}
        assert "Max-Age=31449600" in cookie
        assert "Path=/" in cookie
        assert "samesite=lax" in cookie.lower()
        assert "httponly" not in attributes
        assert "secure" not in attributes
        assert_security_headers(response)

    async def test_secure_cookie_setting_marks_the_cookie_secure(self, harness: Any) -> None:
        async with harness.client(SECURE_COOKIES="true") as client:
            response = await client.get("/bootstrap.json")

        assert "secure" in response.headers["set-cookie"].lower().split("; ")

    async def test_bootstrap_renews_a_valid_cookie_and_replaces_an_invalid_one(self, harness: Any) -> None:
        token = "a" * 32
        async with harness.client() as client:
            kept = await client.get("/bootstrap.json", headers={"cookie": f"csrftoken={token}"})
            client.cookies.clear()
            replaced = await client.get("/bootstrap.json", headers={"cookie": "csrftoken=short"})

        assert kept.cookies["csrftoken"] == token
        assert replaced.cookies["csrftoken"] != "short"

    async def test_wrong_methods_get_an_empty_405_with_allow(self, harness: Any, ids: dict[str, str]) -> None:
        async with harness.client() as client:
            post_bootstrap = await harness.post(client, "/bootstrap.json", {})
            get_llm = [await client.get(path.format(**ids)) for path in LLM_POSTS]
            post_detail = await harness.post(client, f"/papers/{ids['main']}/detail.json", {})

        for response, allow in [(post_bootstrap, "GET"), *[(r, "POST") for r in get_llm], (post_detail, "GET")]:
            assert response.status_code == 405
            assert response.content == b""
            assert response.headers["allow"] == allow
            assert response.headers["content-type"].startswith("text/html")
            assert_security_headers(response)

    async def test_papers_alias_redirects_with_the_query(self, harness: Any) -> None:
        async with harness.client() as client:
            bare = await client.get("/papers/")
            with_query = await client.get("/papers/?q=rag&page=2&sort=upvotes")

        assert (bare.status_code, bare.headers["location"]) == (302, "/")
        assert (with_query.status_code, with_query.headers["location"]) == (302, "/?q=rag&page=2&sort=upvotes")
        assert with_query.content == b""

    async def test_unknown_routes_are_json_404_without_the_spa(self, harness: Any, ids: dict[str, str]) -> None:
        async with harness.client() as client:
            responses = [await client.get("/"), await client.get(f"/papers/{ids['main']}/"), await client.get("/nope")]

        for response in responses:
            assert response.status_code == 404
            assert response.json() == {"error": ROUTE_NOT_FOUND_MESSAGE}

    @pytest.mark.parametrize(
        "path",
        ["/admin/", "/login/", "/auth/login/", "/auth/signup/", "/settings/", "/settings/api-key/", "/favorites/"],
    )
    async def test_removed_account_routes_do_not_exist(self, harness: Any, path: str) -> None:
        async with harness.client() as client:
            get = await client.get(path)
            post = await harness.post(client, path, {})

        assert (get.status_code, get.json()) == (404, {"error": ROUTE_NOT_FOUND_MESSAGE})
        assert post.status_code == 404

    async def test_spa_fallback_serves_files_and_the_index(
        self, harness: Any, tmp_path: Path, ids: dict[str, str]
    ) -> None:
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
        (tmp_path / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
        (tmp_path.parent / "secret.txt").write_text("secret", encoding="utf-8")
        harness.frontend_dist = tmp_path

        async with harness.client(SERVE_FRONTEND_DIST="true") as client:
            index = await client.get(f"/papers/{ids['main']}/")
            asset = await client.get("/assets/app.js")
            traversal = await client.get("/assets/../../secret.txt")
            missing_asset = await client.get("/assets/missing.js")
            null_byte = await client.get("/assets/a%00b.js")
            api = await client.get("/papers/list.json")
            post = await client.post("/somewhere")

        assert (index.status_code, index.text) == (200, "<!doctype html><div id=root></div>")
        assert index.headers["content-type"] == "text/html; charset=utf-8"
        assert (asset.status_code, asset.text) == (200, "console.log(1)")
        assert "secret" not in traversal.text
        assert (missing_asset.status_code, missing_asset.json()) == (404, {"error": ROUTE_NOT_FOUND_MESSAGE})
        assert null_byte.status_code == 404
        assert api.headers["content-type"] == "application/json"
        assert post.status_code == 404

    async def test_missing_build_is_a_plain_503(self, harness: Any, tmp_path: Path) -> None:
        harness.frontend_dist = tmp_path / "missing"

        async with harness.client(SERVE_FRONTEND_DIST="true") as client:
            response = await client.get("/")

        assert response.status_code == 503
        assert response.headers["content-type"].startswith("text/plain")
        assert "npm run build" in response.text

    async def test_unknown_hosts_are_rejected_before_routing(self, harness: Any) -> None:
        async with harness.client() as client:
            response = await client.get("/bootstrap.json", headers={"host": "evil.example"})

        assert response.status_code == 400
        assert response.json() == {"error": INVALID_REQUEST_MESSAGE}
        assert_security_headers(response)


class TestCsrf:
    async def test_rejections_are_json_403(self, harness: Any, ids: dict[str, str]) -> None:
        path = f"/papers/{ids['main']}/analyze/"
        token = "a" * 32
        async with harness.client() as client:
            no_cookie = await client.post(path)
            no_header = await client.post(path, headers={"cookie": f"csrftoken={token}"})
            mismatch = await client.post(path, headers={"cookie": f"csrftoken={token}", "x-csrftoken": "b" * 32})
            bad_origin = await harness.post(client, path, headers={"origin": "http://evil.example"})
            trusted = await harness.post(client, path, headers={"origin": "http://localhost:5173"})

        for response in (no_cookie, no_header, mismatch, bad_origin):
            assert response.status_code == 403
            assert response.headers["content-type"] == "application/json"
            assert response.json() == {"error": CSRF_FAILURE_MESSAGE}
            assert_security_headers(response)
        assert trusted.status_code == 200

    async def test_rejected_requests_do_not_count_toward_the_rate_limit(
        self, harness: Any, ids: dict[str, str]
    ) -> None:
        async with harness.client(RATE_LIMIT_ENABLED="true", RATE_LIMIT_LLM_PER_MINUTE="1") as client:
            for _ in range(3):
                assert (await client.post(f"/papers/{ids['main']}/analyze/")).status_code == 403
            accepted = await harness.post(client, f"/papers/{ids['cached']}/analyze/")

        assert accepted.status_code == 200


class TestRateLimit:
    async def test_llm_endpoints_share_one_bucket_per_ip(self, harness: Any, ids: dict[str, str]) -> None:
        async with harness.client(RATE_LIMIT_ENABLED="true", RATE_LIMIT_LLM_PER_MINUTE="6") as client:
            statuses = [
                (await harness.post(client, path.format(main=ids["cached"]), {"message": ""})).status_code
                for path in LLM_POSTS
            ]
            limited = await harness.post(client, "/papers/assistant/stream/", {"message": "질문"})
            other_ip = await harness.post(
                client, f"/papers/{ids['cached']}/analyze/", headers={"x-real-ip": "203.0.113.8, 10.0.0.1"}
            )
            gets = [await client.get(f"/papers/{ids['main']}/analyze/") for _ in range(3)]

        assert statuses == [200, 400, 400, 400, 400, 400]
        assert limited.status_code == 429
        body = limited.json()
        assert body == {"error": "요청이 너무 많습니다. 60초 후 다시 시도하세요.", "retry_after": 60}
        assert limited.headers["retry-after"] == "60"
        assert other_ip.status_code == 200
        assert {response.status_code for response in gets} == {405}
        assert harness.llm_requests() == []

    async def test_detail_has_its_own_bucket_and_the_list_is_not_limited(
        self, harness: Any, ids: dict[str, str]
    ) -> None:
        async with harness.client(RATE_LIMIT_ENABLED="true", RATE_LIMIT_DETAIL_PER_MINUTE="2") as client:
            details = [(await client.get(f"/papers/{ids['other']}/detail.json")).status_code for _ in range(3)]
            analyze = await harness.post(client, f"/papers/{ids['cached']}/analyze/")
            lists = {(await client.get("/papers/list.json")).status_code for _ in range(5)}

        assert details == [200, 200, 429]
        assert analyze.status_code == 200
        assert lists == {200}

    async def test_a_new_minute_opens_a_new_window(self, harness: Any, ids: dict[str, str]) -> None:
        async with harness.client(RATE_LIMIT_ENABLED="true", RATE_LIMIT_DETAIL_PER_MINUTE="1") as client:
            first = await client.get(f"/papers/{ids['other']}/detail.json")
            second = await client.get(f"/papers/{ids['other']}/detail.json")
            harness.clock.now += 60
            third = await client.get(f"/papers/{ids['other']}/detail.json")

        assert [first.status_code, second.status_code, third.status_code] == [200, 429, 200]

    async def test_disabled_header_uses_the_socket_address(self, harness: Any, ids: dict[str, str]) -> None:
        async with harness.client(
            RATE_LIMIT_ENABLED="true", RATE_LIMIT_DETAIL_PER_MINUTE="1", RATE_LIMIT_IP_HEADER="none"
        ) as client:
            first = await client.get(f"/papers/{ids['other']}/detail.json", headers={"x-real-ip": "1.1.1.1"})
            second = await client.get(f"/papers/{ids['other']}/detail.json", headers={"x-real-ip": "2.2.2.2"})

        assert [first.status_code, second.status_code] == [200, 429]


class TestListAndDetail:
    async def test_list_payload(self, harness: Any, ids: dict[str, str]) -> None:
        async with harness.client() as client:
            response = await client.get("/papers/list.json?q=PREFERENCE&sort=upvotes&mode=ai&page=abc")

        payload = response.json()
        assert [item["arxiv_id"] for item in payload.pop("items")] == [ids["other"], ids["main"]]
        assert payload == {
            "page": 1,
            "page_size": 21,
            "total_items": 2,
            "total_pages": 1,
            "query": "PREFERENCE",
            "sort": "upvotes",
            "mode": "ai",
        }

    async def test_list_items_have_the_legacy_fields(self, harness: Any, helpers: ModuleType) -> None:
        harness.store.papers = [helpers.make_paper("x", published_at=None, pdf_url=None, upvotes=0, authors=[])]

        async with harness.client() as client:
            (item,) = (await client.get("/papers/list.json")).json()["items"]

        assert item == {
            "arxiv_id": "x",
            "title": "Paper x",
            "authors": [],
            "abstract": "Abstract of x.",
            "published_at": None,
            "upvotes": 0,
            "pdf_url": None,
        }

    async def test_list_errors_do_not_leak(self, harness: Any) -> None:
        harness.store.error = RuntimeError("password=hunter2")

        async with harness.client() as client:
            response = await client.get("/papers/list.json")

        assert response.status_code == 500
        assert response.json() == {"error": DATA_LOAD_ERROR_MESSAGE}

    async def test_detail_with_local_and_external_related_papers(self, harness: Any, ids: dict[str, str]) -> None:
        harness.arxiv.papers = [
            ArxivPaper(
                arxiv_id="2609.90001",
                title="Fake External",
                abstract="x",
                authors=["Cy"],
                published_at="2026-09-01T00:00:00Z",
                pdf_url="https://arxiv.org/pdf/2609.90001v1",
            )
        ]

        async with harness.client() as client:
            response = await client.get(f"/papers/{ids['main']}v3/detail.json")

        paper = response.json()["paper"]
        related = paper.pop("related_papers")
        assert paper["arxiv_id"] == ids["main"]
        assert paper["published_at"] == "2024-01-01T00:00:00+00:00"
        assert [(item["arxiv_id"], item["source"], item["relation_score"]) for item in related] == [
            (ids["other"], "local", 3.84),
            ("2609.90001", "arxiv", 0.0),
        ]
        assert harness.arxiv.queries == [("direct preference optimization revisited", 8)]
        assert related[-1] == {
            "arxiv_id": "2609.90001",
            "title": "Fake External",
            "authors": ["Cy"],
            "abstract": "x",
            "published_at": "2026-09-01T00:00:00Z",
            "upvotes": 0,
            "pdf_url": "https://arxiv.org/pdf/2609.90001v1",
            "source": "arxiv",
            "relation_score": 0.0,
        }

    async def test_external_search_results_are_reused_across_detail_requests(
        self, harness: Any, ids: dict[str, str]
    ) -> None:
        harness.arxiv.papers = [ArxivPaper(arxiv_id="2609.90001", title="Fake External", abstract="x")]

        async with harness.client() as client:
            first = await client.get(f"/papers/{ids['main']}/detail.json")
            second = await client.get(f"/papers/{ids['main']}/detail.json")

        assert first.json() == second.json()
        assert len(harness.arxiv.queries) == 1

    async def test_failed_external_searches_are_retried(self, harness: Any, ids: dict[str, str]) -> None:
        harness.arxiv.error = ConnectionError("arxiv down")

        async with harness.client() as client:
            await client.get(f"/papers/{ids['main']}/detail.json")
            await client.get(f"/papers/{ids['main']}/detail.json")

        assert len(harness.arxiv.queries) == 2

    async def test_detail_not_found_and_errors(self, harness: Any) -> None:
        async with harness.client() as client:
            missing = await client.get("/papers/2609.99999/detail.json")
            harness.store.error = RuntimeError("connection to secret-host failed")
            failing = await client.get("/papers/2609.99999/detail.json")

        assert (missing.status_code, missing.json()) == (404, {"error": PAPER_NOT_FOUND_MESSAGE})
        assert (failing.status_code, failing.json()) == (500, {"error": DATA_LOAD_ERROR_MESSAGE})


class TestGeneration:
    async def test_cached_overview_is_served_even_without_a_key(self, harness: Any, ids: dict[str, str]) -> None:
        async with harness.client(OPENAI_API_KEY=" ") as client:
            response = await harness.post(client, f"/papers/{ids['cached']}/analyze/", raw="not json")

        assert response.json() == {"overview": "cached overview", "key_findings": ["k1"], "cached": True}
        assert harness.llm_requests() == []

    async def test_missing_overview_is_generated_with_gpt5_mini_and_cached(
        self, harness: Any, ids: dict[str, str]
    ) -> None:
        harness.fake_openai.chat_handler = lambda request: (
            "- 첫 번째 핵심 포인트입니다" if "핵심 포인트" in request["messages"][0]["content"] else "새 개요"
        )

        async with harness.client() as client:
            first = await harness.post(client, f"/papers/{ids['main']}/analyze/")
            second = await harness.post(client, f"/papers/{ids['main']}/analyze/")

        assert first.json() == {"overview": "새 개요", "key_findings": ["첫 번째 핵심 포인트입니다."], "cached": False}
        assert second.json() == {**first.json(), "cached": True}
        assert [body["model"] for body in harness.llm_requests()] == ["gpt-5-mini", "gpt-5-mini"]
        assert harness.store.overviews[(ids["main"], ANALYSIS_PROMPT_VERSION)].model == "gpt-5-mini"

    async def test_results_cached_under_version_1_are_regenerated(
        self, harness: Any, helpers: ModuleType, ids: dict[str, str]
    ) -> None:
        stale = helpers.AiOverview(overview="stale", key_findings=[], model="gpt-5-mini", generated_at=None)
        harness.store.overviews = {(ids["main"], 1): stale}
        harness.store.summaries = {
            (ids["main"], "gpt-5", 1): helpers.DetailedSummary(summary="stale", model="gpt-5", generated_at=None)
        }

        async with harness.client() as client:
            overview = await harness.post(client, f"/papers/{ids['main']}/analyze/")
            summary = await harness.post(client, f"/papers/{ids['main']}/summary/", {"model": "gpt-5"})

        assert overview.json()["cached"] is False
        assert summary.json()["cached"] is False
        assert harness.store.overviews[(ids["main"], ANALYSIS_PROMPT_VERSION)].overview != "stale"
        assert harness.store.summaries[(ids["main"], "gpt-5", SUMMARY_PROMPT_VERSION)].summary != "stale"

    async def test_overview_errors(self, harness: Any, ids: dict[str, str]) -> None:
        async with harness.client(OPENAI_API_KEY=" ") as client:
            unknown = await harness.post(client, "/papers/2609.99999/analyze/")
            no_key = await harness.post(client, f"/papers/{ids['main']}/analyze/")
        harness.fake_openai.chat_handler = harness.fakes.default_chat_handler
        async with harness.client(OPENAI_API_KEY=harness.fake_openai.expected_api_key) as client:
            no_context = await harness.post(client, f"/papers/{ids['other']}/analyze/")

        assert (unknown.status_code, unknown.json()) == (404, {"error": PAPER_NOT_FOUND_MESSAGE})
        assert (no_key.status_code, no_key.json()) == (503, {"error": AI_UNAVAILABLE_MESSAGE})
        assert (no_context.status_code, no_context.json()) == (500, {"error": GENERATION_ERROR_MESSAGE})
        assert harness.llm_requests() == []

    async def test_summary_validation_order(self, harness: Any, ids: dict[str, str]) -> None:
        async with harness.client(OPENAI_API_KEY=" ") as client:
            responses = [
                await harness.post(client, f"/papers/{ids['main']}/summary/", raw="{not json"),
                await harness.post(client, f"/papers/{ids['main']}/summary/", [1, 2]),
                await harness.post(client, f"/papers/{ids['main']}/summary/", {}),
                await harness.post(client, "/papers/2609.99999/summary/", {"model": "gpt-4o"}),
                await harness.post(client, "/papers/2609.99999/summary/", {"model": "gpt-5-mini"}),
                await harness.post(client, f"/papers/{ids['cached']}/summary/", {"model": " gpt-5-mini "}),
                await harness.post(client, f"/papers/{ids['cached']}/summary/", {"model": "gpt-5"}),
            ]

        assert [(response.status_code, response.json()) for response in responses] == [
            (400, {"error": INVALID_REQUEST_MESSAGE}),
            (400, {"error": INVALID_REQUEST_MESSAGE}),
            (400, {"error": UNSUPPORTED_MODEL_MESSAGE}),
            (400, {"error": UNSUPPORTED_MODEL_MESSAGE}),
            (404, {"error": PAPER_NOT_FOUND_MESSAGE}),
            (200, {"summary": "cached summary", "cached": True, "model": "gpt-5-mini"}),
            (503, {"error": AI_UNAVAILABLE_MESSAGE}),
        ]

    async def test_summary_is_generated_per_model_and_cached(self, harness: Any, ids: dict[str, str]) -> None:
        async with harness.client() as client:
            first = await harness.post(client, f"/papers/{ids['main']}/summary/", {"model": "gpt-5"})
            again = await harness.post(client, f"/papers/{ids['main']}/summary/", {"model": "gpt-5"})

        assert first.json()["cached"] is False and first.json()["model"] == "gpt-5"
        assert again.json() == {**first.json(), "cached": True}
        assert {body["model"] for body in harness.llm_requests()} == {"gpt-5"}
        assert (
            harness.store.summaries[(ids["main"], "gpt-5", SUMMARY_PROMPT_VERSION)].summary == first.json()["summary"]
        )

    async def test_generation_failures_are_generic(self, harness: Any, ids: dict[str, str]) -> None:
        def fail(request: Any) -> Any:
            raise harness.fakes.FakeHTTPError(400, "postgres://user:pw@host")

        harness.fake_openai.chat_handler = fail
        async with harness.client() as client:
            analyze = await harness.post(client, f"/papers/{ids['main']}/analyze/")
            summary = await harness.post(client, f"/papers/{ids['main']}/summary/", {"model": "gpt-5-mini"})

        for response in (analyze, summary):
            assert (response.status_code, response.json()) == (500, {"error": GENERATION_ERROR_MESSAGE})
        assert (ids["main"], 1) not in harness.store.overviews


class TestChat:
    @pytest.mark.parametrize(
        "path",
        ["/papers/{main}/chat/", "/papers/{main}/chat/stream/", "/papers/assistant/chat/", "/papers/assistant/stream/"],
    )
    async def test_invalid_inputs_are_400_before_other_checks(
        self, harness: Any, ids: dict[str, str], path: str
    ) -> None:
        url = path.format(**ids)
        async with harness.client(OPENAI_API_KEY=" ") as client:
            responses = [
                await harness.post(client, url, {"message": "   "}),
                await harness.post(client, url, {"message": "\u0001\u0002\u007f"}),
                await harness.post(client, url, {"message": "가" * 4001}),
                await harness.post(client, url, {"message": "질문", "history": "not a list"}),
                await harness.post(client, url, {"history": []}),
                await harness.post(client, url, raw="{not json"),
                await harness.post(client, url, [1]),
            ]

        assert [response.json()["error"] for response in responses] == [
            "메시지를 입력하세요.",
            "메시지를 입력하세요.",
            "메시지는 4,000자 이하로 입력하세요.",
            INVALID_REQUEST_MESSAGE,
            "메시지를 입력하세요.",
            INVALID_REQUEST_MESSAGE,
            INVALID_REQUEST_MESSAGE,
        ]
        assert {response.status_code for response in responses} == {400}

    async def test_not_found_comes_before_the_key_check(self, harness: Any) -> None:
        async with harness.client(OPENAI_API_KEY=" ") as client:
            chat = await harness.post(client, "/papers/2609.99999/chat/", {"message": "질문"})
            stream = await harness.post(client, "/papers/2609.99999/chat/stream/", {"message": "질문"})
            agent = await harness.post(client, "/papers/assistant/stream/", {"message": "질문"})

        assert [(r.status_code, r.json()["error"]) for r in (chat, stream)] == [(404, PAPER_NOT_FOUND_MESSAGE)] * 2
        assert (agent.status_code, agent.json()) == (503, {"error": AI_UNAVAILABLE_MESSAGE})

    async def test_paper_chat_stream_bytes(self, harness: Any, helpers: ModuleType, ids: dict[str, str]) -> None:
        harness.fake_openai.chat_handler = lambda request: "근거는 [2]입니다."

        async with harness.client() as client:
            response = await harness.post(client, f"/papers/{ids['main']}/chat/stream/", {"message": "method?"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert (response.headers["cache-control"], response.headers["x-accel-buffering"]) == ("no-cache", "no")
        events = helpers.parse_sse(response.text)
        assert "".join(event["chunk"] for event in events[:-2]) == "근거는 [2]입니다."
        assert events[-2] == {
            "citations": [
                {
                    "arxiv_id": ids["main"],
                    "title": "Direct Preference Optimization Revisited",
                    "url": f"https://arxiv.org/pdf/{ids['main']}.pdf",
                    "section_title": "3 Method",
                    "chunk_id": 10,
                    "in_answer": True,
                }
            ]
        }
        assert events[-1] == "[DONE]"
        assert response.text.isascii()
        assert harness.retriever.calls == [("method?", 5, ids["main"])]

    async def test_paper_chat_json_reports_the_retrieval_mode(self, harness: Any, ids: dict[str, str]) -> None:
        harness.retriever.contexts = {}
        harness.fake_openai.chat_handler = lambda request: "답변"

        async with harness.client() as client:
            response = await harness.post(
                client,
                f"/papers/{ids['main']}v2/chat/",
                {"message": "\u0007질문\u0000", "history": [{"role": "user", "content": " 질문 "}]},
            )

        payload = response.json()
        assert (payload["answer"], payload["retrieval_mode"]) == ("답변", "first_chunks")
        assert [citation["chunk_id"] for citation in payload["citations"]] == [None, 1, 2]
        (body,) = harness.llm_requests()
        assert [message["role"] for message in body["messages"]] == ["system", "user"]
        assert body["messages"][1]["content"].endswith("## 질문\n질문")

    async def test_stream_errors_become_an_error_event(
        self, harness: Any, helpers: ModuleType, ids: dict[str, str]
    ) -> None:
        harness.retriever.error = RuntimeError("db down")

        async with harness.client() as client:
            stream = await harness.post(client, f"/papers/{ids['main']}/chat/stream/", {"message": "q"})
            plain = await harness.post(client, f"/papers/{ids['main']}/chat/", {"message": "q"})

        assert helpers.parse_sse(stream.text) == [{"error": STREAM_ERROR_MESSAGE}, "[DONE]"]
        assert (plain.status_code, plain.json()) == (500, {"error": STREAM_ERROR_MESSAGE})

    async def test_agent_stream_and_json(self, harness: Any, helpers: ModuleType) -> None:
        harness.fake_openai.chat_handler = lambda request: "도구 없이 답합니다."

        async with harness.client() as client:
            stream = await harness.post(
                client,
                "/papers/assistant/stream/",
                {"message": "안녕", "history": [{"role": "assistant", "content": "이전"}]},
            )
            plain = await harness.post(client, "/papers/assistant/chat/", {"message": "안녕"})

        events = helpers.parse_sse(stream.text)
        assert "".join(event["chunk"] for event in events[:-2]) == "도구 없이 답합니다."
        assert events[-2:] == [{"citations": []}, "[DONE]"]
        assert plain.json() == {"answer": "도구 없이 답합니다.", "citations": []}
        roles = [[message["role"] for message in body["messages"]] for body in harness.llm_requests()]
        assert roles == [["system", "assistant", "user"], ["system", "user"]]

    async def test_agent_model_failures(self, harness: Any, helpers: ModuleType) -> None:
        def fail(request: Any) -> Any:
            raise harness.fakes.FakeHTTPError(400, "simulated model failure")

        harness.fake_openai.chat_handler = fail
        async with harness.client() as client:
            stream = await harness.post(client, "/papers/assistant/stream/", {"message": "질문"})
            plain = await harness.post(client, "/papers/assistant/chat/", {"message": "질문"})

        assert helpers.parse_sse(stream.text) == [{"error": STREAM_ERROR_MESSAGE}, "[DONE]"]
        assert (plain.status_code, plain.json()) == (500, {"error": STREAM_ERROR_MESSAGE})
