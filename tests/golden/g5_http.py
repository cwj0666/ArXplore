import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from tests.fakes.openai_server import (
    ARXIV_QUERY_PATH,
    CHAT_COMPLETIONS_PATH,
    EMBEDDINGS_PATH,
    ArxivFeed,
    ChatReply,
    EmbeddingVectors,
    FakeHTTPError,
    FakeOpenAIServer,
    RecordedRequest,
    ToolCall,
    default_chat_handler,
)
from tests.golden.adapters import HttpAdapter, HttpResult
from tests.golden.store import DATA_DIR, diff

FORMAT = 1
GOLDEN_PATH = DATA_DIR / "g5_http.json.gz"
SPA_DIST = DATA_DIR / "g5_spa"
MAIN_PAPER = "2608.30597"
CACHED_PAPER = "2609.28470"
UNKNOWN_PAPER = "2609.99999"
SYNTHETIC_PAPER = "2609.99001"
SEARCH_QUERY = "retrieval augmented generation benchmark"
HEADER_ALLOWLIST = (
    "allow",
    "cache-control",
    "content-type",
    "cross-origin-opener-policy",
    "location",
    "referrer-policy",
    "retry-after",
    "set-cookie",
    "vary",
    "x-accel-buffering",
    "x-content-type-options",
    "x-frame-options",
)
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
RETRY_AFTER_PLACEHOLDER = "<retry_after>"
UNSET = object()


@dataclass(frozen=True)
class Step:
    method: str
    path: str
    payload: Any = UNSET
    raw: str | None = None
    csrf: bool = True
    headers: tuple[tuple[str, str], ...] = ()
    repeat: int = 1

    def body(self) -> bytes | None:
        if self.raw is not None:
            return self.raw.encode("utf-8")
        if self.payload is UNSET:
            return None
        return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")

    def describe(self) -> dict[str, Any]:
        described: dict[str, Any] = {"method": self.method, "path": self.path}
        if self.headers:
            described["headers"] = [list(pair) for pair in self.headers]
        if self.raw is not None:
            described["raw"] = self.raw
        elif self.payload is not UNSET:
            described["json"] = self.payload
        if not self.csrf:
            described["csrf"] = False
        return described


@dataclass(frozen=True)
class Scenario:
    name: str
    steps: tuple[Step, ...]
    chat: str = "default"
    embeddings: str = "ok"
    arxiv: str = "default"
    openai_key: bool = True
    llm_layout: str = "sequence"
    fresh_minute: bool = False


@dataclass(frozen=True)
class Bundle:
    name: str
    scenarios: tuple[Scenario, ...]
    setup_sql: tuple[str, ...] = field(default=())


def get(path: str, **options: Any) -> Step:
    return Step("GET", path, **options)


def post(path: str, payload: Any = UNSET, **options: Any) -> Step:
    return Step("POST", path, payload, **options)


def _messages(request: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    messages = request.get("messages")
    return [message for message in messages if isinstance(message, Mapping)] if isinstance(messages, list) else []


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, Mapping))
    return ""


def system_text(request: Mapping[str, Any]) -> str:
    return next(
        (_text(message.get("content")) for message in _messages(request) if message.get("role") == "system"), ""
    )


def _after_tool(request: Mapping[str, Any]) -> str | None:
    messages = _messages(request)
    if messages and messages[-1].get("role") == "tool":
        return _text(messages[-1].get("content"))
    return None


def _first_search_hit(tool_output: str) -> tuple[str, str]:
    title = re.search(r"^\[1\] 제목: (.*?) \| arxiv_id:", tool_output, re.MULTILINE)
    url = re.search(r"^출처\(URL\): (\S+)", tool_output, re.MULTILINE)
    return (title.group(1) if title else "제목 없음", url.group(1) if url else "https://arxiv.org/abs/0000.00000")


def _first_trending_hit(tool_output: str) -> tuple[str, str]:
    match = re.search(r"^\[1\] (.*?) \| URL: (\S+)", tool_output, re.MULTILINE)
    return (match.group(1), match.group(2)) if match else ("제목 없음", "https://arxiv.org/abs/0000.00000")


SEARCH_CALL = ToolCall(name="search_paper_chunks_tool", arguments=json.dumps({"query": SEARCH_QUERY}))
TRENDING_CALL = ToolCall(name="get_trending_papers_tool", arguments="{}")
LONG_PREAMBLE = "먼저 관련 논문을 찾아보겠습니다. " * 8
KEY_FINDINGS_REPLY = "\n".join(
    [
        "- 3.5% 향상을 보고합니다.",
        "3.5% 정확도 향상이 이 논문의 핵심 결과입니다",
        "- 짧음",
        "1) 두 번째 핵심 포인트는 충분히 길게 적었습니다",
        "• 세 번째 핵심 포인트: 결과 해석",
        "• 세 번째 핵심 포인트: 결과 해석",
        "* 다섯 번째 포인트는 문장부호가 없습니다",
        "- 여섯 번째 포인트입니다!",
        "- 일곱 번째 포인트입니다?",
        "- 여덟 번째 포인트입니다.",
    ]
)
OVERVIEW_REPLY = (
    "이 논문은 잡음이 섞인 선호 데이터에서 정책을 정렬하는 방법을 다룹니다.\n\n"
    "두 번째 문단에서는 실험 설정과 결과를 설명합니다. 결과는 기존 방법보다 안정적입니다."
)
NUMBERED_REPLY = (
    "요청하신 내용을 정리했습니다 [1]. 방법은 [2, 3]에서 설명합니다. 범위 밖 번호 [9]도 섞었습니다. "
    "링크 텍스트 [4](https://example.com/four)는 인용이 아닙니다."
)


def _agent_link(title: str, url: str) -> str:
    return f"검색 결과를 정리했습니다.\n\n- **[{title}]({url})** — 질문과 관련된 논문입니다."


def reply_key_findings_or_overview(request: Mapping[str, Any]) -> Any:
    return KEY_FINDINGS_REPLY if "핵심 포인트" in system_text(request) else OVERVIEW_REPLY


def reply_model_error(request: Mapping[str, Any]) -> Any:
    raise FakeHTTPError(400, "simulated model failure", error_type="invalid_request_error")


def reply_final_summary_error(request: Mapping[str, Any]) -> Any:
    if "깊이 있게 분석" in system_text(request):
        raise FakeHTTPError(400, "simulated final summary failure", error_type="invalid_request_error")
    return default_chat_handler(request)


def reply_abort_midway(request: Mapping[str, Any]) -> Any:
    return ChatReply(content="중간에 끊기는 답변입니다. " * 3, abort_after_events=4)


def reply_search_then_link(request: Mapping[str, Any]) -> Any:
    tool_output = _after_tool(request)
    return SEARCH_CALL if tool_output is None else _agent_link(*_first_search_hit(tool_output))


def reply_search_then_plain(request: Mapping[str, Any]) -> Any:
    return (
        SEARCH_CALL
        if _after_tool(request) is None
        else "검색 결과에 관련 논문이 여러 편 있습니다. 링크 없이 요약합니다."
    )


def reply_search_then_foreign_link(request: Mapping[str, Any]) -> Any:
    if _after_tool(request) is None:
        return SEARCH_CALL
    return "- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** — 검색 결과에 없는 논문입니다."


def reply_long_preamble_then_link(request: Mapping[str, Any]) -> Any:
    tool_output = _after_tool(request)
    if tool_output is None:
        return ChatReply(content=LONG_PREAMBLE, tool_calls=(SEARCH_CALL,))
    return _agent_link(*_first_search_hit(tool_output))


def reply_short_preamble_then_link(request: Mapping[str, Any]) -> Any:
    tool_output = _after_tool(request)
    if tool_output is None:
        return ChatReply(content="검색하겠습니다.", tool_calls=(SEARCH_CALL,))
    return _agent_link(*_first_search_hit(tool_output))


def reply_always_search(request: Mapping[str, Any]) -> Any:
    return SEARCH_CALL


def reply_empty(request: Mapping[str, Any]) -> Any:
    return ChatReply(content="")


def reply_trending_then_link(request: Mapping[str, Any]) -> Any:
    tool_output = _after_tool(request)
    return TRENDING_CALL if tool_output is None else _agent_link(*_first_trending_hit(tool_output))


def reply_direct_long_answer(request: Mapping[str, Any]) -> Any:
    return "도구 없이 바로 답합니다. " * 12


REPLY_SCRIPTS: dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "default": default_chat_handler,
    "numbered": lambda request: NUMBERED_REPLY,
    "plain": lambda request: "제공된 발췌문으로는 답하기 어렵습니다.",
    "overview": reply_key_findings_or_overview,
    "model_error": reply_model_error,
    "final_summary_error": reply_final_summary_error,
    "abort_midway": reply_abort_midway,
    "search_then_link": reply_search_then_link,
    "search_then_plain": reply_search_then_plain,
    "search_then_foreign_link": reply_search_then_foreign_link,
    "long_preamble_then_link": reply_long_preamble_then_link,
    "short_preamble_then_link": reply_short_preamble_then_link,
    "always_search": reply_always_search,
    "empty": reply_empty,
    "trending_then_link": reply_trending_then_link,
    "direct_long_answer": reply_direct_long_answer,
}


def _fail_embeddings(request: Mapping[str, Any]) -> list[list[float]]:
    raise FakeHTTPError(401, "simulated embedding failure", error_type="invalid_api_key")


def _fail_arxiv(query: Mapping[str, list[str]]) -> str:
    raise FakeHTTPError(503, "simulated arXiv outage")


def _history(count: int) -> list[Any]:
    history: list[Any] = [
        {"role": "system", "content": "무시되는 system 메시지"},
        "not a dict",
        {"role": "user", "content": 42},
        {"role": "assistant", "content": "   "},
        {"role": "tool", "content": "무시되는 도구 메시지"},
    ]
    for index in range(count):
        role = "user" if index % 2 == 0 else "assistant"
        history.append({"role": role, "content": f"{role} 메시지 {index}"})
    history.append({"role": "assistant", "content": "긴 답변 " + "가" * 4100})
    return history


HISTORY_QUESTION = "이 논문의 방법을 요약해 주세요."
HISTORY_WITH_REPEAT = [*_history(24), {"role": "user", "content": f"  {HISTORY_QUESTION}  "}]
CHAT_PATHS = (f"/papers/{MAIN_PAPER}/chat/", f"/papers/{MAIN_PAPER}/chat/stream/")
AGENT_PATHS = ("/papers/assistant/chat/", "/papers/assistant/stream/")
LLM_PATHS = (
    f"/papers/{MAIN_PAPER}/analyze/",
    f"/papers/{MAIN_PAPER}/summary/",
    *CHAT_PATHS,
    *AGENT_PATHS,
)


def _both(paths: tuple[str, ...], payload: Any, **options: Any) -> tuple[Step, ...]:
    return tuple(post(path, payload, **options) for path in paths)


PAGES = Bundle(
    "pages",
    (
        Scenario("bootstrap", (get("/bootstrap.json"),)),
        Scenario("bootstrap_rejects_post", (post("/bootstrap.json", {}),)),
        Scenario("list_default", (get("/papers/list.json"),)),
        Scenario("list_query", (get("/papers/list.json?q=diffusion"),)),
        Scenario("list_query_upvotes_mode", (get("/papers/list.json?q=language&sort=upvotes&mode=ai"),)),
        Scenario("list_query_case_and_spaces", (get("/papers/list.json?q=%20%20Diffusion%20Models%20"),)),
        Scenario("list_query_word_order", (get("/papers/list.json?q=models%20diffusion"),)),
        Scenario("list_page_two", (get("/papers/list.json?page=2"),)),
        Scenario("list_page_out_of_range", (get("/papers/list.json?page=999"),)),
        Scenario(
            "list_page_invalid",
            (get("/papers/list.json?page=abc"), get("/papers/list.json?page=0"), get("/papers/list.json?page=-3")),
        ),
        Scenario("list_zero_results", (get("/papers/list.json?q=zzqqxx-no-such-term"),)),
        Scenario("list_unknown_sort_and_mode", (get("/papers/list.json?sort=oldest&mode=weird"),)),
        Scenario("list_korean_query", (get("/papers/list.json?q=%EB%AA%A8%EB%8D%B8"),)),
        Scenario("list_rejects_post", (post("/papers/list.json", {}),)),
        Scenario(
            "spa_shell",
            (get("/"), get("/?q=rag"), get("/papers/assistant/"), get(f"/papers/{MAIN_PAPER}/")),
        ),
        Scenario("papers_redirect", (get("/papers/"), get("/papers/?q=rag&page=2&sort=upvotes"))),
        Scenario(
            "csrf_failures",
            (
                post(f"/papers/{MAIN_PAPER}/analyze/", csrf=False),
                post(f"/papers/{MAIN_PAPER}/analyze/", csrf=False, headers=(("cookie", "csrftoken=" + "a" * 32),)),
                post(
                    f"/papers/{MAIN_PAPER}/analyze/",
                    csrf=False,
                    headers=(("cookie", "csrftoken=" + "a" * 32), ("x-csrftoken", "b" * 32)),
                ),
                post(f"/papers/{UNKNOWN_PAPER}/analyze/", headers=(("origin", "http://evil.example"),)),
                post(f"/papers/{UNKNOWN_PAPER}/analyze/", headers=(("origin", "http://localhost:5173"),)),
            ),
        ),
        Scenario(
            "llm_endpoints_reject_get",
            tuple(get(path) for path in LLM_PATHS),
        ),
    ),
)

DETAIL = Bundle(
    "detail",
    (
        Scenario("detail_found", (get(f"/papers/{MAIN_PAPER}/detail.json"),)),
        Scenario("detail_not_found", (get(f"/papers/{UNKNOWN_PAPER}/detail.json"),)),
        Scenario(
            "detail_id_normalization",
            (get(f"/papers/{MAIN_PAPER}v3/detail.json"), get(f"/papers/{MAIN_PAPER}.pdf/detail.json")),
        ),
        Scenario("detail_external_related", (get(f"/papers/{SYNTHETIC_PAPER}/detail.json"),)),
        Scenario("detail_external_failure", (get(f"/papers/{SYNTHETIC_PAPER}/detail.json"),), arxiv="fail"),
        Scenario("detail_rejects_post", (post(f"/papers/{MAIN_PAPER}/detail.json", {}),)),
    ),
    setup_sql=(
        f"""
        INSERT INTO papers (arxiv_id, title, authors, abstract, primary_category, categories, pdf_url,
                            published_at, updated_at, upvotes, source)
        VALUES ('{SYNTHETIC_PAPER}', 'Zqxvbnm Wplkjhg Yrtcvbn', '["Golden Harness"]'::jsonb,
                'Qzxw vbnmq plkjh ytrew.', NULL, '[]'::jsonb, 'https://arxiv.org/pdf/{SYNTHETIC_PAPER}',
                '2026-09-01T00:00:00+00', '2026-09-01T00:00:00+00', 0, 'hf_daily_papers')
        """,
    ),
)

ANALYZE = Bundle(
    "analyze",
    (
        Scenario("analyze_cached", (post(f"/papers/{CACHED_PAPER}/analyze/"),)),
        Scenario(
            "analyze_generate_then_cached",
            (post(f"/papers/{MAIN_PAPER}/analyze/", {"ignored": True}), post(f"/papers/{MAIN_PAPER}/analyze/")),
            chat="overview",
        ),
        Scenario("analyze_not_found", (post(f"/papers/{UNKNOWN_PAPER}/analyze/"),)),
        Scenario(
            "analyze_without_key",
            (
                post(f"/papers/{UNKNOWN_PAPER}/analyze/"),
                post(f"/papers/{CACHED_PAPER}/analyze/"),
                post("/papers/2609.26796/analyze/"),
            ),
            openai_key=False,
        ),
        Scenario("analyze_model_failure", (post("/papers/2609.26796/analyze/"),), chat="model_error"),
    ),
)

SUMMARY = Bundle(
    "summary",
    (
        Scenario(
            "summary_invalid_requests",
            (
                post(f"/papers/{MAIN_PAPER}/summary/", raw="{not json"),
                post(f"/papers/{MAIN_PAPER}/summary/", [1, 2]),
                post(f"/papers/{MAIN_PAPER}/summary/", {}),
                post(f"/papers/{MAIN_PAPER}/summary/", {"model": "gpt-4o"}),
                post(f"/papers/{UNKNOWN_PAPER}/summary/", {"model": "gpt-4o"}),
                post(f"/papers/{UNKNOWN_PAPER}/summary/", {"model": "gpt-5-mini"}),
            ),
        ),
        Scenario(
            "summary_cached",
            (post(f"/papers/{CACHED_PAPER}/summary/", {"model": "gpt-5-mini"}),),
            llm_layout="summary_graph",
        ),
        Scenario(
            "summary_generate_gpt5_mini",
            (
                post(f"/papers/{MAIN_PAPER}/summary/", {"model": "gpt-5-mini"}),
                post(f"/papers/{MAIN_PAPER}/summary/", {"model": "gpt-5-mini"}),
            ),
            llm_layout="summary_graph",
        ),
        Scenario(
            "summary_generate_gpt5",
            (post(f"/papers/{MAIN_PAPER}/summary/", {"model": " gpt-5 "}),),
            llm_layout="summary_graph",
        ),
        Scenario(
            "summary_without_key",
            (
                post(f"/papers/{CACHED_PAPER}/summary/", {"model": "gpt-5-mini"}),
                post(f"/papers/{CACHED_PAPER}/summary/", {"model": "gpt-5"}),
                post(f"/papers/{MAIN_PAPER}/summary/", {"model": "gpt-4o"}),
            ),
            openai_key=False,
        ),
        Scenario(
            "summary_model_failure",
            (post("/papers/2609.26796/summary/", {"model": "gpt-5-mini"}),),
            chat="final_summary_error",
            llm_layout="summary_graph",
        ),
    ),
)

PAPER_CHAT = Bundle(
    "paper_chat",
    (
        Scenario(
            "chat_invalid_requests",
            (
                *_both(CHAT_PATHS, {"message": "   "}),
                *_both(CHAT_PATHS, {"message": "\u0001\u0002\u007f"}),
                *_both(CHAT_PATHS, {"message": "가" * 4001}),
                *_both(CHAT_PATHS, {"message": "질문", "history": "not a list"}),
                *_both(CHAT_PATHS, {"history": []}),
                post(CHAT_PATHS[1], raw="{not json"),
                post(CHAT_PATHS[1], [1]),
                post(f"/papers/{UNKNOWN_PAPER}/chat/", {"message": "질문"}),
                post(f"/papers/{UNKNOWN_PAPER}/chat/stream/", {"message": "질문"}),
            ),
        ),
        Scenario(
            "chat_without_key",
            (
                *_both(CHAT_PATHS, {"message": "방법을 설명해 주세요."}),
                post(CHAT_PATHS[1], {"message": ""}),
                post(f"/papers/{UNKNOWN_PAPER}/chat/stream/", {"message": "질문"}),
            ),
            openai_key=False,
        ),
        Scenario(
            "chat_numbered_citations",
            _both(CHAT_PATHS, {"message": "What is the proposed method?"}),
            chat="numbered",
        ),
        Scenario(
            "chat_without_citations",
            _both(CHAT_PATHS, {"message": "What is the proposed method?"}),
            chat="plain",
        ),
        Scenario(
            "chat_history_normalization",
            _both(CHAT_PATHS, {"message": f"\u0007{HISTORY_QUESTION}\u0000", "history": HISTORY_WITH_REPEAT}),
            chat="numbered",
        ),
        Scenario(
            "chat_message_at_limit",
            (post(CHAT_PATHS[1], {"message": "가" * 3999 + "?"}),),
            chat="plain",
        ),
        Scenario(
            "chat_first_chunks_fallback",
            _both(CHAT_PATHS, {"message": "이 논문이 해결하려는 문제를 한 문장으로 말해줘"}),
            chat="numbered",
            embeddings="fail",
        ),
        Scenario(
            "chat_version_suffix",
            (post(f"/papers/{MAIN_PAPER}v2/chat/stream/", {"message": "What is the proposed method?"}),),
            chat="numbered",
        ),
        Scenario(
            "chat_model_failure",
            _both(CHAT_PATHS, {"message": "What is the proposed method?"}),
            chat="model_error",
        ),
        Scenario(
            "chat_stream_aborted_midway",
            _both(CHAT_PATHS, {"message": "What is the proposed method?"}),
            chat="abort_midway",
        ),
    ),
)

AGENT = Bundle(
    "agent",
    (
        Scenario(
            "agent_invalid_requests",
            (
                *_both(AGENT_PATHS, {"message": ""}),
                *_both(AGENT_PATHS, {"message": "가" * 4001}),
                *_both(AGENT_PATHS, {"message": "질문", "history": {"role": "user"}}),
                post(AGENT_PATHS[1], raw="[]"),
            ),
        ),
        Scenario("agent_without_key", (*_both(AGENT_PATHS, {"message": "RAG 논문 추천해줘"}),), openai_key=False),
        Scenario(
            "agent_search_then_link", _both(AGENT_PATHS, {"message": "RAG 논문 추천해줘"}), chat="search_then_link"
        ),
        Scenario(
            "agent_search_then_plain", _both(AGENT_PATHS, {"message": "RAG 논문 추천해줘"}), chat="search_then_plain"
        ),
        Scenario(
            "agent_link_not_in_results",
            _both(AGENT_PATHS, {"message": "RAG 논문 추천해줘"}),
            chat="search_then_foreign_link",
        ),
        Scenario(
            "agent_long_preamble_leaks",
            _both(AGENT_PATHS, {"message": "RAG 논문 추천해줘"}),
            chat="long_preamble_then_link",
        ),
        Scenario(
            "agent_short_preamble_dropped",
            _both(AGENT_PATHS, {"message": "RAG 논문 추천해줘"}),
            chat="short_preamble_then_link",
        ),
        Scenario("agent_step_limit", _both(AGENT_PATHS, {"message": "RAG 논문 추천해줘"}), chat="always_search"),
        Scenario("agent_empty_answer", _both(AGENT_PATHS, {"message": "RAG 논문 추천해줘"}), chat="empty"),
        Scenario(
            "agent_trending_then_link",
            _both(AGENT_PATHS, {"message": "요즘 뜨는 논문 알려줘"}),
            chat="trending_then_link",
        ),
        Scenario(
            "agent_direct_answer",
            _both(
                AGENT_PATHS,
                {
                    "message": "안녕",
                    "history": [
                        {"role": "user", "content": "이전 질문"},
                        {"role": "assistant", "content": "이전 답변"},
                    ],
                },
            ),
            chat="direct_long_answer",
        ),
        Scenario(
            "agent_search_embedding_failure",
            (post(AGENT_PATHS[1], {"message": "RAG 논문 추천해줘"}),),
            chat="search_then_link",
            embeddings="fail",
        ),
        Scenario("agent_model_failure", _both(AGENT_PATHS, {"message": "RAG 논문 추천해줘"}), chat="model_error"),
    ),
)

RATE_LIMIT = Bundle(
    "rate_limit",
    (
        Scenario(
            "llm_bucket_is_shared",
            (
                post(f"/papers/{UNKNOWN_PAPER}/analyze/", repeat=5),
                post(f"/papers/{UNKNOWN_PAPER}/summary/", raw="{", repeat=5),
                post(CHAT_PATHS[0], {"message": ""}, repeat=5),
                post(CHAT_PATHS[1], {"message": ""}, repeat=5),
                post(AGENT_PATHS[0], {"message": ""}, repeat=5),
                post(AGENT_PATHS[1], {"message": ""}, repeat=5),
                post(f"/papers/{UNKNOWN_PAPER}/analyze/"),
                post(AGENT_PATHS[1], {"message": "질문"}),
                get(f"/papers/{MAIN_PAPER}/analyze/"),
                get(f"/papers/{UNKNOWN_PAPER}/detail.json"),
                post(f"/papers/{UNKNOWN_PAPER}/analyze/", headers=(("x-real-ip", "203.0.113.7"),)),
                post(f"/papers/{UNKNOWN_PAPER}/analyze/", headers=(("x-real-ip", "203.0.113.8, 10.0.0.1"),)),
            ),
            fresh_minute=True,
        ),
        Scenario(
            "detail_bucket_is_separate",
            (
                get(f"/papers/{UNKNOWN_PAPER}/detail.json", repeat=60),
                get(f"/papers/{UNKNOWN_PAPER}/detail.json"),
                post(f"/papers/{UNKNOWN_PAPER}/analyze/"),
                get("/papers/list.json?page=2"),
            ),
            fresh_minute=True,
        ),
    ),
)

BUNDLES: dict[str, Bundle] = {
    bundle.name: bundle for bundle in (PAGES, DETAIL, ANALYZE, SUMMARY, PAPER_CHAT, AGENT, RATE_LIMIT)
}


def parse_cookie(value: str) -> dict[str, Any]:
    parts = [part.strip() for part in value.split(";") if part.strip()]
    name, _, cookie_value = parts[0].partition("=")
    attributes: dict[str, Any] = {}
    for part in parts[1:]:
        key, separator, attribute = part.partition("=")
        normalized = key.strip().lower()
        if not separator:
            attributes[normalized] = True
        elif normalized == "expires":
            attributes[normalized] = "<date>"
        else:
            attributes[normalized] = attribute.strip()
    return {"name": name.strip(), "value": "<token>" if cookie_value else "", "attributes": attributes}


def parse_sse(text: str) -> list[Any]:
    events: list[Any] = []
    for block in text.split("\n\n"):
        if not block:
            continue
        data = block.removeprefix("data: ")
        events.append(data if data == "[DONE]" else json.loads(data))
    return events


def _header_values(result: HttpResult, name: str) -> list[str]:
    return [value for key, value in result.headers if key == name]


def normalize_response(step: Step, result: HttpResult) -> dict[str, Any]:
    text = result.body.decode("utf-8")
    headers: dict[str, list[Any]] = {}
    for name in HEADER_ALLOWLIST:
        values = _header_values(result, name)
        if not values:
            continue
        if name == "set-cookie":
            headers[name] = [parse_cookie(value) for value in values]
        elif name == "retry-after":
            headers[name] = [RETRY_AFTER_PLACEHOLDER for _ in values]
        else:
            headers[name] = values
    content_type = (headers.get("content-type") or [""])[0]
    parsed = json.loads(text) if content_type.startswith("application/json") else None
    if result.status == 429:
        text, parsed = _mask_retry_after(result, text, parsed)
    record: dict[str, Any] = {"request": step.describe(), "status": result.status, "headers": headers, "body": text}
    if parsed is not None:
        record["json"] = parsed
    elif content_type.startswith("text/event-stream"):
        record["events"] = parse_sse(text)
    return record


def _mask_retry_after(result: HttpResult, text: str, payload: Any) -> tuple[str, Any]:
    values = _header_values(result, "retry-after")
    seconds = payload.get("retry_after") if isinstance(payload, dict) else None
    if not values or str(seconds) != values[0] or not 1 <= int(seconds) <= 60:
        raise AssertionError(f"inconsistent Retry-After: header {values}, body {seconds}")
    pattern = re.compile(rf"(?<!\d){seconds}(?!\d)")
    masked_payload = {
        key: RETRY_AFTER_PLACEHOLDER if key == "retry_after" else pattern.sub(RETRY_AFTER_PLACEHOLDER, str(value))
        for key, value in payload.items()
    }
    return pattern.sub(RETRY_AFTER_PLACEHOLDER, text), masked_payload


LEGACY_RULES = "legacy"
V2_RULES = "v2"
RULE_REQUEST = "request"
RULE_STATUS = "status code"
RULE_MEDIA_TYPE = "content-type media type"
RULE_ALLOW = "allow methods"
RULE_COOKIES = "set-cookie attributes without value"
RULE_SSE_BODY = "SSE body bytes"
RULE_JSON_BODY = "JSON body value"
RULE_OTHER_BODY = "other body bytes"
V2_EXACT_HEADERS = (
    "cache-control",
    "x-accel-buffering",
    "retry-after",
    "location",
    "x-content-type-options",
    "x-frame-options",
    "referrer-policy",
    "cross-origin-opener-policy",
)
V2_COOKIE_ATTRIBUTES = ("path", "max-age", "samesite", "secure", "httponly")


def _media_types(values: list[str]) -> list[str]:
    return [value.split(";", 1)[0].strip().lower() for value in values]


def _allowed_methods(values: list[str]) -> list[str]:
    return sorted({method.strip().upper() for value in values for method in value.split(",") if method.strip()})


def _cookie_view(cookie: Mapping[str, Any]) -> dict[str, Any]:
    attributes = {str(key).lower(): value for key, value in (cookie.get("attributes") or {}).items()}
    view: dict[str, Any] = {"name": cookie.get("name")}
    for name in V2_COOKIE_ATTRIBUTES:
        value = attributes.get(name)
        if name in ("secure", "httponly"):
            view[name] = value is True
        elif name == "samesite" and isinstance(value, str):
            view[name] = value.lower()
        else:
            view[name] = value
    return view


def v2_rule_views(record: Mapping[str, Any]) -> dict[str, Any]:
    headers = record.get("headers") or {}
    views: dict[str, Any] = {
        RULE_REQUEST: record.get("request"),
        RULE_STATUS: record.get("status"),
        RULE_MEDIA_TYPE: _media_types(headers.get("content-type", [])),
        RULE_ALLOW: _allowed_methods(headers.get("allow", [])),
        RULE_COOKIES: [_cookie_view(cookie) for cookie in headers.get("set-cookie", [])],
    }
    for name in V2_EXACT_HEADERS:
        views[f"header {name}"] = [str(value).strip() for value in headers.get(name, [])]
    if "events" in record:
        views[RULE_SSE_BODY] = record.get("body")
    elif "json" in record:
        views[RULE_JSON_BODY] = record.get("json")
    else:
        views[RULE_OTHER_BODY] = record.get("body")
    return views


def compare_responses_legacy(expected: list[Any], actual: list[Any]) -> list[str]:
    return diff(expected, actual)


def compare_responses_v2(expected: list[Any], actual: list[Any]) -> list[str]:
    problems: list[str] = []
    if len(expected) != len(actual):
        problems.append(f"response count {len(expected)} -> {len(actual)}")
    for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
        request = left.get("request") or {}
        label = f"response {index} {request.get('method', '?')} {request.get('path', '?')}"
        problems.extend(f"{label}: {problem}" for problem in diff(v2_rule_views(left), v2_rule_views(right)))
    return problems


RESPONSE_COMPARISONS: dict[str, Callable[[list[Any], list[Any]], list[str]]] = {
    LEGACY_RULES: compare_responses_legacy,
    V2_RULES: compare_responses_v2,
}


def response_comparison(implementation: str) -> Callable[[list[Any], list[Any]], list[str]]:
    return RESPONSE_COMPARISONS[V2_RULES if implementation == V2_RULES else LEGACY_RULES]


def _without_stream_options(body: Any) -> Any:
    if isinstance(body, dict):
        return {key: value for key, value in body.items() if key != "stream_options"}
    return body


def _last_user_text(body: Mapping[str, Any]) -> str:
    users = [_text(message.get("content")) for message in _messages(body) if message.get("role") == "user"]
    return users[-1] if users else ""


def group_summary_graph(bodies: list[Any]) -> dict[str, Any]:
    sections: dict[str, list[Any]] = {}
    buckets: dict[str, list[Any]] = {}
    others: list[Any] = []
    for body in bodies:
        user = _last_user_text(body)
        section = re.search(r"섹션 분류: (\w+)", user)
        bucket = re.search(r"^bucket: (\w+)", user, re.MULTILINE)
        if section:
            sections.setdefault(section.group(1), []).append(body)
        elif bucket:
            buckets.setdefault(bucket.group(1), []).append(body)
        else:
            others.append(body)
    return {
        "section_calls": sections,
        "bucket_calls": buckets,
        "other_calls": others,
        "last_call_is_other": bool(bodies) and bool(others) and bodies[-1] is others[-1],
    }


def normalize_fake_requests(requests: list[RecordedRequest], llm_layout: str) -> dict[str, Any]:
    llm = [_without_stream_options(request.body) for request in requests if request.path == CHAT_COMPLETIONS_PATH]
    return {
        "llm_requests": group_summary_graph(llm) if llm_layout == "summary_graph" and llm else llm,
        "llm_statuses": [request.status for request in requests if request.path == CHAT_COMPLETIONS_PATH],
        "embedding_requests": [
            {"body": request.body, "status": request.status} for request in requests if request.path == EMBEDDINGS_PATH
        ],
        "arxiv_requests": [
            {"query": request.query, "status": request.status}
            for request in requests
            if request.path == ARXIV_QUERY_PATH
        ],
    }


def wait_for_fresh_minute(margin_seconds: float = 45.0) -> None:
    seconds = time.time() % 60
    if seconds > margin_seconds:
        time.sleep(60 - seconds + 0.5)


def csrf_token(adapter: HttpAdapter) -> str:
    result = adapter.request("GET", "/bootstrap.json", headers=[], body=None)
    for value in _header_values(result, "set-cookie"):
        name, _, rest = value.partition("=")
        if name.strip() == "csrftoken":
            return rest.split(";", 1)[0]
    raise AssertionError("bootstrap.json did not set a csrftoken cookie")


def _step_headers(step: Step, token: str | None) -> list[tuple[str, str]]:
    headers = [(name.lower(), value) for name, value in step.headers]
    if step.body() is not None:
        headers.append(("content-type", "application/json"))
    if step.csrf and step.method not in SAFE_METHODS and token is not None:
        headers.extend([("cookie", f"csrftoken={token}"), ("x-csrftoken", token)])
    return headers


def run_scenario(adapter: HttpAdapter, server: FakeOpenAIServer, scenario: Scenario) -> dict[str, Any]:
    adapter.reset_state()
    server.reset()
    server.chat_handler = REPLY_SCRIPTS[scenario.chat]
    server.embedding_handler = _fail_embeddings if scenario.embeddings == "fail" else EmbeddingVectors()
    server.arxiv_feed = _fail_arxiv if scenario.arxiv == "fail" else ArxivFeed()
    needs_token = any(step.csrf and step.method not in SAFE_METHODS for step in scenario.steps)
    token = csrf_token(adapter) if needs_token else None
    if scenario.fresh_minute:
        wait_for_fresh_minute()
    responses: list[dict[str, Any]] = []
    with adapter.openai_key(scenario.openai_key):
        for step in scenario.steps:
            for _ in range(step.repeat):
                result = adapter.request(step.method, step.path, headers=_step_headers(step, token), body=step.body())
                responses.append(normalize_response(step, result))
    return {
        "responses": responses,
        **normalize_fake_requests(server.requests, scenario.llm_layout),
        "fake_server_errors": [error.as_dict() for error in server.errors],
    }


def run_bundle(adapter: HttpAdapter, server: FakeOpenAIServer, bundle: Bundle) -> dict[str, Any]:
    return {scenario.name: run_scenario(adapter, server, scenario) for scenario in bundle.scenarios}
