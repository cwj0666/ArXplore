import json
import math
import operator
import subprocess
import sys
import xml.etree.ElementTree as ET
from functools import reduce
from pathlib import Path

import httpx
import openai
import pytest
from langchain_openai import ChatOpenAI

from tests.fakes.openai_server import (
    ARXIV_QUERY_PATH,
    CHAT_COMPLETIONS_PATH,
    DEFAULT_CHUNK_CHARS,
    EMBEDDINGS_PATH,
    FAKE_API_KEY,
    FIXED_CREATED,
    SYSTEM_FINGERPRINT,
    ArxivEntry,
    ArxivFeed,
    ChatReply,
    EmbeddingVectors,
    FakeHTTPError,
    FakeOpenAIServer,
    ScriptedChatHandler,
    ToolCall,
    default_chat_handler,
    deterministic_unit_vector,
    embedding_key,
    load_embedding_table,
    to_float32,
    write_embedding_table,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_papers",
        "description": "Search the paper index.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


@pytest.fixture
def server():
    with FakeOpenAIServer() as running:
        yield running


def make_client(server: FakeOpenAIServer, *, api_key: str = FAKE_API_KEY) -> openai.OpenAI:
    return openai.OpenAI(api_key=api_key, base_url=server.base_url, max_retries=0)


def make_chat_model(server: FakeOpenAIServer, **kwargs) -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", api_key=FAKE_API_KEY, base_url=server.base_url, max_retries=0, **kwargs)


def expected_reply(user_text: str) -> str:
    return default_chat_handler({"messages": [{"role": "user", "content": user_text}]}).content


def test_chat_completion_returns_default_reply_and_records_request(server):
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "hello"},
    ]
    completion = make_client(server).chat.completions.create(model="gpt-4o", messages=messages, temperature=0)

    assert completion.id == "chatcmpl-fake-000001"
    assert completion.created == FIXED_CREATED
    assert completion.system_fingerprint == SYSTEM_FINGERPRINT
    assert completion.choices[0].message.content == expected_reply("hello")
    assert completion.choices[0].finish_reason == "stop"
    [request] = server.requests
    assert (request.seq, request.method, request.path, request.status) == (1, "POST", CHAT_COMPLETIONS_PATH, 200)
    assert request.authorized is True
    assert request.body["model"] == "gpt-4o"
    assert request.body["temperature"] == 0
    assert request.body["messages"] == messages
    assert "authorization" not in request.headers
    assert server.errors == []


def test_default_reply_depends_only_on_the_last_user_message():
    first = default_chat_handler(
        {"messages": [{"role": "system", "content": "a"}, {"role": "user", "content": "same question"}]}
    )
    second = default_chat_handler(
        {
            "messages": [
                {"role": "system", "content": "b"},
                {"role": "user", "content": [{"type": "text", "text": "same "}, {"type": "text", "text": "question"}]},
            ]
        }
    )
    other = default_chat_handler({"messages": [{"role": "user", "content": "another question"}]})

    assert first == second
    assert first.content != other.content
    assert first.content.startswith("Fake reply ")


def test_streaming_splits_content_into_fixed_pieces_and_reports_usage(server):
    stream = make_client(server).chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "stream please"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = list(stream)

    assert chunks[0].choices[0].delta.role == "assistant"
    pieces = [
        chunk.choices[0].delta.content for chunk in chunks[1:] if chunk.choices and chunk.choices[0].delta.content
    ]
    assert "".join(pieces) == expected_reply("stream please")
    assert all(len(piece) == DEFAULT_CHUNK_CHARS for piece in pieces[:-1])
    assert 0 < len(pieces[-1]) <= DEFAULT_CHUNK_CHARS
    assert chunks[-2].choices[0].finish_reason == "stop"
    assert chunks[-1].choices == []
    assert chunks[-1].usage.total_tokens == chunks[-1].usage.prompt_tokens + chunks[-1].usage.completion_tokens
    assert {chunk.id for chunk in chunks} == {"chatcmpl-fake-000001"}
    assert server.requests[0].body["stream_options"] == {"include_usage": True}


def test_chunk_size_is_configurable():
    with FakeOpenAIServer(chunk_chars=3) as running:
        stream = make_client(running).chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "x"}], stream=True
        )
        pieces = [
            chunk.choices[0].delta.content for chunk in stream if chunk.choices and chunk.choices[0].delta.content
        ]

    assert all(len(piece) == 3 for piece in pieces[:-1])
    assert "".join(pieces) == expected_reply("x")


@pytest.mark.parametrize("stream", [False, True])
def test_identical_request_sequences_produce_identical_bytes(stream):
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "byte check"}], "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    headers = {"Authorization": f"Bearer {FAKE_API_KEY}"}
    captured = []
    for _ in range(2):
        with FakeOpenAIServer() as running:
            responses = [
                httpx.post(f"{running.base_url}/chat/completions", json=body, headers=headers).content for _ in range(2)
            ]
        captured.append(responses)

    assert captured[0] == captured[1]
    assert captured[0][0] != captured[0][1]


def test_chat_openai_stream_reassembles_the_reply(server):
    model = make_chat_model(server, temperature=0)
    pieces = [chunk.content for chunk in model.stream("논문 요약해 줘")]

    assert "".join(pieces) == expected_reply("논문 요약해 줘")
    assert len([piece for piece in pieces if piece]) > 1
    [request] = server.requests_for(CHAT_COMPLETIONS_PATH)
    assert request.body["stream"] is True
    assert request.body["temperature"] == 0
    assert request.body["messages"] == [{"role": "user", "content": "논문 요약해 줘"}]


def test_bind_tools_invoke_returns_the_scripted_tool_call(server):
    server.chat_handler = ScriptedChatHandler([ToolCall(name="search_papers", arguments='{"query": "diffusion"}')])
    message = make_chat_model(server).bind_tools([SEARCH_TOOL]).invoke("find diffusion papers")

    assert message.tool_calls == [
        {"name": "search_papers", "args": {"query": "diffusion"}, "id": "call_fake_000001_0", "type": "tool_call"}
    ]
    assert message.content == ""
    [request] = server.requests
    assert request.body["tools"] == [SEARCH_TOOL]
    assert request.body["stream"] is False


def test_streamed_tool_call_arguments_are_reassembled(server):
    arguments = json.dumps({"query": "retrieval augmented generation for long documents"})
    server.chat_handler = ScriptedChatHandler(
        [ChatReply(content="Searching first. ", tool_calls=(ToolCall(name="search_papers", arguments=arguments),))]
    )
    chunks = list(make_chat_model(server).bind_tools([SEARCH_TOOL]).stream("find papers"))
    merged = reduce(operator.add, chunks)

    assert merged.content == "Searching first. "
    assert merged.tool_calls == [
        {
            "name": "search_papers",
            "args": {"query": "retrieval augmented generation for long documents"},
            "id": "call_fake_000001_0",
            "type": "tool_call",
        }
    ]
    argument_pieces = [piece for chunk in chunks for piece in chunk.tool_call_chunks if piece.get("args")]
    assert len(argument_pieces) > 1
    assert merged.response_metadata["finish_reason"] == "tool_calls"


def test_tool_call_only_stream_puts_the_role_on_the_first_tool_chunk(server):
    server.chat_handler = ScriptedChatHandler([[ToolCall(name="search_papers", arguments="{}", id="call_given")]])
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "q"}], "stream": True}
    response = httpx.post(
        f"{server.base_url}/chat/completions", json=body, headers={"Authorization": f"Bearer {FAKE_API_KEY}"}
    )
    events = [line.removeprefix("data: ") for line in response.text.split("\n\n") if line]

    assert events[-1] == "[DONE]"
    first = json.loads(events[0])["choices"][0]["delta"]
    assert first["role"] == "assistant"
    assert first["content"] is None
    assert first["tool_calls"][0]["id"] == "call_given"
    assert json.loads(events[-2])["choices"][0]["finish_reason"] == "tool_calls"


def test_embeddings_are_deterministic_float32_unit_vectors(server):
    client = make_client(server)
    first = client.embeddings.create(model="text-embedding-3-large", input=["alpha", "beta"], dimensions=1536)
    again = client.embeddings.create(model="text-embedding-3-large", input="alpha", dimensions=1536)
    as_float = client.embeddings.create(
        model="text-embedding-3-large", input=["alpha"], dimensions=1536, encoding_format="float"
    )
    short = client.embeddings.create(model="text-embedding-3-large", input="alpha", dimensions=8)

    alpha, beta = (item.embedding for item in first.data)
    assert len(alpha) == 1536
    assert math.isclose(math.fsum(value * value for value in alpha), 1.0, abs_tol=1e-5)
    assert alpha == again.data[0].embedding == as_float.data[0].embedding
    assert alpha == deterministic_unit_vector(embedding_key("alpha"), 1536)
    assert alpha == to_float32(alpha)
    assert alpha != beta
    assert len(short.data[0].embedding) == 8
    bodies = [request.body for request in server.requests_for(EMBEDDINGS_PATH)]
    assert [body["encoding_format"] for body in bodies] == ["base64", "base64", "float", "base64"]
    assert bodies[0]["input"] == ["alpha", "beta"]
    assert bodies[1]["input"] == "alpha"
    assert server.errors == []


def test_embedding_dimensions_default_to_1536(server):
    response = httpx.post(
        f"{server.base_url}/embeddings",
        json={"model": "text-embedding-3-large", "input": "no dimensions"},
        headers={"Authorization": f"Bearer {FAKE_API_KEY}"},
    )

    assert len(response.json()["data"][0]["embedding"]) == 1536


def test_embedding_table_vectors_take_precedence(server, tmp_path):
    stored = [0.5, -0.5, 0.25, 0.125]
    table_path = tmp_path / "query_embeddings.json.gz"
    write_embedding_table(
        table_path, {embedding_key("stored query"): stored}, model="text-embedding-3-large", dimensions=4
    )
    vectors = EmbeddingVectors(load_embedding_table(table_path))
    server.embedding_handler = vectors

    response = make_client(server).embeddings.create(
        model="text-embedding-3-large", input=["stored query", "other"], dimensions=4
    )

    assert response.data[0].embedding == stored
    assert response.data[1].embedding == deterministic_unit_vector(embedding_key("other"), 4)
    assert vectors.table_hits == 1
    with pytest.raises(openai.BadRequestError):
        make_client(server).embeddings.create(model="text-embedding-3-large", input="stored query", dimensions=8)
    [error] = server.errors
    assert (error.path, error.status) == (EMBEDDINGS_PATH, 400)


def test_unknown_paths_return_404_and_are_recorded_as_errors(server):
    with pytest.raises(openai.NotFoundError):
        make_client(server).models.list()
    response = httpx.get(f"{server.url}/nowhere")

    assert response.status_code == 404
    assert [(error.method, error.path, error.status) for error in server.errors] == [
        ("GET", "/v1/models", 404),
        ("GET", "/nowhere", 404),
    ]
    assert [request.status for request in server.requests] == [404, 404]


def test_wrong_method_on_a_known_path_is_an_error(server):
    response = httpx.get(f"{server.base_url}/chat/completions", headers={"Authorization": f"Bearer {FAKE_API_KEY}"})

    assert response.status_code == 405
    assert [error.status for error in server.errors] == [405]


def test_unexpected_api_key_is_rejected_and_recorded(server):
    with pytest.raises(openai.AuthenticationError):
        make_client(server, api_key="sk-not-the-fake").chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )

    [request] = server.requests
    assert request.authorized is False
    assert [error.status for error in server.errors] == [401]


def test_any_api_key_is_accepted_when_no_key_is_expected():
    with FakeOpenAIServer(expected_api_key=None) as running:
        make_client(running, api_key="sk-anything").chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
        [request] = running.requests

    assert request.authorized is None
    assert running.errors == []


def test_handlers_can_simulate_api_errors_without_recording_errors(server):
    def rate_limited(request):
        raise FakeHTTPError(429, "slow down", error_type="rate_limit_exceeded")

    server.chat_handler = rate_limited
    with pytest.raises(openai.RateLimitError):
        make_client(server).chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])

    assert [request.status for request in server.requests] == [429]
    assert server.errors == []


def test_crashing_or_exhausted_handlers_are_recorded_as_errors(server):
    server.chat_handler = ScriptedChatHandler(["only one reply"])
    client = make_client(server)
    first = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "1"}])
    with pytest.raises(openai.InternalServerError):
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "2"}])

    assert first.choices[0].message.content == "only one reply"
    [error] = server.errors
    assert error.status == 500
    assert "exhausted" in error.message


def test_reset_clears_records_and_restarts_ids(server):
    client = make_client(server)
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "1"}])
    server.reset()
    completion = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "2"}])

    assert completion.id == "chatcmpl-fake-000001"
    assert [request.seq for request in server.requests] == [1]


def test_record_path_appends_json_lines(tmp_path):
    record_path = tmp_path / "requests.jsonl"
    with FakeOpenAIServer(record_path=record_path) as running:
        make_client(running).chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "log"}])
        httpx.get(f"{running.url}/missing")

    lines = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines()]
    assert [(line["kind"], line["path"], line["status"]) for line in lines] == [
        ("request", CHAT_COMPLETIONS_PATH, 200),
        ("request", "/missing", 404),
        ("error", "/missing", 404),
    ]
    assert lines[0]["body"]["messages"] == [{"role": "user", "content": "log"}]


def test_arxiv_feed_filters_by_id_list_and_limits_searches(server):
    by_id = httpx.get(server.arxiv_url, params={"id_list": "2609.90002v1,2609.99999"})
    search = httpx.get(server.arxiv_url, params={"search_query": 'all:"x"', "start": 0, "max_results": 1})

    by_id_entries = ET.fromstring(by_id.text).findall("atom:entry", ATOM)
    assert by_id.headers["content-type"].startswith("application/atom+xml")
    assert [entry.find("atom:id", ATOM).text for entry in by_id_entries] == ["http://arxiv.org/abs/2609.90002v1"]
    entry = by_id_entries[0]
    assert entry.find("arxiv:primary_category", ATOM).attrib["term"] == "cs.CV"
    assert [link.attrib["href"] for link in entry.findall("atom:link", ATOM) if link.attrib.get("title") == "pdf"] == [
        "https://arxiv.org/pdf/2609.90002v1"
    ]
    search_entries = ET.fromstring(search.text).findall("atom:entry", ATOM)
    assert [item.find("atom:id", ATOM).text for item in search_entries] == ["http://arxiv.org/abs/2609.90001v1"]
    assert [request.query for request in server.requests_for(ARXIV_QUERY_PATH)][0] == {
        "id_list": ["2609.90002v1,2609.99999"]
    }
    assert server.errors == []


def test_arxiv_feed_can_be_replaced(server):
    server.arxiv_feed = ArxivFeed([ArxivEntry(arxiv_id="2609.12345", title="A & B <test>", summary="s", version=3)])
    custom = ET.fromstring(httpx.get(server.arxiv_url, params={"id_list": "2609.12345"}).text)
    server.arxiv_feed = "<feed/>"
    fixed = httpx.get(server.arxiv_url, params={"search_query": "anything"}).text

    assert custom.find("atom:entry/atom:title", ATOM).text == "A & B <test>"
    assert custom.find("atom:entry/atom:id", ATOM).text == "http://arxiv.org/abs/2609.12345v3"
    assert fixed == "<feed/>"


def test_environ_redirects_sdk_clients(server, monkeypatch):
    for key, value in server.environ().items():
        monkeypatch.setenv(key, value)

    openai.OpenAI(max_retries=0).embeddings.create(model="text-embedding-3-large", input="env", dimensions=4)
    ChatOpenAI(model="gpt-4o", max_retries=0).invoke("env")

    assert [request.path for request in server.requests] == [EMBEDDINGS_PATH, CHAT_COMPLETIONS_PATH]
    assert all(request.authorized for request in server.requests)


def test_cli_serves_requests():
    process = subprocess.Popen(
        [sys.executable, "-m", "tests.fakes.openai_server", "--port", "0"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        base_url = process.stdout.readline().strip()
        client = openai.OpenAI(api_key=FAKE_API_KEY, base_url=base_url, max_retries=0)
        completion = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "cli"}])
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert base_url.startswith("http://127.0.0.1:")
    assert base_url.endswith("/v1")
    assert completion.choices[0].message.content == expected_reply("cli")


def test_strict_embedding_table_rejects_unknown_inputs_and_records_every_input(server):
    vectors = EmbeddingVectors({embedding_key("known"): [1.0, 0.0]}, strict=True)
    server.embedding_handler = vectors
    client = make_client(server)

    known = client.embeddings.create(model="text-embedding-3-large", input="known", dimensions=2)
    with pytest.raises(openai.BadRequestError):
        client.embeddings.create(model="text-embedding-3-large", input=["known", "unknown"], dimensions=2)

    assert known.data[0].embedding == [1.0, 0.0]
    assert vectors.inputs == ["known", "known", "unknown"]
    [error] = server.errors
    assert (error.path, error.status) == (EMBEDDINGS_PATH, 400)
