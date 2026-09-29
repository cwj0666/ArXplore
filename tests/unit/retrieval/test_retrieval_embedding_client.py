import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from openai import AuthenticationError, OpenAIError

from arxplore.config import AppSettings
from arxplore.context import override_openai_runtime
from arxplore.embedding import EmbeddingClient, EmbeddingKeyMissingError, clean_embedding_input


@dataclass
class EmbeddingServer:
    base_url: str = ""
    status: int = 200
    requests: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [body for _, body in self.requests]

    @property
    def authorizations(self) -> list[str]:
        return [authorization for authorization, _ in self.requests]


def _handler(server: EmbeddingServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            server.requests.append((self.headers.get("Authorization", ""), body))
            if server.status != 200:
                payload: dict[str, Any] = {"error": {"message": "bad key", "type": "invalid_api_key"}}
            else:
                data = [
                    {"object": "embedding", "index": index, "embedding": [float(len(text)), float(index)]}
                    for index, text in enumerate(body["input"])
                ]
                usage = {"prompt_tokens": len(data), "total_tokens": len(data)}
                payload = {"object": "list", "data": data, "model": body["model"], "usage": usage}
            content = json.dumps(payload).encode()
            self.send_response(server.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args: Any) -> None:
            pass

    return Handler


@pytest.fixture
def embedding_server() -> Iterator[EmbeddingServer]:
    server = EmbeddingServer()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler(server))
    server.base_url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _client(
    server: EmbeddingServer,
    *,
    model: str = "text-embedding-3-large",
    api_key: str | None = "sk-server",
    batch_size: int = 64,
) -> EmbeddingClient:
    return EmbeddingClient(
        model=model, dimensions=1536, batch_size=batch_size, api_key=api_key, base_url=server.base_url
    )


async def test_embeds_in_batches_with_dimensions_for_embedding_3_models(embedding_server: EmbeddingServer) -> None:
    client = _client(embedding_server, batch_size=2)

    vectors = await client.embed(["a", "bb", "ccc"])
    await client.aclose()

    assert vectors == [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]]
    assert [body["input"] for body in embedding_server.bodies] == [["a", "bb"], ["ccc"]]
    assert all(body["model"] == "text-embedding-3-large" for body in embedding_server.bodies)
    assert all(body["dimensions"] == 1536 for body in embedding_server.bodies)


async def test_default_batch_is_64(embedding_server: EmbeddingServer) -> None:
    client = _client(embedding_server)

    await client.embed([f"text {index}" for index in range(130)])
    await client.aclose()

    assert [len(body["input"]) for body in embedding_server.bodies] == [64, 64, 2]


async def test_older_models_do_not_send_dimensions(embedding_server: EmbeddingServer) -> None:
    client = _client(embedding_server, model="text-embedding-ada-002")

    await client.embed(["query"])
    await client.aclose()

    assert "dimensions" not in embedding_server.bodies[0]


async def test_inputs_are_stripped_before_surrogates_are_removed(embedding_server: EmbeddingServer) -> None:
    client = _client(embedding_server)

    await client.embed(["  policy\ud800 loss \n", " \udfff x"])
    await client.aclose()

    assert embedding_server.bodies[0]["input"] == ["policy loss", " x"]


def test_input_cleaning_rule() -> None:
    assert clean_embedding_input("  a\ud800b  ") == "ab"
    assert clean_embedding_input("\ud800 a") == " a"


async def test_no_texts_means_no_request(embedding_server: EmbeddingServer) -> None:
    client = _client(embedding_server, api_key=None)

    assert await client.embed([]) == []
    assert embedding_server.requests == []


async def test_request_scoped_key_wins_over_the_server_key(embedding_server: EmbeddingServer) -> None:
    client = _client(embedding_server)

    with override_openai_runtime(api_key="sk-request"):
        assert client.resolve_api_key() == "sk-request"
        await client.embed(["q"])
    with override_openai_runtime(api_key="   "):
        assert client.resolve_api_key() == "sk-server"
        await client.embed(["q"])
    await client.embed(["q"])
    await client.aclose()

    assert embedding_server.authorizations == ["Bearer sk-request", "Bearer sk-server", "Bearer sk-server"]


async def test_without_any_key_the_client_is_unavailable(embedding_server: EmbeddingServer) -> None:
    client = _client(embedding_server, api_key="  ")

    assert client.is_available() is False
    with pytest.raises(EmbeddingKeyMissingError) as caught:
        await client.embed(["q"])
    assert not isinstance(caught.value, OpenAIError)
    assert embedding_server.requests == []

    with override_openai_runtime(api_key="sk-request"):
        assert client.is_available() is True


async def test_api_errors_surface_as_openai_errors(embedding_server: EmbeddingServer) -> None:
    embedding_server.status = 401
    client = _client(embedding_server)

    with pytest.raises(AuthenticationError):
        await client.embed(["q"])
    await client.aclose()

    assert len(embedding_server.requests) == 1


def test_from_settings_reads_the_embedding_settings() -> None:
    settings = AppSettings.model_validate(
        {
            "OPENAI_API_KEY": "sk-settings",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
            "OPENAI_EMBEDDING_DIMENSIONS": "512",
            "EMBEDDING_BATCH_SIZE": "16",
        }
    )

    client = EmbeddingClient.from_settings(settings, base_url="http://127.0.0.1:9/v1")

    assert (client.model, client.dimensions, client.batch_size, client.base_url) == (
        "text-embedding-3-small",
        512,
        16,
        "http://127.0.0.1:9/v1",
    )
    assert client.resolve_api_key() == "sk-settings"


def test_from_settings_uses_the_configured_base_url_unless_one_is_given() -> None:
    settings = AppSettings.model_validate({"OPENAI_API_KEY": "sk-settings", "OPENAI_BASE_URL": "http://llm.test/v1"})

    assert EmbeddingClient.from_settings(settings).base_url == "http://llm.test/v1"
    assert EmbeddingClient.from_settings(settings, base_url="http://127.0.0.1:9/v1").base_url == "http://127.0.0.1:9/v1"
    assert EmbeddingClient.from_settings(AppSettings.model_validate({})).base_url is None
