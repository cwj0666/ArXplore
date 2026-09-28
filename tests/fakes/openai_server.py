import argparse
import base64
import gzip
import hashlib
import json
import math
import socket
import struct
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self
from urllib.parse import parse_qs, urlsplit
from xml.sax.saxutils import escape, quoteattr

FAKE_API_KEY = "sk-fake-arxplore"
FIXED_CREATED = 1767225600
SYSTEM_FINGERPRINT = "fp_arxplore_fake"
DEFAULT_CHUNK_CHARS = 8
DEFAULT_EMBEDDING_DIMENSIONS = 1536
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
EMBEDDINGS_PATH = "/v1/embeddings"
ARXIV_QUERY_PATH = "/arxiv/api/query"
EMBEDDING_TABLE_FORMAT = 1

_REPLY_WORDS = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
)


class FakeHTTPError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        error_type: str = "fake_error",
        record_error: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.error_type = error_type
        self.record_error = record_error


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: str = "{}"
    id: str | None = None


@dataclass(frozen=True)
class ChatReply:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None


ChatHandler = Callable[[Mapping[str, Any]], ChatReply | str | ToolCall | Sequence[ToolCall]]
EmbeddingHandler = Callable[[Mapping[str, Any]], Sequence[Sequence[float]]]
ArxivFeedSource = str | Callable[[Mapping[str, list[str]]], str]


@dataclass(frozen=True)
class RecordedRequest:
    seq: int
    method: str
    path: str
    query: dict[str, list[str]]
    body: Any
    status: int
    authorized: bool | None
    headers: dict[str, str] = field(default_factory=dict, compare=False, repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "method": self.method,
            "path": self.path,
            "query": self.query,
            "body": self.body,
            "status": self.status,
            "authorized": self.authorized,
        }


@dataclass(frozen=True)
class RecordedError:
    seq: int
    method: str
    path: str
    status: int
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "method": self.method,
            "path": self.path,
            "status": self.status,
            "message": self.message,
        }


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"] for part in content if isinstance(part, Mapping) and isinstance(part.get("text"), str)
        )
    return ""


def last_user_text(request: Mapping[str, Any]) -> str:
    messages = request.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, Mapping) and message.get("role") == "user":
                return message_text(message.get("content"))
    return ""


def default_chat_handler(request: Mapping[str, Any]) -> ChatReply:
    digest = hashlib.sha256(last_user_text(request).encode("utf-8", "surrogatepass")).hexdigest()
    words = " ".join(_REPLY_WORDS[int(char, 16)] for char in digest[:12])
    return ChatReply(content=f"Fake reply {digest[:12]}: {words}.")


class ScriptedChatHandler:
    def __init__(self, replies: Iterable[ChatReply | str | ToolCall | Sequence[ToolCall]]) -> None:
        self._replies = list(replies)
        self._lock = threading.Lock()
        self.calls = 0

    def __call__(self, request: Mapping[str, Any]) -> ChatReply | str | ToolCall | Sequence[ToolCall]:
        with self._lock:
            if self.calls >= len(self._replies):
                raise LookupError(f"scripted chat replies exhausted after {len(self._replies)} calls")
            reply = self._replies[self.calls]
            self.calls += 1
        return reply

    @property
    def remaining(self) -> int:
        with self._lock:
            return len(self._replies) - self.calls


def normalize_reply(value: Any) -> ChatReply:
    if isinstance(value, ChatReply):
        return value
    if isinstance(value, str):
        return ChatReply(content=value)
    if isinstance(value, ToolCall):
        return ChatReply(tool_calls=(value,))
    if isinstance(value, Sequence) and all(isinstance(item, ToolCall) for item in value):
        return ChatReply(tool_calls=tuple(value))
    raise TypeError(f"chat handler returned an unsupported value: {type(value).__name__}")


def split_fixed(text: str, size: int) -> list[str]:
    step = max(1, size)
    return [text[index : index + step] for index in range(0, len(text), step)]


def chat_usage(request: Mapping[str, Any], reply: ChatReply) -> dict[str, int]:
    prompt_chars = len(json.dumps(request.get("messages"), ensure_ascii=False, sort_keys=True))
    completion_chars = len(reply.content) + sum(len(call.name) + len(call.arguments) for call in reply.tool_calls)
    prompt_tokens = max(1, prompt_chars // 4)
    completion_tokens = max(1, completion_chars // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def resolve_tool_calls(reply: ChatReply, number: int) -> list[dict[str, str]]:
    return [
        {
            "id": call.id or f"call_fake_{number:06d}_{index}",
            "name": call.name,
            "arguments": call.arguments,
        }
        for index, call in enumerate(reply.tool_calls)
    ]


def chat_completion_payload(
    *,
    completion_id: str,
    model: str,
    reply: ChatReply,
    tool_calls: list[dict[str, str]],
    usage: dict[str, int],
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": reply.content if reply.content or not tool_calls else None,
    }
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for call in tool_calls
        ]
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": reply.finish_reason or ("tool_calls" if tool_calls else "stop"),
            }
        ],
        "usage": usage,
    }


def chat_completion_chunks(
    *,
    completion_id: str,
    model: str,
    reply: ChatReply,
    tool_calls: list[dict[str, str]],
    usage: dict[str, int],
    include_usage: bool,
    chunk_chars: int,
) -> list[dict[str, Any]]:
    def envelope(choices: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": FIXED_CREATED,
            "model": model,
            "system_fingerprint": SYSTEM_FINGERPRINT,
            "choices": choices,
        }
        if include_usage:
            payload["usage"] = None
        return payload

    def delta_chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
        return envelope([{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish_reason}])

    chunks: list[dict[str, Any]] = []
    if reply.content or not tool_calls:
        chunks.append(delta_chunk({"role": "assistant", "content": ""}))
        chunks.extend(delta_chunk({"content": piece}) for piece in split_fixed(reply.content, chunk_chars))
    for index, call in enumerate(tool_calls):
        header = {
            "index": index,
            "id": call["id"],
            "type": "function",
            "function": {"name": call["name"], "arguments": ""},
        }
        if chunks:
            chunks.append(delta_chunk({"tool_calls": [header]}))
        else:
            chunks.append(delta_chunk({"role": "assistant", "content": None, "tool_calls": [header]}))
        chunks.extend(
            delta_chunk({"tool_calls": [{"index": index, "function": {"arguments": piece}}]})
            for piece in split_fixed(call["arguments"], chunk_chars)
        )
    chunks.append(delta_chunk({}, reply.finish_reason or ("tool_calls" if tool_calls else "stop")))
    if include_usage:
        usage_chunk = envelope([])
        usage_chunk["usage"] = usage
        chunks.append(usage_chunk)
    return chunks


def embedding_inputs(request: Mapping[str, Any]) -> list[Any]:
    raw = request.get("input")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list) and raw and all(isinstance(item, int) for item in raw):
        return [raw]
    if isinstance(raw, list) and raw and all(isinstance(item, str | list) for item in raw):
        return list(raw)
    raise FakeHTTPError(400, "input must be a string, a token list or a non-empty list of them", record_error=True)


def embedding_key(value: Any) -> str:
    if isinstance(value, str):
        data = value.encode("utf-8", "surrogatepass")
    else:
        data = json.dumps(value, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(data).hexdigest()


def to_float32(values: Sequence[float]) -> list[float]:
    count = len(values)
    return list(struct.unpack(f"<{count}f", struct.pack(f"<{count}f", *values)))


def deterministic_unit_vector(key: str, dimensions: int) -> list[float]:
    raw: list[float] = []
    counter = 0
    while len(raw) < dimensions:
        block = hashlib.sha256(f"{key}:{counter}".encode("ascii")).digest()
        raw.extend(value / 2147483648.0 - 1.0 for value in struct.unpack("<8I", block))
        counter += 1
    raw = raw[:dimensions]
    norm = math.sqrt(math.fsum(value * value for value in raw))
    return to_float32([value / norm for value in raw])


def encode_vector_base64(vector: Sequence[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")


def decode_vector(value: Any) -> list[float]:
    if isinstance(value, str):
        raw = base64.b64decode(value)
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))
    return [float(item) for item in value]


def load_embedding_table(path: str | Path) -> dict[str, list[float]]:
    data = Path(path).read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    payload = json.loads(data)
    vectors = payload.get("vectors", payload) if isinstance(payload, Mapping) else {}
    return {str(key): decode_vector(value) for key, value in vectors.items()}


def write_embedding_table(
    path: str | Path,
    vectors: Mapping[str, Sequence[float]],
    *,
    model: str,
    dimensions: int,
) -> None:
    payload = {
        "format": EMBEDDING_TABLE_FORMAT,
        "model": model,
        "dimensions": dimensions,
        "vectors": {key: encode_vector_base64(vectors[key]) for key in sorted(vectors)},
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
    target = Path(path)
    target.write_bytes(gzip.compress(data, compresslevel=9, mtime=0) if target.suffix == ".gz" else data)


class EmbeddingVectors:
    def __init__(self, table: Mapping[str, Sequence[float]] | None = None, *, strict: bool = False) -> None:
        self.table = dict(table or {})
        self.strict = strict
        self.table_hits = 0
        self.inputs: list[Any] = []
        self._lock = threading.Lock()

    def __call__(self, request: Mapping[str, Any]) -> list[list[float]]:
        dimensions = int(request.get("dimensions") or DEFAULT_EMBEDDING_DIMENSIONS)
        vectors: list[list[float]] = []
        for item in embedding_inputs(request):
            key = embedding_key(item)
            with self._lock:
                self.inputs.append(item)
            stored = self.table.get(key)
            if stored is None:
                if self.strict:
                    raise FakeHTTPError(400, f"no stored vector for input {key[:12]}", record_error=True)
                vectors.append(deterministic_unit_vector(key, dimensions))
                continue
            if len(stored) != dimensions:
                raise FakeHTTPError(
                    400,
                    f"stored vector {key[:12]} has {len(stored)} dimensions, request asked for {dimensions}",
                    record_error=True,
                )
            with self._lock:
                self.table_hits += 1
            vectors.append(list(stored))
        return vectors


@dataclass(frozen=True)
class ArxivEntry:
    arxiv_id: str
    title: str
    summary: str
    authors: tuple[str, ...] = ()
    categories: tuple[str, ...] = ("cs.CL",)
    published: str = "2026-09-01T00:00:00Z"
    updated: str | None = None
    version: int = 1
    comment: str | None = None


DEFAULT_ARXIV_ENTRIES = (
    ArxivEntry(
        arxiv_id="2609.90001",
        title="Fake External Study of Retrieval Augmented Generation",
        summary="We describe a fake external paper about retrieval augmented generation for testing.",
        authors=("Ada Fake", "Ben Placeholder"),
        categories=("cs.CL", "cs.IR"),
    ),
    ArxivEntry(
        arxiv_id="2609.90002",
        title="Fake External Study of Vision Language Models",
        summary="We describe a fake external paper about vision language models for testing.",
        authors=("Cy Stub",),
        categories=("cs.CV",),
        published="2026-09-02T00:00:00Z",
    ),
)


def _arxiv_entry_xml(entry: ArxivEntry) -> list[str]:
    versioned = f"{entry.arxiv_id}v{entry.version}"
    lines = [
        "  <entry>",
        f"    <id>http://arxiv.org/abs/{versioned}</id>",
        f"    <title>{escape(entry.title)}</title>",
        f"    <updated>{entry.updated or entry.published}</updated>",
        f'    <link href="https://arxiv.org/abs/{versioned}" rel="alternate" type="text/html"/>',
        f'    <link href="https://arxiv.org/pdf/{versioned}" rel="related" type="application/pdf" title="pdf"/>',
        f"    <summary>{escape(entry.summary)}</summary>",
    ]
    lines.extend(
        f'    <category term={quoteattr(category)} scheme="http://arxiv.org/schemas/atom"/>'
        for category in entry.categories
    )
    lines.append(f"    <published>{entry.published}</published>")
    if entry.comment:
        lines.append(f"    <arxiv:comment>{escape(entry.comment)}</arxiv:comment>")
    if entry.categories:
        lines.append(f"    <arxiv:primary_category term={quoteattr(entry.categories[0])}/>")
    for author in entry.authors:
        lines.extend(["    <author>", f"      <name>{escape(author)}</name>", "    </author>"])
    lines.append("  </entry>")
    return lines


def build_arxiv_feed(entries: Iterable[ArxivEntry]) -> str:
    selected = list(entries)
    lines = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        '<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"'
        ' xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns="http://www.w3.org/2005/Atom">',
        "  <id>https://arxiv.org/api/fake</id>",
        "  <title>arXiv Query: fake</title>",
        "  <updated>2026-09-29T00:00:00Z</updated>",
        f"  <opensearch:itemsPerPage>{len(selected)}</opensearch:itemsPerPage>",
        f"  <opensearch:totalResults>{len(selected)}</opensearch:totalResults>",
        "  <opensearch:startIndex>0</opensearch:startIndex>",
    ]
    for entry in selected:
        lines.extend(_arxiv_entry_xml(entry))
    lines.append("</feed>")
    return "\n".join(lines) + "\n"


def _base_arxiv_id(value: str) -> str:
    candidate = value.strip()
    head, separator, tail = candidate.rpartition("v")
    if separator and head and tail.isdigit():
        return head
    return candidate


class ArxivFeed:
    def __init__(self, entries: Iterable[ArxivEntry] = DEFAULT_ARXIV_ENTRIES) -> None:
        self.entries = tuple(entries)

    def __call__(self, query: Mapping[str, list[str]]) -> str:
        requested = [
            _base_arxiv_id(item) for value in query.get("id_list", []) for item in value.split(",") if item.strip()
        ]
        if requested:
            by_id = {entry.arxiv_id: entry for entry in self.entries}
            return build_arxiv_feed(by_id[arxiv_id] for arxiv_id in requested if arxiv_id in by_id)
        try:
            max_results = int((query.get("max_results") or ["10"])[0])
        except ValueError as exc:
            raise FakeHTTPError(400, "max_results must be an integer", record_error=True) from exc
        return build_arxiv_feed(self.entries[: max(0, max_results)])


@dataclass
class _Response:
    status: int
    content_type: str
    body: bytes = b""
    events: list[bytes] | None = None


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _error_payload(message: str, error_type: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": error_type, "param": None, "code": None}}


class _RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 30

    def setup(self) -> None:
        super().setup()
        self.server.fake._track_connection(self.connection, open_=True)

    def finish(self) -> None:
        try:
            super().finish()
        finally:
            self.server.fake._track_connection(self.connection, open_=False)

    def do_GET(self) -> None:
        self.server.fake._dispatch(self)

    def do_POST(self) -> None:
        self.server.fake._dispatch(self)

    def do_PUT(self) -> None:
        self.server.fake._dispatch(self)

    def do_PATCH(self) -> None:
        self.server.fake._dispatch(self)

    def do_DELETE(self) -> None:
        self.server.fake._dispatch(self)

    def log_message(self, *args: Any) -> None:
        return None


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], fake: "FakeOpenAIServer") -> None:
        self.fake = fake
        super().__init__(address, _RequestHandler)


class FakeOpenAIServer:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        chat_handler: ChatHandler | None = None,
        embedding_handler: EmbeddingHandler | None = None,
        arxiv_feed: ArxivFeedSource | None = None,
        chunk_chars: int = DEFAULT_CHUNK_CHARS,
        expected_api_key: str | None = FAKE_API_KEY,
        record_path: str | Path | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.chat_handler: ChatHandler = chat_handler or default_chat_handler
        self.embedding_handler: EmbeddingHandler = embedding_handler or EmbeddingVectors()
        self.arxiv_feed: ArxivFeedSource = arxiv_feed if arxiv_feed is not None else ArxivFeed()
        self.chunk_chars = chunk_chars
        self.expected_api_key = expected_api_key
        self.record_path = Path(record_path) if record_path else None
        self._lock = threading.Lock()
        self._requests: list[RecordedRequest] = []
        self._errors: list[RecordedError] = []
        self._seq = 0
        self._completion_number = 0
        self._connections: set[socket.socket] = set()
        self._httpd: _HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._routes: dict[tuple[str, str], Callable[[Any, dict[str, list[str]]], _Response]] = {
            ("POST", CHAT_COMPLETIONS_PATH): self._chat_completions,
            ("POST", EMBEDDINGS_PATH): self._embeddings,
            ("GET", ARXIV_QUERY_PATH): self._arxiv_query,
        }

    def start(self) -> Self:
        if self._httpd is not None:
            return self
        self._httpd = _HTTPServer((self.host, self.port), self)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.05}, name="fake-openai-server", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        httpd, thread = self._httpd, self._thread
        if httpd is None:
            return
        httpd.shutdown()
        httpd.server_close()
        with self._lock:
            connections = list(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if thread is not None:
            thread.join(timeout=5)
        self._httpd = None
        self._thread = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc_info: Any) -> None:
        self.stop()

    @property
    def url(self) -> str:
        if self._httpd is None:
            raise RuntimeError("fake server is not running")
        return f"http://{self.host}:{self.port}"

    @property
    def base_url(self) -> str:
        return f"{self.url}/v1"

    @property
    def arxiv_url(self) -> str:
        return f"{self.url}{ARXIV_QUERY_PATH}"

    def environ(self) -> dict[str, str]:
        return {
            "OPENAI_API_KEY": self.expected_api_key or FAKE_API_KEY,
            "OPENAI_BASE_URL": self.base_url,
            "OPENAI_API_BASE": self.base_url,
            "ARXIV_API_BASE_URL": self.arxiv_url,
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "LANGSMITH_API_KEY": "fake-disabled",
        }

    @property
    def requests(self) -> list[RecordedRequest]:
        with self._lock:
            return sorted(self._requests, key=lambda request: request.seq)

    @property
    def errors(self) -> list[RecordedError]:
        with self._lock:
            return sorted(self._errors, key=lambda error: error.seq)

    def requests_for(self, path: str) -> list[RecordedRequest]:
        return [request for request in self.requests if request.path == path]

    def reset(self) -> None:
        with self._lock:
            self._requests.clear()
            self._errors.clear()
            self._seq = 0
            self._completion_number = 0

    def _track_connection(self, connection: socket.socket, *, open_: bool) -> None:
        with self._lock:
            if open_:
                self._connections.add(connection)
            else:
                self._connections.discard(connection)

    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _next_completion_number(self) -> int:
        with self._lock:
            self._completion_number += 1
            return self._completion_number

    def _append_record(self, entry: RecordedRequest | RecordedError) -> None:
        with self._lock:
            if isinstance(entry, RecordedRequest):
                self._requests.append(entry)
                line = {"kind": "request", **entry.as_dict()}
            else:
                self._errors.append(entry)
                line = {"kind": "error", **entry.as_dict()}
            if self.record_path is not None:
                with self.record_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")

    def _dispatch(self, handler: BaseHTTPRequestHandler) -> None:
        seq = self._next_seq()
        split = urlsplit(handler.path)
        path = split.path
        query = parse_qs(split.query, keep_blank_values=True)
        length = int(handler.headers.get("Content-Length") or 0)
        raw = handler.rfile.read(length) if length > 0 else b""
        headers = {key.lower(): value for key, value in handler.headers.items() if key.lower() != "authorization"}
        body, body_is_json = self._parse_body(raw)
        authorized = self._check_api_key(handler.headers.get("Authorization"), path)
        try:
            if authorized is False:
                raise FakeHTTPError(401, "unexpected api key", error_type="invalid_api_key", record_error=True)
            route = self._routes.get((handler.command, path))
            if route is None:
                known_path = any(route_path == path for _, route_path in self._routes)
                status = 405 if known_path else 404
                raise FakeHTTPError(status, f"no fake route for {handler.command} {path}", record_error=True)
            if handler.command == "POST" and not body_is_json:
                raise FakeHTTPError(400, "request body must be a JSON object", record_error=True)
            response = route(body, query)
        except FakeHTTPError as exc:
            response = _Response(
                exc.status, "application/json", _json_bytes(_error_payload(exc.message, exc.error_type))
            )
            self._append_record(
                RecordedRequest(seq, handler.command, path, query, body, exc.status, authorized, headers)
            )
            if exc.record_error:
                self._append_record(RecordedError(seq, handler.command, path, exc.status, exc.message))
            self._send(handler, response)
            return
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            response = _Response(500, "application/json", _json_bytes(_error_payload(message, "fake_server_error")))
            self._append_record(RecordedRequest(seq, handler.command, path, query, body, 500, authorized, headers))
            self._append_record(RecordedError(seq, handler.command, path, 500, message))
            self._send(handler, response)
            return
        self._append_record(
            RecordedRequest(seq, handler.command, path, query, body, response.status, authorized, headers)
        )
        self._send(handler, response)

    @staticmethod
    def _parse_body(raw: bytes) -> tuple[Any, bool]:
        if not raw:
            return None, False
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return raw.decode("utf-8", "replace"), False
        return parsed, isinstance(parsed, dict)

    def _check_api_key(self, authorization: str | None, path: str) -> bool | None:
        if self.expected_api_key is None or not path.startswith("/v1/"):
            return None
        return authorization == f"Bearer {self.expected_api_key}"

    def _send(self, handler: BaseHTTPRequestHandler, response: _Response) -> None:
        try:
            handler.send_response(response.status)
            handler.send_header("Content-Type", response.content_type)
            if response.events is None:
                handler.send_header("Content-Length", str(len(response.body)))
                handler.end_headers()
                handler.wfile.write(response.body)
                return
            handler.send_header("Cache-Control", "no-cache")
            handler.send_header("Transfer-Encoding", "chunked")
            handler.end_headers()
            for event in response.events:
                handler.wfile.write(f"{len(event):X}\r\n".encode("ascii") + event + b"\r\n")
            handler.wfile.write(b"0\r\n\r\n")
        except ConnectionError:
            handler.close_connection = True

    def _chat_completions(self, body: Any, query: dict[str, list[str]]) -> _Response:
        if not isinstance(body.get("messages"), list):
            raise FakeHTTPError(400, "messages must be a list", error_type="invalid_request_error", record_error=True)
        reply = normalize_reply(self.chat_handler(body))
        number = self._next_completion_number()
        completion_id = f"chatcmpl-fake-{number:06d}"
        tool_calls = resolve_tool_calls(reply, number)
        usage = chat_usage(body, reply)
        model = str(body.get("model") or "")
        if not body.get("stream"):
            payload = chat_completion_payload(
                completion_id=completion_id, model=model, reply=reply, tool_calls=tool_calls, usage=usage
            )
            return _Response(200, "application/json", _json_bytes(payload))
        stream_options = body.get("stream_options") if isinstance(body.get("stream_options"), dict) else {}
        chunks = chat_completion_chunks(
            completion_id=completion_id,
            model=model,
            reply=reply,
            tool_calls=tool_calls,
            usage=usage,
            include_usage=bool(stream_options.get("include_usage")),
            chunk_chars=self.chunk_chars,
        )
        events = [b"data: " + _json_bytes(chunk) + b"\n\n" for chunk in chunks]
        events.append(b"data: [DONE]\n\n")
        return _Response(200, "text/event-stream; charset=utf-8", events=events)

    def _embeddings(self, body: Any, query: dict[str, list[str]]) -> _Response:
        inputs = embedding_inputs(body)
        encoding = body.get("encoding_format") or "float"
        if encoding not in ("float", "base64"):
            raise FakeHTTPError(400, f"unsupported encoding_format {encoding}", record_error=True)
        vectors = list(self.embedding_handler(body))
        if len(vectors) != len(inputs):
            raise FakeHTTPError(500, "embedding handler returned a wrong vector count", record_error=True)
        data = [
            {
                "object": "embedding",
                "index": index,
                "embedding": encode_vector_base64(vector) if encoding == "base64" else to_float32(list(vector)),
            }
            for index, vector in enumerate(vectors)
        ]
        tokens = sum(max(1, len(item) // 4) if isinstance(item, str) else len(item) for item in inputs)
        payload = {
            "object": "list",
            "data": data,
            "model": str(body.get("model") or ""),
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }
        return _Response(200, "application/json", _json_bytes(payload))

    def _arxiv_query(self, body: Any, query: dict[str, list[str]]) -> _Response:
        feed = self.arxiv_feed(query) if callable(self.arxiv_feed) else self.arxiv_feed
        return _Response(200, "application/atom+xml; charset=utf-8", feed.encode("utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.fakes.openai_server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--embedding-table", type=Path)
    parser.add_argument("--arxiv-feed", type=Path)
    parser.add_argument("--record", type=Path)
    parser.add_argument("--api-key", default=FAKE_API_KEY)
    parser.add_argument("--any-api-key", action="store_true")
    args = parser.parse_args(argv)

    embedding_handler = EmbeddingVectors(load_embedding_table(args.embedding_table) if args.embedding_table else None)
    arxiv_feed = args.arxiv_feed.read_text(encoding="utf-8") if args.arxiv_feed else None
    server = FakeOpenAIServer(
        host=args.host,
        port=args.port,
        embedding_handler=embedding_handler,
        arxiv_feed=arxiv_feed,
        chunk_chars=args.chunk_chars,
        expected_api_key=None if args.any_api_key else args.api_key,
        record_path=args.record,
    )
    server.start()
    print(server.base_url, flush=True)
    for key, value in server.environ().items():
        print(f"export {key}={value}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
