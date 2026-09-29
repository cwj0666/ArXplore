import importlib
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

IMPLEMENTATION_ENV = "GOLDEN_IMPL"
DEFAULT_IMPLEMENTATION = "v2"


class AdapterUnavailable(LookupError):
    pass


class ParsedPdf(TypedDict):
    text: str
    sections: list[dict[str, Any]]
    source: str
    quality_metrics: dict[str, Any]
    parser_metadata: dict[str, Any]
    artifacts: dict[str, Any]


class ParserAdapter(Protocol):
    def describe(self) -> dict[str, str]: ...

    def extract_page_texts(self, content: bytes) -> list[str]: ...

    def parse_page_texts(self, page_texts: list[str]) -> ParsedPdf: ...

    def parse_pdf_bytes(self, content: bytes) -> ParsedPdf: ...

    def build_chunks(self, text: str, sections: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

    def summarize_chunks(self, chunks: list[dict[str, Any]]) -> dict[str, Any]: ...

    def parse_layout_response(self, payload: list[Any]) -> ParsedPdf | None: ...


@dataclass(frozen=True)
class HttpResult:
    status: int
    headers: list[tuple[str, str]]
    body: bytes


class HttpAdapter(Protocol):
    def describe(self) -> dict[str, str]: ...

    def request(self, method: str, path: str, *, headers: list[tuple[str, str]], body: bytes | None) -> HttpResult: ...

    def reset_state(self) -> None: ...

    def openai_key(self, configured: bool) -> AbstractContextManager[None]: ...

    def close(self) -> None: ...


class HttpAdapterFactory(Protocol):
    def __call__(self, *, frontend_dist: Path) -> HttpAdapter: ...


class RetrievalAdapter(Protocol):
    def describe(self) -> dict[str, str]: ...

    def eval_search(self, method: str, query: str, *, k: int, adjacency_window: int) -> list[dict[str, Any]]: ...

    def retrieve_contexts(
        self, query: str, *, limit: int, arxiv_id: str | None, retrieval_mode: str
    ) -> tuple[list[dict[str, Any]], str]: ...

    def paper_chat_sources(
        self, arxiv_id: str, question: str, *, retrieval_mode: str
    ) -> tuple[list[dict[str, Any]], str] | None: ...

    def close(self) -> None: ...


def implementation_name() -> str:
    return os.environ.get(IMPLEMENTATION_ENV, "").strip() or DEFAULT_IMPLEMENTATION


def load_adapter(kind: str, implementation: str | None = None, **options: Any) -> Any:
    name = implementation or implementation_name()
    module_name = f"{__name__}.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise AdapterUnavailable(f"no golden adapter module {module_name}") from exc
        raise
    factory = getattr(module, f"{kind}_adapter", None)
    if factory is None:
        raise AdapterUnavailable(f"golden adapter {name!r} does not provide {kind!r}")
    return factory(**options)


def load_parser_adapter(implementation: str | None = None) -> ParserAdapter:
    return load_adapter("parser", implementation)


def load_retrieval_adapter(implementation: str | None = None) -> RetrievalAdapter:
    return load_adapter("retrieval", implementation)


def load_http_adapter(implementation: str | None = None, **options: Any) -> HttpAdapter:
    return load_adapter("http", implementation, **options)
