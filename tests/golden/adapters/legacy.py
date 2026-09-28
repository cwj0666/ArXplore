import io
import os
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from src.integrations.fulltext_parser import FulltextParser
from src.integrations.pdf_parser.types import FulltextParseResult
from tests.golden.adapters import ParsedPdf

PYPDF_SOURCE = "pdf"


def _as_parsed(result: FulltextParseResult) -> ParsedPdf:
    return {
        "text": result.text,
        "sections": [dict(section) for section in result.sections],
        "source": result.source,
        "quality_metrics": dict(result.quality_metrics),
        "parser_metadata": dict(result.parser_metadata),
        "artifacts": dict(result.artifacts),
    }


class _ReplayLayoutClient:
    def __init__(self, payload: list[Any]) -> None:
        self._payload = payload

    def is_configured(self) -> bool:
        return True

    def analyze_pdf_bytes(self, content: bytes, **options: Any) -> list[dict[str, Any]]:
        from src.integrations.layout_parser_client import LayoutParserClient

        return [LayoutParserClient._normalize_segment(segment) for segment in self._payload]


class LegacyParserAdapter:
    def __init__(self) -> None:
        self._parser = FulltextParser()

    def describe(self) -> dict[str, str]:
        return {"implementation": "legacy", "pypdf": version("pypdf")}

    def extract_page_texts(self, content: bytes) -> list[str]:
        return [page.extract_text() or "" for page in PdfReader(io.BytesIO(content)).pages]

    def parse_page_texts(self, page_texts: list[str]) -> ParsedPdf:
        parser = self._parser
        pages = [parser._normalize_extracted_page_text(text) for text in page_texts]
        return self._from_extracted_text(parser._normalize_text("\n\n".join(page for page in pages if page)))

    def parse_pdf_bytes(self, content: bytes) -> ParsedPdf:
        return self._from_extracted_text(self._parser._extract_pdf_text(content))

    def build_chunks(self, text: str, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._parser.build_chunks(text, sections=sections)

    def summarize_chunks(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        return self._parser.summarize_chunks(chunks)

    def parse_layout_response(self, payload: list[Any]) -> ParsedPdf | None:
        parser = FulltextParser(layout_parser_client=_ReplayLayoutClient(payload))
        result = parser._parse_with_layout_parser(b"%PDF-replayed-layout-response")
        return None if result is None else _as_parsed(result)

    def _from_extracted_text(self, text: str) -> ParsedPdf:
        parser = self._parser
        if not text:
            return _as_parsed(parser._build_fallback_result(""))
        sections = parser._extract_sections(text) or [{"title": "Full Text", "text": text}]
        metrics = parser._build_fulltext_quality_metrics(text=text, sections=sections, source=PYPDF_SOURCE)
        return _as_parsed(
            FulltextParseResult(text=text, sections=sections, source=PYPDF_SOURCE, quality_metrics=metrics)
        )


def parser_adapter() -> LegacyParserAdapter:
    return LegacyParserAdapter()


@contextmanager
def _retrieval_mode(mode: str) -> Iterator[None]:
    from src.shared.settings import get_settings

    previous = os.environ.get("RETRIEVAL_MODE")
    os.environ["RETRIEVAL_MODE"] = mode
    get_settings.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("RETRIEVAL_MODE", None)
        else:
            os.environ["RETRIEVAL_MODE"] = previous
        get_settings.cache_clear()


class LegacyRetrievalAdapter:
    def __init__(self) -> None:
        from src.integrations.db import close_all_pools
        from src.integrations.paper_retriever import PaperRetriever
        from src.shared.settings import get_settings

        get_settings.cache_clear()
        close_all_pools()
        self._retriever = PaperRetriever()

    def describe(self) -> dict[str, str]:
        return {"implementation": "legacy"}

    def eval_search(self, method: str, query: str, *, k: int, adjacency_window: int) -> list[dict[str, Any]]:
        search = {
            "lexical": self._retriever.search_paper_contexts,
            "vector": self._retriever.search_paper_contexts_by_vector,
            "hybrid": self._retriever.search_paper_contexts_by_hybrid,
        }[method]
        return search(query, limit=k, adjacency_window=adjacency_window)

    def retrieve_contexts(
        self, query: str, *, limit: int, arxiv_id: str | None, retrieval_mode: str
    ) -> tuple[list[dict[str, Any]], str]:
        from src.core.agent.retrieval import retrieve_contexts

        with _retrieval_mode(retrieval_mode):
            return retrieve_contexts(query, retriever=self._retriever, limit=limit, arxiv_id=arxiv_id)

    def paper_chat_sources(
        self, arxiv_id: str, question: str, *, retrieval_mode: str
    ) -> tuple[list[dict[str, Any]], str] | None:
        from src.core.agent.paper_chat import retrieve_paper_chat_sources

        paper = self._retriever.repository.get_paper(arxiv_id)
        if paper is None:
            return None
        with _retrieval_mode(retrieval_mode):
            result = retrieve_paper_chat_sources(dict(paper), question, retriever=self._retriever)
        return result.sources, result.retrieval_mode

    def close(self) -> None:
        from src.integrations.db import close_all_pools

        close_all_pools()


def retrieval_adapter() -> LegacyRetrievalAdapter:
    return LegacyRetrievalAdapter()


class LegacyHttpAdapter:
    def __init__(self, *, frontend_dist: Path) -> None:
        import django
        import httpx
        from django.conf import settings
        from django.core.wsgi import get_wsgi_application

        if os.environ.get("DJANGO_SETTINGS_MODULE") != "arxplore_web.settings":
            raise RuntimeError("the HTTP golden runs the production settings module arxplore_web.settings")
        application = get_wsgi_application()
        settings.FRONTEND_DIST_DIR = frontend_dist
        self._django_version = django.get_version()
        self._client = httpx.Client(
            transport=httpx.WSGITransport(app=application, remote_addr="127.0.0.1"), base_url="http://localhost"
        )
        self._check_environment()

    @staticmethod
    def _check_environment() -> None:
        from src.shared.settings import get_settings

        settings = get_settings()
        problems = []
        if os.environ.get("LAYOUT_PARSER_BASE_URL") != "" or settings.layout_parser_base_url:
            problems.append("LAYOUT_PARSER_BASE_URL was refilled")
        if settings.openai_api_key != os.environ.get("OPENAI_API_KEY") or not settings.openai_api_key:
            problems.append("OPENAI_API_KEY in settings is not the harness key")
        if problems:
            raise RuntimeError("; ".join(problems))

    def describe(self) -> dict[str, str]:
        return {"implementation": "legacy", "framework": f"django {self._django_version}"}

    def request(self, method: str, path: str, *, headers: list[tuple[str, str]], body: bytes | None) -> Any:
        from tests.golden.adapters import HttpResult

        response = self._client.request(method, path, headers=headers, content=body)
        self._client.cookies.clear()
        return HttpResult(
            response.status_code,
            [(name.lower(), value) for name, value in response.headers.multi_items()],
            response.content,
        )

    def reset_state(self) -> None:
        from django.core.cache import cache

        cache.clear()

    @contextmanager
    def openai_key(self, configured: bool) -> Iterator[None]:
        if configured:
            yield
            return
        from src.shared.settings import get_settings

        previous = os.environ.get("OPENAI_API_KEY", "")
        os.environ["OPENAI_API_KEY"] = ""
        get_settings.cache_clear()
        try:
            if get_settings().openai_api_key:
                raise RuntimeError("the OpenAI key came back from a .env file; run the HTTP golden outside the repo")
            yield
        finally:
            os.environ["OPENAI_API_KEY"] = previous
            get_settings.cache_clear()

    def close(self) -> None:
        from src.integrations.db import close_all_pools

        self._client.close()
        close_all_pools()


def http_adapter(*, frontend_dist: Path) -> LegacyHttpAdapter:
    return LegacyHttpAdapter(frontend_dist=frontend_dist)
