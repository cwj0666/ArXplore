from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.core.agent import retrieval, tools
from src.core.agent.citations import collect_tool_hits
from src.core.agent.tools import (
    NO_SEARCH_RESULTS_MESSAGE,
    _format_context_papers,
    get_trending_papers_tool,
    search_paper_chunks_tool,
)
from src.integrations.embedding_client import EmbeddingClient
from src.shared import override_openai_runtime


def _retriever_context(**overrides) -> dict:
    context = {
        "chunk_id": 3,
        "arxiv_id": "2401.00001",
        "paper_title": "Direct Preference Optimization Revisited",
        "chunk_text": "hit chunk only",
        "context_text": "previous chunk\n\nhit chunk only\n\nnext chunk",
        "section_title": "3 Method",
        "content_role": "body",
        "score": 0.42,
    }
    context.update(overrides)
    return context


def test_formatter_uses_retriever_paper_title():
    formatted = _format_context_papers([_retriever_context()])

    assert "제목: Direct Preference Optimization Revisited" in formatted
    assert "제목 없음" not in formatted


def test_formatter_falls_back_to_title_key():
    context = _retriever_context()
    del context["paper_title"]
    context["title"] = "Legacy Title"

    assert "제목: Legacy Title" in _format_context_papers([context])


def test_formatter_prefers_context_text_over_chunk_text():
    formatted = _format_context_papers([_retriever_context()])

    assert "previous chunk" in formatted
    assert "next chunk" in formatted


def test_formatter_uses_chunk_text_when_context_text_missing():
    formatted = _format_context_papers([_retriever_context(context_text="")])

    assert "내용: hit chunk only" in formatted


def test_formatter_includes_section_title():
    assert "섹션: 3 Method" in _format_context_papers([_retriever_context()])


def test_formatter_url_prefers_pdf_url_then_abs_link():
    with_pdf = _format_context_papers([_retriever_context(pdf_url="https://arxiv.org/pdf/2401.00001")])
    without_pdf = _format_context_papers([_retriever_context()])

    assert "출처(URL): https://arxiv.org/pdf/2401.00001" in with_pdf
    assert "출처(URL): https://arxiv.org/abs/2401.00001" in without_pdf


def test_formatter_includes_every_citation_field():
    formatted = _format_context_papers([_retriever_context()])

    assert formatted.startswith(
        "[1] 제목: Direct Preference Optimization Revisited | arxiv_id: 2401.00001 | 섹션: 3 Method | chunk_id: 3\n"
        "출처(URL): https://arxiv.org/abs/2401.00001\n"
        "내용: previous chunk"
    )


def test_formatter_marks_missing_section_and_chunk_id():
    formatted = _format_context_papers([_retriever_context(section_title="", chunk_id=None)])

    assert "섹션: - | chunk_id: -" in formatted


def test_formatter_handles_empty_results():
    assert _format_context_papers([]) == NO_SEARCH_RESULTS_MESSAGE
    assert "지어내지 말고" in NO_SEARCH_RESULTS_MESSAGE


def test_search_tool_docstring_describes_hybrid_search():
    description = search_paper_chunks_tool.description

    assert "PostgreSQL 전문 검색과 pgvector 벡터 검색을 RRF로 결합" in description


def _retriever(contexts: list[dict], *, embedding_available: bool = True) -> MagicMock:
    retriever = MagicMock()
    retriever.embedding_client.is_available.return_value = embedding_available
    retriever.search_paper_contexts_by_hybrid.return_value = contexts
    retriever.search_paper_contexts.return_value = contexts
    return retriever


def test_search_tool_uses_hybrid_when_embedding_key_available():
    retriever = _retriever([_retriever_context()])

    with patch.object(tools, "PaperRetriever", return_value=retriever):
        output = search_paper_chunks_tool.invoke({"query": "preference optimization"})

    retriever.search_paper_contexts_by_hybrid.assert_called_once_with("preference optimization", arxiv_id=None, limit=5)
    retriever.search_paper_contexts.assert_not_called()
    assert "Direct Preference Optimization Revisited" in output


def test_search_tool_uses_lexical_without_embedding_key():
    retriever = _retriever([_retriever_context()], embedding_available=False)

    with patch.object(tools, "PaperRetriever", return_value=retriever):
        search_paper_chunks_tool.invoke({"query": "dpo"})

    retriever.search_paper_contexts_by_hybrid.assert_not_called()
    retriever.search_paper_contexts.assert_called_once_with("dpo", arxiv_id=None, limit=5)


def test_search_tool_honors_lexical_retrieval_mode():
    retriever = _retriever([_retriever_context()])

    with (
        patch.object(tools, "PaperRetriever", return_value=retriever),
        patch.object(retrieval, "get_settings", return_value=SimpleNamespace(retrieval_mode="lexical")),
    ):
        search_paper_chunks_tool.invoke({"query": "dpo"})

    retriever.search_paper_contexts_by_hybrid.assert_not_called()


def test_search_tool_records_hits_for_citations():
    retriever = _retriever([_retriever_context()])

    with patch.object(tools, "PaperRetriever", return_value=retriever), collect_tool_hits() as hits:
        search_paper_chunks_tool.invoke({"query": "dpo"})

    assert hits == [
        {
            "arxiv_id": "2401.00001",
            "title": "Direct Preference Optimization Revisited",
            "url": "https://arxiv.org/abs/2401.00001",
            "section_title": "3 Method",
            "chunk_id": 3,
        }
    ]


def test_trending_tool_joins_lines_with_real_newline():
    repo = MagicMock()
    repo.list_recent_papers.return_value = [
        {"arxiv_id": "2401.00001", "title": "Low Votes", "upvotes": 1},
        {"arxiv_id": "2401.00002", "title": "High Votes", "upvotes": 10, "pdf_url": "https://arxiv.org/pdf/2401.00002"},
    ]

    with patch.object(tools, "PaperRepository", return_value=repo):
        output = get_trending_papers_tool.invoke({})

    lines = output.split("\n")
    assert "\\n" not in output
    assert len(lines) == 2
    assert lines[0].startswith("[1] High Votes | URL: https://arxiv.org/pdf/2401.00002")
    assert lines[1].startswith("[2] Low Votes | URL: https://arxiv.org/abs/2401.00001")


def test_trending_tool_records_hits_with_same_urls():
    repo = MagicMock()
    repo.list_recent_papers.return_value = [
        {"arxiv_id": "2401.00002", "title": "High Votes", "upvotes": 10, "pdf_url": "https://arxiv.org/pdf/2401.00002"},
    ]

    with patch.object(tools, "PaperRepository", return_value=repo), collect_tool_hits() as hits:
        output = get_trending_papers_tool.invoke({})

    assert hits[0]["url"] == "https://arxiv.org/pdf/2401.00002"
    assert hits[0]["url"] in output
    assert hits[0]["chunk_id"] is None


class TestEmbeddingKeyResolution:
    def _client(self, server_key: str | None) -> EmbeddingClient:
        settings = SimpleNamespace(
            openai_api_key=server_key,
            openai_embedding_model="text-embedding-3-large",
            openai_embedding_dimensions=1536,
            embedding_batch_size=64,
        )
        return EmbeddingClient(settings=settings)

    def test_session_key_wins_over_server_key(self):
        client = self._client("sk-server")

        with override_openai_runtime(api_key="sk-session"):
            assert client.resolve_api_key() == "sk-session"
            assert client._get_client().api_key == "sk-session"

    def test_server_key_is_used_without_session_key(self):
        client = self._client("sk-server")

        assert client.resolve_api_key() == "sk-server"
        assert client._get_client().api_key == "sk-server"

    def test_blank_session_key_falls_back_to_server_key(self):
        client = self._client("sk-server")

        with override_openai_runtime(api_key="  "):
            assert client.resolve_api_key() == "sk-server"

    def test_client_is_rebuilt_when_key_changes(self):
        client = self._client("sk-server")
        with override_openai_runtime(api_key="sk-a"):
            first = client._get_client()
        with override_openai_runtime(api_key="sk-b"):
            second = client._get_client()

        assert first.api_key == "sk-a"
        assert second.api_key == "sk-b"

    def test_no_key_means_unavailable_and_retrieval_degrades_to_lexical(self):
        client = self._client(None)
        retriever = MagicMock()
        retriever.embedding_client = client
        retriever.search_paper_contexts.return_value = []

        assert client.resolve_api_key() is None
        assert client.is_available() is False
        assert retrieval.resolve_retrieval_mode(retriever) == "lexical"
        contexts, mode = retrieval.retrieve_contexts("q", retriever=retriever, limit=5)
        assert (contexts, mode) == ([], "lexical")
        retriever.search_paper_contexts_by_hybrid.assert_not_called()

    def test_session_key_alone_enables_hybrid(self):
        retriever = MagicMock()
        retriever.embedding_client = self._client(None)

        with override_openai_runtime(api_key="sk-session"):
            assert retrieval.resolve_retrieval_mode(retriever) == "hybrid"
