from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from langchain_core.language_models import FakeListChatModel

from papers import services
from src.core.agent import paper_chat, retrieval
from src.core.agent.paper_chat import (
    ABSTRACT_SECTION_TITLE,
    FIRST_CHUNKS_MODE,
    PAPER_CHAT_CONTEXT_LIMIT,
    format_paper_chat_context,
    retrieve_paper_chat_sources,
    stream_paper_chat_answer,
)
from src.core.tracing import build_paper_chat_trace_config, resolve_trace_runtime

PAPER = {
    "arxiv_id": "2401.00001",
    "title": "Direct Preference Optimization Revisited",
    "abstract": "We revisit DPO.",
    "pdf_url": None,
}


def _context(chunk_id: int, section: str = "3 Method", text: str | None = None) -> dict:
    return {
        "chunk_id": chunk_id,
        "arxiv_id": PAPER["arxiv_id"],
        "paper_title": PAPER["title"],
        "chunk_text": f"chunk {chunk_id}",
        "context_text": text if text is not None else f"context {chunk_id}",
        "section_title": section,
    }


def _retriever(*, hybrid=None, lexical=None, first_chunks=None, embedding_available: bool = True) -> MagicMock:
    retriever = MagicMock()
    retriever.embedding_client.is_available.return_value = embedding_available
    retriever.search_paper_contexts_by_hybrid.return_value = hybrid or []
    retriever.search_paper_contexts.return_value = lexical or []
    retriever.repository.list_paper_chunks.return_value = first_chunks or []
    return retriever


def _settings(mode: str) -> SimpleNamespace:
    return SimpleNamespace(retrieval_mode=mode)


class TestRetrieval:
    def test_hybrid_search_is_scoped_to_paper_and_abstract_comes_first(self):
        retriever = _retriever(hybrid=[_context(10), _context(11, "4 Experiments")])

        result = retrieve_paper_chat_sources(PAPER, "what is the loss?", retriever=retriever)

        retriever.search_paper_contexts_by_hybrid.assert_called_once_with(
            "what is the loss?", arxiv_id="2401.00001", limit=PAPER_CHAT_CONTEXT_LIMIT
        )
        retriever.search_paper_contexts.assert_not_called()
        assert result.retrieval_mode == "hybrid"
        assert [source["section_title"] for source in result.sources] == [
            ABSTRACT_SECTION_TITLE,
            "3 Method",
            "4 Experiments",
        ]
        assert result.sources[0]["text"] == "We revisit DPO."
        assert result.sources[0]["chunk_id"] is None
        assert [source["chunk_id"] for source in result.sources[1:]] == [10, 11]
        assert {source["url"] for source in result.sources} == {"https://arxiv.org/abs/2401.00001"}

    def test_lexical_setting_skips_hybrid(self):
        retriever = _retriever(lexical=[_context(10)])

        with patch.object(retrieval, "get_settings", return_value=_settings("lexical")):
            result = retrieve_paper_chat_sources(PAPER, "q", retriever=retriever)

        retriever.search_paper_contexts_by_hybrid.assert_not_called()
        retriever.search_paper_contexts.assert_called_once_with(
            "q", arxiv_id="2401.00001", limit=PAPER_CHAT_CONTEXT_LIMIT
        )
        assert result.retrieval_mode == "lexical"

    def test_missing_embedding_key_degrades_to_lexical(self):
        retriever = _retriever(lexical=[_context(10)], embedding_available=False)

        result = retrieve_paper_chat_sources(PAPER, "q", retriever=retriever)

        retriever.search_paper_contexts_by_hybrid.assert_not_called()
        assert result.retrieval_mode == "lexical"

    def test_embedding_api_error_degrades_to_lexical(self):
        retriever = _retriever(lexical=[_context(10)])
        request = httpx.Request("POST", "https://api.openai.com/v1/embeddings")
        retriever.search_paper_contexts_by_hybrid.side_effect = openai.APIConnectionError(request=request)

        result = retrieve_paper_chat_sources(PAPER, "q", retriever=retriever)

        assert result.retrieval_mode == "lexical"
        assert [source["chunk_id"] for source in result.sources[1:]] == [10]

    def test_empty_retrieval_falls_back_to_first_chunks(self):
        first_chunks = [
            {
                "chunk_id": 1,
                "arxiv_id": PAPER["arxiv_id"],
                "chunk_index": 0,
                "chunk_text": "intro text",
                "section_title": "1 Introduction",
            },
            {
                "chunk_id": 2,
                "arxiv_id": PAPER["arxiv_id"],
                "chunk_index": 1,
                "chunk_text": "more",
                "section_title": None,
            },
        ]
        retriever = _retriever(first_chunks=first_chunks)

        result = retrieve_paper_chat_sources(PAPER, "한국어 질문", retriever=retriever)

        retriever.repository.list_paper_chunks.assert_called_once_with("2401.00001", limit=PAPER_CHAT_CONTEXT_LIMIT)
        assert result.retrieval_mode == FIRST_CHUNKS_MODE
        assert [(s["chunk_id"], s["section_title"], s["text"]) for s in result.sources[1:]] == [
            (1, "1 Introduction", "intro text"),
            (2, None, "more"),
        ]

    def test_duplicate_and_empty_contexts_are_dropped(self):
        retriever = _retriever(hybrid=[_context(10), _context(10), _context(11, text="")])
        retriever.search_paper_contexts_by_hybrid.return_value[2]["chunk_text"] = ""

        result = retrieve_paper_chat_sources(PAPER, "q", retriever=retriever)

        assert [source["chunk_id"] for source in result.sources] == [None, 10]

    def test_paper_without_abstract_has_no_abstract_source(self):
        retriever = _retriever(hybrid=[_context(10)])

        result = retrieve_paper_chat_sources({**PAPER, "abstract": ""}, "q", retriever=retriever)

        assert [source["chunk_id"] for source in result.sources] == [10]


def test_context_format_numbers_sources_with_section_and_chunk_id():
    retriever = _retriever(hybrid=[_context(10)])
    result = retrieve_paper_chat_sources(PAPER, "q", retriever=retriever)

    formatted = format_paper_chat_context(result.sources)

    assert "[1] 섹션: Abstract | chunk_id: -\nWe revisit DPO." in formatted
    assert "[2] 섹션: 3 Method | chunk_id: 10\ncontext 10" in formatted


class TestStreaming:
    def _retrieval(self):
        return retrieve_paper_chat_sources(PAPER, "q", retriever=_retriever(hybrid=[_context(10), _context(11)]))

    def test_streams_chunks_then_citations_for_referenced_sources(self):
        llm = FakeListChatModel(responses=["손실 함수는 [2]에 나옵니다."])

        events = list(stream_paper_chat_answer("q", paper=PAPER, retrieval=self._retrieval(), llm=llm))

        assert all("chunk" in event for event in events[:-1])
        assert len(events) > 2
        assert "".join(event["chunk"] for event in events[:-1]) == "손실 함수는 [2]에 나옵니다."
        assert events[-1] == {
            "citations": [
                {
                    "arxiv_id": "2401.00001",
                    "title": PAPER["title"],
                    "url": "https://arxiv.org/abs/2401.00001",
                    "section_title": "3 Method",
                    "chunk_id": 10,
                    "in_answer": True,
                }
            ]
        }

    def test_unreferenced_answer_returns_all_sources_unverified(self):
        llm = FakeListChatModel(responses=["번호 없는 답변입니다."])

        events = list(stream_paper_chat_answer("q", paper=PAPER, retrieval=self._retrieval(), llm=llm))

        citations = events[-1]["citations"]
        assert [(c["section_title"], c["chunk_id"], c["in_answer"]) for c in citations] == [
            ("Abstract", None, False),
            ("3 Method", 10, False),
            ("3 Method", 11, False),
        ]

    def test_out_of_range_reference_is_ignored(self):
        llm = FakeListChatModel(responses=["근거 [9]"])

        events = list(stream_paper_chat_answer("q", paper=PAPER, retrieval=self._retrieval(), llm=llm))

        assert all(c["in_answer"] is False for c in events[-1]["citations"])


def test_paper_chat_trace_uses_own_stage_and_runtime():
    config = build_paper_chat_trace_config(
        runtime="production", user="tester", extra_metadata={"retrieval_mode": "hybrid"}
    )

    assert config["run_name"] == "paper_chat"
    assert config["metadata"]["runtime"] == "production"
    assert config["metadata"]["retrieval_mode"] == "hybrid"


def test_trace_runtime_resolution():
    assert resolve_trace_runtime("development") == "dev"
    assert resolve_trace_runtime("production") == "production"
    assert resolve_trace_runtime("PROD") == "production"
    assert resolve_trace_runtime("staging") == "dev"


class _User:
    is_authenticated = True

    def get_username(self) -> str:
        return "tester"


class TestServices:
    def _repo(self, paper=PAPER):
        repo = MagicMock()
        repo.get_paper.return_value = paper
        return repo

    def test_stream_paper_chat_uses_session_key_and_paper_scope(self):
        repo = self._repo()
        retriever = _retriever(hybrid=[_context(10)])
        seen_keys: list[str | None] = []

        def build_llm(**_kwargs):
            from src.shared import get_runtime_openai_api_key

            seen_keys.append(get_runtime_openai_api_key())
            return FakeListChatModel(responses=["답 [2]"])

        with (
            patch.object(services, "get_paper_repository", return_value=repo),
            patch("src.integrations.paper_retriever.PaperRetriever", return_value=retriever) as retriever_cls,
            patch.object(paper_chat, "build_chat_llm", side_effect=build_llm),
        ):
            prepared = services.prepare_paper_chat("2401.00001", "질문", [], user=_User(), session_api_key="sk-user")
            events = list(services.stream_paper_chat(prepared))

        assert seen_keys == ["sk-user"]
        assert retriever_cls.call_args.kwargs["repository"] is repo
        assert retriever.search_paper_contexts_by_hybrid.call_args.kwargs["arxiv_id"] == "2401.00001"
        assert events[-1]["citations"][0]["chunk_id"] == 10

    def test_answer_paper_chat_returns_answer_citations_and_mode(self):
        retriever = _retriever(first_chunks=[{"chunk_id": 1, "chunk_text": "intro", "section_title": "1 Introduction"}])

        with (
            patch.object(services, "get_paper_repository", return_value=self._repo()),
            patch("src.integrations.paper_retriever.PaperRetriever", return_value=retriever),
            patch.object(paper_chat, "build_chat_llm", return_value=FakeListChatModel(responses=["답변"])),
        ):
            payload = services.answer_paper_chat("2401.00001", "질문", [], user=_User(), session_api_key="sk-user")

        assert payload["answer"] == "답변"
        assert payload["retrieval_mode"] == FIRST_CHUNKS_MODE
        assert [c["chunk_id"] for c in payload["citations"]] == [None, 1]

    def test_prepare_paper_chat_raises_not_found(self):
        repo = self._repo(paper=None)

        with (
            patch.object(services, "get_paper_repository", return_value=repo),
            pytest.raises(services.PaperNotFoundError),
        ):
            services.prepare_paper_chat("2401.99999", "질문", [], user=_User(), session_api_key="sk-user")
