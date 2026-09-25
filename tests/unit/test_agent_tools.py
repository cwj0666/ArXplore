from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.core.agent import tools
from src.core.agent.tools import _format_context_papers, get_trending_papers_tool, search_paper_chunks_tool


def _retriever_context(**overrides) -> dict:
    context = {
        "chunk_id": "2401.00001:3",
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


def test_formatter_handles_empty_results():
    assert _format_context_papers([]) == "검색된 관련 논문이 없습니다."


def test_search_tool_docstring_describes_full_text_search():
    description = search_paper_chunks_tool.description

    assert "벡터" not in description
    assert "전문 검색" in description


def test_search_tool_formats_retriever_contexts():
    retriever = MagicMock()
    retriever.search_paper_contexts.return_value = [_retriever_context()]

    with patch.object(tools, "PaperRetriever", return_value=retriever):
        output = search_paper_chunks_tool.invoke({"query": "preference optimization"})

    retriever.search_paper_contexts.assert_called_once_with("preference optimization", limit=5)
    assert "Direct Preference Optimization Revisited" in output


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
