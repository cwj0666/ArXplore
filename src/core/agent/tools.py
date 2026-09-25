from typing import Any

from langchain_core.tools import tool

from src.core.rag_types import context_text, paper_url, source_from_context
from src.integrations.paper_repository import PaperRepository
from src.integrations.paper_retriever import PaperRetriever

from .citations import record_tool_hits
from .retrieval import retrieve_contexts

SEARCH_RESULT_LIMIT = 5
TRENDING_PAPER_LIMIT = 10
NO_SEARCH_RESULTS_MESSAGE = (
    "검색된 관련 논문이 없습니다. 이 결과를 근거로 논문을 추천하거나 지어내지 말고, 검색 결과가 없다고 답하세요."
)


def _format_context_papers(context_papers: list[dict[str, Any]]) -> str:
    if not context_papers:
        return NO_SEARCH_RESULTS_MESSAGE

    formatted_docs = []
    for index, paper in enumerate(context_papers, 1):
        source = source_from_context(paper)
        chunk_id = source["chunk_id"]
        header = (
            f"[{index}] 제목: {source['title']}"
            f" | arxiv_id: {source['arxiv_id'] or '-'}"
            f" | 섹션: {source['section_title'] or '-'}"
            f" | chunk_id: {chunk_id if chunk_id is not None else '-'}"
        )
        content = context_text(paper) or "내용 없음"
        formatted_docs.append(f"{header}\n출처(URL): {source['url']}\n내용: {content}\n")

    return "\n".join(formatted_docs)


@tool
def search_paper_chunks_tool(query: str) -> str:
    """논문 본문 청크를 검색합니다. PostgreSQL 전문 검색과 pgvector 벡터 검색을 RRF로 결합한 hybrid 검색을 쓰고,
    질의 임베딩을 만들 수 없으면 전문 검색만 씁니다. 특정 주제에 대한 정보 조사가 필요할 때 사용하세요."""
    contexts, _mode = retrieve_contexts(query, retriever=PaperRetriever(), limit=SEARCH_RESULT_LIMIT)
    record_tool_hits(source_from_context(context) for context in contexts)
    return _format_context_papers(contexts)


@tool
def get_trending_papers_tool() -> str:
    """요즘 최신 트렌디한 논문이 무엇인지, 혹은 추천수가 가장 높은 최근 논문이 무엇인지 파악해야 할 때 이 도구를 사용합니다.
    검색어에 상관없이 최신 DB 통계를 반환합니다."""
    repo = PaperRepository()
    papers = repo.list_recent_papers(limit=TRENDING_PAPER_LIMIT)
    papers.sort(key=lambda x: x.get("upvotes", 0) or 0, reverse=True)
    if not papers:
        return "최근 논문 데이터가 없습니다. 논문을 지어내지 말고 데이터가 없다고 답하세요."

    formatted = []
    hits = []
    for i, p in enumerate(papers, 1):
        pdf_link = paper_url(p["arxiv_id"], p.get("pdf_url"))
        formatted.append(f"[{i}] {p['title']} | URL: {pdf_link} (추천수: {p.get('upvotes', 0)})")
        hits.append(
            {
                "arxiv_id": p["arxiv_id"],
                "title": p.get("title") or "",
                "url": pdf_link,
                "section_title": None,
                "chunk_id": None,
            }
        )
    record_tool_hits(hits)
    return "\n".join(formatted)
