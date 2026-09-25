from typing import Any

from langchain_core.tools import tool

from src.integrations.paper_repository import PaperRepository
from src.integrations.paper_retriever import PaperRetriever


def _format_context_papers(context_papers: list[dict[str, Any]]) -> str:
    if not context_papers:
        return "검색된 관련 논문이 없습니다."

    formatted_docs = []
    for i, paper in enumerate(context_papers, 1):
        arxiv_id = paper.get("arxiv_id") or "Unknown ID"
        title = paper.get("paper_title") or paper.get("title") or "제목 없음"
        section_title = paper.get("section_title")
        content = (
            paper.get("context_text")
            or paper.get("chunk_text")
            or paper.get("text")
            or paper.get("abstract")
            or "내용 없음"
        )
        pdf_url = paper.get("pdf_url") or f"https://arxiv.org/abs/{arxiv_id}"

        header = f"[{i}] 출처(URL): {pdf_url} | 제목: {title}"
        if section_title:
            header += f" | 섹션: {section_title}"
        formatted_docs.append(f"{header}\n내용: {content}\n")

    return "\n".join(formatted_docs)


@tool
def search_paper_chunks_tool(query: str) -> str:
    """사용자의 기술 키워드로 PostgreSQL 전문 검색(full-text search)을 수행해 관련 논문 텍스트 청크를 가져옵니다.
    특정 주제에 대한 정보 조사가 필요할 때 사용하세요."""
    retriever = PaperRetriever()
    contexts = retriever.search_paper_contexts(query, limit=5)
    return _format_context_papers(contexts)


@tool
def get_trending_papers_tool() -> str:
    """요즘 최신 트렌디한 논문이 무엇인지, 혹은 추천수가 가장 높은 최근 논문이 무엇인지 파악해야 할 때 이 도구를 사용합니다.
    검색어에 상관없이 최신 DB 통계를 반환합니다."""
    repo = PaperRepository()
    papers = repo.list_recent_papers(limit=10)
    papers.sort(key=lambda x: x.get("upvotes", 0) or 0, reverse=True)
    if not papers:
        return "최근 논문 데이터가 없습니다."

    formatted = []
    for i, p in enumerate(papers, 1):
        pdf_link = p.get("pdf_url") or f"https://arxiv.org/abs/{p['arxiv_id']}"
        formatted.append(f"[{i}] {p['title']} | URL: {pdf_link} (추천수: {p.get('upvotes', 0)})")
    return "\n".join(formatted)
