from collections.abc import Sequence
from typing import Protocol

from langchain_core.tools import BaseTool, tool

from arxplore.domain.papers import Paper
from arxplore.domain.retrieval import RetrievalContext
from arxplore.llm.citations import (
    UNTITLED,
    CitationSource,
    optional_int,
    optional_str,
    paper_url,
    record_tool_hits,
)
from arxplore.llm.paper_chat import ContextRetriever

SEARCH_TOOL_NAME = "search_paper_chunks_tool"
SEARCH_TOOL_DESCRIPTION = (
    "논문 본문 청크를 검색합니다. PostgreSQL 전문 검색과 pgvector 벡터 검색의 점수를 결합한 hybrid 검색을 쓰고,\n"
    "    질의 임베딩을 만들 수 없으면 전문 검색만 씁니다. 특정 주제에 대한 정보 조사가 필요할 때 사용하세요."
)
TRENDING_TOOL_NAME = "get_trending_papers_tool"
TRENDING_TOOL_DESCRIPTION = (
    "요즘 최신 트렌디한 논문이 무엇인지, 혹은 추천수가 가장 높은 최근 논문이 무엇인지 파악해야 할 때 이 도구를 사용합니다.\n"
    "    검색어에 상관없이 최신 DB 통계를 반환합니다."
)
SEARCH_RESULT_LIMIT = 5
TRENDING_PAPER_LIMIT = 10
NO_SEARCH_RESULTS_MESSAGE = (
    "검색된 관련 논문이 없습니다. 이 결과를 근거로 논문을 추천하거나 지어내지 말고, 검색 결과가 없다고 답하세요."
)
NO_TRENDING_PAPERS_MESSAGE = "최근 논문 데이터가 없습니다. 논문을 지어내지 말고 데이터가 없다고 답하세요."
EMPTY_CONTENT = "내용 없음"
MISSING_FIELD = "-"


class RecentPaperReader(Protocol):
    async def list_recent_papers(self, *, limit: int) -> Sequence[Paper]: ...


def context_citation_source(context: RetrievalContext) -> CitationSource:
    hit = context.hit
    arxiv_id = str(hit.arxiv_id or "")
    return CitationSource(
        arxiv_id=arxiv_id,
        title=str(hit.paper_title or UNTITLED),
        url=paper_url(arxiv_id),
        section_title=optional_str(hit.section_title),
        chunk_id=optional_int(hit.chunk_id),
    )


def format_search_results(contexts: Sequence[RetrievalContext]) -> str:
    if not contexts:
        return NO_SEARCH_RESULTS_MESSAGE
    documents = []
    for index, context in enumerate(contexts, 1):
        source = context_citation_source(context)
        chunk_id = source.chunk_id if source.chunk_id is not None else MISSING_FIELD
        header = (
            f"[{index}] 제목: {source.title}"
            f" | arxiv_id: {source.arxiv_id or MISSING_FIELD}"
            f" | 섹션: {source.section_title or MISSING_FIELD}"
            f" | chunk_id: {chunk_id}"
        )
        content = context.text or context.hit.chunk_text or EMPTY_CONTENT
        documents.append(f"{header}\n출처(URL): {source.url}\n내용: {content}\n")
    return "\n".join(documents)


def rank_trending_papers(papers: Sequence[Paper]) -> list[Paper]:
    return sorted(papers, key=lambda paper: paper.upvotes or 0, reverse=True)


def trending_citation_source(paper: Paper) -> CitationSource:
    return CitationSource(
        arxiv_id=paper.arxiv_id,
        title=paper.title or "",
        url=paper_url(paper.arxiv_id, paper.pdf_url),
        section_title=None,
        chunk_id=None,
    )


def format_trending_papers(papers: Sequence[Paper]) -> str:
    lines = []
    for index, paper in enumerate(papers, 1):
        link = paper_url(paper.arxiv_id, paper.pdf_url)
        lines.append(f"[{index}] {paper.title} | URL: {link} (추천수: {paper.upvotes})")
    return "\n".join(lines)


def build_agent_tools(retriever: ContextRetriever, papers: RecentPaperReader) -> list[BaseTool]:
    @tool(SEARCH_TOOL_NAME, description=SEARCH_TOOL_DESCRIPTION)
    async def search_paper_chunks_tool(query: str) -> str:
        result = await retriever.retrieve_contexts(query, limit=SEARCH_RESULT_LIMIT)
        record_tool_hits(context_citation_source(context) for context in result.contexts)
        return format_search_results(result.contexts)

    @tool(TRENDING_TOOL_NAME, description=TRENDING_TOOL_DESCRIPTION)
    async def get_trending_papers_tool() -> str:
        ranked = rank_trending_papers(await papers.list_recent_papers(limit=TRENDING_PAPER_LIMIT))
        if not ranked:
            return NO_TRENDING_PAPERS_MESSAGE
        record_tool_hits(trending_citation_source(paper) for paper in ranked)
        return format_trending_papers(ranked)

    return [search_paper_chunks_tool, get_trending_papers_tool]
