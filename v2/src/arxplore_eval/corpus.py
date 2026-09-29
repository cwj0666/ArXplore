from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from arxplore_eval.dataset import EvalQuery

CORPUS_COUNTS_SQL = """
    SELECT
        (SELECT COUNT(*) FROM papers) AS papers,
        (SELECT COUNT(DISTINCT arxiv_id) FROM paper_chunks) AS papers_with_chunks,
        (SELECT COUNT(*) FROM paper_chunks) AS chunks,
        (SELECT COUNT(*) FROM paper_embeddings) AS embeddings
"""
FULLTEXT_SOURCES_SQL = "SELECT source, COUNT(*) AS count FROM paper_fulltexts GROUP BY source ORDER BY source"
EXISTING_PAPERS_SQL = "SELECT arxiv_id FROM papers WHERE arxiv_id = ANY(%s)"
EXISTING_CHUNKS_SQL = "SELECT id FROM paper_chunks WHERE id = ANY(%s)"
CHUNK_TEXTS_SQL = "SELECT id, chunk_text FROM paper_chunks WHERE id = ANY(%s)"
DATABASE_INFO_SQL = """
    SELECT
        current_setting('server_version') AS server_version,
        (SELECT extversion FROM pg_extension WHERE extname = 'vector') AS vector_version,
        current_setting('shared_buffers') AS shared_buffers
"""
PREVIEW_IDS = 10

type Connection = AsyncConnection[Any]


class PreconditionError(RuntimeError):
    pass


@dataclass(frozen=True)
class CorpusSummary:
    papers: int
    papers_with_chunks: int
    chunks: int
    embeddings: int
    fulltext_sources: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        return (
            f"papers {self.papers} (청크 보유 {self.papers_with_chunks}), "
            f"chunks {self.chunks}, embeddings {self.embeddings}"
        )

    def describe_sources(self) -> str:
        return ", ".join(f"{key} {value}" for key, value in self.fulltext_sources.items()) or "없음"


async def _rows(connection: Connection, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
    async with connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(sql, params)
        return await cursor.fetchall()


async def corpus_summary(connection: Connection) -> CorpusSummary:
    counts = (await _rows(connection, CORPUS_COUNTS_SQL))[0]
    sources = {str(row["source"]): int(row["count"]) for row in await _rows(connection, FULLTEXT_SOURCES_SQL)}
    return CorpusSummary(
        papers=int(counts["papers"]),
        papers_with_chunks=int(counts["papers_with_chunks"]),
        chunks=int(counts["chunks"]),
        embeddings=int(counts["embeddings"]),
        fulltext_sources=sources,
    )


async def existing_paper_ids(connection: Connection, arxiv_ids: Sequence[str]) -> set[str]:
    if not arxiv_ids:
        return set()
    return {str(row["arxiv_id"]) for row in await _rows(connection, EXISTING_PAPERS_SQL, (list(arxiv_ids),))}


async def existing_chunk_ids(connection: Connection, chunk_ids: Sequence[int]) -> set[int]:
    if not chunk_ids:
        return set()
    return {int(row["id"]) for row in await _rows(connection, EXISTING_CHUNKS_SQL, (list(chunk_ids),))}


async def fetch_chunk_texts(connection: Connection, chunk_ids: Sequence[int]) -> dict[int, str]:
    if not chunk_ids:
        return {}
    rows = await _rows(connection, CHUNK_TEXTS_SQL, (list(chunk_ids),))
    return {int(row["id"]): str(row["chunk_text"] or "") for row in rows}


async def retrieval_preflight(
    connection: Connection, queries: Iterable[EvalQuery], *, needs_embeddings: bool
) -> CorpusSummary:
    selected = list(queries)
    arxiv_ids = sorted({value for query in selected for value in query.relevant_arxiv_ids})
    chunk_ids = sorted({value for query in selected for value in query.relevant_chunk_ids})
    summary = await corpus_summary(connection)
    found_papers = await existing_paper_ids(connection, arxiv_ids)
    found_chunks = await existing_chunk_ids(connection, chunk_ids)

    problems: list[str] = []
    missing_papers = [value for value in arxiv_ids if value not in found_papers]
    missing_chunks = [value for value in chunk_ids if value not in found_chunks]
    if missing_papers:
        problems.append(f"정답 arxiv_id {len(missing_papers)}개가 papers에 없습니다: {missing_papers[:PREVIEW_IDS]}")
    if missing_chunks:
        problems.append(
            f"정답 chunk_id {len(missing_chunks)}개가 paper_chunks에 없습니다(재청킹·백필 후 id가 바뀌었을 수 있음): "
            f"{missing_chunks[:PREVIEW_IDS]}"
        )
    if not summary.chunks:
        problems.append("paper_chunks가 비어 있습니다.")
    if needs_embeddings and not summary.embeddings:
        problems.append(
            "paper_embeddings가 비어 있어 vector/hybrid 결과가 의미 없습니다. --methods lexical로 실행하세요."
        )
    if problems:
        raise PreconditionError("\n".join(problems))
    return summary


def paper_chat_targets(queries: Iterable[EvalQuery], modes: Sequence[str]) -> list[str]:
    if "paper_chat" not in modes:
        return []
    return sorted(
        {
            query.paper_chat_arxiv_id
            for query in queries
            if query.supports_mode("paper_chat") and not query.expect_rejected and query.paper_chat_arxiv_id
        }
    )


async def generation_preflight(
    connection: Connection, queries: Iterable[EvalQuery], *, modes: Sequence[str]
) -> CorpusSummary:
    targets = paper_chat_targets(queries, modes)
    summary = await corpus_summary(connection)
    found = await existing_paper_ids(connection, targets)
    problems: list[str] = []
    missing = [value for value in targets if value not in found]
    if missing:
        problems.append(f"상세 챗 대상 arxiv_id {len(missing)}개가 papers에 없습니다: {missing[:PREVIEW_IDS]}")
    if not summary.chunks:
        problems.append("paper_chunks가 비어 있습니다.")
    if problems:
        raise PreconditionError("\n".join(problems))
    return summary


async def database_info(connection: Connection) -> dict[str, str]:
    row = (await _rows(connection, DATABASE_INFO_SQL))[0]
    return {
        "server_version": str(row["server_version"]),
        "vector_version": str(row["vector_version"]),
        "shared_buffers": str(row["shared_buffers"]),
    }
