import re
from collections.abc import Mapping, Sequence, Set
from dataclasses import replace
from decimal import Decimal
from typing import Any

from psycopg.rows import dict_row

from arxplore.db.pool import ConnectionSource, use_connection
from arxplore.db.repositories.embeddings import vector_literal
from arxplore.domain.retrieval import RetrievalHit, VectorScores
from arxplore.parsing.roles import FRONT_MATTER_ROLE, TABLE_LIKE_ROLE, is_references_section_title
from arxplore.retrieval.lexical import PER_PAPER_CANDIDATE_CAP
from arxplore.retrieval.query import (
    citation_intent_requested,
    query_tokens,
    requested_section_intents,
    section_intent_bonus,
)
from arxplore.retrieval.signals import looks_reference_like_text
from arxplore.retrieval.snippet import build_search_snippet

MIN_VECTOR_CANDIDATES = 80
MIN_SCOPED_VECTOR_CANDIDATES = 40
VECTOR_CANDIDATES_PER_RESULT = 8
SCOPED_VECTOR_CANDIDATES_PER_RESULT = 4
MAX_HNSW_EF_SEARCH = 1000
DEFAULT_HNSW_EF_SEARCH = 40
MAX_OVERLAP_BONUS = 0.12
OVERLAP_BONUS_PER_TOKEN = 0.03
APPENDIX_QUERY_KEYWORDS = ("appendix", "supplement", "additional analysis")
APPENDIX_SECTION_KEYWORDS = (
    "appendix",
    "additional analysis",
    "supplementary",
    "experimental details",
    "implementation details",
)
CONCLUSION_QUERY_KEYWORDS = ("conclusion", "limitation", "discussion")
CONCLUSION_SECTION_KEYWORDS = ("conclusion", "discussion", "limitations")
PENALIZED_CONTENT_ROLES = frozenset({FRONT_MATTER_ROLE, TABLE_LIKE_ROLE})

_TEXT_TOKEN = re.compile(r"[a-z0-9]+")

_CONTENT_ROLE_ADJUSTMENT_SQL = """
    CASE
        WHEN COALESCE(c.metadata->>'content_role', '') = 'references' THEN -0.22
        WHEN COALESCE(c.metadata->>'content_role', '') = 'toc' THEN -0.28
        WHEN COALESCE(c.metadata->>'content_role', '') = 'front_matter' THEN -0.16
        WHEN COALESCE(c.metadata->>'content_role', '') = 'table_like' THEN -0.1
        WHEN COALESCE(c.metadata->>'content_role', '') = 'figure_caption' THEN -0.08
        WHEN COALESCE(c.metadata->>'content_role', '') = 'appendix' THEN -0.1
        ELSE 0
    END
"""

_SECTION_BOOST_SQL = """
    CASE
        WHEN c.section_title ILIKE 'Abstract' THEN 0.12
        WHEN c.section_title ILIKE '%%Introduction%%' THEN 0.09
        WHEN c.section_title ILIKE '%%Method%%' OR c.section_title ILIKE '%%Approach%%' THEN 0.03
        WHEN c.section_title ILIKE '%%Related Work%%' THEN 0.02
        WHEN c.section_title ILIKE '%%Conclusion%%' THEN -0.03
        WHEN c.section_title ILIKE '%%Discussion%%' THEN -0.03
        WHEN c.section_title ILIKE '%%Appendix%%' THEN -0.1
        WHEN c.section_title ILIKE '%%Additional Analysis%%' THEN -0.1
        WHEN c.section_title ILIKE '%%Experimental Details%%' THEN -0.08
        WHEN c.section_title ILIKE '%%Implementation Details%%' THEN -0.08
        ELSE 0
    END
"""


def resolve_candidate_limit(limit: int, *, scoped: bool = False) -> int:
    normalized_limit = max(1, int(limit))
    if scoped:
        return max(normalized_limit * SCOPED_VECTOR_CANDIDATES_PER_RESULT, MIN_SCOPED_VECTOR_CANDIDATES)
    return max(normalized_limit * VECTOR_CANDIDATES_PER_RESULT, MIN_VECTOR_CANDIDATES)


def resolve_hnsw_ef_search(candidate_limit: int) -> int | None:
    if candidate_limit <= DEFAULT_HNSW_EF_SEARCH:
        return None
    return min(int(candidate_limit), MAX_HNSW_EF_SEARCH)


def build_vector_search_query(
    embedding: Sequence[float],
    *,
    limit: int,
    arxiv_id: str | None = None,
    model_name: str | None = None,
    min_similarity: float = 0.0,
    per_paper_cap: int | None = PER_PAPER_CANDIDATE_CAP,
) -> tuple[str, dict[str, Any]]:
    candidate_limit = resolve_candidate_limit(limit, scoped=bool(arxiv_id))
    params: dict[str, Any] = {
        "query": vector_literal(embedding),
        "candidate_limit": candidate_limit,
        "limit": max(1, int(limit)),
    }
    cap_filter_sql = ""
    if not arxiv_id and per_paper_cap is not None and int(per_paper_cap) > 0:
        params["per_paper_cap"] = int(per_paper_cap)
        cap_filter_sql = "WHERE paper_rank <= %(per_paper_cap)s"
    if model_name:
        params["model_name"] = model_name

    if arxiv_id:
        params["arxiv_id"] = arxiv_id
        candidates_sql = f"""
            scoped AS MATERIALIZED (
                SELECT e.chunk_id, e.embedding <=> %(query)s::vector AS distance
                FROM paper_chunks sc
                JOIN paper_embeddings e ON e.chunk_id = sc.id
                WHERE sc.arxiv_id = %(arxiv_id)s
                    {"AND e.model_name = %(model_name)s" if model_name else ""}
            ),
            candidates AS (
                SELECT chunk_id, distance
                FROM scoped
                ORDER BY distance ASC, chunk_id DESC
                LIMIT %(candidate_limit)s
            )"""
    else:
        candidates_sql = f"""
            candidates AS (
                SELECT e.chunk_id, e.embedding <=> %(query)s::vector AS distance
                FROM paper_embeddings e
                {"WHERE e.model_name = %(model_name)s" if model_name else ""}
                ORDER BY e.embedding <=> %(query)s::vector
                LIMIT %(candidate_limit)s
            )"""

    min_similarity_sql = ""
    if min_similarity and float(min_similarity) > 0:
        min_similarity_sql = "AND raw_similarity_score >= %(min_similarity)s"
        params["min_similarity"] = float(min_similarity)

    sql = f"""
        WITH {candidates_sql},
        ranked AS (
            SELECT
                c.id,
                c.arxiv_id,
                p.title,
                p.abstract,
                c.chunk_text,
                c.chunk_index,
                c.section_title,
                COALESCE(c.metadata->>'content_role', '') AS content_role,
                1 - k.distance AS raw_similarity_score,
                {_CONTENT_ROLE_ADJUSTMENT_SQL} AS content_role_adjustment,
                {_SECTION_BOOST_SQL} AS section_boost
            FROM candidates k
            JOIN paper_chunks c ON c.id = k.chunk_id
            JOIN papers p ON p.arxiv_id = c.arxiv_id
        ),
        kept AS (
            SELECT
                ranked.*,
                row_number() OVER (
                    PARTITION BY arxiv_id
                    ORDER BY (raw_similarity_score + content_role_adjustment + section_boost) DESC, id DESC
                ) AS paper_rank
            FROM ranked
            WHERE content_role <> 'toc'
                {min_similarity_sql}
        )
        SELECT
            id,
            arxiv_id,
            title,
            abstract,
            chunk_text,
            chunk_index,
            section_title,
            (raw_similarity_score + content_role_adjustment + section_boost) AS similarity_score,
            raw_similarity_score,
            content_role,
            content_role_adjustment,
            section_boost
        FROM kept
        {cap_filter_sql}
        ORDER BY (raw_similarity_score + content_role_adjustment + section_boost) DESC, id DESC
        LIMIT %(limit)s
    """
    return sql, params


def _as_float(value: float | Decimal | None) -> float:
    return float(value) if value is not None else 0.0


def vector_hit_from_row(query: str, row: Mapping[str, Any]) -> RetrievalHit:
    paper_title = str(row["title"] or "")
    paper_abstract = str(row["abstract"] or "")
    chunk_text = str(row["chunk_text"] or "")
    return RetrievalHit(
        chunk_id=int(row["id"]),
        arxiv_id=str(row["arxiv_id"]),
        chunk_index=int(row["chunk_index"]),
        paper_title=paper_title,
        paper_abstract=paper_abstract,
        chunk_text=chunk_text,
        section_title=str(row["section_title"] or ""),
        content_role=str(row["content_role"] or ""),
        method="vector",
        score=_as_float(row["similarity_score"]),
        snippet=build_search_snippet(query, chunk_text, paper_abstract, paper_title),
        vector=VectorScores(
            raw_similarity_score=_as_float(row["raw_similarity_score"]),
            content_role_adjustment=_as_float(row["content_role_adjustment"]),
            section_boost=_as_float(row["section_boost"]),
        ),
    )


async def fetch_vector_candidates(
    db: ConnectionSource,
    embedding: Sequence[float],
    *,
    query: str,
    limit: int,
    arxiv_id: str | None = None,
    model_name: str | None = None,
    min_similarity: float = 0.0,
) -> list[RetrievalHit]:
    sql, params = build_vector_search_query(
        embedding,
        limit=limit,
        arxiv_id=arxiv_id,
        model_name=model_name,
        min_similarity=max(0.0, min_similarity),
    )
    ef_search = None if arxiv_id else resolve_hnsw_ef_search(params["candidate_limit"])
    async with (
        use_connection(db) as connection,
        connection.transaction(),
        connection.cursor(row_factory=dict_row) as cursor,
    ):
        if ef_search is not None:
            await cursor.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),), prepare=False)
        await cursor.execute(sql, params, prepare=False)
        rows = await cursor.fetchall()
    return [vector_hit_from_row(query, row) for row in rows]


def lexical_overlap_bonus(tokens: Set[str], text: str) -> float:
    if not tokens:
        return 0.0
    text_tokens = set(_TEXT_TOKEN.findall(text.lower()))
    if not text_tokens:
        return 0.0
    overlap = len(tokens & text_tokens)
    if overlap == 0:
        return 0.0
    return min(MAX_OVERLAP_BONUS, OVERLAP_BONUS_PER_TOKEN * overlap)


def rerank_vector_hits(query: str, hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
    tokens = query_tokens(query)
    lowered_query = query.lower()
    appendix_requested = any(keyword in lowered_query for keyword in APPENDIX_QUERY_KEYWORDS)
    conclusion_requested = any(keyword in lowered_query for keyword in CONCLUSION_QUERY_KEYWORDS)
    references_requested = citation_intent_requested(query)
    intents = requested_section_intents(query)

    reranked: list[RetrievalHit] = []
    for hit in hits:
        section_lowered = hit.section_title.lower()
        adjustment = lexical_overlap_bonus(tokens, f"{hit.section_title} {hit.chunk_text}")
        if not appendix_requested and any(keyword in section_lowered for keyword in APPENDIX_SECTION_KEYWORDS):
            adjustment -= 0.08
        if not conclusion_requested and any(keyword in section_lowered for keyword in CONCLUSION_SECTION_KEYWORDS):
            adjustment -= 0.03
        if not references_requested and (
            is_references_section_title(hit.section_title) or "acknowledg" in section_lowered
        ):
            adjustment -= 0.18
        if hit.content_role in PENALIZED_CONTENT_ROLES:
            adjustment -= 0.02
        if not references_requested and looks_reference_like_text(hit.chunk_text):
            adjustment -= 0.14
        adjustment += section_intent_bonus(intents, section_lowered)

        scores = replace(hit.vector, rerank_adjustment=adjustment) if hit.vector is not None else None
        reranked.append(replace(hit, score=hit.score + adjustment, vector=scores))

    return sorted(reranked, key=lambda hit: (hit.score, hit.chunk_id), reverse=True)
