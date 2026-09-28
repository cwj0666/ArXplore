import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from decimal import Decimal
from typing import Any

from psycopg.rows import dict_row

from arxplore.db.pool import ConnectionSource, use_connection
from arxplore.domain.retrieval import LexicalScores, RetrievalHit
from arxplore.parsing.roles import FRONT_MATTER_ROLE, REFERENCES_ROLE, is_references_section_title
from arxplore.retrieval.query import reference_intent_requested, requested_section_intents, section_intent_bonus
from arxplore.retrieval.signals import looks_outline_like_text, looks_reference_like_text
from arxplore.retrieval.snippet import build_search_snippet

PER_PAPER_CANDIDATE_CAP = 3
STRICT_MATCH_BONUS = 1.0
MIN_PARTIAL_MATCH_POOL = 500
PARTIAL_MATCH_POOL_PER_RESULT = 20
FILTERED_CONTENT_ROLES = frozenset({REFERENCES_ROLE, FRONT_MATTER_ROLE})

SECTION_NUMBER_PREFIX_PATTERN = r"(?:\d+(?:\.\d+)*[.)]?|[IVXLC]+[.)]?|[IVXLCivxlc]+[.)])"
REFERENCES_TITLE_PATTERN = r"(?:references?|bibliography|works\s+cited|literature\s+cited)(?:\s+and\s+notes)?"

_LIKE_SPECIAL_CHARACTERS = re.compile(r"([\\%_])")
_FTS_SEPARATORS = re.compile(r"[-/]+")
_UNESCAPED_LOWERCASE_LETTER = re.compile(r"(?<!\\)[a-z]")


def ascii_case_insensitive(pattern: str) -> str:
    return _UNESCAPED_LOWERCASE_LETTER.sub(lambda match: f"[{match.group().upper()}{match.group()}]", pattern)


REFERENCES_SECTION_TITLE_SQL_REGEX = (
    rf"\A\s*(?:{SECTION_NUMBER_PREFIX_PATTERN}\s+)?[\s.:]*"
    rf"{ascii_case_insensitive(REFERENCES_TITLE_PATTERN)}[\s.:]*\Z"
)

_QUERY_TERMS_SQL = r"""
    SELECT
        string_agg('''' || replace(replace(lexeme, '\', '\\'), '''', '''''') || '''', ' | ')::tsquery AS any_term,
        array_agg(lexeme) AS lexemes,
        count(*) AS lexeme_count
    FROM unnest(
        tsvector_to_array(to_tsvector('english', %(query)s) || to_tsvector('english', %(fts_query)s))
    ) AS lexeme
"""


def escape_like(value: str) -> str:
    return _LIKE_SPECIAL_CHARACTERS.sub(r"\\\1", value)


def build_lexical_candidates_query(
    query: str,
    *,
    limit: int,
    arxiv_id: str | None = None,
    per_paper_cap: int | None = PER_PAPER_CANDIDATE_CAP,
) -> tuple[str, dict[str, Any]] | None:
    normalized_query = " ".join(query.split())
    if not normalized_query:
        return None

    chunk_scope_sql = ""
    paper_scope_sql = ""
    normalized_limit = max(1, int(limit))
    params: dict[str, Any] = {
        "query": normalized_query,
        "fts_query": _FTS_SEPARATORS.sub(" ", normalized_query),
        "like_pattern": f"%{escape_like(normalized_query)}%",
        "limit": normalized_limit,
        "partial_pool": max(normalized_limit * PARTIAL_MATCH_POOL_PER_RESULT, MIN_PARTIAL_MATCH_POOL),
    }
    if arxiv_id:
        chunk_scope_sql = "AND c.arxiv_id = %(arxiv_id)s"
        paper_scope_sql = "AND p.arxiv_id = %(arxiv_id)s"
        params["arxiv_id"] = arxiv_id

    cap_filter_sql = ""
    if not arxiv_id and per_paper_cap is not None and int(per_paper_cap) > 0:
        params["per_paper_cap"] = int(per_paper_cap)
        cap_filter_sql = "WHERE paper_rank <= %(per_paper_cap)s"

    sql = f"""
        WITH query_terms AS ({_QUERY_TERMS_SQL}),
        matched AS (
            SELECT c.id
            FROM paper_chunks c
            WHERE c.chunk_vector @@ (SELECT any_term FROM query_terms)
                {chunk_scope_sql}
            UNION
            SELECT c.id
            FROM papers p
            JOIN paper_chunks c ON c.arxiv_id = p.arxiv_id
            WHERE p.title_abstract_vector @@ (SELECT any_term FROM query_terms)
                {paper_scope_sql}
        ),
        screened AS MATERIALIZED (
            SELECT
                c.id AS chunk_id,
                (
                    v.search_vector @@ websearch_to_tsquery('english', %(query)s)
                    OR v.search_vector @@ plainto_tsquery('english', %(fts_query)s)
                ) AS strict_match,
                (length(v.search_vector) - length(ts_delete(v.search_vector, q.lexemes)))::double precision
                    / GREATEST(q.lexeme_count, 1) AS coverage
            FROM matched m
            JOIN paper_chunks c ON c.id = m.id
            JOIN papers p ON p.arxiv_id = c.arxiv_id
            CROSS JOIN query_terms q
            CROSS JOIN LATERAL (
                SELECT p.title_abstract_vector || c.chunk_vector AS search_vector OFFSET 0
            ) v
            WHERE coalesce(c.metadata->>'content_role', '') <> 'toc'
        ),
        pool AS (
            SELECT chunk_id, strict_match, coverage
            FROM screened
            WHERE strict_match
            UNION ALL
            (
                SELECT chunk_id, strict_match, coverage
                FROM screened
                WHERE NOT strict_match
                ORDER BY coverage DESC, chunk_id DESC
                LIMIT %(partial_pool)s
            )
        ),
        candidates AS (
            SELECT
                k.chunk_id,
                c.arxiv_id,
                p.title AS paper_title,
                p.abstract AS paper_abstract,
                c.chunk_text,
                c.section_title,
                COALESCE(c.metadata->>'content_role', '') AS content_role,
                k.strict_match,
                k.coverage,
                v.search_vector,
                q.any_term
            FROM pool k
            JOIN paper_chunks c ON c.id = k.chunk_id
            JOIN papers p ON p.arxiv_id = c.arxiv_id
            CROSS JOIN query_terms q
            CROSS JOIN LATERAL (
                SELECT p.title_abstract_vector || c.chunk_vector AS search_vector OFFSET 0
            ) v
        ),
        ranked AS (
            SELECT
                chunk_id,
                arxiv_id,
                strict_match,
                coverage,
                CASE
                    WHEN strict_match THEN
                        ts_rank_cd(search_vector, websearch_to_tsquery('english', %(query)s))
                        + 0.65 * ts_rank_cd(search_vector, plainto_tsquery('english', %(fts_query)s))
                        + {STRICT_MATCH_BONUS}
                    ELSE coverage * ts_rank_cd(search_vector, any_term, 32)
                END AS fts_score,
                CASE
                    WHEN NOT strict_match THEN 0
                    WHEN paper_title ILIKE %(like_pattern)s THEN 0.45
                    WHEN paper_abstract ILIKE %(like_pattern)s THEN 0.2
                    WHEN chunk_text ILIKE %(like_pattern)s THEN 0.15
                    ELSE 0
                END AS ilike_bonus,
                CASE
                    WHEN content_role = 'references' THEN -0.24
                    WHEN content_role = 'toc' THEN -0.28
                    WHEN content_role = 'front_matter' THEN -0.14
                    WHEN content_role = 'table_like' THEN -0.12
                    WHEN content_role = 'figure_caption' THEN -0.08
                    WHEN content_role = 'appendix' THEN -0.08
                    ELSE 0
                END AS content_role_adjustment,
                CASE
                    WHEN section_title ILIKE 'Abstract' THEN 0.16
                    WHEN section_title ILIKE '%%Introduction%%' THEN 0.1
                    WHEN section_title ILIKE '%%Method%%' OR section_title ILIKE '%%Approach%%' THEN 0.04
                    WHEN section_title ILIKE '%%Related Work%%' THEN 0.02
                    WHEN section_title ILIKE '%%Conclusion%%' THEN -0.02
                    WHEN section_title ILIKE '%%Discussion%%' THEN -0.02
                    WHEN section_title ILIKE '%%Appendix%%' THEN -0.08
                    WHEN section_title ILIKE '%%Additional Analysis%%' THEN -0.08
                    WHEN section_title ILIKE '%%Experimental Details%%' THEN -0.06
                    WHEN section_title ILIKE '%%Implementation Details%%' THEN -0.06
                    ELSE 0
                END AS section_boost,
                CASE
                    WHEN section_title ILIKE '%%Table of Contents%%' THEN -0.12
                    WHEN section_title ~ '{REFERENCES_SECTION_TITLE_SQL_REGEX}' THEN -0.08
                    WHEN section_title = 'Front Matter' THEN -0.03
                    ELSE 0
                END AS structural_adjustment
            FROM candidates
        ),
        scored AS MATERIALIZED (
            SELECT
                ranked.*,
                (fts_score + ilike_bonus + content_role_adjustment + section_boost + structural_adjustment) AS score
            FROM ranked
        ),
        kept AS (
            SELECT
                scored.*,
                row_number() OVER (PARTITION BY arxiv_id ORDER BY score DESC, chunk_id DESC) AS paper_rank
            FROM scored
            WHERE score > 0.01
        ),
        top_hits AS (
            SELECT *
            FROM kept
            {cap_filter_sql}
            ORDER BY score DESC, chunk_id DESC
            LIMIT %(limit)s
        )
        SELECT
            t.chunk_id,
            t.arxiv_id,
            p.title AS paper_title,
            p.abstract AS paper_abstract,
            c.chunk_text,
            c.chunk_index,
            c.section_title,
            COALESCE(c.metadata->>'content_role', '') AS content_role,
            t.fts_score,
            t.ilike_bonus,
            t.content_role_adjustment,
            t.section_boost,
            t.structural_adjustment,
            t.score,
            t.strict_match,
            t.coverage
        FROM top_hits t
        JOIN paper_chunks c ON c.id = t.chunk_id
        JOIN papers p ON p.arxiv_id = t.arxiv_id
        ORDER BY t.score DESC, t.chunk_id DESC
    """
    return sql, params


def _as_float(value: float | Decimal | None) -> float:
    return float(value) if value is not None else 0.0


def lexical_hit_from_row(query: str, row: Mapping[str, Any]) -> RetrievalHit:
    paper_title = str(row["paper_title"] or "")
    paper_abstract = str(row["paper_abstract"] or "")
    chunk_text = str(row["chunk_text"] or "")
    return RetrievalHit(
        chunk_id=int(row["chunk_id"]),
        arxiv_id=str(row["arxiv_id"]),
        chunk_index=int(row["chunk_index"]),
        paper_title=paper_title,
        paper_abstract=paper_abstract,
        chunk_text=chunk_text,
        section_title=str(row["section_title"] or ""),
        content_role=str(row["content_role"] or ""),
        method="lexical",
        score=_as_float(row["score"]),
        snippet=build_search_snippet(query, chunk_text, paper_abstract, paper_title),
        lexical=LexicalScores(
            fts_score=_as_float(row["fts_score"]),
            ilike_bonus=_as_float(row["ilike_bonus"]),
            content_role_adjustment=_as_float(row["content_role_adjustment"]),
            section_boost=_as_float(row["section_boost"]),
            structural_adjustment=_as_float(row["structural_adjustment"]),
            coverage=_as_float(row["coverage"]),
            strict_match=bool(row["strict_match"]),
        ),
    )


async def fetch_lexical_candidates(
    db: ConnectionSource,
    query: str,
    *,
    limit: int,
    arxiv_id: str | None = None,
    per_paper_cap: int | None = PER_PAPER_CANDIDATE_CAP,
) -> list[RetrievalHit]:
    built = build_lexical_candidates_query(query, limit=limit, arxiv_id=arxiv_id, per_paper_cap=per_paper_cap)
    if built is None:
        return []
    sql, params = built
    async with use_connection(db) as connection, connection.cursor(row_factory=dict_row) as cursor:
        await cursor.execute(sql, params, prepare=False)
        rows = await cursor.fetchall()
    return [lexical_hit_from_row(params["query"], row) for row in rows]


def rerank_lexical_hits(query: str, hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
    intents = requested_section_intents(query)
    reranked: list[RetrievalHit] = []
    for hit in hits:
        bonus = section_intent_bonus(intents, hit.section_title)
        scores = replace(hit.lexical, section_intent_bonus=bonus) if hit.lexical is not None else None
        reranked.append(replace(hit, score=hit.score + bonus, lexical=scores))
    return sorted(reranked, key=lambda hit: (hit.score, hit.chunk_id), reverse=True)


def filter_lexical_hits(query: str, hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
    if reference_intent_requested(query):
        return list(hits)
    return [
        hit
        for hit in hits
        if hit.content_role not in FILTERED_CONTENT_ROLES
        and not is_references_section_title(hit.section_title)
        and "front matter" not in hit.section_title.lower()
        and not looks_reference_like_text(hit.chunk_text)
        and not looks_outline_like_text(hit.chunk_text)
    ]
