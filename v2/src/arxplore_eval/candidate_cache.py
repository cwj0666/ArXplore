import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Self

from arxplore.domain.retrieval import LexicalScores, RetrievalHit, VectorScores
from arxplore.retrieval.diversity import apply_paper_diversity as apply_hit_diversity
from arxplore.retrieval.fusion import fuse_hybrid_hits
from arxplore.retrieval.service import Retriever
from arxplore_eval.dataset import EvalQuery, parse_query
from arxplore_eval.research_fusion import (
    PRODUCT_FUSION,
    Candidate,
    HybridFusionConfig,
    apply_paper_diversity,
    candidate_from_hit,
    fuse_hybrid_candidates,
    strict_match_of,
    to_float,
)

CACHE_FORMAT = 1
CHANNELS = ("hybrid", "vector", "lexical")
HYBRID = "hybrid"

type RankKey = tuple[Any, str]


class CacheError(ValueError):
    pass


def compact_candidate(candidate: Mapping[str, Any], *, channel: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "chunk_id": candidate.get("chunk_id"),
        "arxiv_id": str(candidate.get("arxiv_id") or ""),
        "score": to_float(candidate.get("score")),
        "content_role": str(candidate.get("content_role") or ""),
        "section_title": str(candidate.get("section_title") or ""),
    }
    breakdown = candidate.get("score_breakdown") or {}
    if channel == "lexical" and "strict_match" in breakdown:
        record["strict_match"] = breakdown["strict_match"]
    return record


def restore_candidate(record: Mapping[str, Any]) -> Candidate:
    candidate = {key: value for key, value in record.items() if key != "strict_match"}
    candidate["score_breakdown"] = {"strict_match": record["strict_match"]} if "strict_match" in record else {}
    return candidate


def compact_hit(hit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": hit.get("chunk_id"),
        "arxiv_id": str(hit.get("arxiv_id") or ""),
        "content_role": str(hit.get("content_role") or ""),
    }


@dataclass(frozen=True)
class CachedQuery:
    query: EvalQuery
    normalized_query: str
    lexical: tuple[Candidate, ...]
    vector: tuple[Candidate, ...]
    live: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.query.id

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": "query",
            "eval_query": self.query.to_dict(),
            "normalized_query": self.normalized_query,
            "lexical": [compact_candidate(candidate, channel="lexical") for candidate in self.lexical],
            "vector": [compact_candidate(candidate, channel="vector") for candidate in self.vector],
            "live": {channel: [compact_hit(hit) for hit in hits] for channel, hits in self.live.items()},
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any], *, location: str = "") -> Self:
        try:
            return cls(
                query=parse_query(record["eval_query"], location=location),
                normalized_query=str(record["normalized_query"]),
                lexical=tuple(restore_candidate(item) for item in record["lexical"]),
                vector=tuple(restore_candidate(item) for item in record["vector"]),
                live={channel: tuple(hits) for channel, hits in (record.get("live") or {}).items()},
            )
        except (KeyError, TypeError) as exc:
            raise CacheError(f"{location}: 질의 레코드 형식 오류: {exc}") from exc


def _live(hits: Iterable[RetrievalHit]) -> tuple[dict[str, Any], ...]:
    return tuple(compact_hit(candidate_from_hit(hit)) for hit in hits)


async def capture_query(retriever: Retriever, query: EvalQuery, *, k: int) -> CachedQuery:
    live_hybrid = await retriever.search_hybrid(query.query, limit=k)
    inputs = await retriever.hybrid_inputs(query.query, limit=k)
    live_vector = await retriever.search_vector(query.query, limit=k)
    live_lexical = await retriever.search_lexical(query.query, limit=k)
    return CachedQuery(
        query=query,
        normalized_query=inputs.query,
        lexical=tuple(candidate_from_hit(hit) for hit in inputs.lexical),
        vector=tuple(candidate_from_hit(hit) for hit in inputs.vector),
        live={"hybrid": _live(live_hybrid), "vector": _live(live_vector), "lexical": _live(live_lexical)},
    )


def write_cache(path: str | Path, header: Mapping[str, Any], queries: Iterable[CachedQuery]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"kind": "header", "format": CACHE_FORMAT, **header}, ensure_ascii=False)]
    lines.extend(json.dumps(query.to_record(), ensure_ascii=False) for query in queries)
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    with target.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle:
        handle.write(payload)


def read_cache(path: str | Path) -> tuple[dict[str, Any], list[CachedQuery]]:
    source = Path(path)
    header: dict[str, Any] | None = None
    queries: list[CachedQuery] = []
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{source}:{line_number}"
            record = json.loads(line)
            if header is None:
                if record.get("kind") != "header":
                    raise CacheError(f"{location}: 첫 줄은 머리말(kind=header)이어야 합니다")
                if record.get("format") != CACHE_FORMAT:
                    raise CacheError(f"{location}: 캐시 형식 {record.get('format')!r}은 지원하지 않습니다")
                header = record
                continue
            queries.append(CachedQuery.from_record(record, location=location))
    if header is None:
        raise CacheError(f"{source}: 빈 캐시입니다")
    ids = [query.id for query in queries]
    if len(set(ids)) != len(ids):
        raise CacheError(f"{source}: 질의 id가 중복됩니다")
    return header, queries


def cache_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _as_tuple(value: Any) -> Any:
    return tuple(_as_tuple(item) for item in value) if isinstance(value, list) else value


def recorded_fusion_config(header: Mapping[str, Any]) -> HybridFusionConfig:
    recorded = header.get("fusion_default")
    if not recorded:
        return PRODUCT_FUSION
    known = {item.name for item in fields(HybridFusionConfig)}
    unknown = sorted(set(recorded) - known)
    if unknown:
        raise CacheError(f"머리말 fusion_default에 알 수 없는 필드가 있습니다: {unknown}")
    return HybridFusionConfig(**{name: _as_tuple(value) for name, value in recorded.items()})


def replay(
    cached: CachedQuery, *, k: int, channel: str = HYBRID, config: HybridFusionConfig = PRODUCT_FUSION
) -> list[Candidate]:
    if channel == HYBRID:
        fused = fuse_hybrid_candidates(cached.normalized_query, list(cached.lexical), list(cached.vector), config)
        return apply_paper_diversity(fused, limit=k)
    if channel == "vector":
        return apply_paper_diversity(list(cached.vector), limit=k)
    if channel == "lexical":
        return apply_paper_diversity(list(cached.lexical), limit=k)
    raise ValueError(f"unknown channel {channel!r}; choose from {CHANNELS}")


def cached_hit(candidate: Mapping[str, Any], *, channel: str) -> RetrievalHit:
    chunk_id = candidate.get("chunk_id")
    if chunk_id is None:
        raise CacheError("chunk_id가 없는 후보는 제품 융합으로 재생할 수 없습니다")
    strict_match = strict_match_of(candidate)
    lexical = None
    vector = None
    if channel == "lexical" and strict_match is not None:
        lexical = LexicalScores(
            fts_score=0.0,
            ilike_bonus=0.0,
            content_role_adjustment=0.0,
            section_boost=0.0,
            structural_adjustment=0.0,
            coverage=0.0,
            strict_match=bool(strict_match),
        )
    if channel == "vector":
        vector = VectorScores(raw_similarity_score=0.0, content_role_adjustment=0.0, section_boost=0.0)
    return RetrievalHit(
        chunk_id=int(chunk_id),
        arxiv_id=str(candidate.get("arxiv_id") or ""),
        chunk_index=0,
        paper_title="",
        paper_abstract="",
        chunk_text="",
        section_title=str(candidate.get("section_title") or ""),
        content_role=str(candidate.get("content_role") or ""),
        method="lexical" if channel == "lexical" else "vector",
        score=to_float(candidate.get("score")),
        snippet="",
        lexical=lexical,
        vector=vector,
    )


def replay_product(cached: CachedQuery, *, k: int) -> list[RetrievalHit]:
    lexical = [cached_hit(candidate, channel="lexical") for candidate in cached.lexical]
    vector = [cached_hit(candidate, channel="vector") for candidate in cached.vector]
    return apply_hit_diversity(fuse_hybrid_hits(lexical, vector), limit=k)


@dataclass(frozen=True)
class Mismatch:
    query_id: str
    channel: str
    live: list[RankKey]
    replayed: list[RankKey]


def rank_keys(hits: Sequence[Mapping[str, Any]]) -> list[RankKey]:
    return [(hit.get("chunk_id"), str(hit.get("arxiv_id") or "")) for hit in hits]


def hit_rank_keys(hits: Sequence[RetrievalHit]) -> list[RankKey]:
    return [(hit.chunk_id, hit.arxiv_id) for hit in hits]


def compare_live(
    queries: Sequence[CachedQuery],
    *,
    k: int,
    channel: str = HYBRID,
    config: HybridFusionConfig = PRODUCT_FUSION,
) -> list[Mismatch]:
    mismatches: list[Mismatch] = []
    for cached in queries:
        replayed = rank_keys(replay(cached, k=k, channel=channel, config=config))
        live = rank_keys(cached.live[channel]) if channel in cached.live else None
        if live != replayed:
            mismatches.append(Mismatch(cached.id, channel, live or [], replayed))
    return mismatches


def gate(queries: Sequence[CachedQuery], *, k: int, config: HybridFusionConfig = PRODUCT_FUSION) -> list[Mismatch]:
    return compare_live(queries, k=k, channel=HYBRID, config=config)


def product_check(
    queries: Sequence[CachedQuery], *, k: int, config: HybridFusionConfig = PRODUCT_FUSION
) -> list[Mismatch]:
    mismatches: list[Mismatch] = []
    for cached in queries:
        research = rank_keys(replay(cached, k=k, channel=HYBRID, config=config))
        product = hit_rank_keys(replay_product(cached, k=k))
        if research != product:
            mismatches.append(Mismatch(cached.id, HYBRID, product, research))
    return mismatches
