"""hybrid 융합 입력 캐시: 질의별 lexical/vector 후보와 제품 hybrid 결과를 저장하고 융합만 다시 재생한다.

캐시는 gzip JSONL이다. 첫 줄은 머리말(`kind: header`: 커밋, k, 코퍼스 요약, 질의셋 digest, 기록 시점의 기본 융합 설정),
나머지는 질의 한 줄씩(`kind: query`)이다. 후보는 융합·다양성·지표에 필요한 필드만 남긴다:
`chunk_id`, `arxiv_id`, `score`, `content_role`, `section_title`, lexical만 `strict_match`. 순서가 곧 채널 순위다.

`capture_query`는 `PaperRetriever.hybrid_fusion_inputs`(제품 hybrid 경로가 융합에 넘기는 입력 그대로)와 제품
`search_paper_chunks_by_hybrid` 결과를 함께 기록한다. `replay`는 저장한 후보에 `fuse_hybrid_candidates` →
`apply_paper_diversity`를 적용하고, `gate`는 기록 시점의 제품 설정(머리말 `fusion_default`, `recorded_fusion_config`)으로
재생한 순위가 기록한 제품 결과와 같은지 본다. 제품 기본값이 바뀐 뒤에도 예전 캐시의 게이트가 그대로 성립한다.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from eval.dataset import EvalQuery, parse_query
from src.integrations.hybrid_fusion import (
    DEFAULT_HYBRID_FUSION,
    HybridFusionConfig,
    apply_paper_diversity,
    fuse_hybrid_candidates,
)

CACHE_FORMAT = 1
CHANNELS = ("hybrid", "vector", "lexical")


class CacheError(ValueError):
    pass


def compact_candidate(candidate: dict, *, channel: str) -> dict[str, Any]:
    """채널 후보 하나에서 융합·다양성·지표가 읽는 필드만 남긴다. lexical은 `strict_match`가 있을 때만 싣는다."""
    record: dict[str, Any] = {
        "chunk_id": candidate.get("chunk_id"),
        "arxiv_id": str(candidate.get("arxiv_id") or ""),
        "score": float(candidate.get("score") or 0.0),
        "content_role": str(candidate.get("content_role") or ""),
        "section_title": str(candidate.get("section_title") or ""),
    }
    breakdown = candidate.get("score_breakdown") or {}
    if channel == "lexical" and "strict_match" in breakdown:
        record["strict_match"] = breakdown["strict_match"]
    return record


def restore_candidate(record: dict[str, Any]) -> dict[str, Any]:
    """캐시 후보를 융합 입력 shape로 되돌린다(`strict_match`는 `score_breakdown` 안으로)."""
    candidate = {key: value for key, value in record.items() if key != "strict_match"}
    candidate["score_breakdown"] = {"strict_match": record["strict_match"]} if "strict_match" in record else {}
    return candidate


def compact_hit(hit: dict) -> dict[str, Any]:
    return {
        "chunk_id": hit.get("chunk_id"),
        "arxiv_id": str(hit.get("arxiv_id") or ""),
        "content_role": str(hit.get("content_role") or ""),
    }


@dataclass(frozen=True)
class CachedQuery:
    query: EvalQuery
    normalized_query: str
    lexical: tuple[dict, ...]
    vector: tuple[dict, ...]
    live: dict[str, tuple[dict, ...]] = field(default_factory=dict)

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
    def from_record(cls, record: dict[str, Any], *, location: str = "") -> CachedQuery:
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


def capture_query(retriever: Any, query: EvalQuery, *, k: int) -> CachedQuery:
    """제품 경로로 한 질의의 융합 입력과 live 결과(hybrid·vector·lexical, 각각 limit=k)를 기록한다.

    live hybrid와 융합 입력은 따로 호출하므로 DB를 두 번 조회한다. 임베딩은 호출자가 질의별로 메모해 두어야
    두 호출이 같은 벡터를 쓴다(`scripts/eval_dump_candidates.py`의 `MemoEmbeddingClient`).
    """
    live_hybrid = retriever.search_paper_chunks_by_hybrid(query.query, limit=k)
    normalized_query, lexical, vector = retriever.hybrid_fusion_inputs(query.query, limit=k)
    live_vector = retriever.search_paper_chunks_by_vector(query.query, limit=k)
    live_lexical = retriever.search_paper_chunks(query.query, limit=k)
    return CachedQuery(
        query=query,
        normalized_query=normalized_query,
        lexical=tuple(lexical),
        vector=tuple(vector),
        live={"hybrid": tuple(live_hybrid), "vector": tuple(live_vector), "lexical": tuple(live_lexical)},
    )


def write_cache(path: str | Path, header: dict[str, Any], queries: Iterable[CachedQuery]) -> None:
    """gzip JSONL로 쓴다. gzip 머리의 파일 이름을 비우고 mtime을 0으로 고정해 같은 내용이면 같은 바이트가 된다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"kind": "header", "format": CACHE_FORMAT, **header}, ensure_ascii=False)]
    lines.extend(json.dumps(query.to_record(), ensure_ascii=False) for query in queries)
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle:
        handle.write(payload)


def read_cache(path: str | Path) -> tuple[dict[str, Any], list[CachedQuery]]:
    path = Path(path)
    header: dict[str, Any] | None = None
    queries: list[CachedQuery] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
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
        raise CacheError(f"{path}: 빈 캐시입니다")
    ids = [query.id for query in queries]
    if len(set(ids)) != len(ids):
        raise CacheError(f"{path}: 질의 id가 중복됩니다")
    return header, queries


def cache_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _as_tuple(value: Any) -> Any:
    return tuple(_as_tuple(item) for item in value) if isinstance(value, list) else value


def recorded_fusion_config(header: dict[str, Any]) -> HybridFusionConfig:
    """머리말의 `fusion_default`(기록 시점 `asdict(DEFAULT_HYBRID_FUSION)`)를 설정으로 되돌린다. 없는 필드는 지금 기본값을
    쓰므로, 필드가 늘기 전 캐시(예: 2026-09-28, 가중 RRF 규칙)도 그때 설정 그대로 복원된다. 머리말에 없으면 지금 기본값."""
    recorded = header.get("fusion_default")
    if not recorded:
        return DEFAULT_HYBRID_FUSION
    known = {item.name for item in fields(HybridFusionConfig)}
    unknown = sorted(set(recorded) - known)
    if unknown:
        raise CacheError(f"머리말 fusion_default에 알 수 없는 필드가 있습니다: {unknown}")
    return HybridFusionConfig(**{name: _as_tuple(value) for name, value in recorded.items()})


def replay(cached: CachedQuery, *, k: int, channel: str = "hybrid", config: HybridFusionConfig | None = None) -> list:
    """저장한 후보로 한 채널의 최종 순위를 다시 만든다. hybrid는 융합 → 다양성, vector/lexical은 다양성만."""
    if channel == "hybrid":
        fused = fuse_hybrid_candidates(
            cached.normalized_query, list(cached.lexical), list(cached.vector), config or DEFAULT_HYBRID_FUSION
        )
        return apply_paper_diversity(fused, limit=k)
    if channel == "vector":
        return apply_paper_diversity(list(cached.vector), limit=k)
    if channel == "lexical":
        return apply_paper_diversity(list(cached.lexical), limit=k)
    raise ValueError(f"unknown channel {channel!r}; choose from {CHANNELS}")


@dataclass(frozen=True)
class Mismatch:
    query_id: str
    channel: str
    live: list[tuple[Any, str]]
    replayed: list[tuple[Any, str]]


def _ids(hits: Sequence[dict]) -> list[tuple[Any, str]]:
    return [(hit.get("chunk_id"), str(hit.get("arxiv_id") or "")) for hit in hits]


def compare_live(
    queries: Sequence[CachedQuery], *, k: int, channel: str = "hybrid", config: HybridFusionConfig | None = None
) -> list[Mismatch]:
    """`config`(없으면 기본 설정) 재생 순위(chunk_id, arxiv_id)가 기록한 live 순위와 다른 질의. live가 없는 질의도 불일치로 센다."""
    mismatches: list[Mismatch] = []
    for cached in queries:
        replayed = _ids(replay(cached, k=k, channel=channel, config=config))
        live = _ids(cached.live.get(channel, ())) if channel in cached.live else None
        if live != replayed:
            mismatches.append(Mismatch(cached.id, channel, live or [], replayed))
    return mismatches


def gate(queries: Sequence[CachedQuery], *, k: int, config: HybridFusionConfig | None = None) -> list[Mismatch]:
    """재생 게이트: 기록 시점 제품 설정(`config`, 보통 `recorded_fusion_config(header)`. 없으면 지금 기본값) 재생이 모든
    질의에서 기록한 제품 hybrid 결과와 같아야 한다. 불일치 목록을 돌려준다."""
    return compare_live(queries, k=k, channel="hybrid", config=config)
