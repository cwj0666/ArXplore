from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from arxplore.domain.retrieval import HybridInputs, RetrievalHit
from arxplore.retrieval.diversity import MAX_CHUNKS_PER_PAPER
from arxplore.retrieval.diversity import apply_paper_diversity as apply_hit_diversity
from arxplore.retrieval.fusion import LEXICAL_WEIGHT
from arxplore.retrieval.lexical import STRICT_MATCH_BONUS
from arxplore.retrieval.query import query_tokens

type Candidate = dict[str, Any]

WEIGHTING_MODES = ("rules", "static", "confidence_linear", "convex")
SCORE_NORMALIZATIONS = ("minmax", "theoretical")
CHANNELS = ("lexical", "vector")
LEXICAL_SCORE_FLOOR = 0.0
LOWEST_LEXICAL_CONTENT_ROLE_ADJUSTMENT = -0.24
LOWEST_LEXICAL_SECTION_BOOST = -0.08
LOWEST_LEXICAL_STRUCTURAL_ADJUSTMENT = -0.12
STRICT_LEXICAL_SCORE_FLOOR = round(
    STRICT_MATCH_BONUS
    + LOWEST_LEXICAL_CONTENT_ROLE_ADJUSTMENT
    + LOWEST_LEXICAL_SECTION_BOOST
    + LOWEST_LEXICAL_STRUCTURAL_ADJUSTMENT,
    6,
)
VECTOR_SCORE_FLOOR = -1.0


@dataclass(frozen=True)
class HybridFusionConfig:
    rank_constant: float = 60.0
    overlap_bonus: float = 0.015
    drop_partial_lexical: bool = True
    weighting: str = "convex"
    long_query_min_tokens: int = 5
    long_query_multipliers: tuple[float, float] = (0.85, 1.05)
    confidence_bands: tuple[tuple[float, float, float], ...] = ((0.3, 0.45, 1.1), (0.5, 0.7, 1.05))
    no_overlap_top_n: int = 5
    no_overlap_max_confidence: float = 0.4
    no_overlap_multipliers: tuple[float, float] = (0.75, 1.08)
    weight_floors: tuple[float, float] = (0.2, 0.5)
    static_weights: tuple[float, float] = (1.0, 1.0)
    confidence_min_weight: float = 0.5
    confidence_tau: float = 0.5
    quality_weight: bool = True
    quality_tiers: tuple[tuple[float, float], ...] = ((0.2, 0.2), (0.3, 0.4), (0.5, 0.65), (0.8, 0.85))
    convex_alpha: float = LEXICAL_WEIGHT
    score_normalization: str = "minmax"
    score_floors: tuple[float, float] = (LEXICAL_SCORE_FLOOR, VECTOR_SCORE_FLOOR)

    def __post_init__(self) -> None:
        if self.weighting not in WEIGHTING_MODES:
            raise ValueError(f"weighting must be one of {WEIGHTING_MODES}, got {self.weighting!r}")
        if self.score_normalization not in SCORE_NORMALIZATIONS:
            raise ValueError(
                f"score_normalization must be one of {SCORE_NORMALIZATIONS}, got {self.score_normalization!r}"
            )
        if not 0.0 <= self.convex_alpha <= 1.0:
            raise ValueError(f"convex_alpha must be within [0, 1], got {self.convex_alpha}")
        if self.rank_constant <= 0:
            raise ValueError(f"rank_constant must be > 0, got {self.rank_constant}")
        if self.weighting == "confidence_linear" and self.confidence_tau <= 0:
            raise ValueError(f"confidence_tau must be > 0, got {self.confidence_tau}")

    @property
    def convex(self) -> bool:
        return self.weighting == "convex"


PRODUCT_FUSION = HybridFusionConfig()
LEGACY_RULES_FUSION = HybridFusionConfig(weighting="rules")
STANDARD_RRF_FUSION = HybridFusionConfig(
    weighting="static", quality_weight=False, overlap_bonus=0.0, drop_partial_lexical=False
)


def to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except TypeError, ValueError:
        return 0.0


def strict_match_of(candidate: Mapping[str, Any]) -> Any:
    return (candidate.get("score_breakdown") or {}).get("strict_match")


def lexical_confidence(candidate: Mapping[str, Any]) -> float:
    score = to_float(candidate.get("score"))
    strict_match = strict_match_of(candidate)
    if strict_match is None:
        return score
    if not strict_match:
        return 0.0
    return score - STRICT_MATCH_BONUS


def channel_score_range(scores: Sequence[float], normalization: str, floor: float) -> tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    high = max(scores)
    if normalization == "minmax":
        return min(scores), high
    if normalization == "theoretical":
        return floor, high
    raise ValueError(f"unknown normalization {normalization!r}; choose from {SCORE_NORMALIZATIONS}")


def normalize_channel_scores(scores: Sequence[float], normalization: str, floor: float = 0.0) -> list[float]:
    low, high = channel_score_range(scores, normalization, floor)
    span = high - low
    if span <= 0:
        return [1.0 for _ in scores]
    return [(score - low) / span for score in scores]


def fusion_lexical_candidates(
    lexical_candidates: Sequence[Candidate],
    vector_candidates: Sequence[Candidate],
    config: HybridFusionConfig = PRODUCT_FUSION,
) -> list[Candidate]:
    if vector_candidates and config.drop_partial_lexical:
        return [candidate for candidate in lexical_candidates if strict_match_of(candidate) is not False]
    return list(lexical_candidates)


def _rules_weights(
    query: str,
    lexical_candidates: Sequence[Candidate],
    vector_candidates: Sequence[Candidate],
    config: HybridFusionConfig,
    lexical_top_score: float,
) -> dict[str, float]:
    weights = {"lexical": 1.0, "vector": 1.0}
    top_n = config.no_overlap_top_n
    lexical_top_ids = {int(candidate.get("chunk_id") or 0) for candidate in lexical_candidates[:top_n]}
    vector_top_ids = {int(candidate.get("chunk_id") or 0) for candidate in vector_candidates[:top_n]}
    overlap_count = len(lexical_top_ids & vector_top_ids)

    if len(query_tokens(query)) >= config.long_query_min_tokens:
        weights["lexical"] *= config.long_query_multipliers[0]
        weights["vector"] *= config.long_query_multipliers[1]
    for upper, lexical_multiplier, vector_multiplier in config.confidence_bands:
        if lexical_top_score < upper:
            weights["lexical"] *= lexical_multiplier
            weights["vector"] *= vector_multiplier
            break
    if overlap_count == 0 and lexical_top_score < config.no_overlap_max_confidence:
        weights["lexical"] *= config.no_overlap_multipliers[0]
        weights["vector"] *= config.no_overlap_multipliers[1]

    return {
        "lexical": max(config.weight_floors[0], weights["lexical"]),
        "vector": max(config.weight_floors[1], weights["vector"]),
    }


def resolve_hybrid_method_weights(
    query: str,
    lexical_candidates: Sequence[Candidate],
    vector_candidates: Sequence[Candidate],
    config: HybridFusionConfig = PRODUCT_FUSION,
) -> dict[str, float]:
    if config.convex:
        return {"lexical": config.convex_alpha, "vector": 1.0 - config.convex_alpha}
    lexical_top_score = lexical_confidence(lexical_candidates[0]) if lexical_candidates else 0.0
    if config.weighting == "static":
        return {"lexical": config.static_weights[0], "vector": config.static_weights[1]}
    if config.weighting == "confidence_linear":
        ratio = min(1.0, max(0.0, lexical_top_score / config.confidence_tau))
        w_min = config.confidence_min_weight
        return {"lexical": w_min + (1.0 - w_min) * ratio, "vector": 1.0}
    return _rules_weights(query, lexical_candidates, vector_candidates, config, lexical_top_score)


def hybrid_quality_weight(
    method: str, candidate: Mapping[str, Any], config: HybridFusionConfig = PRODUCT_FUSION
) -> float:
    if method != "lexical" or not config.quality_weight:
        return 1.0
    score = lexical_confidence(candidate)
    for upper, weight in config.quality_tiers:
        if score < upper:
            return weight
    return 1.0


def _normalized_channels(
    lexical_candidates: Sequence[Candidate], vector_candidates: Sequence[Candidate], config: HybridFusionConfig
) -> dict[str, list[float]]:
    return {
        method: normalize_channel_scores(
            [to_float(candidate.get("score")) for candidate in candidates], config.score_normalization, floor
        )
        for method, candidates, floor in (
            ("lexical", lexical_candidates, config.score_floors[0]),
            ("vector", vector_candidates, config.score_floors[1]),
        )
    }


def _fused_sort_key(candidate: Candidate) -> tuple[float, int, int]:
    return (
        to_float(candidate.get("score")),
        len(candidate.get("matched_methods") or []),
        int(candidate.get("chunk_id") or 0),
    )


def fuse_hybrid_candidates(
    query: str,
    lexical_candidates: Sequence[Candidate],
    vector_candidates: Sequence[Candidate],
    config: HybridFusionConfig = PRODUCT_FUSION,
) -> list[Candidate]:
    lexical = fusion_lexical_candidates(lexical_candidates, vector_candidates, config)
    vector = list(vector_candidates)
    method_weights = resolve_hybrid_method_weights(query, lexical, vector, config)
    normalized = _normalized_channels(lexical, vector, config) if config.convex else {}
    merged: dict[int, Candidate] = {}
    fallback_key_seed = -1

    for method, candidates in (("lexical", lexical), ("vector", vector)):
        for index, candidate in enumerate(candidates):
            chunk_id = int(candidate.get("chunk_id") or fallback_key_seed)
            if not candidate.get("chunk_id"):
                fallback_key_seed -= 1

            entry = merged.setdefault(
                chunk_id,
                {
                    **candidate,
                    "retrieval_method": "hybrid",
                    "score_source": "hybrid",
                    "matched_methods": [],
                    "score_breakdown": {},
                    "score": 0.0,
                    "similarity_score": 0.0,
                },
            )

            rank = index + 1
            method_score = to_float(candidate.get("score"))
            method_breakdown = dict(candidate.get("score_breakdown") or {})

            if method not in entry["matched_methods"]:
                entry["matched_methods"].append(method)

            breakdown = entry["score_breakdown"]
            breakdown[f"{method}_rank"] = rank
            quality_weight = 1.0
            if config.convex:
                contribution = method_weights[method] * normalized[method][index]
                breakdown[f"{method}_normalized_score"] = normalized[method][index]
            else:
                quality_weight = hybrid_quality_weight(method, candidate, config)
                contribution = (method_weights[method] * quality_weight) / (config.rank_constant + rank)
                breakdown[f"{method}_rrf_score"] = contribution
            breakdown[f"{method}_score"] = method_score
            breakdown[f"{method}_weight"] = method_weights[method]
            if not config.convex:
                breakdown[f"{method}_quality_weight"] = quality_weight
            breakdown[f"{method}_score_breakdown"] = method_breakdown
            entry["score"] = to_float(entry.get("score")) + contribution
            entry["similarity_score"] = entry["score"]

            if (method == "vector" and "lexical" not in entry["matched_methods"]) or method == "lexical":
                entry["snippet"] = candidate.get("snippet") or entry.get("snippet")

    fused = list(merged.values())
    if not config.convex:
        for candidate in fused:
            if len(candidate.get("matched_methods") or []) > 1 and config.overlap_bonus > 0:
                candidate["score"] = to_float(candidate.get("score")) + config.overlap_bonus
                candidate["similarity_score"] = candidate["score"]
                candidate["score_breakdown"]["cross_method_overlap_bonus"] = config.overlap_bonus

    fused.sort(key=_fused_sort_key, reverse=True)
    return fused


def _diversity_stand_in(position: int, candidate: Mapping[str, Any]) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=position,
        arxiv_id=str(candidate.get("arxiv_id") or ""),
        chunk_index=0,
        paper_title="",
        paper_abstract="",
        chunk_text="",
        section_title="",
        content_role="",
        method="hybrid",
        score=0.0,
        snippet="",
    )


def apply_paper_diversity(
    candidates: Sequence[Candidate],
    *,
    limit: int,
    arxiv_id: str | None = None,
    max_chunks_per_paper: int = MAX_CHUNKS_PER_PAPER,
) -> list[Candidate]:
    stand_ins = [_diversity_stand_in(position, candidate) for position, candidate in enumerate(candidates)]
    kept = apply_hit_diversity(stand_ins, limit=limit, arxiv_id=arxiv_id, max_chunks_per_paper=max_chunks_per_paper)
    return [candidates[hit.chunk_id] for hit in kept]


def candidate_from_hit(hit: RetrievalHit) -> Candidate:
    breakdown: dict[str, Any] = {}
    if hit.lexical is not None:
        breakdown = asdict(hit.lexical)
    elif hit.vector is not None:
        breakdown = asdict(hit.vector)
    return {
        "chunk_id": hit.chunk_id,
        "arxiv_id": hit.arxiv_id,
        "chunk_index": hit.chunk_index,
        "paper_title": hit.paper_title,
        "paper_abstract": hit.paper_abstract,
        "chunk_text": hit.chunk_text,
        "section_title": hit.section_title,
        "content_role": hit.content_role,
        "retrieval_method": hit.method,
        "score": hit.score,
        "snippet": hit.snippet,
        "score_breakdown": breakdown,
    }


def fuse_hits(inputs: HybridInputs, config: HybridFusionConfig, *, limit: int) -> list[RetrievalHit]:
    if not inputs.query:
        return []
    originals = {
        "lexical": {hit.chunk_id: hit for hit in inputs.lexical},
        "vector": {hit.chunk_id: hit for hit in inputs.vector},
    }
    fused = fuse_hybrid_candidates(
        inputs.query,
        [candidate_from_hit(hit) for hit in inputs.lexical],
        [candidate_from_hit(hit) for hit in inputs.vector],
        config,
    )
    kept = apply_paper_diversity(fused, limit=max(1, limit))
    return [
        replace(
            originals[entry["matched_methods"][0]][int(entry["chunk_id"])],
            method="hybrid",
            score=to_float(entry["score"]),
        )
        for entry in kept
    ]
