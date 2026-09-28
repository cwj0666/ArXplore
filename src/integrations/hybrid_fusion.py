"""hybrid 검색의 lexical/vector 후보 융합과 논문 다양성 규칙(순수 함수).

`PaperRetriever`는 `DEFAULT_HYBRID_FUSION`으로 `fuse_hybrid_candidates`를 부른 뒤 `apply_paper_diversity`로 자른다.
평가 하니스(`eval/fusion_sweep.py`)는 같은 함수를 다른 설정으로 불러 저장해 둔 후보에서 융합만 다시 재생한다.
DB·임베딩 호출이 없으므로 같은 입력이면 같은 출력이다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.integrations.paper_repository import STRICT_MATCH_BONUS

WEIGHTING_MODES = ("rules", "static", "confidence_linear")
_QUERY_STOPWORDS = frozenset({"the", "and", "for", "with", "from", "that", "this"})


@dataclass(frozen=True)
class HybridFusionConfig:
    """hybrid 융합 상수 묶음. 기본값이 제품 경로다(9880127에서 정한 값, 근거 기록 없음).

    채널별 RRF 점수는 `채널 가중치 × 후보 품질 가중 / (rank_constant + 순위)`이고, 두 채널에 모두 나온 청크에는
    `overlap_bonus`를 더한다. 채널 가중치는 `weighting`이 정한다.

    - `rules`: 질의 토큰 수, lexical 1위 confidence, 상위 N개 겹침으로 곱셈 보정한 뒤 하한을 둔다(현재 제품 규칙).
    - `static`: `static_weights`(lexical, vector)를 그대로 쓴다. (1, 1)이면 표준 RRF 가중치다.
    - `confidence_linear`: `w_lex = w_min + (1 - w_min) × clip(conf / tau, 0, 1)`, `w_vec = 1`. conf는 lexical 1위 confidence.

    `drop_partial_lexical`이 켜져 있으면 vector 결과가 있을 때 lexical 부분 일치 행(`strict_match`가 False)을 뺀다.
    """

    rank_constant: float = 60.0
    overlap_bonus: float = 0.015
    drop_partial_lexical: bool = True
    weighting: str = "rules"
    long_query_min_tokens: int = 5
    long_query_multipliers: tuple[float, float] = (0.85, 1.05)
    # (confidence 상한(미만), lexical 배수, vector 배수). 앞에서부터 처음 맞는 구간 하나만 적용한다.
    confidence_bands: tuple[tuple[float, float, float], ...] = ((0.3, 0.45, 1.1), (0.5, 0.7, 1.05))
    no_overlap_top_n: int = 5
    no_overlap_max_confidence: float = 0.4
    no_overlap_multipliers: tuple[float, float] = (0.75, 1.08)
    weight_floors: tuple[float, float] = (0.2, 0.5)
    static_weights: tuple[float, float] = (1.0, 1.0)
    confidence_min_weight: float = 0.5
    confidence_tau: float = 0.5
    quality_weight: bool = True
    # (lexical confidence 상한(미만), 품질 가중). 모든 상한 이상이면 1.0.
    quality_tiers: tuple[tuple[float, float], ...] = ((0.2, 0.2), (0.3, 0.4), (0.5, 0.65), (0.8, 0.85))

    def __post_init__(self) -> None:
        if self.weighting not in WEIGHTING_MODES:
            raise ValueError(f"weighting must be one of {WEIGHTING_MODES}, got {self.weighting!r}")
        if self.rank_constant <= 0:
            raise ValueError(f"rank_constant must be > 0, got {self.rank_constant}")
        if self.weighting == "confidence_linear" and self.confidence_tau <= 0:
            raise ValueError(f"confidence_tau must be > 0, got {self.confidence_tau}")


DEFAULT_HYBRID_FUSION = HybridFusionConfig()
# 가중치·품질 가중·교차 보너스가 없는 표준 RRF(k=60). 예전 `eval.runner.plain_rrf`와 같게 부분 일치 행도 버리지 않는다.
STANDARD_RRF_FUSION = HybridFusionConfig(
    weighting="static", quality_weight=False, overlap_bonus=0.0, drop_partial_lexical=False
)


def to_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def query_tokens(query: str) -> set[str]:
    """짧은 영문 질의에서 의미 있는 토큰만 뽑는다."""
    return {
        token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) >= 3 and token not in _QUERY_STOPWORDS
    }


def lexical_confidence(candidate: dict) -> float:
    """hybrid 가중치 판단에 쓰는 lexical 점수. strict 일치 행은 `STRICT_MATCH_BONUS`를 뺀 점수,
    일부 lexeme만 일치한 행은 0이다. `strict_match`가 없는 후보는 점수를 그대로 쓴다."""
    score = to_float(candidate.get("score"))
    strict_match = (candidate.get("score_breakdown") or {}).get("strict_match")
    if strict_match is None:
        return score
    if not strict_match:
        return 0.0
    return score - STRICT_MATCH_BONUS


def resolve_hybrid_method_weights(
    query: str,
    lexical_candidates: list[dict],
    vector_candidates: list[dict],
    config: HybridFusionConfig = DEFAULT_HYBRID_FUSION,
) -> dict[str, float]:
    """질의 성격과 lexical confidence를 보고 채널 가중치를 정한다. `lexical_candidates`는 부분 일치 행을 뺀 뒤의 목록이다."""
    lexical_top_score = lexical_confidence(lexical_candidates[0]) if lexical_candidates else 0.0
    if config.weighting == "static":
        return {"lexical": config.static_weights[0], "vector": config.static_weights[1]}
    if config.weighting == "confidence_linear":
        ratio = min(1.0, max(0.0, lexical_top_score / config.confidence_tau))
        w_min = config.confidence_min_weight
        return {"lexical": w_min + (1.0 - w_min) * ratio, "vector": 1.0}

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


def hybrid_quality_weight(method: str, candidate: dict, config: HybridFusionConfig = DEFAULT_HYBRID_FUSION) -> float:
    """RRF에 후보 자체의 confidence를 반영한다. vector 후보와 품질 가중을 끈 설정은 1.0."""
    if method != "lexical" or not config.quality_weight:
        return 1.0
    score = lexical_confidence(candidate)
    for upper, weight in config.quality_tiers:
        if score < upper:
            return weight
    return 1.0


def fuse_hybrid_candidates(
    query: str,
    lexical_candidates: list[dict],
    vector_candidates: list[dict],
    config: HybridFusionConfig = DEFAULT_HYBRID_FUSION,
) -> list[dict]:
    """lexical/vector 결과를 reciprocal rank fusion으로 병합해 점수 순으로 정렬한다. 논문 다양성은 적용하지 않는다.

    `config.drop_partial_lexical`이 켜져 있고 vector 결과가 있으면 lexical 결과 중 일부 lexeme만 일치한 행
    (`strict_match`가 False)은 융합에서 뺀다. vector 결과가 없으면 lexical 결과를 모두 쓴다.
    동점은 일치한 채널 수, chunk_id 순으로 내림차순이다. chunk_id가 없는 후보는 서로 합치지 않는다.
    """
    if vector_candidates and config.drop_partial_lexical:
        lexical_candidates = [
            candidate
            for candidate in lexical_candidates
            if (candidate.get("score_breakdown") or {}).get("strict_match") is not False
        ]
    method_weights = resolve_hybrid_method_weights(query, lexical_candidates, vector_candidates, config)
    merged: dict[int, dict] = {}
    fallback_key_seed = -1

    for method, candidates in (("lexical", lexical_candidates), ("vector", vector_candidates)):
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
            quality_weight = hybrid_quality_weight(method, candidate, config)
            rrf_score = (method_weights[method] * quality_weight) / (config.rank_constant + rank)
            method_score = to_float(candidate.get("score"))
            method_breakdown = dict(candidate.get("score_breakdown") or {})

            if method not in entry["matched_methods"]:
                entry["matched_methods"].append(method)

            entry["score"] = to_float(entry.get("score")) + rrf_score
            entry["similarity_score"] = entry["score"]
            entry["score_breakdown"][f"{method}_rank"] = rank
            entry["score_breakdown"][f"{method}_rrf_score"] = rrf_score
            entry["score_breakdown"][f"{method}_score"] = method_score
            entry["score_breakdown"][f"{method}_weight"] = method_weights[method]
            entry["score_breakdown"][f"{method}_quality_weight"] = quality_weight
            entry["score_breakdown"][f"{method}_score_breakdown"] = method_breakdown

            if method == "vector" and "lexical" not in entry["matched_methods"]:
                entry["snippet"] = candidate.get("snippet") or entry.get("snippet")
            elif method == "lexical":
                entry["snippet"] = candidate.get("snippet") or entry.get("snippet")

    merged_candidates = list(merged.values())
    for candidate in merged_candidates:
        overlap_bonus = config.overlap_bonus if len(candidate.get("matched_methods") or []) > 1 else 0.0
        if overlap_bonus > 0:
            candidate["score"] = to_float(candidate.get("score")) + overlap_bonus
            candidate["similarity_score"] = candidate["score"]
            candidate["score_breakdown"]["cross_method_overlap_bonus"] = overlap_bonus

    merged_candidates.sort(
        key=lambda item: (
            to_float(item.get("score")),
            len(item.get("matched_methods") or []),
            int(item.get("chunk_id") or 0),
        ),
        reverse=True,
    )
    return merged_candidates


def apply_paper_diversity(
    candidates: list[dict],
    *,
    limit: int,
    arxiv_id: str | None = None,
    max_chunks_per_paper: int = 2,
) -> list[dict]:
    """논문마다 앞에서부터 `max_chunks_per_paper`개까지만 고르고, 후보가 모자랄 때만
    상한을 넘긴 청크를 원래 순서대로 채운다. 논문 범위 검색이나 후보가 `limit` 이하면 순서대로 자른다."""
    normalized_limit = max(1, limit)
    if arxiv_id or len(candidates) <= normalized_limit:
        return candidates[:normalized_limit]

    selected: list[dict] = []
    overflow: list[dict] = []
    paper_counts: dict[str, int] = {}

    for candidate in candidates:
        candidate_arxiv_id = str(candidate.get("arxiv_id") or "")
        count = paper_counts.get(candidate_arxiv_id, 0)
        if candidate_arxiv_id and count >= max_chunks_per_paper:
            overflow.append(candidate)
            continue

        selected.append(candidate)
        if candidate_arxiv_id:
            paper_counts[candidate_arxiv_id] = count + 1
        if len(selected) >= normalized_limit:
            return selected[:normalized_limit]

    for candidate in overflow:
        selected.append(candidate)
        if len(selected) >= normalized_limit:
            break

    return selected[:normalized_limit]
