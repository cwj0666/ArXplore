"""hybrid 융합 설정 비교: 캐시한 후보로 설정별 순위를 재생하고 반복 교차검증·1-SE 규칙·paired bootstrap으로 고른다.

절차(사전 등록: docs/worklog/phase-4/2026-09-28_01_*.md)

1. `build_specs`가 후보 설정을 만든다. R0(vector·lexical 단독), F0(표준 RRF, k 격자), F1(고정 가중치 w_lex × k),
   F2(confidence 선형 가중 w_min × tau × k), C(가중 규칙, 2026-09-29까지의 제품 설정 `LEGACY_RULES_FUSION`),
   CF(C의 방법 가중·품질 가중·교차 보너스 2^3 요인 배치).
   F1·F2는 품질 가중 on/off × 교차 보너스 {0, 0.015}와 교차한다. REF는 비교용(예전 `hybrid_plainrrf` ablation 등)이다.
   lexical 부분 일치 행 제거(`drop_partial_lexical`)는 REF를 뺀 모든 설정에서 제품과 같게 켠다.
2. `evaluate_specs`가 설정마다 질의별 `QueryResult`(eval.runner 지표 코드)를 만든다.
3. `fold_plan` + `cross_validate`: 클러스터 층화 5-fold를 10번 반복한다. ko/en 쌍(id 끝 `-ko`/`-en`만 다른 질의,
   `cluster_key`)은 한 클러스터로 같은 fold에 들어간다. 가족마다 학습 fold에서 MRR이 가장 높은 설정을 고르고
   남긴 fold에서 잰다. 가족별 CV 평균 ± SE(반복마다 fold 점수 표준편차/√fold 수의 평균)를 낸다.
4. `select_one_se`: 선택 가능한 가족(F0·F1·F2·C·CF) 전체 설정 중 전체 데이터 MRR 최고 설정의 CV SE를 기준으로,
   최고값 − 1 SE 이상인 설정 중 상수가 가장 적은 설정을 고른다(동률이면 MRR이 높은 쪽, 그다음 이름 순).
5. `paired_bootstrap`: 클러스터 단위 재표집(10,000회, 고정 seed)으로 ΔMRR·Δhit@1의 95% 백분위 구간을 낸다.
   통계량은 질의 평균(뽑힌 클러스터들의 질의 차이 합 / 뽑힌 질의 수)이다.

convex combination 비교(사전 등록: docs/worklog/phase-4/2026-09-29_01_*.md)는 `build_convex_specs`의 CC(선택 대상)·CCK(보조)
가족을 위 설정에 더해 같은 절차로 돌리고, pick을 CC 안에서만 고른다(`run_sweep(selectable_families=("CC",))`).
`render_convex_sections`가 정규화별 CV, α 곡선, 퇴화 정규화 수, 질의별 차이를 덧붙인다.

seed가 같으면 결과가 같다. 난수는 `random.Random(seed)`만 쓴다.
"""

from __future__ import annotations

import csv
import math
import re
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from random import Random
from typing import Any

from eval.candidate_cache import CachedQuery, replay
from eval.metrics import mean, percentile
from eval.report import format_rate
from eval.runner import QueryResult, aggregate, hit_cutoffs, score_hits
from src.integrations.hybrid_fusion import (
    LEGACY_RULES_FUSION,
    LEXICAL_SCORE_FLOOR,
    STANDARD_RRF_FUSION,
    STRICT_LEXICAL_SCORE_FLOOR,
    VECTOR_SCORE_FLOOR,
    HybridFusionConfig,
    channel_score_range,
    fusion_lexical_candidates,
    to_float,
)

RANK_CONSTANTS = (10, 20, 40, 60, 100)
STATIC_LEXICAL_WEIGHTS = tuple(round(0.1 * step, 1) for step in range(11))
CONFIDENCE_MIN_WEIGHTS = tuple(round(0.1 * step, 1) for step in range(1, 10))
CONFIDENCE_TAUS = (0.2, 0.3, 0.5, 0.8)
OVERLAP_BONUSES = (0.0, 0.015)
QUALITY_SWITCHES = (False, True)
# convex combination: lexical 가중치 α 격자(α = 1.0은 lexical이 빈 질의에서 모든 점수가 0이 되어 뺀다)와 정규화 3종.
CONVEX_ALPHAS = tuple(round(0.05 * step, 2) for step in range(20))
CONVEX_NORMALIZATIONS = {
    "mm": ("minmax", (LEXICAL_SCORE_FLOOR, VECTOR_SCORE_FLOOR)),
    "tmm": ("theoretical", (LEXICAL_SCORE_FLOOR, VECTOR_SCORE_FLOOR)),
    "tmms": ("theoretical", (STRICT_LEXICAL_SCORE_FLOOR, VECTOR_SCORE_FLOOR)),
}
# 부분 일치 행을 버리지 않는 보조 가족에는 strict 하한(tmms)이 성립하지 않아 넣지 않는다.
CONVEX_KEEP_NORMALIZATIONS = ("mm", "tmm")
CONVEX = "CC"
CONVEX_KEEP = "CCK"
SELECTABLE_FAMILIES = ("F0", "F1", "F2", "C", "CF")
FAMILY_ORDER = ("R0_vector", "R0_lexical", "F0", "F1", "F2", "C", "CF", CONVEX, CONVEX_KEEP)
REPORT_SUBSETS = ("all", "known_item", "manual", "llm_synth", "ko", "en")
DEFAULT_SEED = 20260928
DEFAULT_FOLDS = 5
DEFAULT_REPEATS = 10
DEFAULT_RESAMPLES = 10_000
# 설정 이름은 예전 리포트와 비교할 수 있게 그대로 둔다. C는 2026-09-29까지의 제품 규칙이고 지금 제품 기본은 `CC_mm_a0.35`다.
CURRENT = "C_current"
PLAIN_RRF_K60 = "F0_rrf_k60"
VECTOR_ONLY = "vector_only"
LEXICAL_ONLY = "lexical_only"
_PAIR_SUFFIX = re.compile(r"-(?:ko|en)$")


def count_parameters(config: HybridFusionConfig) -> int:
    """1-SE 규칙의 단순성 기준: 설정이 실제로 쓰는 자유 상수 수.

    rank_constant 1개는 모든 설정에 있다. 교차 보너스 1, 품질 가중 구간(상한 + 값) 8, 고정 가중치는 1이 아닌 값마다 1,
    confidence 선형은 w_min·tau 2, 가중 규칙(C)은 토큰 기준 1 + 배수 2 + 신뢰도 구간 6 + 겹침 규칙 4 + 하한 2 = 15.
    `drop_partial_lexical` 같은 구조 스위치는 세지 않는다. 가중 규칙 C는 25개다.
    convex는 α 1개다. 정규화 종류와 이론 하한은 점수 정의에서 정한 값이라 세지 않는다.
    """
    if config.weighting == "convex":
        return 1
    count = 1
    if config.overlap_bonus > 0:
        count += 1
    if config.quality_weight:
        count += 2 * len(config.quality_tiers)
    if config.weighting == "static":
        count += sum(1 for weight in config.static_weights if weight != 1.0)
    elif config.weighting == "confidence_linear":
        count += 2
    else:
        count += (
            1
            + len(config.long_query_multipliers)
            + 3 * len(config.confidence_bands)
            + 2
            + len(config.no_overlap_multipliers)
            + len(config.weight_floors)
        )
    return count


@dataclass(frozen=True)
class SweepSpec:
    name: str
    family: str
    channel: str = "hybrid"
    config: HybridFusionConfig | None = None
    selectable: bool = True

    @property
    def n_params(self) -> int:
        return count_parameters(self.config) if self.config is not None else 0


def _rrf(k: float, *, lexical_weight: float = 1.0, quality: bool = False, bonus: float = 0.0) -> HybridFusionConfig:
    return HybridFusionConfig(
        rank_constant=float(k),
        weighting="static",
        static_weights=(lexical_weight, 1.0),
        quality_weight=quality,
        overlap_bonus=bonus,
    )


def build_specs() -> list[SweepSpec]:
    """비교할 설정 전체. 이름은 설정을 그대로 드러내고 가족 안에서 유일하다."""
    specs = [
        SweepSpec(VECTOR_ONLY, "R0_vector", channel="vector", selectable=False),
        SweepSpec(LEXICAL_ONLY, "R0_lexical", channel="lexical", selectable=False),
    ]
    specs.extend(SweepSpec(f"F0_rrf_k{k}", "F0", config=_rrf(k)) for k in RANK_CONSTANTS)
    for quality, bonus, k, weight in product(QUALITY_SWITCHES, OVERLAP_BONUSES, RANK_CONSTANTS, STATIC_LEXICAL_WEIGHTS):
        specs.append(
            SweepSpec(
                f"F1_w{weight:.1f}_k{k}_q{int(quality)}_b{bonus:g}",
                "F1",
                config=_rrf(k, lexical_weight=weight, quality=quality, bonus=bonus),
            )
        )
    for quality, bonus, k, w_min, tau in product(
        QUALITY_SWITCHES, OVERLAP_BONUSES, RANK_CONSTANTS, CONFIDENCE_MIN_WEIGHTS, CONFIDENCE_TAUS
    ):
        config = HybridFusionConfig(
            rank_constant=float(k),
            weighting="confidence_linear",
            confidence_min_weight=w_min,
            confidence_tau=tau,
            quality_weight=quality,
            overlap_bonus=bonus,
        )
        specs.append(SweepSpec(f"F2_wmin{w_min:.1f}_tau{tau:g}_k{k}_q{int(quality)}_b{bonus:g}", "F2", config=config))
    specs.append(SweepSpec(CURRENT, "C", config=LEGACY_RULES_FUSION))
    for rules, quality, bonus in product((1, 0), (1, 0), (1, 0)):
        config = replace(
            LEGACY_RULES_FUSION,
            weighting="rules" if rules else "static",
            quality_weight=bool(quality),
            overlap_bonus=LEGACY_RULES_FUSION.overlap_bonus if bonus else 0.0,
        )
        specs.append(SweepSpec(f"CF_m{rules}q{quality}b{bonus}", "CF", config=config))
    specs.append(
        SweepSpec(
            "REF_C_keep_partial",
            "REF",
            config=replace(LEGACY_RULES_FUSION, drop_partial_lexical=False),
            selectable=False,
        )
    )
    specs.append(SweepSpec("REF_plainrrf_ablation", "REF", config=STANDARD_RRF_FUSION, selectable=False))
    return specs


def convex_config(normalization: str, alpha: float, *, drop_partial_lexical: bool = True) -> HybridFusionConfig:
    mode, floors = CONVEX_NORMALIZATIONS[normalization]
    return HybridFusionConfig(
        weighting="convex",
        convex_alpha=alpha,
        score_normalization=mode,
        score_floors=floors,
        drop_partial_lexical=drop_partial_lexical,
    )


def build_convex_specs() -> list[SweepSpec]:
    """CC(부분 일치 행 제거 켬, 선택 대상) 정규화 3종 × α 20개 = 60개, CCK(제거 끔, 보고만) mm·tmm × α 20개 = 40개.
    이름은 `CC_<정규화>_a<α>`라 이름 순이 mm < tmm < tmms, 같은 정규화 안에서는 α가 작은 쪽(vector에 가까운 쪽)이 앞이다."""
    specs = [
        SweepSpec(f"{CONVEX}_{normalization}_a{alpha:.2f}", CONVEX, config=convex_config(normalization, alpha))
        for normalization, alpha in product(CONVEX_NORMALIZATIONS, CONVEX_ALPHAS)
    ]
    specs.extend(
        SweepSpec(
            f"{CONVEX_KEEP}_{normalization}_a{alpha:.2f}",
            CONVEX_KEEP,
            config=convex_config(normalization, alpha, drop_partial_lexical=False),
            selectable=False,
        )
        for normalization, alpha in product(CONVEX_KEEP_NORMALIZATIONS, CONVEX_ALPHAS)
    )
    return specs


def convex_groups(specs: Sequence[SweepSpec]) -> dict[str, list[str]]:
    """정규화별 하위 가족(`CC_mm`, `CCK_tmm` 등) → 설정 이름. CV를 따로 내 정규화 선택의 영향을 본다."""
    groups: dict[str, list[str]] = {}
    for spec in specs:
        if spec.family in (CONVEX, CONVEX_KEEP):
            groups.setdefault(spec.name.rsplit("_a", 1)[0], []).append(spec.name)
    return groups


def evaluate_specs(
    queries: Sequence[CachedQuery],
    specs: Sequence[SweepSpec],
    *,
    k: int,
    progress: Callable[[int, int, SweepSpec], None] | None = None,
) -> dict[str, list[QueryResult]]:
    """설정마다 캐시 질의 순서대로 재생해 `score_hits`로 채점한다."""
    results: dict[str, list[QueryResult]] = {}
    for index, spec in enumerate(specs, start=1):
        if progress is not None:
            progress(index, len(specs), spec)
        results[spec.name] = [
            score_hits(
                cached.query,
                spec.name,
                replay(cached, k=k, channel=spec.channel, config=spec.config),
                k=k,
                latency_ms=None,
            )
            for cached in queries
        ]
    return results


def per_query(results: Sequence[QueryResult], metric: str) -> list[float]:
    """질의 순서대로 논문 단위 지표 값. 검색 평가 대상 질의는 정답 논문이 있으므로 None이 없어야 한다."""
    values = [row.paper_metrics.get(metric) for row in results]
    missing = [row.query_id for row, value in zip(results, values, strict=True) if value is None]
    if missing:
        raise ValueError(f"{metric} is undefined for queries {missing[:5]}")
    return [float(value) for value in values]


def cluster_key(query_id: str) -> str:
    """ko/en 쌍 클러스터 키: id 끝의 `-ko`/`-en`을 뗀 값. 짝이 없는 질의는 혼자 한 클러스터다."""
    return _PAIR_SUFFIX.sub("", query_id)


def cluster_strata(queries: Sequence[Any]) -> tuple[list[str], list[str]]:
    """질의별 (클러스터 키, 층). 층은 클러스터 단위로 정한다: source × (쌍이면 `pair`, 혼자면 lang).

    쌍은 ko/en을 하나씩 담으므로 lang을 층에 넣지 않아도 fold마다 언어가 섞인다. 혼자인 질의만 lang으로 나눈다.
    """
    groups = [cluster_key(query.id) for query in queries]
    members: dict[str, list[Any]] = {}
    for group, query in zip(groups, queries, strict=True):
        members.setdefault(group, []).append(query)
    strata_by_group = {
        group: "+".join(sorted({query.source for query in items}))
        + (":pair" if len(items) > 1 else f":{items[0].lang}")
        for group, items in members.items()
    }
    return groups, [strata_by_group[group] for group in groups]


def clusters_of(groups: Sequence[str]) -> list[list[int]]:
    """클러스터 키 목록을 처음 나온 순서의 위치 묶음으로 바꾼다."""
    clusters: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        clusters.setdefault(group, []).append(index)
    return list(clusters.values())


def stratified_folds(groups: Sequence[str], strata: Sequence[str], n_folds: int, rng: Random) -> list[int]:
    """클러스터를 통째로 fold에 배정한다. 층마다 클러스터를 섞은 뒤, 그 층의 질의가 가장 적은 fold(동률이면 전체 질의가
    가장 적은 fold, 그다음 번호 순)에 하나씩 넣는다. 같은 클러스터의 질의는 같은 층이어야 한다."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if len(groups) != len(strata):
        raise ValueError("groups and strata must be the same length")
    clusters = clusters_of(groups)
    for members in clusters:
        if len({strata[index] for index in members}) != 1:
            raise ValueError(f"cluster {groups[members[0]]!r} spans several strata")
    folds = [0] * len(groups)
    fold_sizes = [0] * n_folds
    for name in sorted(set(strata)):
        in_stratum = [members for members in clusters if strata[members[0]] == name]
        rng.shuffle(in_stratum)
        stratum_sizes = [0] * n_folds
        for members in in_stratum:
            fold = min(
                range(n_folds), key=lambda candidate: (stratum_sizes[candidate], fold_sizes[candidate], candidate)
            )
            for index in members:
                folds[index] = fold
            stratum_sizes[fold] += len(members)
            fold_sizes[fold] += len(members)
    return folds


def fold_plan(
    groups: Sequence[str], strata: Sequence[str], *, n_folds: int, repeats: int, seed: int
) -> list[list[int]]:
    """반복마다 `Random(seed + 반복 번호)`로 만든 클러스터 층화 fold 배정. 모든 가족이 같은 배정을 쓴다."""
    return [stratified_folds(groups, strata, n_folds, Random(seed + repeat)) for repeat in range(repeats)]


@dataclass(frozen=True)
class CvResult:
    family: str
    n_configs: int
    mean: float
    se: float
    choices: tuple[tuple[str, int], ...]


def _tuning_key(spec: SweepSpec, train_mean: float) -> tuple[float, int, str]:
    return (-train_mean, spec.n_params, spec.name)


def cross_validate(
    family: str,
    specs: Sequence[SweepSpec],
    scores: Mapping[str, Sequence[float]],
    plan: Sequence[Sequence[int]],
) -> CvResult:
    """학습 fold 평균이 가장 높은 설정(동률이면 상수가 적은 쪽, 이름 순)을 골라 남긴 fold에서 잰다.

    반복마다 평균은 질의 가중(남긴 fold 점수 합 / 질의 수), SE는 fold 점수 표본 표준편차 / √fold 수이고,
    반복 평균을 돌려준다. `choices`는 fold마다 고른 설정의 빈도(많은 순)다.
    """
    if not specs:
        raise ValueError(f"family {family!r} has no configs")
    n_queries = len(next(iter(scores.values())))
    totals = {spec.name: sum(scores[spec.name]) for spec in specs}
    repeat_means: list[float] = []
    repeat_ses: list[float] = []
    choices: Counter[str] = Counter()
    for folds in plan:
        n_folds = max(folds) + 1
        fold_sizes = Counter(folds)
        fold_sums = {spec.name: [0.0] * n_folds for spec in specs}
        for spec in specs:
            sums = fold_sums[spec.name]
            for value, fold in zip(scores[spec.name], folds, strict=True):
                sums[fold] += value
        held_out_total = 0.0
        fold_scores: list[float] = []
        for fold in range(n_folds):
            train_size = n_queries - fold_sizes[fold]
            chosen = min(
                specs,
                key=lambda spec: _tuning_key(spec, (totals[spec.name] - fold_sums[spec.name][fold]) / train_size),
            )
            choices[chosen.name] += 1
            held_out = fold_sums[chosen.name][fold]
            held_out_total += held_out
            fold_scores.append(held_out / fold_sizes[fold])
        repeat_means.append(held_out_total / n_queries)
        repeat_ses.append(statistics.stdev(fold_scores) / math.sqrt(n_folds))
    return CvResult(
        family=family,
        n_configs=len(specs),
        mean=statistics.fmean(repeat_means),
        se=statistics.fmean(repeat_ses),
        choices=tuple(sorted(choices.items(), key=lambda item: (-item[1], item[0]))),
    )


@dataclass(frozen=True)
class Selection:
    best: str
    best_score: float
    se: float
    threshold: float
    pick: str
    pick_score: float
    within: tuple[str, ...]


def select_one_se(
    specs: Sequence[SweepSpec], full_scores: Mapping[str, float], se_of: Callable[[str], float]
) -> Selection:
    """1-SE 규칙. 최고 설정(전체 데이터 점수, 동률이면 상수가 적은 쪽·이름 순)의 SE만큼 낮춘 기준 이상인 설정 중
    상수가 가장 적은 설정을 고른다. `within`은 기준을 넘은 설정을 (상수 수, -점수, 이름) 순으로 담는다."""
    best = min(specs, key=lambda spec: (-full_scores[spec.name], spec.n_params, spec.name))
    best_score = full_scores[best.name]
    se = se_of(best.name)
    threshold = best_score - se
    within = sorted(
        (spec for spec in specs if full_scores[spec.name] >= threshold),
        key=lambda spec: (spec.n_params, -full_scores[spec.name], spec.name),
    )
    pick = within[0]
    return Selection(
        best=best.name,
        best_score=best_score,
        se=se,
        threshold=threshold,
        pick=pick.name,
        pick_score=full_scores[pick.name],
        within=tuple(spec.name for spec in within),
    )


def bootstrap_draws(n: int, n_resamples: int, seed: int) -> list[list[int]]:
    """클러스터 번호 0..n-1에서 n개씩 복원 추출한 표본 `n_resamples`개. 모든 비교가 같은 표본을 쓴다(공통 난수)."""
    rng = Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(n_resamples)]


@dataclass(frozen=True)
class BootstrapResult:
    label: str
    subset: str
    metric: str
    n: int
    n_clusters: int
    delta: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    draws: Sequence[Sequence[int]],
    *,
    clusters: Sequence[Sequence[int]] | None = None,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """질의별 차이 a−b의 평균과 클러스터 재표집 평균의 백분위 신뢰구간(선형 보간).

    `clusters`는 질의 위치 묶음이고 `draws`의 원소는 클러스터 번호다. 표본마다 통계량은 뽑힌 클러스터들의 질의 차이 합을
    뽑힌 질의 수로 나눈 질의 평균이다. `clusters`가 없으면 질의 하나가 클러스터 하나다.
    """
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be non-empty and the same length")
    deltas = [left - right for left, right in zip(a, b, strict=True)]
    clusters = [[index] for index in range(len(deltas))] if clusters is None else clusters
    if sorted(index for members in clusters for index in members) != list(range(len(deltas))):
        raise ValueError("clusters must partition the query positions")
    cluster_sums = [sum(deltas[index] for index in members) for members in clusters]
    cluster_sizes = [len(members) for members in clusters]
    resampled = [
        sum(cluster_sums[cluster] for cluster in draw) / sum(cluster_sizes[cluster] for cluster in draw)
        for draw in draws
    ]
    tail = (1.0 - confidence) / 2 * 100
    return statistics.fmean(deltas), percentile(resampled, tail), percentile(resampled, 100 - tail)


def compare(
    label: str,
    a: str,
    b: str,
    results: Mapping[str, Sequence[QueryResult]],
    *,
    metrics: Sequence[str],
    subset: str,
    positions: Sequence[int],
    clusters: Sequence[Sequence[int]],
    draws: Sequence[Sequence[int]],
) -> list[BootstrapResult]:
    """`positions` 질의만 골라 a−b를 지표마다 클러스터 paired bootstrap한다. `clusters`는 positions 안의 위치 묶음,
    `draws`는 클러스터 수로 뽑은 표본이다."""
    rows: list[BootstrapResult] = []
    for metric in metrics:
        left = per_query(results[a], metric)
        right = per_query(results[b], metric)
        delta, low, high = paired_bootstrap(
            [left[index] for index in positions],
            [right[index] for index in positions],
            draws,
            clusters=clusters,
        )
        rows.append(BootstrapResult(label, subset, metric, len(positions), len(clusters), delta, low, high))
    return rows


@dataclass(frozen=True)
class SweepReport:
    k: int
    specs: tuple[SweepSpec, ...]
    results: dict[str, list[QueryResult]]
    full_scores: dict[str, float]
    cv: dict[str, CvResult]
    config_se: dict[str, float]
    selection: Selection
    comparisons: tuple[BootstrapResult, ...]
    decision: tuple[str, ...]
    settings: dict[str, Any]
    subset_sizes: dict[str, tuple[int, int]]
    selectable_families: tuple[str, ...] = SELECTABLE_FAMILIES


def _subset_positions(queries: Sequence[CachedQuery], subset: str) -> list[int]:
    if subset == "all":
        return list(range(len(queries)))
    if subset in ("ko", "en"):
        return [index for index, cached in enumerate(queries) if cached.query.lang == subset]
    return [index for index, cached in enumerate(queries) if cached.query.source == subset]


def run_sweep(
    queries: Sequence[CachedQuery],
    *,
    k: int,
    specs: Sequence[SweepSpec] | None = None,
    seed: int = DEFAULT_SEED,
    n_folds: int = DEFAULT_FOLDS,
    repeats: int = DEFAULT_REPEATS,
    n_resamples: int = DEFAULT_RESAMPLES,
    progress: Callable[[int, int, SweepSpec], None] | None = None,
    selectable_families: Sequence[str] = SELECTABLE_FAMILIES,
    cv_groups: Mapping[str, Sequence[str]] | None = None,
    rrf_check: bool = False,
) -> SweepReport:
    """재생 → 채점 → 가족별 CV → 1-SE 선택 → paired bootstrap → 사전 등록 판정.

    pick은 `selectable_families`의 선택 가능 설정에서 고른다. `cv_groups`(이름 → 설정 이름)는 가족 CV에 더해 따로 CV를 낸다.
    `rrf_check`면 판정에 규칙 (4)(pick − F0(k=60), 논문 주장 점검)를 덧붙인다.
    """
    specs = list(build_specs() if specs is None else specs)
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("spec names must be unique")
    for required in (CURRENT, PLAIN_RRF_K60, VECTOR_ONLY):
        if required not in names:
            raise ValueError(f"specs must include {required!r}")
    primary = f"mrr@{k}"
    results = evaluate_specs(queries, specs, k=k, progress=progress)
    scores = {name: per_query(rows, primary) for name, rows in results.items()}
    full_scores = {name: statistics.fmean(values) for name, values in scores.items()}

    groups, strata = cluster_strata([cached.query for cached in queries])
    plan = fold_plan(groups, strata, n_folds=n_folds, repeats=repeats, seed=seed)
    families: dict[str, list[SweepSpec]] = {}
    for spec in specs:
        families.setdefault(spec.family, []).append(spec)
    cv = {
        family: cross_validate(family, members, scores, plan) for family, members in families.items() if family != "REF"
    }
    by_name = {spec.name: spec for spec in specs}
    for group, members in (cv_groups or {}).items():
        if group in cv:
            raise ValueError(f"cv group {group!r} collides with a family name")
        cv[group] = cross_validate(group, [by_name[name] for name in members], scores, plan)

    selectable_families = tuple(selectable_families)
    selectable = [spec for spec in specs if spec.selectable and spec.family in selectable_families]
    if not selectable:
        raise ValueError(f"no selectable configs in families {selectable_families}")
    config_se: dict[str, float] = {}

    def se_of(name: str) -> float:
        if name not in config_se:
            spec = next(spec for spec in specs if spec.name == name)
            config_se[name] = cross_validate(name, [spec], scores, plan).se
        return config_se[name]

    selection = select_one_se(selectable, full_scores, se_of)
    for name in (CURRENT, PLAIN_RRF_K60, VECTOR_ONLY, LEXICAL_ONLY, selection.pick):
        se_of(name)

    metrics = (primary, "hit@1")
    pairs = [
        ("pick − C", selection.pick, CURRENT),
        ("pick − F0(k=60)", selection.pick, PLAIN_RRF_K60),
        ("pick − vector_only", selection.pick, VECTOR_ONLY),
        ("C − F0(k=60)", CURRENT, PLAIN_RRF_K60),
        ("C − vector_only", CURRENT, VECTOR_ONLY),
    ]
    comparisons: list[BootstrapResult] = []
    for offset, subset in enumerate(("all", "manual")):
        positions = _subset_positions(queries, subset)
        if not positions:
            continue
        clusters = clusters_of([groups[index] for index in positions])
        draws = bootstrap_draws(len(clusters), n_resamples, seed + offset)
        for label, a, b in pairs:
            comparisons.extend(
                compare(
                    label,
                    a,
                    b,
                    results,
                    metrics=metrics,
                    subset=subset,
                    positions=positions,
                    clusters=clusters,
                    draws=draws,
                )
            )
    subset_sizes = {}
    for subset in REPORT_SUBSETS:
        positions = _subset_positions(queries, subset)
        subset_sizes[subset] = (len(positions), len({groups[index] for index in positions}))

    decision = pre_registered_decision(selection, comparisons, primary=primary, rrf_check=rrf_check)
    return SweepReport(
        k=k,
        specs=tuple(specs),
        results=results,
        full_scores=full_scores,
        cv=cv,
        config_se=config_se,
        selection=selection,
        comparisons=tuple(comparisons),
        decision=tuple(decision),
        settings={"seed": seed, "folds": n_folds, "repeats": repeats, "resamples": n_resamples},
        subset_sizes=subset_sizes,
        selectable_families=selectable_families,
    )


def _find(comparisons: Sequence[BootstrapResult], label: str, subset: str, metric: str) -> BootstrapResult | None:
    return next(
        (row for row in comparisons if row.label == label and row.subset == subset and row.metric == metric), None
    )


def pre_registered_decision(
    selection: Selection, comparisons: Sequence[BootstrapResult], *, primary: str, rrf_check: bool = False
) -> list[str]:
    """사전 등록한 판정 규칙을 기계적으로 적용한 문장들. 최종 채택은 사람이 이 문장과 표를 보고 정한다.
    `rrf_check`면 규칙 (4)(pick − F0(k=60), 제품 판정에는 쓰지 않는 논문 주장 점검)를 덧붙인다."""
    lines: list[str] = []
    versus_current = _find(comparisons, "pick − C", "all", primary)
    manual = _find(comparisons, "pick − C", "manual", primary)
    final = selection.pick
    if selection.pick == CURRENT:
        lines.append(f"1-SE 선택이 가중 규칙({CURRENT})이다 → C 유지.")
        final = CURRENT
    elif manual is not None and manual.high < 0:
        lines.append(
            f"manual 부분집합 비회귀 조건 실패: pick − C ΔMRR 95% CI [{manual.low:+.3f}, {manual.high:+.3f}]가 0 아래 → 교체하지 않는다."
        )
        final = CURRENT
    elif versus_current is None:
        lines.append("pick − C 비교가 없다.")
    elif versus_current.high < 0:
        lines.append(
            f"규칙 (2): C가 유의하게 낫다(pick − C ΔMRR CI [{versus_current.low:+.3f}, {versus_current.high:+.3f}]) → "
            "C 유지, 요인 배치(CF) 표로 어느 요인이 기여하는지 기록."
        )
        final = CURRENT
    elif versus_current.low > 0:
        lines.append(
            f"pick이 C보다 유의하게 낫다(ΔMRR CI [{versus_current.low:+.3f}, {versus_current.high:+.3f}]) → {selection.pick}로 교체."
        )
    else:
        lines.append(
            f"규칙 (1): pick − C ΔMRR CI [{versus_current.low:+.3f}, {versus_current.high:+.3f}]가 0을 포함 → "
            f"더 단순한 {selection.pick}로 교체."
        )
    if manual is not None and final != CURRENT:
        lines.append(f"manual 비회귀 조건 충족: pick − C ΔMRR CI [{manual.low:+.3f}, {manual.high:+.3f}] (상한 ≥ 0).")
    versus_vector = _find(comparisons, f"{'pick' if final != CURRENT else 'C'} − vector_only", "all", primary)
    if versus_vector is not None:
        chosen = selection.pick if final != CURRENT else CURRENT
        if versus_vector.low > 0:
            lines.append(
                f"{chosen}는 vector 단독보다 유의하게 낫다(ΔMRR CI [{versus_vector.low:+.3f}, {versus_vector.high:+.3f}])."
            )
        else:
            lines.append(
                f"규칙 (3): {chosen}가 vector 단독보다 유의하게 낫지 않다(ΔMRR CI [{versus_vector.low:+.3f}, "
                f"{versus_vector.high:+.3f}]) → vector 단독 근거로 보고하고 결정은 사용자에게 넘긴다."
            )
    versus_rrf = _find(comparisons, "pick − F0(k=60)", "all", primary) if rrf_check else None
    if versus_rrf is not None:
        interval = f"ΔMRR CI [{versus_rrf.low:+.3f}, {versus_rrf.high:+.3f}]"
        if versus_rrf.low > 0:
            verdict = f"{selection.pick}가 표준 RRF보다 유의하게 낫다({interval}) → 이 설정에서 'CC > RRF'를 재현했다."
        elif versus_rrf.high < 0:
            verdict = f"표준 RRF가 {selection.pick}보다 유의하게 낫다({interval}) → 'CC > RRF'와 반대다."
        else:
            verdict = f"{selection.pick}와 표준 RRF가 구분되지 않는다({interval}) → 'CC > RRF'를 재현하지 못했다."
        lines.append(f"규칙 (4)(논문 주장 점검, 제품 판정에는 쓰지 않음): {verdict}")
    return lines


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _config_fields(spec: SweepSpec) -> dict[str, Any]:
    config = spec.config
    if config is None:
        return {"channel": spec.channel}
    return {
        "channel": spec.channel,
        "rank_constant": config.rank_constant,
        "weighting": config.weighting,
        "w_lex": config.static_weights[0] if config.weighting == "static" else "",
        "w_min": config.confidence_min_weight if config.weighting == "confidence_linear" else "",
        "tau": config.confidence_tau if config.weighting == "confidence_linear" else "",
        "quality_weight": int(config.quality_weight),
        "overlap_bonus": config.overlap_bonus,
        "drop_partial_lexical": int(config.drop_partial_lexical),
        "alpha": config.convex_alpha if config.weighting == "convex" else "",
        "normalization": config.score_normalization if config.weighting == "convex" else "",
        "score_floors": "/".join(f"{floor:g}" for floor in config.score_floors) if config.weighting == "convex" else "",
    }


def spec_summary(report: SweepReport, name: str) -> dict[str, Any]:
    """설정 하나의 부분집합별 집계(eval.runner.aggregate)."""
    rows = aggregate(report.results[name], subsets=REPORT_SUBSETS)
    return {row.subset: row for row in rows}


def write_csv(report: SweepReport, path: str | Path) -> None:
    k = report.k
    paper_names = [f"hit@{cutoff}" for cutoff in hit_cutoffs(k)] + [f"mrr@{k}", f"recall@{k}"]
    fields = [
        "name",
        "family",
        "selectable",
        "n_params",
        "channel",
        "rank_constant",
        "weighting",
        "w_lex",
        "w_min",
        "tau",
        "quality_weight",
        "overlap_bonus",
        "drop_partial_lexical",
        "alpha",
        "normalization",
        "score_floors",
        *paper_names,
        "chunk_hit@5",
        "n_chunk_queries",
        *(f"mrr@{k}:{subset}" for subset in REPORT_SUBSETS[1:]),
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in report.specs:
            summary = spec_summary(report, spec.name)
            overall = summary["all"]
            row: dict[str, Any] = {
                "name": spec.name,
                "family": spec.family,
                "selectable": int(spec.selectable and spec.family in report.selectable_families),
                "n_params": spec.n_params,
                **_config_fields(spec),
            }
            for metric in paper_names:
                row[metric] = _csv_float(overall.paper.get(metric))
            row["chunk_hit@5"] = _csv_float(overall.chunk.get("hit@5"))
            row["n_chunk_queries"] = overall.n_chunk_queries
            for subset in REPORT_SUBSETS[1:]:
                row[f"mrr@{k}:{subset}"] = (
                    _csv_float(summary[subset].paper.get(f"mrr@{k}")) if subset in summary else ""
                )
            writer.writerow(row)


def _csv_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _ci(row: BootstrapResult) -> str:
    return f"[{row.low:+.3f}, {row.high:+.3f}]"


def render_markdown(report: SweepReport, *, title: str, meta: Mapping[str, str]) -> str:
    k = report.k
    primary = f"mrr@{k}"
    by_name = {spec.name: spec for spec in report.specs}
    selection = report.selection
    lines = [f"# {title}", ""]
    lines.extend(f"- {key}: {value}" for key, value in meta.items())
    lines.extend(
        [
            f"- 절차: 주 지표 논문 MRR@{k}. ko/en 쌍을 한 클러스터로 묶은 층화(source × 쌍/lang) "
            f"{report.settings['folds']}-fold × {report.settings['repeats']}회 반복 CV, 1-SE 규칙, "
            f"클러스터 paired bootstrap {report.settings['resamples']:,}회, seed {report.settings['seed']}",
            f"- 질의 / 클러스터: {report.subset_sizes['all'][0]} / {report.subset_sizes['all'][1]}",
            "",
            "## 사전 등록 규칙 판정",
            "",
        ]
    )
    lines.extend(f"- {line}" for line in report.decision)

    lines.extend(["", "## 가족별 교차검증", ""])
    family_rows = []
    for family in FAMILY_ORDER:
        cv = report.cv.get(family)
        if cv is None:
            continue
        members = [spec for spec in report.specs if spec.family == family]
        best = min(members, key=lambda spec: (-report.full_scores[spec.name], spec.n_params, spec.name))
        top_choice, top_count = cv.choices[0]
        total = sum(count for _, count in cv.choices)
        family_rows.append(
            [
                family,
                str(cv.n_configs),
                f"{cv.mean:.3f} ± {cv.se:.3f}",
                f"`{best.name}` {report.full_scores[best.name]:.3f}",
                f"`{top_choice}` {top_count}/{total}",
            ]
        )
    lines.extend(
        _table(
            ["가족", "설정 수", f"CV MRR@{k} (평균 ± SE)", "전체 데이터 최고", "fold 선택 최다"],
            family_rows,
        )
    )
    lines.extend(
        [
            "",
            "CV 값은 학습 fold에서 고른 설정을 남긴 fold에서 잰 값이라 조정 낙관 편향이 빠져 있다. 전체 데이터 최고값은 편향이 있다.",
            "",
            "## 1-SE 선택",
            "",
            f"- 최고: `{selection.best}` MRR@{k} {selection.best_score:.4f}, CV SE {selection.se:.4f} → 기준 {selection.threshold:.4f}",
            f"- 선택(pick): `{selection.pick}` MRR@{k} {selection.pick_score:.4f}, 상수 {by_name[selection.pick].n_params}개",
            f"- 기준 이상 설정 {len(selection.within)}개 중 상수가 적은 순 10개:",
            "",
        ]
    )
    lines.extend(
        _table(
            ["설정", "상수 수", f"MRR@{k}"],
            [
                [f"`{name}`", str(by_name[name].n_params), f"{report.full_scores[name]:.4f}"]
                for name in selection.within[:10]
            ],
        )
    )

    key_names = list(
        dict.fromkeys([selection.pick, selection.best, CURRENT, PLAIN_RRF_K60, VECTOR_ONLY, LEXICAL_ONLY])
    ) + [spec.name for spec in report.specs if spec.family == "REF"]
    summaries = {name: spec_summary(report, name) for name in key_names}
    paper_names = [f"hit@{cutoff}" for cutoff in hit_cutoffs(k)] + [primary, f"recall@{k}"]
    lines.extend(["", "## 주요 설정 (논문 단위 전체, 청크 hit@5는 청크 정답 질의만)", ""])
    rows = []
    for name in key_names:
        overall = summaries[name]["all"]
        rows.append(
            [
                f"`{name}`",
                str(by_name[name].n_params),
                *(format_rate(overall.paper.get(metric)) for metric in paper_names),
                f"{format_rate(overall.chunk.get('hit@5'))} ({overall.n_chunk_queries})",
                format_rate(report.config_se.get(name)) if name in report.config_se else "",
            ]
        )
    lines.extend(
        _table(
            ["설정", "상수 수", *(name.replace("mrr@", "MRR@") for name in paper_names), "청크 hit@5 (n)", "CV SE"],
            rows,
        )
    )

    lines.extend(["", f"## 부분집합 MRR@{k}", ""])
    rows = []
    for name in key_names:
        cells = []
        for subset in REPORT_SUBSETS:
            row = summaries[name].get(subset)
            cells.append(format_rate(row.paper.get(primary)) if row else "n/a")
        rows.append([f"`{name}`", *cells])
    header = [
        f"{subset} (n {report.subset_sizes[subset][0]} / 클러스터 {report.subset_sizes[subset][1]})"
        for subset in REPORT_SUBSETS
    ]
    lines.extend(_table(["설정", *header], rows))

    lines.extend(["", "## 가중 규칙 C 요인 배치 (방법 가중 m × 품질 가중 q × 교차 보너스 b)", ""])
    factorial = [spec for spec in report.specs if spec.family == "CF"]
    lines.extend(
        _table(
            ["설정", f"MRR@{k}", "hit@1"],
            [
                [
                    f"`{spec.name}`",
                    f"{report.full_scores[spec.name]:.4f}",
                    format_rate(mean(row.paper_metrics.get("hit@1") for row in report.results[spec.name])),
                ]
                for spec in factorial
            ],
        )
    )
    if len(factorial) == 8:
        effects = []
        for letter, label in (("m", "방법 가중"), ("q", "품질 가중"), ("b", "교차 보너스")):
            on = [report.full_scores[spec.name] for spec in factorial if f"{letter}1" in spec.name.split("_")[1]]
            off = [report.full_scores[spec.name] for spec in factorial if f"{letter}0" in spec.name.split("_")[1]]
            effects.append(f"{label} {statistics.fmean(on) - statistics.fmean(off):+.4f}")
        lines.extend(["", f"주효과(켬 − 끔, 나머지 두 요인 평균): {', '.join(effects)}"])

    lines.extend(["", "## Paired bootstrap (a − b, 클러스터 재표집, 95% 백분위 구간)", ""])
    lines.extend(
        _table(
            ["비교", "부분집합", "지표", "n", "클러스터", "Δ", "95% CI", "0 제외"],
            [
                [
                    row.label,
                    row.subset,
                    row.metric.replace("mrr@", "MRR@"),
                    str(row.n),
                    str(row.n_clusters),
                    f"{row.delta:+.4f}",
                    _ci(row),
                    "예" if row.excludes_zero else "아니오",
                ]
                for row in report.comparisons
            ],
        )
    )
    lines.extend(
        [
            "",
            "pick은 같은 질의로 고른 설정이므로 pick − X 구간은 선택 편향만큼 낙관적이다. C·F0·vector_only는 이 질의셋보다 먼저 정해졌다.",
            "",
            f"## 전체 데이터 MRR@{k} 상위 20개",
            "",
        ]
    )
    ranked = sorted(report.specs, key=lambda spec: (-report.full_scores[spec.name], spec.n_params, spec.name))[:20]
    lines.extend(
        _table(
            ["설정", "가족", "상수 수", f"MRR@{k}"],
            [
                [f"`{spec.name}`", spec.family, str(spec.n_params), f"{report.full_scores[spec.name]:.4f}"]
                for spec in ranked
            ],
        )
    )
    return "\n".join(lines) + "\n"


def paper_rank(row: QueryResult) -> int | None:
    """정답 논문의 첫 순위(1부터). 상위 k에 없으면 None. 논문 RR@k의 역수와 같다."""
    mrr = next((value for key, value in row.paper_metrics.items() if key.startswith("mrr@")), None)
    if not mrr:
        return None
    return round(1 / mrr)


@dataclass(frozen=True)
class QueryDifference:
    query_id: str
    source: str
    lang: str
    a_rank: int | None
    b_rank: int | None

    @property
    def a_better(self) -> bool:
        return (self.a_rank or math.inf) < (self.b_rank or math.inf)


def query_differences(report: SweepReport, a: str, b: str) -> list[QueryDifference]:
    """두 설정의 정답 논문 순위(논문 RR@k)가 다른 질의. 캐시 질의 순서를 따른다."""
    differences = []
    for left, right in zip(report.results[a], report.results[b], strict=True):
        a_rank, b_rank = paper_rank(left), paper_rank(right)
        if a_rank != b_rank:
            differences.append(QueryDifference(left.query_id, left.source, left.lang, a_rank, b_rank))
    return differences


def degenerate_normalizations(queries: Sequence[CachedQuery], config: HybridFusionConfig) -> dict[str, int]:
    """convex 정규화가 퇴화(상한 − 하한 ≤ 0, 모두 1.0)하는 질의 수를 채널별로 센다. `empty_lexical`은 융합에 들어가는
    lexical 후보가 없는 질의 수다."""
    counts = {"lexical": 0, "vector": 0, "empty_lexical": 0}
    for cached in queries:
        lexical = fusion_lexical_candidates(list(cached.lexical), list(cached.vector), config)
        if not lexical:
            counts["empty_lexical"] += 1
        for method, candidates, floor in (
            ("lexical", lexical, config.score_floors[0]),
            ("vector", list(cached.vector), config.score_floors[1]),
        ):
            if not candidates:
                continue
            low, high = channel_score_range(
                [to_float(candidate.get("score")) for candidate in candidates], config.score_normalization, floor
            )
            if high - low <= 0:
                counts[method] += 1
    return counts


def _rank_cell(rank: int | None) -> str:
    return "10위 밖" if rank is None else f"{rank}위"


def _cv_cell(report: SweepReport, name: str, family: str | None = None) -> str:
    """설정 하나는 그 설정의 CV(평균 = 전체 데이터 값), 가족을 주면 가족 CV."""
    if family is not None:
        cv = report.cv[family]
        return f"{cv.mean:.3f} ± {cv.se:.3f} ({family} 가족)"
    se = report.config_se.get(name)
    return f"{report.full_scores[name]:.3f} ± {se:.3f}" if se is not None else ""


def _hit1(report: SweepReport, name: str) -> float | None:
    return mean(row.paper_metrics.get("hit@1") for row in report.results[name])


def render_convex_sections(
    report: SweepReport,
    queries: Sequence[CachedQuery],
    *,
    watch_queries: Sequence[str] = (),
    focus_query: str | None = None,
) -> str:
    """convex combination 비교(2026-09-29_01 사전 등록)의 서술 항목. `render_markdown` 뒤에 붙인다."""
    k = report.k
    by_name = {spec.name: spec for spec in report.specs}
    pick = report.selection.pick
    convex_specs = [spec for spec in report.specs if spec.family == CONVEX]
    keep_specs = [spec for spec in report.specs if spec.family == CONVEX_KEEP]
    if not convex_specs:
        return ""

    def best_of(members: Sequence[SweepSpec]) -> SweepSpec:
        return min(members, key=lambda spec: (-report.full_scores[spec.name], spec.n_params, spec.name))

    best_convex = best_of(convex_specs)
    lines = ["", "## Convex combination 요약", ""]
    rows = []
    entries: list[tuple[str, str, str | None]] = [
        (CURRENT, "가중 규칙 C", None),
        (PLAIN_RRF_K60, "표준 RRF k=60", None),
        (VECTOR_ONLY, "vector 단독", None),
        (pick, "CC pick", CONVEX),
    ]
    if best_convex.name != pick:
        entries.append((best_convex.name, "전체 데이터 최고 CC", CONVEX))
    if keep_specs:
        entries.append((best_of(keep_specs).name, "전체 데이터 최고 CCK(보조)", CONVEX_KEEP))
    for name, label, family in entries:
        rows.append(
            [
                f"{label} `{name}`",
                str(by_name[name].n_params),
                f"{report.full_scores[name]:.3f}",
                format_rate(_hit1(report, name)),
                _cv_cell(report, name, family),
            ]
        )
    lines.extend(_table(["설정", "상수 수", f"MRR@{k}", "hit@1", f"CV MRR@{k} (평균 ± SE)"], rows))
    lines.extend(
        [
            "",
            "설정 하나의 CV는 고를 것이 없어 평균이 전체 데이터 값과 같고 SE만 의미가 있다. CC·CCK는 가족 CV(학습 fold에서 고른 설정을",
            "남긴 fold에서 잰 값)다.",
            "",
            "## 정규화별 하위 가족 교차검증",
            "",
        ]
    )
    groups = convex_groups(report.specs)
    group_rows = []
    for group, members in groups.items():
        cv = report.cv.get(group)
        best = best_of([by_name[name] for name in members])
        choice, count = cv.choices[0] if cv else ("", 0)
        total = sum(value for _, value in cv.choices) if cv else 0
        group_rows.append(
            [
                group,
                str(len(members)),
                f"{cv.mean:.3f} ± {cv.se:.3f}" if cv else "",
                f"`{best.name}` {report.full_scores[best.name]:.3f}",
                f"`{choice}` {count}/{total}" if cv else "",
            ]
        )
    lines.extend(
        _table(["하위 가족", "설정 수", f"CV MRR@{k} (평균 ± SE)", "전체 데이터 최고", "fold 선택 최다"], group_rows)
    )

    lines.extend(["", f"## α 곡선 (전체 데이터 MRR@{k}, α는 lexical 가중치)", ""])
    alpha_rows = []
    for alpha in CONVEX_ALPHAS:
        cells = []
        for group in groups:
            name = f"{group}_a{alpha:.2f}"
            cells.append(f"{report.full_scores[name]:.4f}" if name in report.full_scores else "")
        alpha_rows.append([f"{alpha:.2f}", *cells])
    lines.extend(_table(["α", *groups], alpha_rows))

    lines.extend(["", "## 퇴화 정규화 (상한 − 하한 ≤ 0이라 채널 후보가 모두 1.0인 질의 수)", ""])
    degenerate_rows = []
    for group in groups:
        config = by_name[groups[group][0]].config
        counts = degenerate_normalizations(queries, config)
        degenerate_rows.append([group, str(counts["lexical"]), str(counts["vector"]), str(counts["empty_lexical"])])
    lines.extend(_table(["하위 가족", "lexical 퇴화", "vector 퇴화", "lexical 빈 질의"], degenerate_rows))

    for other, label in ((CURRENT, "C"), (VECTOR_ONLY, "vector 단독")):
        differences = query_differences(report, pick, other)
        wins = sum(1 for item in differences if item.a_better)
        lines.extend(
            [
                "",
                f"## 질의별 차이: pick `{pick}` 대 {label} `{other}`",
                "",
                f"정답 논문 순위가 다른 질의 {len(differences)}/{len(report.results[pick])}개 (pick 우세 {wins}, "
                f"{label} 우세 {len(differences) - wins}).",
                "",
            ]
        )
        if differences:
            lines.extend(
                _table(
                    ["질의", "source", "lang", "pick", label, "우세"],
                    [
                        [
                            item.query_id,
                            item.source,
                            item.lang,
                            _rank_cell(item.a_rank),
                            _rank_cell(item.b_rank),
                            "pick" if item.a_better else label,
                        ]
                        for item in differences
                    ],
                )
            )

    index_of = {row.query_id: index for index, row in enumerate(report.results[CURRENT])}
    watched = [query_id for query_id in watch_queries if query_id in index_of]
    if watched:
        columns = [CURRENT, PLAIN_RRF_K60, VECTOR_ONLY, LEXICAL_ONLY, pick]
        lines.extend(["", "## 기준선에서 hybrid와 vector가 갈린 질의의 정답 논문 순위", ""])
        lines.extend(
            _table(
                ["질의", *(f"`{name}`" for name in columns)],
                [
                    [query_id, *(_rank_cell(paper_rank(report.results[name][index_of[query_id]])) for name in columns)]
                    for query_id in watched
                ],
            )
        )

    if focus_query and focus_query in index_of:
        index = index_of[focus_query]
        lines.extend(["", f"## {focus_query}: CC 설정별 정답 논문 순위", ""])
        focus_rows = []
        for group, members in groups.items():
            ranks = [paper_rank(report.results[name][index]) for name in members]
            first = [by_name[name].config.convex_alpha for name, rank in zip(members, ranks, strict=True) if rank == 1]
            focus_rows.append(
                [
                    group,
                    f"{len(first)}/{len(members)}",
                    ", ".join(f"{alpha:.2f}" for alpha in first) or "없음",
                    ", ".join(
                        f"{rank_label}: {count}"
                        for rank_label, count in sorted(Counter(_rank_cell(rank) for rank in ranks).items())
                    ),
                ]
            )
        lines.extend(_table(["하위 가족", "1위인 설정", "1위인 α", "순위 분포"], focus_rows))
    return "\n".join(lines) + "\n"
