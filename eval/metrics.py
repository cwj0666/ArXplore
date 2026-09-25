"""순위 기반 검색 지표.

모든 함수는 "검색 결과의 앞 k개 위치"를 기준으로 계산한다. 같은 논문의 청크가 여러 번 나오면
논문 단위 ranking에 중복이 생기는데, 중복은 위치를 차지하지만 recall을 부풀리지 않는다.
relevant 집합이 비어 있으면 지표를 정의할 수 없으므로 None을 반환하고 집계에서 제외한다.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Hashable, Iterable, Sequence


def _top_k(ranked: Sequence[Hashable], k: int) -> Sequence[Hashable]:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return ranked[:k]


def hit_at_k(ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float | None:
    """앞 k개 안에 relevant 항목이 하나라도 있으면 1.0."""
    if not relevant:
        return None
    relevant_set = set(relevant)
    return 1.0 if any(item in relevant_set for item in _top_k(ranked, k)) else 0.0


def mrr_at_k(ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float | None:
    """앞 k개 안에서 첫 relevant 항목 위치(1부터)의 역수. 없으면 0.0."""
    if not relevant:
        return None
    relevant_set = set(relevant)
    for position, item in enumerate(_top_k(ranked, k), start=1):
        if item in relevant_set:
            return 1.0 / position
    return 0.0


def recall_at_k(ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float | None:
    """앞 k개가 덮는 서로 다른 relevant 항목의 비율."""
    if not relevant:
        return None
    relevant_set = set(relevant)
    found = relevant_set.intersection(_top_k(ranked, k))
    return len(found) / len(relevant_set)


def percentile(values: Iterable[float], p: float) -> float | None:
    """선형 보간 백분위수(numpy 기본 방식과 동일). 값이 없으면 None."""
    if not 0 <= p <= 100:
        raise ValueError(f"p must be within [0, 100], got {p}")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * (p / 100)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean(values: Iterable[float | None]) -> float | None:
    """None을 제외한 산술 평균. 남는 값이 없으면 None."""
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)
