import math
from collections.abc import Collection, Hashable, Iterable, Sequence


def _top_k(ranked: Sequence[Hashable], k: int) -> Sequence[Hashable]:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    return ranked[:k]


def hit_at_k(ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float | None:
    if not relevant:
        return None
    relevant_set = set(relevant)
    return 1.0 if any(item in relevant_set for item in _top_k(ranked, k)) else 0.0


def mrr_at_k(ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float | None:
    if not relevant:
        return None
    relevant_set = set(relevant)
    for position, item in enumerate(_top_k(ranked, k), start=1):
        if item in relevant_set:
            return 1.0 / position
    return 0.0


def recall_at_k(ranked: Sequence[Hashable], relevant: Collection[Hashable], k: int) -> float | None:
    if not relevant:
        return None
    relevant_set = set(relevant)
    found = relevant_set.intersection(_top_k(ranked, k))
    return len(found) / len(relevant_set)


def percentile(values: Iterable[float], p: float) -> float | None:
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
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)
