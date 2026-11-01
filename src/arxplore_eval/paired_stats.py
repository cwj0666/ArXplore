from collections.abc import Callable, Sequence
from dataclasses import dataclass

from arxplore_eval.fusion_sweep import (
    DEFAULT_CONFIDENCE,
    bootstrap_draws,
    cluster_key,
    clusters_of,
    paired_bootstrap,
)

DEFAULT_SEED = 20261005
DEFAULT_RESAMPLES = 10_000

type DrawsFn = Callable[[int], Sequence[Sequence[int]]]


def draws_cache(resamples: int, seed: int) -> DrawsFn:
    cache: dict[int, list[list[int]]] = {}

    def draws(n_clusters: int) -> list[list[int]]:
        if n_clusters not in cache:
            cache[n_clusters] = bootstrap_draws(n_clusters, resamples, seed)
        return cache[n_clusters]

    return draws


@dataclass(frozen=True)
class PairedDelta:
    n: int
    n_clusters: int
    mean_a: float
    mean_b: float
    delta: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def paired_delta(
    values: Sequence[tuple[str, float, float]],
    *,
    draws: DrawsFn,
    confidence: float = DEFAULT_CONFIDENCE,
) -> PairedDelta | None:
    if not values:
        return None
    a_values = [left for _, left, _ in values]
    b_values = [right for _, _, right in values]
    clusters = clusters_of([cluster_key(query_id) for query_id, _, _ in values])
    delta, low, high = paired_bootstrap(
        b_values, a_values, draws(len(clusters)), clusters=clusters, confidence=confidence
    )
    return PairedDelta(
        n=len(values),
        n_clusters=len(clusters),
        mean_a=sum(a_values) / len(a_values),
        mean_b=sum(b_values) / len(b_values),
        delta=delta,
        low=low,
        high=high,
    )


def signed(value: float) -> str:
    return f"{value:+.3f}"


def interval_text(delta: float, low: float, high: float, excludes_zero: bool) -> str:
    marker = " *" if excludes_zero else ""
    return f"{signed(delta)} [{signed(low)}, {signed(high)}]{marker}"
