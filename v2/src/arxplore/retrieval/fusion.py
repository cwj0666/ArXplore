from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from arxplore.domain.retrieval import ChannelFusion, FusionScores, RetrievalChannel, RetrievalHit

LEXICAL_WEIGHT = 0.35


def normalize_channel_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    span = max(scores) - low
    if span <= 0:
        return [1.0 for _ in scores]
    return [(score - low) / span for score in scores]


def fusion_lexical_hits(lexical: Sequence[RetrievalHit], vector: Sequence[RetrievalHit]) -> list[RetrievalHit]:
    if not vector:
        return list(lexical)
    return [hit for hit in lexical if hit.lexical is None or hit.lexical.strict_match]


@dataclass(slots=True)
class _FusedEntry:
    first: RetrievalHit
    snippet: str
    score: float = 0.0
    channels: dict[RetrievalChannel, ChannelFusion] = field(default_factory=dict)

    def fused_hit(self) -> RetrievalHit:
        fusion = FusionScores(lexical=self.channels.get("lexical"), vector=self.channels.get("vector"))
        return replace(self.first, method="hybrid", score=self.score, snippet=self.snippet, fusion=fusion)


def fuse_hybrid_hits(
    lexical: Sequence[RetrievalHit],
    vector: Sequence[RetrievalHit],
    *,
    lexical_weight: float = LEXICAL_WEIGHT,
) -> list[RetrievalHit]:
    weights: dict[RetrievalChannel, float] = {"lexical": lexical_weight, "vector": 1.0 - lexical_weight}
    channels: tuple[tuple[RetrievalChannel, Sequence[RetrievalHit]], ...] = (
        ("lexical", fusion_lexical_hits(lexical, vector)),
        ("vector", vector),
    )
    entries: dict[int, _FusedEntry] = {}

    for channel, hits in channels:
        normalized = normalize_channel_scores([hit.score for hit in hits])
        for index, hit in enumerate(hits):
            entry = entries.get(hit.chunk_id)
            if entry is None:
                entry = entries[hit.chunk_id] = _FusedEntry(first=hit, snippet=hit.snippet)
            entry.channels[channel] = ChannelFusion(
                rank=index + 1,
                score=hit.score,
                normalized_score=normalized[index],
                weight=weights[channel],
            )
            entry.score = entry.score + weights[channel] * normalized[index]
            if channel == "lexical" or "lexical" not in entry.channels:
                entry.snippet = hit.snippet or entry.snippet

    fused = [entry.fused_hit() for entry in entries.values()]
    fused.sort(
        key=lambda hit: (hit.score, len(hit.fusion.matched_methods) if hit.fusion else 0, hit.chunk_id),
        reverse=True,
    )
    return fused
