from collections.abc import Sequence

from arxplore.domain.retrieval import RetrievalHit

MAX_CHUNKS_PER_PAPER = 2


def apply_paper_diversity(
    hits: Sequence[RetrievalHit],
    *,
    limit: int,
    arxiv_id: str | None = None,
    max_chunks_per_paper: int = MAX_CHUNKS_PER_PAPER,
) -> list[RetrievalHit]:
    normalized_limit = max(1, limit)
    if arxiv_id or len(hits) <= normalized_limit:
        return list(hits[:normalized_limit])

    selected: list[RetrievalHit] = []
    overflow: list[RetrievalHit] = []
    paper_counts: dict[str, int] = {}

    for hit in hits:
        count = paper_counts.get(hit.arxiv_id, 0)
        if hit.arxiv_id and count >= max_chunks_per_paper:
            overflow.append(hit)
            continue
        selected.append(hit)
        if hit.arxiv_id:
            paper_counts[hit.arxiv_id] = count + 1
        if len(selected) >= normalized_limit:
            return selected[:normalized_limit]

    for hit in overflow:
        selected.append(hit)
        if len(selected) >= normalized_limit:
            break

    return selected[:normalized_limit]
