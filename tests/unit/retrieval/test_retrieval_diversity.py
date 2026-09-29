from arxplore.domain.retrieval import RetrievalHit
from arxplore.retrieval.diversity import apply_paper_diversity


def _ids(hits: list[RetrievalHit]) -> list[int]:
    return [hit.chunk_id for hit in hits]


def test_caps_two_chunks_per_paper_before_overflow(make_hit) -> None:
    hits = [make_hit(1, arxiv_id="A"), make_hit(2, arxiv_id="A"), make_hit(3, arxiv_id="A")]
    hits += [make_hit(4, arxiv_id="B"), make_hit(5, arxiv_id="C")]

    assert _ids(apply_paper_diversity(hits, limit=4)) == [1, 2, 4, 5]


def test_refills_from_overflow_in_original_order(make_hit) -> None:
    hits = [make_hit(chunk_id, arxiv_id="A") for chunk_id in (1, 2, 3, 4)] + [make_hit(5, arxiv_id="B")]

    assert _ids(apply_paper_diversity(hits, limit=4)) == [1, 2, 5, 3]


def test_single_paper_scope_skips_diversity(make_hit) -> None:
    hits = [make_hit(chunk_id, arxiv_id="A") for chunk_id in range(1, 6)]

    assert _ids(apply_paper_diversity(hits, limit=4, arxiv_id="A")) == [1, 2, 3, 4]


def test_short_lists_are_returned_as_is(make_hit) -> None:
    hits = [make_hit(chunk_id, arxiv_id="A") for chunk_id in range(1, 4)]

    assert _ids(apply_paper_diversity(hits, limit=3)) == [1, 2, 3]


def test_missing_arxiv_id_is_not_capped(make_hit) -> None:
    hits = [make_hit(chunk_id, arxiv_id="") for chunk_id in range(1, 5)] + [make_hit(9, arxiv_id="B")]

    assert _ids(apply_paper_diversity(hits, limit=4)) == [1, 2, 3, 4]


def test_limit_below_one_is_clamped(make_hit) -> None:
    hits = [make_hit(1, arxiv_id="A"), make_hit(2, arxiv_id="B")]

    assert _ids(apply_paper_diversity(hits, limit=0)) == [1]


def test_multi_paper_pool_fills_top_k_without_refill(make_hit) -> None:
    hits = [
        *(make_hit(chunk_id, arxiv_id="A") for chunk_id in (1, 2, 3)),
        *(make_hit(chunk_id, arxiv_id="B") for chunk_id in (4, 5, 6)),
        *(make_hit(chunk_id, arxiv_id="C") for chunk_id in (7, 8, 9)),
        make_hit(10, arxiv_id="D"),
    ]

    selected = apply_paper_diversity(hits, limit=7)

    assert _ids(selected) == [1, 2, 4, 5, 7, 8, 10]
    assert len({hit.arxiv_id for hit in selected}) == 4


def test_capped_pool_refills_only_after_other_papers_run_out(make_hit) -> None:
    hits = [
        *(make_hit(chunk_id, arxiv_id="A") for chunk_id in (1, 2, 3)),
        *(make_hit(chunk_id, arxiv_id="B") for chunk_id in (4, 5, 6)),
        *(make_hit(chunk_id, arxiv_id="C") for chunk_id in (7, 8, 9)),
    ]

    assert _ids(apply_paper_diversity(hits, limit=8)) == [1, 2, 4, 5, 7, 8, 3, 6]


def test_a_custom_cap_per_paper(make_hit) -> None:
    hits = [make_hit(chunk_id, arxiv_id="A") for chunk_id in (1, 2, 3)] + [make_hit(4, arxiv_id="B")]

    assert _ids(apply_paper_diversity(hits, limit=3, max_chunks_per_paper=1)) == [1, 4, 2]
