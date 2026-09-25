from __future__ import annotations

import pytest

from eval.metrics import hit_at_k, mean, mrr_at_k, percentile, recall_at_k


class TestHitAtK:
    def test_hit_within_cutoff(self):
        assert hit_at_k(["a", "b", "c"], {"c"}, 3) == 1.0

    def test_miss_outside_cutoff(self):
        assert hit_at_k(["a", "b", "c"], {"c"}, 2) == 0.0

    def test_empty_relevant_is_undefined(self):
        assert hit_at_k(["a"], set(), 5) is None
        assert hit_at_k(["a"], [], 5) is None

    def test_empty_ranking_is_zero(self):
        assert hit_at_k([], {"a"}, 5) == 0.0

    def test_k_larger_than_ranking(self):
        assert hit_at_k(["a", "b"], {"b"}, 10) == 1.0

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            hit_at_k(["a"], {"a"}, 0)


class TestMrrAtK:
    def test_reciprocal_of_first_relevant_position(self):
        assert mrr_at_k(["x", "a", "b"], {"a", "b"}, 10) == pytest.approx(0.5)

    def test_first_position(self):
        assert mrr_at_k(["a", "x"], {"a"}, 10) == 1.0

    def test_relevant_beyond_cutoff_scores_zero(self):
        assert mrr_at_k(["x", "y", "a"], {"a"}, 2) == 0.0

    def test_duplicates_keep_positions(self):
        # The same paper filling several chunk slots still pushes later papers down.
        assert mrr_at_k(["x", "x", "a"], {"a"}, 10) == pytest.approx(1 / 3)

    def test_empty_relevant_is_undefined(self):
        assert mrr_at_k(["a"], (), 10) is None

    def test_k_larger_than_ranking(self):
        assert mrr_at_k(["x"], {"a"}, 10) == 0.0


class TestRecallAtK:
    def test_fraction_of_distinct_relevant_found(self):
        assert recall_at_k(["a", "x", "b"], {"a", "b", "c", "d"}, 10) == pytest.approx(0.5)

    def test_duplicates_do_not_inflate_recall(self):
        assert recall_at_k(["a", "a", "a"], {"a", "b"}, 10) == pytest.approx(0.5)

    def test_duplicates_consume_cutoff_slots(self):
        assert recall_at_k(["a", "a", "b"], {"a", "b"}, 2) == pytest.approx(0.5)

    def test_duplicate_relevant_ids_count_once(self):
        assert recall_at_k(["a"], ["a", "a"], 10) == 1.0

    def test_empty_relevant_is_undefined(self):
        assert recall_at_k(["a"], set(), 10) is None

    def test_k_larger_than_ranking(self):
        assert recall_at_k(["a"], {"a", "b"}, 50) == pytest.approx(0.5)

    def test_works_with_integer_chunk_ids(self):
        assert recall_at_k([11, 12, 13], {13}, 3) == 1.0


class TestPercentile:
    def test_linear_interpolation_matches_numpy_default(self):
        values = [10.0, 20.0, 30.0, 40.0]
        assert percentile(values, 50) == pytest.approx(25.0)
        assert percentile(values, 95) == pytest.approx(38.5)

    def test_bounds(self):
        values = [3.0, 1.0, 2.0]
        assert percentile(values, 0) == 1.0
        assert percentile(values, 100) == 3.0

    def test_single_value(self):
        assert percentile([7.0], 95) == 7.0

    def test_empty_is_none(self):
        assert percentile([], 50) is None

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            percentile([1.0], 101)


class TestMean:
    def test_skips_none(self):
        assert mean([1.0, None, 0.0]) == pytest.approx(0.5)

    def test_all_none_is_none(self):
        assert mean([None, None]) is None

    def test_empty_is_none(self):
        assert mean([]) is None
