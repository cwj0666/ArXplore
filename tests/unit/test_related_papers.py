from __future__ import annotations

from unittest.mock import patch

import pytest

from papers import services
from src.core.paper_chains import _extract_key_findings

SOURCE = {
    "arxiv_id": "2401.00001",
    "title": "Direct Preference Optimization for Language Models",
    "abstract": "",
    "categories": ["cs.CL", "cs.LG"],
    "primary_category": "cs.CL",
}


def _paper(arxiv_id: str, **overrides) -> dict:
    paper = {"arxiv_id": arxiv_id, "title": "", "abstract": "", "categories": [], "primary_category": None}
    paper.update(overrides)
    return paper


class TestKeywordTokens:
    def test_order_dedupe_stopwords_and_min_length(self):
        tokens = services._keyword_tokens_in_order("Using the Vision-Language Model for AI: a Model of vision-language")

        assert tokens == ["vision-language", "for"]

    def test_leading_digits_are_skipped_not_the_whole_word(self):
        # No word boundary in the pattern: "7b-scale" yields "b-scale".
        assert services._keyword_tokens_in_order("3D 7B-scale gpt-4o x2") == ["b-scale", "gpt-4o"]


class TestScoreRelatedPaper:
    def test_weighted_components(self):
        candidate = _paper(
            "2401.00002",
            title="Preference Optimization with Human Feedback",
            categories=["cs.LG", "cs.AI"],
            primary_category="cs.LG",
        )

        # 1 shared category (2.0) + title overlap {preference, optimization} (2 * 0.8).
        assert services._score_related_paper(SOURCE, candidate) == pytest.approx(2.0 + 1.6)

    def test_same_primary_category_bonus(self):
        candidate = _paper("2401.00002", categories=["cs.CL"], primary_category="cs.CL")

        assert services._score_related_paper(SOURCE, candidate) == pytest.approx(2.0 + 1.5)

    def test_missing_primary_category_on_both_sides_gives_no_bonus(self):
        assert services._score_related_paper(_paper("a"), _paper("b")) == 0.0

    def test_abstract_overlap_is_capped_at_twelve_tokens(self):
        shared = " ".join(f"token{i}" for i in range(20))

        score = services._score_related_paper(_paper("a", abstract=shared), _paper("b", abstract=shared))

        assert score == pytest.approx(12 * 0.12)

    def test_stopwords_do_not_count(self):
        score = services._score_related_paper(
            _paper("a", title="Using Models from the Paper"), _paper("b", title="Using Models from the Paper")
        )

        assert score == 0.0


class TestBuildRelatedPapers:
    class _Repo:
        def __init__(self, papers: list[dict]) -> None:
            self.papers = papers

        def list_recent_papers(self, *, limit: int) -> list[dict]:
            return list(self.papers)

    def _build(self, candidates: list[dict], *, external: list[dict] | None = None, limit: int = 5):
        with (
            patch.object(services, "get_paper_repository", return_value=self._Repo(candidates)),
            patch.object(services, "_search_external_related_papers", return_value=external or []) as external_mock,
        ):
            related = services._build_related_papers(SOURCE, limit=limit)
        return related, external_mock

    def test_excludes_self_duplicates_and_zero_scores_and_sorts(self):
        candidates = [
            _paper("2401.00001", categories=["cs.CL"]),
            _paper("x-low", categories=["cs.LG"], upvotes=1),
            _paper("x-high", categories=["cs.CL", "cs.LG"]),
            _paper("x-tie", categories=["cs.LG"], upvotes=9),
            _paper("x-low", categories=["cs.CL", "cs.LG"]),
            _paper("x-zero", categories=["math.CO"]),
        ]

        related, _ = self._build(candidates, limit=2)

        assert [paper["arxiv_id"] for paper in related] == ["x-high", "x-tie"]
        assert related[0]["relation_score"] == pytest.approx(4.0)
        assert all(paper["source"] == "local" for paper in related)

    def test_external_search_only_fills_the_gap(self):
        candidates = [_paper("x-1", categories=["cs.CL"])]
        external = [{"arxiv_id": "ext-1", "source": "arxiv", "relation_score": 0.0}]

        related, external_mock = self._build(candidates, external=external, limit=3)

        assert [paper["arxiv_id"] for paper in related] == ["x-1", "ext-1"]
        assert external_mock.call_args.kwargs["limit"] == 2
        assert external_mock.call_args.kwargs["seen_ids"] == {"2401.00001", "x-1"}

    def test_external_search_skipped_when_local_is_enough(self):
        candidates = [_paper(f"x-{i}", categories=["cs.CL"]) for i in range(3)]

        related, external_mock = self._build(candidates, limit=3)

        assert len(related) == 3
        external_mock.assert_not_called()


class TestExtractKeyFindings:
    def test_strips_bullets_and_numbering_and_adds_period(self):
        text = "- First finding about scaling\n* Second finding here!\n• Third finding bullet\n1. Fourth numbered item\n2) Fifth numbered item?"

        assert _extract_key_findings(text) == [
            "First finding about scaling.",
            "Second finding here!",
            "Third finding bullet.",
            "Fourth numbered item.",
            "Fifth numbered item?",
        ]

    def test_drops_short_lines_and_dedupes_ignoring_punctuation_and_case(self):
        text = "short\n\nThe model improves accuracy.\nthe model improves accuracy\nTHE MODEL, IMPROVES ACCURACY!"

        assert _extract_key_findings(text) == ["The model improves accuracy."]

    def test_max_items(self):
        text = "\n".join(f"Finding number {i} is here" for i in range(10))

        assert len(_extract_key_findings(text)) == 6
        assert len(_extract_key_findings(text, max_items=2)) == 2

    def test_collapses_internal_whitespace(self):
        assert _extract_key_findings("-   spaced    out   finding  ") == ["spaced out finding."]

    def test_leading_decimal_number_is_mistaken_for_list_numbering(self):
        # Known quirk: "3.5%" at line start is read as the list marker "3." and stripped.
        assert _extract_key_findings("3.5% gain on GSM8K over the baseline") == ["5% gain on GSM8K over the baseline."]
