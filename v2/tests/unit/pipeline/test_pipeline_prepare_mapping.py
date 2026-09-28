from datetime import UTC, date, datetime
from typing import Any

import pytest

from arxplore.pipeline.prepare.mapping import (
    build_pdf_url,
    extract_candidate_id,
    extract_hf_authors,
    extract_hf_signals,
    paper_fields_from_hf_item,
    select_candidates,
)

DAY = date(2026, 9, 15)


def _item(arxiv_id: str, **paper: Any) -> dict[str, Any]:
    return {"paper": {"id": arxiv_id, **paper}}


@pytest.mark.parametrize(
    "item,expected",
    [
        ({"paper": {"id": " 2609.00001 "}}, "2609.00001"),
        ({"paper": {"arxivId": "2609.00002"}}, "2609.00002"),
        ({"paper": {"arxiv_id": "2609.00003"}}, "2609.00003"),
        ({"paper": {"id": "", "arxivId": "2609.00004"}}, "2609.00004"),
        ({"paper": {"id": 12345}, "id": "2609.00005"}, "2609.00005"),
        ({"paper": "not a dict", "arxivId": "2609.00006"}, "2609.00006"),
        ({"paper": {"id": "   "}}, None),
        ({"title": "no id"}, None),
    ],
)
def test_candidate_id_prefers_the_paper_object_then_the_item(item: dict[str, Any], expected: str | None) -> None:
    assert extract_candidate_id(item) == expected


def test_signals_prefer_item_values_then_paper_values() -> None:
    signals = extract_hf_signals({"upvotes": 9, "paper": {"upvotes": 4, "githubRepo": "g", "githubStars": "12"}})
    assert (signals.upvotes, signals.github_url, signals.github_stars) == (9, "g", 12)

    signals = extract_hf_signals({"paper": {"upvotes": None}})
    assert (signals.upvotes, signals.github_url, signals.github_stars) == (0, None, None)


@pytest.mark.parametrize(
    "github,expected_url,expected_stars",
    [
        ({"url": "u", "repoUrl": "r", "stars": 3}, "u", 3),
        ({"repoUrl": "r", "repository": "p"}, "r", 5),
        ({"repository": "p", "stars": None}, "p", 5),
        ({"url": "", "stars": 0}, "fallback", 0),
        ("not a dict", "fallback", 5),
    ],
)
def test_github_object_overrides_repo_fields(github: Any, expected_url: str, expected_stars: int) -> None:
    item = {"github": github, "paper": {"githubRepo": "fallback", "githubStars": 5}}
    signals = extract_hf_signals(item)
    assert (signals.github_url, signals.github_stars) == (expected_url, expected_stars)


def test_paper_github_object_is_used_when_the_item_has_none() -> None:
    signals = extract_hf_signals({"paper": {"github": {"url": "from-paper", "stars": 2}, "githubRepo": "repo"}})
    assert (signals.github_url, signals.github_stars) == ("from-paper", 2)


def test_non_numeric_upvotes_raise() -> None:
    with pytest.raises(ValueError):
        extract_hf_signals({"upvotes": "many"})


def test_authors_keep_named_dict_entries_in_order() -> None:
    item = {"paper": {"authors": [{"name": " Ada "}, {"name": ""}, "raw", {"_id": "x"}, {"name": "Ben"}]}}
    assert extract_hf_authors(item) == ["Ada", "Ben"]
    assert extract_hf_authors({"paper": {"authors": "Ada"}}) == []
    assert extract_hf_authors({}) == []


def test_pdf_url_follows_the_legacy_rule() -> None:
    assert build_pdf_url(" 2609.00001 ") == "https://arxiv.org/pdf/2609.00001.pdf"
    assert build_pdf_url("  ") == ""


def test_paper_fields_match_the_legacy_hf_raw_shape() -> None:
    item = _item(
        "2609.00001v2",
        title="  A title  ",
        summary=" An abstract. ",
        authors=[{"name": "Ada"}],
        publishedAt="2026-09-14T20:00:00.000Z",
        upvotes=11,
        githubRepo="https://github.com/a/b",
        githubStars=3,
    )
    assert paper_fields_from_hf_item("2609.00001", item) == {
        "arxiv_id": "2609.00001",
        "title": "A title",
        "authors": ["Ada"],
        "abstract": "An abstract.",
        "primary_category": None,
        "categories": [],
        "pdf_url": "https://arxiv.org/pdf/2609.00001.pdf",
        "published_at": "2026-09-14T20:00:00.000Z",
        "updated_at": "2026-09-14T20:00:00.000Z",
        "upvotes": 11,
        "github_url": "https://github.com/a/b",
        "github_stars": 3,
        "source": "hf_daily_papers_raw",
    }


def test_upsert_parses_the_hf_timestamps() -> None:
    selection = select_candidates(DAY, [_item("2609.00001", publishedAt="2026-09-14T20:00:00.000Z")])
    upsert = selection.candidates[0].to_upsert()
    assert upsert.published_at == datetime(2026, 9, 14, 20, 0, tzinfo=UTC)
    assert upsert.updated_at == upsert.published_at
    assert upsert.source == "hf_daily_papers_raw"


def test_candidates_are_normalized_deduplicated_in_order_and_last_item_wins() -> None:
    payload = [
        _item("2609.00003v1", title="first"),
        "not a dict",
        {"title": "no id"},
        _item("2609.00001"),
        _item("2609.00003v2", title="second"),
        _item("https://arxiv.org/abs/2609.00002"),
    ]
    selection = select_candidates(DAY, payload)

    assert selection.raw_count == 6
    assert selection.deduplicated_ids == ("2609.00003", "2609.00001", "2609.00002")
    assert selection.selected_ids == selection.deduplicated_ids
    assert [candidate.arxiv_id for candidate in selection.candidates] == list(selection.deduplicated_ids)
    assert selection.candidates[0].title == "second"
    assert selection.candidates[0].pdf_url == "https://arxiv.org/pdf/2609.00003.pdf"


@pytest.mark.parametrize("max_papers,expected", [(None, 3), (0, 3), (-1, 3), (2, 2), (5, 3)])
def test_max_papers_cuts_the_deduplicated_ids(max_papers: int | None, expected: int) -> None:
    payload = [_item(f"2609.0000{number}") for number in range(1, 4)]
    selection = select_candidates(DAY, payload, max_papers=max_papers)
    assert len(selection.selected_ids) == expected
    assert len(selection.candidates) == expected
    assert len(selection.deduplicated_ids) == 3


def test_empty_payload_selects_nothing() -> None:
    selection = select_candidates(DAY, [])
    assert (selection.raw_count, selection.deduplicated_ids, selection.candidates) == (0, (), ())
