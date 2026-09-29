import itertools
import random
import re

import pytest

from arxplore.parsing.roles import is_references_section_title, references_section_title_sql_regex
from arxplore.retrieval.lexical import build_lexical_candidates_query

SQL_REGEX = references_section_title_sql_regex()
SQL_RULE = re.compile(SQL_REGEX)

REFERENCE_TITLES = [
    "References",
    "REFERENCES",
    "Reference",
    "7 References",
    "5. References",
    "3) References",
    "4.2 Bibliography",
    "IV References",
    "IV. REFERENCES",
    "iv. References",
    "iii) references",
    "C References",
    "Works Cited",
    "works  cited",
    "Literature Cited",
    "References and Notes",
    "  References:  ",
    ". References",
    "1 . References",
    "7\nReferences",
]
OTHER_TITLES = [
    "",
    "3.2 Reference Model",
    "Preference Optimization",
    "Related Work",
    "References 1",
    "1.References",
    "1: References",
    "A. References",
    "Civil References",
    "iv references",
    "cl Bibliography",
    "Reference s",
    "Cited Works",
]


def _synthetic_titles(count: int) -> list[str]:
    parts = itertools.product(
        ["", "1", "1.2", "IV", "iv", "Iv", "C", "Civil", "x"],
        ["", ".", ")", ":"],
        ["", " ", "\t", "  "],
        ["References", "reference", "BIBLIOGRAPHY", "Works Cited", "Literature  cited", "Notes", "Refs"],
        ["", ".", " :", " and notes", " 2", "s"],
    )
    titles = {f"{prefix}{punctuation}{space}{word}{tail}" for prefix, punctuation, space, word, tail in parts}
    rng = random.Random(20260929)
    alphabet = "RrEeFfNnCcSsIiVvXxLl1290 .:)\t\nBbWwOoKkTtDdAa"
    while len(titles) < count:
        titles.add("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 14))))
    return sorted(titles)


@pytest.mark.parametrize("title", REFERENCE_TITLES)
def test_reference_titles(title: str) -> None:
    assert is_references_section_title(title) is True
    assert SQL_RULE.search(title) is not None


@pytest.mark.parametrize("title", OTHER_TITLES)
def test_other_titles(title: str) -> None:
    assert is_references_section_title(title) is False
    assert SQL_RULE.search(title) is None


def test_sql_rule_agrees_with_the_chunk_role_rule() -> None:
    disagreements = [
        title
        for title in _synthetic_titles(40_000)
        if (SQL_RULE.search(title) is not None) != is_references_section_title(title)
    ]

    assert disagreements == []


def test_the_rule_can_be_inlined_in_a_sql_string_literal() -> None:
    assert "'" not in SQL_REGEX
    assert "%" not in SQL_REGEX


@pytest.mark.parametrize("arxiv_id", [None, "2401.00001"])
def test_lexical_sql_penalizes_titles_with_the_chunk_role_rule(arxiv_id: str | None) -> None:
    built = build_lexical_candidates_query("attention", limit=5, arxiv_id=arxiv_id)

    assert built is not None
    assert f"WHEN section_title ~ '{SQL_REGEX}' THEN -0.08" in built[0]
