from arxplore.retrieval.snippet import SNIPPET_MAX_CHARS, build_search_snippet


def test_centers_on_the_first_matching_term() -> None:
    text = "a" * 200 + " policy " + "b" * 400

    snippet = build_search_snippet("policy", text, "", "")

    start = text.find("policy") - SNIPPET_MAX_CHARS // 3
    assert snippet == "..." + text[start : start + SNIPPET_MAX_CHARS].strip() + "..."


def test_no_ellipsis_when_the_whole_text_fits() -> None:
    assert build_search_snippet("policy", "the policy loss", "", "") == "the policy loss"


def test_falls_back_to_the_abstract_then_the_title() -> None:
    assert build_search_snippet("reward", "no match", "reward model", "T") == "reward model"
    assert build_search_snippet("scaling", "no match", "none", "Scaling Laws") == "Scaling Laws"


def test_short_terms_are_ignored_and_the_first_nonempty_text_is_compacted() -> None:
    text = "line one\n\n  line   two " + "x" * 400

    snippet = build_search_snippet("rl is ok", text, "abstract", "title")

    assert snippet == " ".join(text.split())[:SNIPPET_MAX_CHARS] + "..."


def test_all_empty_gives_an_empty_snippet() -> None:
    assert build_search_snippet("policy", "", "", "") == ""
