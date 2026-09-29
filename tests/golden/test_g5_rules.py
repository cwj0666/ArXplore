import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from tests.golden import g5_http
from tests.golden.store import diff, read_golden

ORIGINAL_GOLDEN = read_golden(g5_http.GOLDEN_PATH) if g5_http.GOLDEN_PATH.is_file() else {"bundles": {}}
RATE_LIMITED_INDEX = 30


def responses(bundle: str, scenario: str) -> list[dict[str, Any]]:
    return copy.deepcopy(ORIGINAL_GOLDEN["bundles"][bundle][scenario]["responses"])


def _set_header(name: str, values: list[str]) -> Callable[[dict[str, Any]], None]:
    def mutate(record: dict[str, Any]) -> None:
        record["headers"][name] = values

    return mutate


def _drop_header(name: str) -> Callable[[dict[str, Any]], None]:
    def mutate(record: dict[str, Any]) -> None:
        record["headers"].pop(name)

    return mutate


def _set_cookie_attribute(name: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(record: dict[str, Any]) -> None:
        record["headers"]["set-cookie"][0]["attributes"][name] = value

    return mutate


def _set_status(record: dict[str, Any]) -> None:
    record["status"] = 201


def _change_json_value(record: dict[str, Any]) -> None:
    record["json"]["items"][0]["title"] = "changed title"


def _change_sse_byte(record: dict[str, Any]) -> None:
    record["body"] = record["body"].replace('"chunk"', '"Chunk"', 1)


def _fill_empty_body(record: dict[str, Any]) -> None:
    record["body"] = "Method Not Allowed"


def _json_body_as_text(record: dict[str, Any]) -> None:
    record.pop("json")


MUTATIONS = [
    ("status code", "pages", "bootstrap", 0, _set_status),
    ("content-type media type", "pages", "list_default", 0, _set_header("content-type", ["text/plain"])),
    ("header x-frame-options", "pages", "list_default", 0, _set_header("x-frame-options", ["SAMEORIGIN"])),
    ("header cache-control", "paper_chat", "chat_numbered_citations", 1, _set_header("cache-control", ["no-store"])),
    ("header x-accel-buffering", "paper_chat", "chat_numbered_citations", 1, _drop_header("x-accel-buffering")),
    ("header location", "pages", "papers_redirect", 1, _set_header("location", ["/?q=rag"])),
    ("header retry-after", "rate_limit", "llm_bucket_is_shared", RATE_LIMITED_INDEX, _drop_header("retry-after")),
    ("allow methods", "pages", "bootstrap_rejects_post", 0, _set_header("allow", ["GET, POST"])),
    ("set-cookie attributes without value", "pages", "bootstrap", 0, _set_cookie_attribute("path", "/api")),
    ("set-cookie attributes without value", "pages", "bootstrap", 0, _set_cookie_attribute("samesite", "Strict")),
    ("set-cookie attributes without value", "pages", "bootstrap", 0, _set_cookie_attribute("httponly", True)),
    ("set-cookie attributes without value", "pages", "bootstrap", 0, _drop_header("set-cookie")),
    ("JSON body value", "pages", "list_default", 0, _change_json_value),
    ("JSON body value", "pages", "list_default", 0, _json_body_as_text),
    ("SSE body bytes", "paper_chat", "chat_numbered_citations", 1, _change_sse_byte),
    ("other body bytes", "pages", "bootstrap_rejects_post", 0, _fill_empty_body),
]


def test_v2_rules_accept_every_original_response():
    for bundle, scenarios in ORIGINAL_GOLDEN["bundles"].items():
        for name, scenario in scenarios.items():
            recorded = scenario["responses"]
            assert g5_http.compare_responses_v2(recorded, copy.deepcopy(recorded)) == [], f"{bundle}:{name}"


def test_v2_rules_ignore_encoding_charset_cookie_case_and_other_headers():
    expected = [
        *responses("pages", "bootstrap"),
        *responses("pages", "list_default"),
        *responses("paper_chat", "chat_numbered_citations"),
        *responses("rate_limit", "llm_bucket_is_shared")[RATE_LIMITED_INDEX:],
    ]
    actual = copy.deepcopy(expected)
    for record in actual:
        headers = record["headers"]
        headers.pop("vary", None)
        headers["server"] = ["uvicorn"]
        headers["date"] = ["Tue, 29 Sep 2026 00:00:00 GMT"]
        headers["content-length"] = ["0"]
        headers["content-type"] = [value.split(";")[0] + "; charset=utf-8" for value in headers["content-type"]]
        if "json" in record:
            record["body"] = json.dumps(record["json"], ensure_ascii=False, indent=2)
        for cookie in headers.get("set-cookie", []):
            cookie["attributes"] = {
                key.upper(): value.lower() if isinstance(value, str) else value
                for key, value in cookie["attributes"].items()
                if key != "expires"
            }

    assert g5_http.compare_responses_v2(expected, actual) == []
    assert diff(expected, actual) != []


@pytest.mark.parametrize(
    ("rule", "bundle", "scenario", "index", "mutate"),
    MUTATIONS,
    ids=[f"{rule}:{mutate.__name__}" for rule, _, _, _, mutate in MUTATIONS],
)
def test_v2_rules_report_the_rule_that_differs(rule, bundle, scenario, index, mutate):
    expected = responses(bundle, scenario)
    actual = copy.deepcopy(expected)
    mutate(actual[index])

    problems = g5_http.compare_responses_v2(expected, actual)

    assert problems
    assert all(problem.startswith(f"response {index} ") for problem in problems)
    assert any(rule in problem for problem in problems), problems


def test_v2_rules_report_a_missing_response():
    expected = responses("pages", "list_page_invalid")

    assert g5_http.compare_responses_v2(expected, expected[:-1]) == ["response count 3 -> 2"]
