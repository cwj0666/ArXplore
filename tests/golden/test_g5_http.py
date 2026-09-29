from typing import Any

import pytest

from tests.golden import g5_http, g5_runner
from tests.golden.adapters import AdapterUnavailable
from tests.golden.store import diff, format_problems, read_golden, resolve_golden

GOLDEN_FILE = resolve_golden(g5_http.GOLDEN_PATH)
GOLDEN = read_golden(GOLDEN_FILE) if GOLDEN_FILE.is_file() else {"bundles": {}}
SCENARIO_IDS = [(bundle, scenario) for bundle in sorted(GOLDEN["bundles"]) for scenario in GOLDEN["bundles"][bundle]]
FAKE_REQUEST_KEYS = ("llm_requests", "llm_statuses", "embedding_requests", "arxiv_requests", "fake_server_errors")


@pytest.fixture(scope="module")
def g5_results(golden_database_url: str) -> dict[str, Any]:
    try:
        return g5_runner.run_bundles(sorted(GOLDEN["bundles"]), database_url=golden_database_url)
    except AdapterUnavailable as exc:
        pytest.skip(str(exc))


def test_golden_covers_every_bundle():
    assert GOLDEN.get("format") == g5_http.FORMAT
    assert sorted(GOLDEN["bundles"]) == sorted(g5_http.BUNDLES)
    assert {name: sorted(scenarios) for name, scenarios in GOLDEN["bundles"].items()} == {
        name: sorted(scenario.name for scenario in bundle.scenarios) for name, bundle in g5_http.BUNDLES.items()
    }


@pytest.mark.parametrize(("bundle", "scenario"), SCENARIO_IDS, ids=[f"{b}:{s}" for b, s in SCENARIO_IDS])
def test_http_responses_match_golden(g5_results, bundle, scenario):
    expected = GOLDEN["bundles"][bundle][scenario]["responses"]
    actual = g5_results[bundle]["scenarios"][scenario]["responses"]

    problems = g5_http.compare_responses_v2(expected, actual)
    assert not problems, format_problems(f"G5 {bundle}:{scenario}", problems)


@pytest.mark.parametrize(("bundle", "scenario"), SCENARIO_IDS, ids=[f"{b}:{s}" for b, s in SCENARIO_IDS])
def test_fake_server_requests_match_golden(g5_results, bundle, scenario):
    expected = {key: GOLDEN["bundles"][bundle][scenario][key] for key in FAKE_REQUEST_KEYS}
    actual = {key: g5_results[bundle]["scenarios"][scenario][key] for key in FAKE_REQUEST_KEYS}

    problems = diff(expected, actual)
    assert not problems, format_problems(f"G4 {bundle}:{scenario}", problems)
