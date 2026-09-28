import json
from typing import Any
from urllib.parse import urlsplit

import pytest

from tests.golden import environment, g3_layout, g3_pypdf
from tests.golden.store import diff, format_problems, read_golden, resolve_golden, sha256_bytes

ENTRIES = g3_pypdf.load_manifest()
IDS = [entry["arxiv_id"] for entry in ENTRIES]
PDF_HINT = f"set {g3_pypdf.PDF_DIR_ENV} or run python -m scripts.goldens.capture_g3_pypdf fetch"


def load_golden(entry: dict[str, Any]) -> dict[str, Any]:
    return read_golden(resolve_golden(g3_layout.golden_path(entry)))


@pytest.fixture(scope="module")
def layout_service() -> str:
    url = g3_layout.layout_url()
    parts = urlsplit(url)
    if not environment.database_reachable(f"tcp://{parts.hostname}:{parts.port or 80}"):
        pytest.skip(f"layout parser unreachable at {url} (set {g3_layout.LAYOUT_URL_ENV})")
    return url


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_layout_golden_matches_the_manifest(entry):
    golden = load_golden(entry)

    assert golden["format"] == g3_layout.FORMAT
    assert (golden["arxiv_id"], golden["version"], golden["pdf_sha256"]) == (
        entry["arxiv_id"],
        entry["version"],
        entry["sha256"],
    )
    assert golden["request"] == g3_layout.REQUEST_OPTIONS
    assert golden["parse"]["source"] == "layout_pdf"


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_replaying_the_layout_response_matches(parser_adapter, entry):
    golden = load_golden(entry)
    actual = g3_layout.replay_record(parser_adapter, golden["response"])

    assert actual is not None
    problems = diff({key: golden[key] for key in g3_layout.PARSED_KEYS}, actual)
    assert not problems, format_problems(f"{entry['arxiv_id']} layout replay", problems)


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_live_layout_response_matches(layout_service, require_input, entry):
    golden = load_golden(entry)
    pdf_bytes = require_input(g3_pypdf.pdf_path(entry), PDF_HINT).read_bytes()
    assert sha256_bytes(pdf_bytes) == entry["sha256"]

    response = json.loads(g3_layout.request_layout(pdf_bytes, layout_service))
    problems = diff(golden["response"], response)
    assert not problems, format_problems(f"{entry['arxiv_id']} layout response", problems)
