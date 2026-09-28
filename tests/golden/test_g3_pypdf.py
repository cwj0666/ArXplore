from typing import Any

import pytest

from tests.golden import g3_pypdf
from tests.golden.store import diff, format_problems, read_golden, sha256_bytes

ENTRIES = g3_pypdf.load_manifest()
IDS = [entry["arxiv_id"] for entry in ENTRIES]
PDF_HINT = f"set {g3_pypdf.PDF_DIR_ENV} or run python -m scripts.goldens.capture_g3_pypdf fetch"


def load_golden(entry: dict[str, Any]) -> dict[str, Any]:
    return read_golden(g3_pypdf.golden_path(entry))


def assert_matches(label: str, expected: Any, actual: Any) -> None:
    problems = diff(expected, actual)
    assert not problems, format_problems(label, problems)


def read_pdf(require_input: Any, entry: dict[str, Any]) -> bytes:
    data = require_input(g3_pypdf.pdf_path(entry), PDF_HINT).read_bytes()
    assert sha256_bytes(data) == entry["sha256"], f"{g3_pypdf.pdf_path(entry)} does not match the manifest sha256"
    return data


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_golden_file_matches_the_manifest(entry):
    golden = load_golden(entry)

    assert golden["format"] == g3_pypdf.FORMAT
    assert (golden["arxiv_id"], golden["version"], golden["pdf_sha256"]) == (
        entry["arxiv_id"],
        entry["version"],
        entry["sha256"],
    )
    assert golden["parse"]["source"] == g3_pypdf.PYPDF_SOURCE


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_parsing_the_golden_pages_matches(parser_adapter, entry):
    golden = load_golden(entry)
    parsed = parser_adapter.parse_page_texts(g3_pypdf.golden_page_texts(golden))

    assert_matches(
        f"{entry['arxiv_id']} parsed from golden pages",
        g3_pypdf.parsed_part(golden),
        g3_pypdf.parsed_record(parser_adapter, parsed),
    )


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_pypdf_page_extraction_matches(parser_adapter, require_input, entry):
    golden = load_golden(entry)
    pages = parser_adapter.extract_page_texts(read_pdf(require_input, entry))

    assert_matches(f"{entry['arxiv_id']} pypdf pages", golden["pages"], g3_pypdf.page_records(pages))


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_parsing_the_pdf_bytes_matches(parser_adapter, require_input, entry):
    golden = load_golden(entry)
    parsed = parser_adapter.parse_pdf_bytes(read_pdf(require_input, entry))

    assert_matches(
        f"{entry['arxiv_id']} parsed from PDF bytes",
        g3_pypdf.parsed_part(golden),
        g3_pypdf.parsed_record(parser_adapter, parsed),
    )
