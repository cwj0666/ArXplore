import csv
import os
from pathlib import Path

import pytest
from psycopg.conninfo import conninfo_to_dict
from typer.testing import CliRunner

from arxplore_eval.cli import app

pytestmark = pytest.mark.integration

GOLDEN_URL_VARIABLE = "GOLDEN_DATABASE_URL"
READ_ONLY_OPTIONS = "-c default_transaction_read_only=on"
REFERENCE_RESULTS = Path(__file__).resolve().parents[3] / "eval" / "results" / "20260929-005408.csv"
LEXICAL_METHODS = ("lexical", "lexical_nodiv", "lexical_nofilter")
COMPARED_FIELDS = ("retrieved_chunk_ids", "retrieved_arxiv_ids", "retrieved_roles", "paper_mrr@10", "noise_count")


def _rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return {(row["query_id"], row["method"]): row for row in csv.DictReader(handle)}


def test_lexical_methods_reproduce_the_recorded_evaluation(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    url = os.environ.get(GOLDEN_URL_VARIABLE, "").strip()
    if not url:
        pytest.skip(
            f"set {GOLDEN_URL_VARIABLE} to the read-only golden corpus to compare with {REFERENCE_RESULTS.name}"
        )
    if not REFERENCE_RESULTS.is_file():
        pytest.skip(f"reference results missing: {REFERENCE_RESULTS}")
    params = conninfo_to_dict(url)
    for key, value in {
        "POSTGRES_HOST": f"{params.get('host', 'localhost')}:{params.get('port', 5432)}",
        "APP_POSTGRES_DB": str(params["dbname"]),
        "POSTGRES_USER": str(params.get("user", "")),
        "POSTGRES_PASSWORD": str(params.get("password") or "unused-by-trust-auth"),
        "PGOPTIONS": READ_ONLY_OPTIONS,
    }.items():
        monkeypatch.setenv(key, value)
    out_dir = clean_env / "results"

    result = CliRunner().invoke(
        app,
        [
            "retrieval",
            "--queries",
            str(REFERENCE_RESULTS.parents[1] / "queries.jsonl"),
            "--methods",
            ",".join(LEXICAL_METHODS),
            "--warmup",
            "0",
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    reference = _rows(REFERENCE_RESULTS)
    produced = _rows(next(out_dir.glob("*[0-9].csv")))
    assert {method for _, method in produced} == set(LEXICAL_METHODS)
    differing = [
        key for key, row in produced.items() if any(row[field] != reference[key][field] for field in COMPARED_FIELDS)
    ]
    assert len(produced) == 3 * 111
    assert differing == []
