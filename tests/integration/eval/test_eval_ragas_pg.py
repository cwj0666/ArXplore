import csv
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
from psycopg.conninfo import conninfo_to_dict
from typer.testing import CliRunner

from arxplore_eval.cli import app

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(importlib.util.find_spec("ragas") is None, reason="ragas is in the eval group"),
]

ANSWERS = Path(__file__).resolve().parents[3] / "eval" / "results" / "answers_20260926-103836.jsonl"
DPO = "2401.00001"
FAIL_MARKER = "RAGAS-JUDGE-FAIL"
FLAKY_MARKER = "RAGAS-JUDGE-FLAKY"
SELECTED = (
    ("cs-1488", "agent"),
    ("cs-2393", "agent"),
    ("ki-2609.26355-ko", "agent"),
    ("ki-2609.26355-ko", "paper_chat"),
    ("ki-2609.26355-en", "paper_chat"),
    ("cs-12637", "agent"),
    ("sf-empty-string", "agent"),
    ("lang-ko", "agent"),
)
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _answers(path: Path, gold_chunk_id: int) -> Path:
    if not ANSWERS.is_file():
        pytest.skip(f"committed answers missing: {ANSWERS}")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with ANSWERS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[(record["id"], record["mode"])] = record
    selected = [dict(records[key]) for key in SELECTED]
    selected[0]["relevant_chunk_ids"] = [gold_chunk_id]
    selected[1]["reference_answer"] = "The reference answer names the method and its main result."
    selected[4]["answer"] = f"{selected[4]['answer']} {FLAKY_MARKER}"
    selected[7]["answer"] = f"{selected[7]['answer']} {FAIL_MARKER}"
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in selected), encoding="utf-8")
    return path


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_generation_scores_real_answers_through_ragas(
    clean_env: Path,
    empty_database: str,
    chunk_ids: dict[tuple[str, int], int],
    fakes: Any,
    ragas_judge: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = conninfo_to_dict(empty_database)
    with fakes.FakeOpenAIServer(chat_handler=ragas_judge) as server:
        for key, value in {
            "POSTGRES_HOST": f"{params.get('host', 'localhost')}:{params.get('port', 5432)}",
            "APP_POSTGRES_DB": str(params["dbname"]),
            "POSTGRES_USER": str(params.get("user", "")),
            "POSTGRES_PASSWORD": str(params.get("password") or "unused-by-trust-auth"),
            "OPENAI_API_KEY": server.expected_api_key,
            "OPENAI_BASE_URL": server.base_url,
        }.items():
            monkeypatch.setenv(key, value)
        answers = _answers(clean_env / "answers.jsonl", chunk_ids[(DPO, 1)])
        out_dir = clean_env / "results"

        result = CliRunner().invoke(
            app,
            ["generation", "--answers", str(answers), "--out-dir", str(out_dir), "--concurrency", "3"],
        )
        chat_requests = [request.body for request in server.requests_for("/v1/chat/completions")]
        embedding_requests = [request.body for request in server.requests_for("/v1/embeddings")]

    assert result.exit_code == 1, result.output
    assert "판정 실패 2건" in result.stderr
    assert ragas_judge.unknown == []
    assert ragas_judge.titles["_TranslatedStrings"] == 5
    assert ragas_judge.failed >= 4 and ragas_judge.flaked >= 1
    assert {request["model"] for request in chat_requests} == {"gpt-5-mini"}
    assert all(request["max_completion_tokens"] == 8192 for request in chat_requests)
    assert all(request["response_format"] == {"type": "json_object"} for request in chat_requests)
    assert embedding_requests and {request["model"] for request in embedding_requests} == {"text-embedding-3-small"}

    samples = {(row["id"], row["mode"]): row for row in _rows(next(out_dir.glob("generation_*[0-9].csv")))}
    assert set(samples) == set(SELECTED)
    gold = samples[("cs-1488", "agent")]
    assert gold["reference_kind"] == "gold_chunks"
    assert all(gold[metric] for metric in ("faithfulness", "answer_relevancy", "context_precision"))
    assert gold["context_recall_skipped"] == "no_reference_answer"
    referenced = samples[("cs-2393", "agent")]
    assert referenced["reference_kind"] == "answer"
    assert all(referenced[metric] for metric in METRICS)
    assert samples[("ki-2609.26355-ko", "agent")]["context_precision_skipped"] == "no_reference"
    assert samples[("sf-empty-string", "agent")]["faithfulness_skipped"] == "rejected"
    recovered = samples[("ki-2609.26355-en", "paper_chat")]
    assert recovered["faithfulness"] and recovered["answer_relevancy"]
    assert not recovered["faithfulness_error"] and not recovered["answer_relevancy_error"]
    failing = samples[("lang-ko", "agent")]
    assert failing["faithfulness_error"].startswith("InstructorRetryException")
    assert failing["answer_relevancy_error"].startswith("InstructorRetryException")
    assert not failing["faithfulness"]

    summary = {(row["mode"], row["subset"]): row for row in _rows(next(out_dir.glob("generation_*_summary.csv")))}
    assert summary[("agent", "all")]["n_metric_errors"] == "2"
    assert summary[("agent", "all")]["context_recall_n"] == "1"
    markdown = next(out_dir.glob("generation_*.md")).read_text(encoding="utf-8")
    assert "judge=gpt-5-mini, embeddings=text-embedding-3-small" in markdown
    assert "| agent | all | 6 |" in markdown and "| paper_chat | all | 2 |" in markdown
