import csv
import json
from pathlib import Path
from typing import Any

import pytest
from psycopg.conninfo import conninfo_to_dict
from typer.testing import CliRunner

from arxplore.api.services.store import PostgresPaperStore
from arxplore.config import AppSettings
from arxplore.llm.models import ChatModelFactory
from arxplore.llm.tracing import Tracer
from arxplore.retrieval.service import Retriever
from arxplore_eval.answers import (
    build_chat_service,
    generate_agent_answer,
    generate_paper_chat_answer,
    load_answer_records,
)
from arxplore_eval.candidate_cache import read_cache
from arxplore_eval.cli import app
from arxplore_eval.dataset import EvalQuery, dump_queries

pytestmark = pytest.mark.integration

DPO = "2401.00001"
SPARSE = "2401.00002"
runner = CliRunner()


def _search_call(fakes: Any, query: str = "policy loss") -> Any:
    return fakes.ToolCall(name="search_paper_chunks_tool", arguments=json.dumps({"query": query}))


def _chat_handler(fakes: Any):
    def handle(request: dict[str, Any]) -> Any:
        messages = request["messages"]
        if not request.get("tools"):
            return "논문에 따르면 손실은 로지스틱 형태입니다 [2]."
        if messages[-1]["role"] == "tool":
            return f"[Direct preference optimization revisited](https://arxiv.org/abs/{DPO})가 관련 논문입니다."
        return _search_call(fakes)

    return handle


@pytest.fixture
def cli_env(clean_env: Path, empty_database: str, chunk_ids, fake_openai, monkeypatch: pytest.MonkeyPatch) -> Path:
    params = conninfo_to_dict(empty_database)
    values = {
        "POSTGRES_HOST": f"{params.get('host', 'localhost')}:{params.get('port', 5432)}",
        "APP_POSTGRES_DB": str(params["dbname"]),
        "POSTGRES_DB": str(params["dbname"]),
        "POSTGRES_USER": str(params.get("user", "")),
        "POSTGRES_PASSWORD": str(params.get("password") or "unused-by-trust-auth"),
        "OPENAI_API_KEY": fake_openai.expected_api_key,
        "OPENAI_BASE_URL": fake_openai.base_url,
        "RETRIEVAL_MODE": "hybrid",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    return clean_env


def _queries(path: Path) -> Path:
    queries = [
        EvalQuery(id="dpo-en", query="policy loss", lang="en", relevant_arxiv_ids=(DPO,), source="manual"),
        EvalQuery(
            id="sparse-en", query="sparse attention kernels", lang="en", relevant_arxiv_ids=(SPARSE,), source="manual"
        ),
        EvalQuery(
            id="sf-empty",
            query=" ",
            lang="en",
            relevant_arxiv_ids=(),
            source="manual",
            expect_rejected=True,
            expected_behavior="clarify",
        ),
    ]
    path.write_text(dump_queries(queries), encoding="utf-8")
    return path


def test_retrieval_command_runs_every_method(cli_env: Path) -> None:
    out_dir = cli_env / "results"
    result = runner.invoke(
        app,
        ["retrieval", "--queries", str(_queries(cli_env / "q.jsonl")), "--ablations", "all", "--out-dir", str(out_dir)],
    )

    assert result.exit_code == 0, result.output
    with next(out_dir.glob("*[0-9].csv")).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 10
    assert all(not row["error"] for row in rows)
    markdown = next(out_dir.glob("*.md")).read_text(encoding="utf-8")
    assert "검색 평가 제외: 1개" in markdown and "구현: arxplore_eval (v2 공개 인터페이스)" in markdown


def test_dump_candidates_then_gate(cli_env: Path) -> None:
    cache = cli_env / "cache" / "candidates.jsonl.gz"
    queries = _queries(cli_env / "q.jsonl")
    dumped = runner.invoke(app, ["dump-candidates", "--queries", str(queries), "--k", "3", "--out", str(cache)])

    assert dumped.exit_code == 0, dumped.output
    header, queries = read_cache(cache)
    assert [cached.id for cached in queries] == ["dpo-en", "sparse-en"]
    assert header["queries"]["skipped_ids"] == ["sf-empty"]
    assert header["fusion_default"]["weighting"] == "convex"
    assert header["corpus"]["papers"] == 4

    gated = runner.invoke(app, ["fusion", "gate", "--cache", str(cache)])
    assert gated.exit_code == 0, gated.output


def test_generation_collects_through_the_product_chat_service(cli_env: Path, fakes, fake_openai) -> None:
    fake_openai.chat_handler = _chat_handler(fakes)
    answers = cli_env / "answers.jsonl"
    out_dir = cli_env / "results"

    result = runner.invoke(
        app,
        [
            "generation",
            "--collect",
            "--behavior-only",
            "--queries",
            str(_queries(cli_env / "q.jsonl")),
            "--out",
            str(answers),
            "--out-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    records = {(record["id"], record["mode"]): record for record in load_answer_records(answers)}
    assert set(records) == {
        ("dpo-en", "agent"),
        ("dpo-en", "paper_chat"),
        ("sparse-en", "agent"),
        ("sparse-en", "paper_chat"),
        ("sf-empty", "agent"),
        ("sf-empty", "paper_chat"),
    }
    agent = records[("dpo-en", "agent")]
    assert agent["retrieval_mode"] == "hybrid"
    assert agent["contexts"] and agent["hit_arxiv_ids"][0] == DPO
    assert agent["citations"] and all(citation["in_answer"] for citation in agent["citations"])
    chat = records[("dpo-en", "paper_chat")]
    assert chat["contexts"][0] == "We revisit direct preference optimization and its policy loss for alignment."
    assert chat["retrieval_mode"] == "hybrid" and chat["arxiv_id"] == DPO
    assert records[("sf-empty", "agent")]["outcome"] == "rejected"
    markdown = next(out_dir.glob("generation_*.md")).read_text(encoding="utf-8")
    assert "| agent | all | 3 | 2 | 0 | 1 | 0 |" in markdown

    resumed = runner.invoke(
        app,
        ["generation", "--collect", "--no-score", "--queries", str(cli_env / "q.jsonl"), "--out", str(answers)],
    )
    assert resumed.exit_code == 0, resumed.output
    assert "완료 6건 건너뜀" in resumed.stderr


async def test_generators_use_the_real_retriever_and_store(eval_pool, query_vectors, fakes, fake_openai) -> None:
    fake_openai.chat_handler = _chat_handler(fakes)
    models = ChatModelFactory(
        server_api_key=fake_openai.expected_api_key, default_model="gpt-4o", base_url=fake_openai.base_url
    )
    service = build_chat_service(
        AppSettings.model_validate({}),
        store=PostgresPaperStore(eval_pool),
        retriever=Retriever(eval_pool, query_vectors),
        models=models,
        tracer=Tracer.disabled(),
    )
    query = EvalQuery(id="dpo-en", query="policy loss", lang="en", relevant_arxiv_ids=(DPO,), source="manual")

    agent = await generate_agent_answer(service, query)
    chat = await generate_paper_chat_answer(service, query)

    assert agent.retrieval_mode == "hybrid"
    assert len(agent.contexts) == len(set(agent.contexts)) > 0
    assert DPO in agent.hit_arxiv_ids
    assert chat.contexts[0].startswith("We revisit direct preference optimization")
    assert chat.hit_arxiv_ids == [DPO]
    assert [citation["in_answer"] for citation in chat.citations] == [True]
