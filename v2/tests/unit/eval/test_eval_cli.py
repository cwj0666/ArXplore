from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from arxplore_eval.answers import GenerationResult, append_answer_record, build_answer_record
from arxplore_eval.candidate_cache import CachedQuery, write_cache
from arxplore_eval.cli import app
from arxplore_eval.commands import fusion as fusion_commands
from arxplore_eval.dataset import EvalQuery, dump_queries
from arxplore_eval.fusion_sweep import CURRENT, PLAIN_RRF_K60, VECTOR_ONLY, build_specs
from arxplore_eval.research_fusion import LEGACY_RULES_FUSION

runner = CliRunner()


def _small_specs() -> list[Any]:
    wanted = {VECTOR_ONLY, "lexical_only", CURRENT, PLAIN_RRF_K60, "F0_rrf_k10", "F1_w0.5_k60_q0_b0", "CF_m0q1b1"}
    return [spec for spec in build_specs() if spec.name in wanted or spec.family == "REF"]


def _whole_candidates(synthetic_cached, count: int, **kwargs: Any) -> list[CachedQuery]:
    queries = []
    seed = 0
    while len(queries) < count:
        cached = synthetic_cached(seed, **kwargs)
        if all(candidate["chunk_id"] is not None for candidate in cached.lexical):
            queries.append(replace(cached, query=replace(cached.query, id=f"q{len(queries):03d}")))
        seed += 1
    return queries


def _cache(tmp_path: Path, queries: list[CachedQuery], name: str, header: dict[str, Any] | None = None) -> Path:
    path = tmp_path / name
    write_cache(path, {"k": 5, "git_revision": "test", **(header or {})}, queries)
    return path


def test_commands_are_registered() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("retrieval", "generation", "fusion", "latency", "build-queries", "dump-candidates"):
        assert command in result.output
    fusion = runner.invoke(app, ["fusion", "--help"])
    assert all(command in fusion.output for command in ("gate", "sweep", "convex"))


class TestFusionCommands:
    def test_gate_passes_fails_and_reports_missing_caches(self, tmp_path, synthetic_cached) -> None:
        queries = _whole_candidates(synthetic_cached, 15)
        good = _cache(tmp_path, queries, "good.jsonl.gz")
        tampered = [*queries[:-1], replace(queries[-1], live={"hybrid": tuple(reversed(queries[-1].live["hybrid"]))})]
        bad = _cache(tmp_path, tampered, "bad.jsonl.gz")

        passed = runner.invoke(app, ["fusion", "gate", "--cache", str(good)])
        assert passed.exit_code == 0, passed.output
        assert "제품 융합 대조 통과" in passed.stderr

        failed = runner.invoke(app, ["fusion", "gate", "--cache", str(bad)])
        assert failed.exit_code == 1
        assert "불일치 q014" in failed.stderr

        missing = runner.invoke(app, ["fusion", "gate", "--cache", str(tmp_path / "missing.jsonl.gz")])
        assert missing.exit_code == 2

    def test_gate_uses_the_recorded_config_of_a_legacy_cache(self, tmp_path, synthetic_cached) -> None:
        legacy = _whole_candidates(synthetic_cached, 15, config=LEGACY_RULES_FUSION)
        recorded = _cache(tmp_path, legacy, "recorded.jsonl.gz", {"fusion_default": asdict(LEGACY_RULES_FUSION)})
        unrecorded = _cache(tmp_path, legacy, "unrecorded.jsonl.gz")

        assert runner.invoke(app, ["fusion", "gate", "--cache", str(recorded)]).exit_code == 0
        assert runner.invoke(app, ["fusion", "gate", "--cache", str(unrecorded)]).exit_code == 1

    def test_caches_without_chunk_ids_cannot_be_replayed_by_the_product(self, tmp_path, synthetic_cached) -> None:
        with_missing_ids = next(
            cached
            for cached in (synthetic_cached(seed) for seed in range(60))
            if any(candidate["chunk_id"] is None for candidate in cached.lexical)
        )
        path = _cache(tmp_path, [with_missing_ids], "partial.jsonl.gz")
        result = runner.invoke(app, ["fusion", "gate", "--cache", str(path)])
        assert result.exit_code == 2
        assert "제품 융합으로 재생할 수 없는" in result.stderr

    def test_sweep_writes_markdown_and_csv_but_not_after_a_failed_gate(
        self, tmp_path, synthetic_cached, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fusion_commands, "build_specs", _small_specs)
        queries = _whole_candidates(synthetic_cached, 15)
        good = _cache(tmp_path, queries, "good.jsonl.gz")
        out_dir = tmp_path / "results"
        options = ["--out-dir", str(out_dir), "--folds", "3", "--repeats", "2", "--resamples", "100"]

        result = runner.invoke(app, ["fusion", "sweep", "--cache", str(good), *options])

        assert result.exit_code == 0, result.output
        markdown = next(out_dir.glob("fusion_*.md")).read_text(encoding="utf-8")
        assert "## 사전 등록 규칙 판정" in markdown and "## 1-SE 선택" in markdown
        assert "게이트: 기록 시점 설정(weighting=convex) 재생 = live hybrid, 15/15" in markdown
        rows = next(out_dir.glob("fusion_*.csv")).read_text(encoding="utf-8").splitlines()
        assert len(rows) == 1 + len(_small_specs())

        tampered = [*queries[:-1], replace(queries[-1], live={"hybrid": tuple(reversed(queries[-1].live["hybrid"]))})]
        bad = _cache(tmp_path, tampered, "bad.jsonl.gz")
        blocked = tmp_path / "blocked"
        assert runner.invoke(app, ["fusion", "sweep", "--cache", str(bad), "--out-dir", str(blocked)]).exit_code == 1
        assert not blocked.exists()

    def test_convex_command_writes_the_convex_report(
        self, tmp_path, synthetic_cached, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fusion_commands, "build_specs", _small_specs)
        good = _cache(tmp_path, _whole_candidates(synthetic_cached, 15), "good.jsonl.gz")
        out_dir = tmp_path / "results"

        result = runner.invoke(
            app,
            [
                "fusion",
                "convex",
                "--cache",
                str(good),
                "--out-dir",
                str(out_dir),
                "--repeats",
                "2",
                "--resamples",
                "100",
                "--watch",
                "q003",
                "--focus",
                "q004",
            ],
        )

        assert result.exit_code == 0, result.output
        markdown = next(out_dir.glob("fusion_*.md")).read_text(encoding="utf-8")
        assert markdown.startswith("# Convex combination fusion sweep")
        assert "## Convex combination 요약" in markdown and "사전 등록: docs/worklog/phase-4/2026-09-29_01" in markdown
        assert "규칙 (4)" in markdown and "## q004: CC 설정별 정답 논문 순위" in markdown
        rows = next(out_dir.glob("fusion_*.csv")).read_text(encoding="utf-8").splitlines()
        assert len(rows) == 1 + len(_small_specs()) + 100
        assert {"alpha", "normalization", "score_floors"} <= set(rows[0].split(","))


def _q(query_id: str, **kwargs: Any) -> EvalQuery:
    base: dict[str, Any] = {
        "query": "question",
        "lang": "en",
        "relevant_arxiv_ids": ("2401.00001",),
        "source": "manual",
    }
    base.update(kwargs)
    return EvalQuery(id=query_id, **base)


def _answers_file(path: Path) -> Path:
    records = [
        build_answer_record(
            _q("lang-1", category="language"),
            "agent",
            result=GenerationResult(
                answer="[P](https://arxiv.org/abs/2401.00001)", contexts=["ctx"], hit_arxiv_ids=["2401.00001"]
            ),
        ),
        build_answer_record(
            _q("lang-2", category="language"),
            "agent",
            result=GenerationResult(answer="[Fake](https://arxiv.org/abs/2099.00001)", contexts=["ctx"]),
        ),
        build_answer_record(
            _q("sf-1", query="", relevant_arxiv_ids=(), category="safety", expect_rejected=True),
            "agent",
            rejection="메시지를 입력하세요.",
        ),
        build_answer_record(
            _q("pc-1", category="paper_chat", relevant_chunk_ids=(11,)),
            "paper_chat",
            result=GenerationResult(answer="답 [1]", contexts=["abstract"], arxiv_id="2401.00001"),
        ),
    ]
    for record in records:
        append_answer_record(path, record)
    return path


class TestGenerationCommand:
    def test_behavior_only_scores_without_key_or_database(self, clean_env) -> None:
        answers = _answers_file(clean_env / "answers.jsonl")
        out_dir = clean_env / "results"

        result = runner.invoke(
            app,
            [
                "generation",
                "--answers",
                str(answers),
                "--behavior-only",
                "--category",
                "language,safety",
                "--out-dir",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        markdown = next(out_dir.glob("generation_*.md")).read_text(encoding="utf-8")
        assert "RAGAS 생략(--behavior-only)" in markdown
        assert "케이스 카테고리: language 2, safety 1" in markdown
        assert "| agent | category:safety | 1 | 0 | 0 | 1 | 0 |" in markdown
        assert "결과 종류: answered 2, refused 0, rejected 1, error 0" in result.stderr
        assert len(list(out_dir.iterdir())) == 3

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["generation"], "--collect와 --answers 중 하나만"),
            (["generation", "--collect", "--answers", "a.jsonl"], "--collect와 --answers 중 하나만"),
            (["generation", "--answers", "a.jsonl", "--no-score"], "--no-score"),
            (["generation", "--collect", "--behavior-only"], "OPENAI_API_KEY"),
            (["generation", "--answers", "a.jsonl"], "OPENAI_API_KEY"),
            (["generation", "--answers", "a.jsonl", "--metrics", "bleu"], "bleu"),
            (["generation", "--answers", "a.jsonl", "--mode", "chat"], "mode"),
        ],
    )
    def test_preconditions_exit_with_two(self, clean_env, argv: list[str], message: str) -> None:
        result = runner.invoke(app, argv)
        assert result.exit_code == 2
        assert message in result.stderr

    def test_missing_answer_file(self, clean_env) -> None:
        result = runner.invoke(app, ["generation", "--answers", "missing.jsonl", "--behavior-only"])
        assert result.exit_code == 2
        assert "답변 파일이 없습니다" in result.stderr


class TestRetrievalCommand:
    def test_placeholder_catalogue_is_refused_before_touching_the_database(self, clean_env, eval_dir) -> None:
        out_dir = clean_env / "r"
        result = runner.invoke(
            app, ["retrieval", "--queries", str(eval_dir / "cases"), "--methods", "lexical", "--out-dir", str(out_dir)]
        )
        assert result.exit_code == 2
        assert "검색 평가 대상이 아닌 케이스" in result.stderr and "--attach-ids" in result.stderr
        assert not out_dir.exists()

    def test_unknown_methods_and_missing_key(self, clean_env) -> None:
        queries = clean_env / "queries.jsonl"
        queries.write_text(dump_queries([_q("q1", query="sparse attention")]), encoding="utf-8")
        unknown = runner.invoke(app, ["retrieval", "--queries", str(queries), "--methods", "bm25"])
        assert unknown.exit_code == 2 and "unknown method" in unknown.stderr
        keyless = runner.invoke(app, ["retrieval", "--queries", str(queries), "--methods", "vector"])
        assert keyless.exit_code == 2 and "OPENAI_API_KEY" in keyless.stderr

    def test_unreachable_database_exits_without_writing(self, clean_env, monkeypatch: pytest.MonkeyPatch) -> None:
        for key, value in {
            "POSTGRES_HOST": "127.0.0.1:9",
            "POSTGRES_DB": "arxplore",
            "POSTGRES_USER": "arxplore",
            "POSTGRES_PASSWORD": "unused",
            "POSTGRES_POOL_TIMEOUT": "1",
            "POSTGRES_CONNECT_TIMEOUT": "1",
        }.items():
            monkeypatch.setenv(key, value)
        queries = clean_env / "queries.jsonl"
        queries.write_text(dump_queries([_q("q1", query="sparse attention")]), encoding="utf-8")
        out_dir = clean_env / "results"

        result = runner.invoke(
            app, ["retrieval", "--queries", str(queries), "--methods", "lexical", "--out-dir", str(out_dir)]
        )

        assert result.exit_code == 2
        assert "PostgreSQL에 연결할 수 없습니다" in result.stderr
        assert not out_dir.exists()


class TestOtherCommands:
    def test_build_queries_combine_rewrites_the_combined_file(self, clean_env, monkeypatch: pytest.MonkeyPatch) -> None:
        eval_home = clean_env / "eval"
        eval_home.mkdir()
        (eval_home / "queries.generated.jsonl").write_text(dump_queries([_q("g1")]), encoding="utf-8")
        (eval_home / "queries.cases.jsonl").write_text(dump_queries([_q("c1")]), encoding="utf-8")
        monkeypatch.setenv("ARXPLORE_EVAL_DIR", str(eval_home))

        result = runner.invoke(app, ["build-queries", "--combine"])

        assert result.exit_code == 0, result.output
        combined = (eval_home / "queries.jsonl").read_bytes()
        assert (
            combined
            == (eval_home / "queries.generated.jsonl").read_bytes() + (eval_home / "queries.cases.jsonl").read_bytes()
        )

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["build-queries", "--write"], "--attach-ids"),
            (["build-queries", "--overwrite", "--append"], "--append"),
            (["build-queries", "--attach-ids", "--set", "a=b"], "--set"),
            (["latency", "--repeats", "1"], "--repeats"),
        ],
    )
    def test_argument_errors(self, clean_env, eval_dir, argv: list[str], message: str) -> None:
        extra = ["--cases", str(eval_dir / "cases")] if "--attach-ids" in argv else []
        result = runner.invoke(app, [*argv, *extra])
        assert result.exit_code == 2
        assert message in result.stderr

    def test_dump_candidates_needs_a_key(self, clean_env) -> None:
        queries = clean_env / "queries.jsonl"
        queries.write_text(dump_queries([_q("q1", query="sparse attention")]), encoding="utf-8")
        result = runner.invoke(app, ["dump-candidates", "--queries", str(queries)])
        assert result.exit_code == 2
        assert "OPENAI_API_KEY" in result.stderr


def test_scoring_without_ragas_fails_before_any_work(clean_env, monkeypatch: pytest.MonkeyPatch) -> None:
    from arxplore_eval.commands import generation as generation_commands
    from arxplore_eval.generation import RagasUnavailableError

    def missing() -> None:
        raise RagasUnavailableError("ragas가 설치되어 있지 않습니다.")

    monkeypatch.setattr(generation_commands, "require_ragas", missing)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    answers = _answers_file(clean_env / "answers.jsonl")

    result = runner.invoke(app, ["generation", "--answers", str(answers), "--out-dir", str(clean_env / "r")])

    assert result.exit_code == 2
    assert "ragas" in result.stderr and "--behavior-only" in result.stderr
    assert not (clean_env / "r").exists()
