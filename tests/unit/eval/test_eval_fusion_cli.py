from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from arxplore_eval.agent_plan import file_sha256
from arxplore_eval.answers import (
    AGENT_ORIGIN_LIVE,
    AGENT_ORIGIN_PLAN_ONLY,
    AGENT_ORIGIN_PLAN_REPLAY,
    AgentTrace,
    GenerationResult,
    append_answer_record,
    build_answer_record,
)
from arxplore_eval.cli import app
from arxplore_eval.commands import generation as generation_commands
from arxplore_eval.commands.generation import DEFAULT_AGENT_SETUP, CollectionOptions, agent_setup
from arxplore_eval.corpus import PreconditionError
from arxplore_eval.dataset import EvalQuery

runner = CliRunner()


def _q(query_id: str = "a-en") -> EvalQuery:
    return EvalQuery(
        id=query_id, query="What is policy loss?", lang="en", relevant_arxiv_ids=("2401.00001",), source="manual"
    )


def _options(tmp_path: Path, **kwargs: Any) -> CollectionOptions:
    base: dict[str, Any] = {
        "queries": tmp_path / "q.jsonl",
        "modes": ["agent"],
        "lang": None,
        "limit": None,
        "category": None,
        "out": tmp_path / "out.jsonl",
        "answer_model": "m",
        "keep_going": False,
        "base_url": None,
        "trace": False,
        "refusal_phrases": [],
    }
    base.update(kwargs)
    return CollectionOptions(**base)


def _record(
    query_id: str,
    *,
    origin: str | None = None,
    fusion: str | None = None,
    plan_sha256: str | None = None,
    error: str | None = None,
    rejection: str | None = None,
    mode: str = "agent",
) -> dict[str, Any]:
    query = _q(query_id)
    if error or rejection:
        return build_answer_record(query, mode, error=error, rejection=rejection)
    agent = None if origin is None else AgentTrace(fusion=fusion, origin=origin, plan_sha256=plan_sha256)
    return build_answer_record(query, mode, result=GenerationResult(answer="답", contexts=["c"], agent=agent))


def _file(path: Path, *records: dict[str, Any]) -> Path:
    for record in records:
        append_answer_record(path, record)
    return path


class TestFlagValidation:
    @pytest.mark.parametrize(
        "flags", [["--fusion", "convex"], ["--plan-only"], ["--replay-plan", "p.jsonl"], ["--fusion", "legacy_rules"]]
    )
    def test_agent_flags_need_collect(self, clean_env: Path, flags: list[str]) -> None:
        result = runner.invoke(app, ["generation", "--answers", "a.jsonl", *flags])
        assert result.exit_code == 2
        assert "--fusion, --plan-only, --replay-plan은 --collect와 함께만" in result.stderr

    def test_plan_only_with_replay_plan(self, clean_env: Path) -> None:
        result = runner.invoke(
            app, ["generation", "--collect", "--mode", "agent", "--plan-only", "--replay-plan", "p.jsonl"]
        )
        assert result.exit_code == 2
        assert "--plan-only와 --replay-plan은 함께 쓸 수 없습니다" in result.stderr

    def test_plan_only_with_fusion(self, clean_env: Path) -> None:
        result = runner.invoke(app, ["generation", "--collect", "--mode", "agent", "--plan-only", "--fusion", "convex"])
        assert result.exit_code == 2
        assert "--plan-only는 융합 전 후보만 기록하므로 --fusion을 쓰지 않습니다" in result.stderr

    def test_replay_plan_without_fusion(self, clean_env: Path) -> None:
        result = runner.invoke(app, ["generation", "--collect", "--mode", "agent", "--replay-plan", "p.jsonl"])
        assert result.exit_code == 2
        assert "--replay-plan에는 --fusion이 필요합니다" in result.stderr

    @pytest.mark.parametrize("mode", ["paper_chat", "both"])
    @pytest.mark.parametrize(
        "flags", [["--fusion", "convex"], ["--plan-only"], ["--fusion", "convex", "--replay-plan", "p.jsonl"]]
    )
    def test_agent_flags_need_agent_mode(self, clean_env: Path, mode: str, flags: list[str]) -> None:
        result = runner.invoke(app, ["generation", "--collect", "--mode", mode, *flags])
        assert result.exit_code == 2
        assert "--mode agent에서만" in result.stderr

    def test_default_mode_is_both_and_refuses_agent_flags(self, clean_env: Path) -> None:
        result = runner.invoke(app, ["generation", "--collect", "--plan-only"])
        assert result.exit_code == 2
        assert "--mode agent에서만" in result.stderr

    def test_unknown_fusion_name_exits_2(self, clean_env: Path) -> None:
        result = runner.invoke(app, ["generation", "--collect", "--mode", "agent", "--fusion", "rrf"])
        assert result.exit_code == 2

    @pytest.mark.parametrize("flags", [["--fusion", "convex"], ["--plan-only"]])
    def test_valid_agent_flags_pass_the_checks_and_reach_the_key_check(self, clean_env: Path, flags: list[str]) -> None:
        result = runner.invoke(app, ["generation", "--collect", "--mode", "agent", *flags])
        assert result.exit_code == 2
        assert "OPENAI_API_KEY" in result.stderr
        assert "--mode agent에서만" not in result.stderr


class TestScoringPlanOnlyRecords:
    def test_answers_with_plan_only_records_are_refused(self, clean_env: Path) -> None:
        path = clean_env / "plans.jsonl"
        base = _record("x-en")
        _file(
            path,
            {**base, "id": "x-en", "plan_only": True},
            {**base, "id": "y-en", "plan_only": False, "answer": "ok"},
            {**base, "id": "z-en", "plan_only": True},
        )
        result = runner.invoke(app, ["generation", "--answers", str(path), "--behavior-only"])
        assert result.exit_code == 2
        assert "계획 캡처(--plan-only) 레코드 2건" in result.stderr
        assert "'x-en', 'z-en'" in result.stderr and "y-en" not in result.stderr


class TestCollectionOptionsSetup:
    def test_default_setup(self, tmp_path: Path) -> None:
        options = _options(tmp_path)
        assert options.agent_setup == DEFAULT_AGENT_SETUP == (None, None, None)
        assert options.skips_scoring is False

    def test_fusion_only_is_live(self, tmp_path: Path) -> None:
        assert _options(tmp_path, fusion="convex").agent_setup == (AGENT_ORIGIN_LIVE, "convex", None)

    def test_plan_only_setup_and_scoring(self, tmp_path: Path) -> None:
        options = _options(tmp_path, plan_only=True)
        assert options.agent_setup == (AGENT_ORIGIN_PLAN_ONLY, None, None)
        assert options.skips_scoring is True

    def test_replay_setup_carries_fusion_and_file_digest(self, tmp_path: Path) -> None:
        plan = tmp_path / "plans.jsonl"
        plan.write_text("plan-bytes", encoding="utf-8")
        options = _options(tmp_path, fusion="legacy_rules", replay_plan=plan)
        assert options.plan_sha256 == file_sha256(plan)
        assert options.agent_setup == (AGENT_ORIGIN_PLAN_REPLAY, "legacy_rules", file_sha256(plan))
        assert options.skips_scoring is False

    def test_replay_digest_changes_with_the_file(self, tmp_path: Path) -> None:
        plan = tmp_path / "plans.jsonl"
        plan.write_text("one", encoding="utf-8")
        first = _options(tmp_path, fusion="convex", replay_plan=plan).agent_setup
        plan.write_text("two", encoding="utf-8")
        assert _options(tmp_path, fusion="convex", replay_plan=plan).agent_setup != first


class TestAgentSetupOfRecords:
    def test_default_record_has_no_setup(self) -> None:
        assert agent_setup(_record("a-en")) == DEFAULT_AGENT_SETUP

    def test_trace_record_setup(self) -> None:
        record = _record("a-en", origin=AGENT_ORIGIN_PLAN_REPLAY, fusion="convex", plan_sha256="abc")
        assert agent_setup(record) == (AGENT_ORIGIN_PLAN_REPLAY, "convex", "abc")

    def test_non_string_values_become_none(self) -> None:
        assert agent_setup({"agent_origin": 3, "fusion": ["x"], "plan_sha256": None}) == (None, None, None)


class TestDoneKeysResumeGuard:
    def test_missing_file_has_nothing_done(self, tmp_path: Path) -> None:
        assert generation_commands._done_keys(tmp_path / "none.jsonl", None, None, ("live", "convex", None)) == set()

    @pytest.mark.parametrize(
        "setup",
        [
            (None, None, None),
            (AGENT_ORIGIN_LIVE, "convex", None),
            (AGENT_ORIGIN_PLAN_ONLY, None, None),
            (AGENT_ORIGIN_PLAN_REPLAY, "convex", "a" * 64),
        ],
    )
    def test_same_setup_resumes(self, tmp_path: Path, setup: generation_commands.AgentSetup) -> None:
        origin, fusion, plan = setup
        path = _file(tmp_path / "o.jsonl", _record("a-en", origin=origin, fusion=fusion, plan_sha256=plan))
        assert generation_commands._done_keys(path, None, None, setup) == {("a-en", "agent")}

    @pytest.mark.parametrize(
        ("existing", "wanted"),
        [
            ((AGENT_ORIGIN_PLAN_REPLAY, "convex", "a" * 64), (AGENT_ORIGIN_PLAN_REPLAY, "legacy_rules", "a" * 64)),
            ((AGENT_ORIGIN_PLAN_REPLAY, "convex", "a" * 64), (AGENT_ORIGIN_PLAN_REPLAY, "convex", "b" * 64)),
            ((AGENT_ORIGIN_PLAN_REPLAY, "convex", "a" * 64), (AGENT_ORIGIN_LIVE, "convex", None)),
            ((AGENT_ORIGIN_LIVE, "convex", None), (None, None, None)),
            ((None, None, None), (AGENT_ORIGIN_LIVE, "convex", None)),
            ((AGENT_ORIGIN_PLAN_ONLY, None, None), (AGENT_ORIGIN_LIVE, "convex", None)),
        ],
    )
    def test_different_setup_is_refused(
        self, tmp_path: Path, existing: generation_commands.AgentSetup, wanted: generation_commands.AgentSetup
    ) -> None:
        origin, fusion, plan = existing
        path = _file(tmp_path / "o.jsonl", _record("a-en", origin=origin, fusion=fusion, plan_sha256=plan))
        with pytest.raises(PreconditionError, match=r"에이전트 레코드 설정.*다른 --out"):
            generation_commands._done_keys(path, None, None, wanted)

    def test_message_names_both_setups(self, tmp_path: Path) -> None:
        path = _file(
            tmp_path / "o.jsonl",
            _record("a-en", origin=AGENT_ORIGIN_PLAN_REPLAY, fusion="convex", plan_sha256="abcdef0123456789"),
        )
        with pytest.raises(PreconditionError) as caught:
            generation_commands._done_keys(path, None, None, (None, None, None))
        text = str(caught.value)
        assert "plan_replay, fusion convex, plan abcdef012345" in text
        assert "제품 기본" in text

    def test_error_and_rejected_records_do_not_count_as_a_setup(self, tmp_path: Path) -> None:
        path = _file(
            tmp_path / "o.jsonl",
            _record("e-en", error="Boom: x"),
            _record("r-en", rejection="too long"),
            _record("a-en", origin=AGENT_ORIGIN_LIVE, fusion="convex"),
        )
        assert generation_commands._done_keys(path, None, None, (AGENT_ORIGIN_LIVE, "convex", None)) == {
            ("a-en", "agent"),
            ("r-en", "agent"),
        }

    def test_paper_chat_records_are_ignored_by_the_agent_guard(self, tmp_path: Path) -> None:
        path = _file(tmp_path / "o.jsonl", _record("p-en", mode="paper_chat"))
        assert generation_commands._done_keys(path, None, None, (AGENT_ORIGIN_LIVE, "convex", None)) == {
            ("p-en", "paper_chat")
        }

    def test_guard_is_skipped_without_an_agent_setup(self, tmp_path: Path) -> None:
        path = _file(tmp_path / "o.jsonl", _record("a-en", origin=AGENT_ORIGIN_PLAN_ONLY))
        assert generation_commands._done_keys(path, None, None, None) == {("a-en", "agent")}

    def test_resume_skips_completed_keys_but_not_errors(self, tmp_path: Path) -> None:
        path = _file(tmp_path / "o.jsonl", _record("a-en"), _record("e-en", error="Boom: x"))
        assert generation_commands._done_keys(path, None, None, DEFAULT_AGENT_SETUP) == {("a-en", "agent")}
