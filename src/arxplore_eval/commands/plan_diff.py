from pathlib import Path
from typing import Annotated

import typer

from arxplore_eval.answers import AnswersError, load_agent_plans
from arxplore_eval.commands.common import (
    TIME_FORMAT,
    default_queries_path,
    default_results_dir,
    echo_err,
    fail,
    now,
    repo_root,
    stamp_of,
)
from arxplore_eval.dataset import DatasetError, load_queries
from arxplore_eval.fusion_variants import FUSION_DESCRIPTIONS
from arxplore_eval.plan_diff import (
    MD_FILE,
    SEARCHES_CSV_FILE,
    STRATA_CSV_FILE,
    diff_plans,
    render_markdown,
    write_searches_csv,
    write_strata_csv,
)
from arxplore_eval.provenance import file_digest, git_revision


def register(app: typer.Typer) -> None:
    @app.command(
        "plan-diff",
        help="--plan-only로 기록한 에이전트 계획의 융합 전 후보로 legacy_rules와 convex의 top-k 차이를 낸다(LLM·DB 호출 없음).",
    )
    def plan_diff_command(
        plan: Annotated[Path, typer.Option(help="generation --collect --mode agent --plan-only로 만든 JSONL")],
        queries: Annotated[Path | None, typer.Option(help="계획을 만들 때 쓴 질의셋 JSONL")] = None,
        out_dir: Annotated[Path | None, typer.Option(help="기본 eval/results/plan_diff_<stamp>")] = None,
    ) -> None:
        queries_path = queries or default_queries_path()
        try:
            loaded = load_queries(queries_path)
            plans, questions = load_agent_plans(plan)
        except FileNotFoundError as exc:
            fail(f"파일을 읽을 수 없습니다: {exc}")
        except (DatasetError, AnswersError) as exc:
            fail(f"입력 형식 오류: {exc}")
        by_id = {query.id: query for query in loaded}
        unknown = sorted(set(plans) - set(by_id))
        if unknown:
            fail(f"질의셋에 없는 계획 id: {unknown[:10]}")
        changed = sorted(query_id for query_id, text in questions.items() if by_id[query_id].query != text)
        if changed:
            fail(f"질의셋과 질문이 다른 계획 id: {changed[:10]}")
        diffs = diff_plans(plans, loaded)
        started_at = now()
        directory = out_dir or default_results_dir() / f"plan_diff_{stamp_of(started_at)}"
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "실행 시각": started_at.strftime(TIME_FORMAT),
            "커밋": git_revision(repo_root()),
            "계획": f"`{plan}` sha256:{file_digest(plan)} ({len(plans)}건)",
            "질의셋": f"`{queries_path}` sha256:{file_digest(queries_path)}",
            **{f"융합 {name}": description for name, description in FUSION_DESCRIPTIONS.items()},
        }
        markdown = render_markdown(diffs, title="에이전트 계획 검색 차이: legacy_rules vs convex", meta=meta)
        (directory / MD_FILE).write_text(markdown, encoding="utf-8")
        write_searches_csv(diffs, directory / SEARCHES_CSV_FILE)
        write_strata_csv(diffs, directory / STRATA_CSV_FILE)
        typer.echo(markdown)
        echo_err(f"결과: {directory}/{{{MD_FILE}, {SEARCHES_CSV_FILE}, {STRATA_CSV_FILE}}}")
