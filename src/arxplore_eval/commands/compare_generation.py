from pathlib import Path
from typing import Annotated

import typer

from arxplore_eval.answers import MODES, PAPER_CHAT_MODE
from arxplore_eval.behavior import DEFAULT_REFUSAL_PHRASES, load_refusal_phrases
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
from arxplore_eval.compare_generation import (
    MD_FILE,
    METRICS_CSV_FILE,
    QUERIES_CSV_FILE,
    REFUSAL_CSV_FILE,
    VARIANCE_CSV_FILE,
    CompareGenerationError,
    RunValues,
    compare_generation,
    pair_generation,
    read_run,
    render_markdown,
    write_metrics_csv,
    write_queries_csv,
    write_refusal_csv,
    write_variance_csv,
)
from arxplore_eval.dataset import DatasetError, load_queries
from arxplore_eval.fusion_sweep import DEFAULT_CONFIDENCE
from arxplore_eval.paired_stats import DEFAULT_RESAMPLES, DEFAULT_SEED
from arxplore_eval.plan_diff import read_strata_csv
from arxplore_eval.provenance import file_digest, git_revision


def _run_lines(label: str, runs: list[RunValues]) -> dict[str, str]:
    lines: dict[str, str] = {}
    for index, run in enumerate(runs, start=1):
        variants = f", prompt_variant {'/'.join(run.prompt_variants)}" if run.prompt_variants else ""
        lines[f"{label} 반복 {index}"] = (
            f"`{run.path}` sha256:{file_digest(run.path)} ({run.kind}, {len(run.values)}건{variants})"
        )
    return lines


def register(app: typer.Typer) -> None:
    @app.command(
        "compare-generation",
        help="두 조건(각각 반복 여러 개)의 생성 평가 결과를 질의별 반복 평균으로 짝지어 차이와 군집 paired bootstrap 구간을 낸다.",
    )
    def compare_generation_command(
        a: Annotated[
            list[Path],
            typer.Option("--a", help="기준 조건의 샘플 CSV(generation_<stamp>.csv) 또는 답변 JSONL. 반복마다 한 번씩"),
        ],
        b: Annotated[list[Path], typer.Option("--b", help="비교 조건의 샘플 CSV 또는 답변 JSONL. 반복마다 한 번씩")],
        queries: Annotated[Path | None, typer.Option(help="두 조건에 쓴 질의셋 JSONL")] = None,
        label_a: Annotated[str, typer.Option(help="기준 조건 이름")] = "a",
        label_b: Annotated[str, typer.Option(help="비교 조건 이름")] = "b",
        mode: Annotated[str, typer.Option(help="비교할 모드(agent 또는 paper_chat)")] = PAPER_CHAT_MODE,
        refusal_phrases: Annotated[
            Path | None, typer.Option(help="답변 JSONL을 읽을 때 쓸 거절 문구 파일(한 줄에 하나)")
        ] = None,
        resamples: Annotated[int, typer.Option(min=1)] = DEFAULT_RESAMPLES,
        seed: Annotated[int, typer.Option()] = DEFAULT_SEED,
        confidence: Annotated[float, typer.Option(min=0.5, max=0.999)] = DEFAULT_CONFIDENCE,
        out_dir: Annotated[Path | None, typer.Option(help="기본 eval/results/compare_generation_<stamp>")] = None,
        strata: Annotated[
            Path | None, typer.Option(help="질의 층 CSV(id, stratum). plan-diff의 strata.csv. 층별 차이를 더한다")
        ] = None,
    ) -> None:
        if mode not in MODES:
            fail(f"알 수 없는 mode: {mode!r}. 사용 가능: {', '.join(MODES)}")
        queries_path = queries or default_queries_path()
        try:
            loaded = load_queries(queries_path)
        except (FileNotFoundError, DatasetError) as exc:
            fail(f"질의셋을 읽을 수 없습니다: {exc}")
        try:
            phrases = load_refusal_phrases(refusal_phrases) if refusal_phrases else list(DEFAULT_REFUSAL_PHRASES)
            strata_by_id = read_strata_csv(strata) if strata else {}
        except (ValueError, OSError) as exc:
            fail(str(exc))
        try:
            runs_a = [read_run(path, mode=mode, refusal_phrases=phrases) for path in a]
            runs_b = [read_run(path, mode=mode, refusal_phrases=phrases) for path in b]
            paired = pair_generation(runs_a, runs_b, loaded, label_a=label_a, label_b=label_b, strata=strata_by_id)
        except FileNotFoundError as exc:
            fail(f"결과 파일을 읽을 수 없습니다: {exc}")
        except (CompareGenerationError, KeyError, ValueError) as exc:
            fail(f"두 조건을 짝지을 수 없습니다: {exc}")

        started_at = now()
        comparison = compare_generation(paired, resamples=resamples, seed=seed, confidence=confidence)
        directory = out_dir or default_results_dir() / f"compare_generation_{stamp_of(started_at)}"
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "실행 시각": started_at.strftime(TIME_FORMAT),
            "커밋": git_revision(repo_root()),
            "모드": mode,
            **_run_lines(f"a = {label_a}", runs_a),
            **_run_lines(f"b = {label_b}", runs_b),
            "질의셋": f"`{queries_path}` sha256:{file_digest(queries_path)}",
            "짝지은 질의": str(len(paired.pairs)),
            "지표": ", ".join(paired.metrics),
            "거절 문구": f"`{refusal_phrases.name}`" if refusal_phrases else "내장 목록",
            "bootstrap": f"{resamples}회, seed {seed}, {confidence * 100:.0f}% 구간, 한영 번역쌍 군집",
        }
        if strata:
            meta["질의 층"] = f"`{strata}` sha256:{file_digest(strata)}"
        markdown = render_markdown(comparison, title=f"생성 비교: {label_b} vs {label_a}", meta=meta)
        (directory / MD_FILE).write_text(markdown, encoding="utf-8")
        write_metrics_csv(comparison.deltas, directory / METRICS_CSV_FILE)
        write_queries_csv(paired, directory / QUERIES_CSV_FILE)
        write_variance_csv(comparison.variance, directory / VARIANCE_CSV_FILE)
        write_refusal_csv(comparison.refusal, directory / REFUSAL_CSV_FILE)
        typer.echo(markdown)
        for line in paired.warnings:
            echo_err(line)
        echo_err(
            f"결과: {directory}/{{{MD_FILE}, {METRICS_CSV_FILE}, {QUERIES_CSV_FILE}, {VARIANCE_CSV_FILE}, {REFUSAL_CSV_FILE}}}"
        )
