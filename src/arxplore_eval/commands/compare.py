from pathlib import Path
from typing import Annotated

import typer

from arxplore_eval.commands.common import (
    TIME_FORMAT,
    default_queries_path,
    default_results_dir,
    echo_err,
    fail,
    now,
    repo_root,
    split_names,
    stamp_of,
)
from arxplore_eval.compare_runs import (
    MD_FILE,
    METRICS_CSV_FILE,
    QUERIES_CSV_FILE,
    CompareError,
    compare_runs,
    pair_runs,
    read_run_csv,
    render_markdown,
    write_metrics_csv,
    write_queries_csv,
)
from arxplore_eval.dataset import DatasetError, load_queries
from arxplore_eval.fusion_sweep import DEFAULT_CONFIDENCE
from arxplore_eval.paired_stats import DEFAULT_RESAMPLES, DEFAULT_SEED
from arxplore_eval.provenance import file_digest, git_revision


def register(app: typer.Typer) -> None:
    @app.command(
        "compare-runs",
        help="두 retrieval 실행 CSV를 질의·방식으로 짝지어 지표 차이와 군집 paired bootstrap 구간을 낸다.",
    )
    def compare_runs_command(
        a: Annotated[Path, typer.Option("--a", help="기준 실행의 질의별 CSV (retrieval 결과 <stamp>.csv)")],
        b: Annotated[Path, typer.Option("--b", help="비교 실행의 질의별 CSV")],
        queries: Annotated[Path | None, typer.Option(help="두 실행에 쓴 질의셋 JSONL")] = None,
        label_a: Annotated[str, typer.Option(help="기준 실행 이름")] = "a",
        label_b: Annotated[str, typer.Option(help="비교 실행 이름")] = "b",
        methods: Annotated[str, typer.Option(help="쉼표 구분. 비우면 두 실행에 모두 있는 방식 전부")] = "",
        resamples: Annotated[int, typer.Option(min=1)] = DEFAULT_RESAMPLES,
        seed: Annotated[int, typer.Option()] = DEFAULT_SEED,
        confidence: Annotated[float, typer.Option(min=0.5, max=0.999)] = DEFAULT_CONFIDENCE,
        out_dir: Annotated[Path | None, typer.Option(help="기본 eval/results/compare_<stamp>")] = None,
    ) -> None:
        queries_path = queries or default_queries_path()
        try:
            loaded = load_queries(queries_path)
        except (FileNotFoundError, DatasetError) as exc:
            fail(f"질의셋을 읽을 수 없습니다: {exc}")
        try:
            run_a = read_run_csv(a)
            run_b = read_run_csv(b)
            paired = pair_runs(run_a, run_b, loaded, label_a=label_a, label_b=label_b, methods=split_names(methods))
        except FileNotFoundError as exc:
            fail(f"실행 CSV를 읽을 수 없습니다: {exc}")
        except (CompareError, KeyError, ValueError) as exc:
            fail(f"두 실행을 짝지을 수 없습니다: {exc}")

        started_at = now()
        comparison = compare_runs(paired, resamples=resamples, seed=seed, confidence=confidence)
        directory = out_dir or default_results_dir() / f"compare_{stamp_of(started_at)}"
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "실행 시각": started_at.strftime(TIME_FORMAT),
            "커밋": git_revision(repo_root()),
            f"a = {label_a}": f"`{a}` sha256:{file_digest(a)}",
            f"b = {label_b}": f"`{b}` sha256:{file_digest(b)}",
            "질의셋": f"`{queries_path}` sha256:{file_digest(queries_path)}",
            "k": str(paired.k),
            "방식": ", ".join(f"{method} {len(paired.pairs[method])}쌍" for method in paired.methods),
            "bootstrap": f"{resamples}회, seed {seed}, {confidence * 100:.0f}% 구간, 한영 번역쌍 군집",
        }
        markdown = render_markdown(comparison, title=f"검색 실행 비교: {label_b} vs {label_a}", meta=meta)
        (directory / MD_FILE).write_text(markdown, encoding="utf-8")
        write_metrics_csv(comparison.deltas, directory / METRICS_CSV_FILE)
        write_queries_csv(paired, directory / QUERIES_CSV_FILE)
        typer.echo(markdown)
        for line in [*paired.warnings, *comparison.identity_warnings]:
            echo_err(line)
        echo_err(f"결과: {directory / MD_FILE}, {directory / METRICS_CSV_FILE}, {directory / QUERIES_CSV_FILE}")
