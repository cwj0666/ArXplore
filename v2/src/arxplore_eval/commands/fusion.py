from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from arxplore_eval.candidate_cache import (
    CachedQuery,
    CacheError,
    Mismatch,
    cache_digest,
    gate,
    product_check,
    read_cache,
    recorded_fusion_config,
)
from arxplore_eval.commands.common import (
    EXIT_FAILURES,
    TIME_FORMAT,
    default_cache_dir,
    default_results_dir,
    echo_err,
    fail,
    now,
    repo_root,
    stamp_of,
)
from arxplore_eval.fusion_sweep import (
    BASELINE_SPLIT_QUERIES,
    CONVEX,
    DEFAULT_FOLDS,
    DEFAULT_REPEATS,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    FOCUS_QUERY,
    SELECTABLE_FAMILIES,
    SweepSpec,
    build_convex_specs,
    build_specs,
    convex_groups,
    render_convex_sections,
    render_markdown,
    run_sweep,
    write_csv,
)
from arxplore_eval.provenance import git_revision
from arxplore_eval.research_fusion import HybridFusionConfig

CACHE_GLOB = "*.jsonl.gz"
SWEEP = "sweep"
CONVEX_COMMAND = "convex"
PROGRESS_EVERY = 100


@dataclass(frozen=True)
class GateOutcome:
    path: Path
    header: dict[str, Any]
    queries: list[CachedQuery]
    k: int
    recorded: HybridFusionConfig


def resolve_cache(value: Path | None) -> Path | None:
    if value is not None:
        return value
    candidates = sorted(default_cache_dir().glob(CACHE_GLOB))
    return candidates[-1] if candidates else None


def _echo_mismatches(mismatches: list[Mismatch], *, live_label: str, replay_label: str) -> None:
    for item in mismatches:
        echo_err(f"불일치 {item.query_id}")
        echo_err(f"  {live_label}: {item.live}")
        echo_err(f"  {replay_label}: {item.replayed}")


def run_gate(cache: Path | None) -> GateOutcome:
    path = resolve_cache(cache)
    if path is None or not path.exists():
        fail(f"캐시가 없습니다: {path or default_cache_dir()}. arxplore-eval dump-candidates로 먼저 만드세요.")
    try:
        header, queries = read_cache(path)
    except (CacheError, OSError, ValueError) as exc:
        fail(f"캐시를 읽을 수 없습니다: {exc}")
    if not queries:
        fail("캐시에 질의가 없습니다.")
    k = int(header["k"])
    echo_err(f"캐시: {path} sha256:{cache_digest(path)} — 질의 {len(queries)}개, k={k}")
    try:
        recorded = recorded_fusion_config(header)
    except (CacheError, TypeError, ValueError) as exc:
        fail(f"캐시 머리말의 융합 설정을 읽을 수 없습니다: {exc}")
    echo_err(f"게이트 설정: 기록 시점 제품 융합 weighting={recorded.weighting}")

    mismatches = gate(queries, k=k, config=recorded)
    _echo_mismatches(mismatches, live_label="live    ", replay_label="replayed")
    if mismatches:
        fail(
            f"게이트 실패: 기록 시점 설정 재생이 live hybrid와 다른 질의 {len(mismatches)}/{len(queries)}개",
            EXIT_FAILURES,
        )
    echo_err(f"게이트 통과: 기록 시점 설정 재생이 live hybrid와 {len(queries)}개 질의 모두 같습니다.")

    try:
        product = product_check(queries, k=k)
    except CacheError as exc:
        fail(f"제품 융합으로 재생할 수 없는 캐시입니다: {exc}")
    _echo_mismatches(product, live_label="product ", replay_label="research")
    if product:
        fail(
            f"제품 융합 대조 실패: research_fusion 제품 설정과 제품 융합의 재생 순위가 다른 질의 "
            f"{len(product)}/{len(queries)}개",
            EXIT_FAILURES,
        )
    echo_err(f"제품 융합 대조 통과: research_fusion 제품 설정 재생이 제품 융합과 {len(queries)}개 질의 모두 같습니다.")
    return GateOutcome(path=path, header=header, queries=queries, k=k, recorded=recorded)


def _progress(index: int, total: int, spec: SweepSpec) -> None:
    if index == 1 or index % PROGRESS_EVERY == 0 or index == total:
        echo_err(f"[{index}/{total}] {spec.name}")


def _meta(outcome: GateOutcome, *, started_at: str, n_specs: int, convex: bool) -> dict[str, str]:
    header = outcome.header
    corpus = header.get("corpus") or {}
    query_info = header.get("queries") or {}
    n = len(outcome.queries)
    meta = {
        "실행 시각": started_at,
        "커밋": git_revision(repo_root()),
        "캐시": f"`{outcome.path.name}` sha256:{cache_digest(outcome.path)} — 기록 {header.get('created_at')}, "
        f"기록 커밋 {header.get('git_revision')}",
        "질의셋": f"`{query_info.get('path')}` sha256:{query_info.get('digest')} — {n}개 "
        f"(lang {query_info.get('langs')}, source {query_info.get('sources')})",
        "설정": f"k={outcome.k}, 채널 후보 {header.get('branch_limit')}개(저장소 요청 {header.get('branch_fetch_limit')}개), "
        f"설정 {n_specs}개",
        "코퍼스": f"papers {corpus.get('papers')} (청크 보유 {corpus.get('papers_with_chunks')}), "
        f"chunks {corpus.get('chunks')}, embeddings {corpus.get('embeddings')}, "
        f"본문 source {corpus.get('fulltext_sources')}",
        "임베딩 모델": str(header.get("embedding_model")),
        "게이트": f"기록 시점 설정(weighting={outcome.recorded.weighting}) 재생 = live hybrid, {n}/{n}",
        "제품 융합 대조": f"research_fusion 제품 설정 재생 = arxplore.retrieval 제품 융합 재생, {n}/{n}",
    }
    if convex:
        meta["사전 등록"] = "docs/worklog/phase-4/2026-09-29_01 (pick은 CC 가족 안에서 고른다)"
    return meta


def run_sweep_command(
    command: str,
    *,
    cache: Path | None,
    seed: int,
    folds: int,
    repeats: int,
    resamples: int,
    out_dir: Path | None,
    watch: list[str],
    focus: str | None,
) -> None:
    outcome = run_gate(cache)
    started_at = now()
    convex = command == CONVEX_COMMAND
    specs = build_specs() + (build_convex_specs() if convex else [])
    report = run_sweep(
        outcome.queries,
        k=outcome.k,
        specs=specs,
        seed=seed,
        n_folds=folds,
        repeats=repeats,
        n_resamples=resamples,
        progress=_progress,
        selectable_families=(CONVEX,) if convex else SELECTABLE_FAMILIES,
        cv_groups=convex_groups(specs) if convex else None,
        rrf_check=convex,
    )
    meta = _meta(outcome, started_at=started_at.strftime(TIME_FORMAT), n_specs=len(report.specs), convex=convex)
    stamp = stamp_of(started_at)
    directory = out_dir or default_results_dir()
    directory.mkdir(parents=True, exist_ok=True)
    title = f"{'Convex combination fusion' if convex else 'Hybrid fusion'} sweep {stamp}"
    markdown = render_markdown(report, title=title, meta=meta)
    if convex:
        markdown += render_convex_sections(
            report, outcome.queries, watch_queries=watch or list(BASELINE_SPLIT_QUERIES), focus_query=focus
        )
    (directory / f"fusion_{stamp}.md").write_text(markdown, encoding="utf-8")
    write_csv(report, directory / f"fusion_{stamp}.csv")
    typer.echo(markdown)
    echo_err(f"결과: {directory / f'fusion_{stamp}'}.{{md,csv}}")


CacheOption = Annotated[Path | None, typer.Option(help="gzip JSONL 캐시. 기본: eval/cache/의 가장 최근 파일")]
SeedOption = Annotated[int, typer.Option()]
FoldsOption = Annotated[int, typer.Option(min=2)]
RepeatsOption = Annotated[int, typer.Option(min=1)]
ResamplesOption = Annotated[int, typer.Option(min=1)]
OutDirOption = Annotated[Path | None, typer.Option()]


def register(app: typer.Typer) -> None:
    fusion = typer.Typer(no_args_is_help=True, help="캐시한 융합 입력으로 재생·게이트·설정 비교를 한다(DB·키 불필요).")
    app.add_typer(fusion, name="fusion")

    @fusion.command("gate", help="기록 시점 설정 재생 = 기록한 live hybrid, research 제품 설정 = 제품 융합을 확인한다.")
    def gate_command(cache: CacheOption = None) -> None:
        run_gate(cache)

    @fusion.command(SWEEP, help="설정 가족별 재생 → 반복 층화 CV → 1-SE 선택 → paired bootstrap.")
    def sweep_command(
        cache: CacheOption = None,
        seed: SeedOption = DEFAULT_SEED,
        folds: FoldsOption = DEFAULT_FOLDS,
        repeats: RepeatsOption = DEFAULT_REPEATS,
        resamples: ResamplesOption = DEFAULT_RESAMPLES,
        out_dir: OutDirOption = None,
    ) -> None:
        run_sweep_command(
            SWEEP,
            cache=cache,
            seed=seed,
            folds=folds,
            repeats=repeats,
            resamples=resamples,
            out_dir=out_dir,
            watch=[],
            focus=None,
        )

    @fusion.command(CONVEX_COMMAND, help="sweep에 convex combination 가족(CC·CCK)을 더하고 pick을 CC 안에서 고른다.")
    def convex_command(
        cache: CacheOption = None,
        seed: SeedOption = DEFAULT_SEED,
        folds: FoldsOption = DEFAULT_FOLDS,
        repeats: RepeatsOption = DEFAULT_REPEATS,
        resamples: ResamplesOption = DEFAULT_RESAMPLES,
        out_dir: OutDirOption = None,
        watch: Annotated[list[str] | None, typer.Option(help="순위를 따로 싣는 질의 id(반복 가능)")] = None,
        focus: Annotated[str | None, typer.Option(help="CC 설정별 순위를 싣는 질의 id")] = FOCUS_QUERY,
    ) -> None:
        run_sweep_command(
            CONVEX_COMMAND,
            cache=cache,
            seed=seed,
            folds=folds,
            repeats=repeats,
            resamples=resamples,
            out_dir=out_dir,
            watch=list(watch or []),
            focus=focus,
        )
