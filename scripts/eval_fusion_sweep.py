"""hybrid 융합 설정 재생·비교. 캐시(scripts/eval_dump_candidates.py)만 읽으므로 DB·API 키가 필요 없다.

    python scripts/eval_fusion_sweep.py gate  --cache eval/cache/candidates_<ts>.jsonl.gz
    python scripts/eval_fusion_sweep.py sweep --cache eval/cache/candidates_<ts>.jsonl.gz
    python scripts/eval_fusion_sweep.py convex --cache eval/cache/candidates_<ts>.jsonl.gz

gate: 기록 시점의 제품 융합 설정(캐시 머리말 `fusion_default`)으로 재생한 순위가 캐시에 기록한 제품 hybrid 순위와
모든 질의에서 같아야 한다.
다르면 질의별 차이를 출력하고 종료 코드 1. sweep도 먼저 gate를 돌리고, 실패하면 아무것도 쓰지 않고 종료 코드 1.
sweep: 설정 가족별 재생 → 반복 층화 CV → 1-SE 선택 → paired bootstrap → eval/results/fusion_<timestamp>.{md,csv}.
절차와 판정 규칙은 eval/fusion_sweep.py와 docs/worklog/phase-4/2026-09-28_01_*.md(사전 등록).
convex: sweep 설정에 convex combination 가족(CC·CCK)을 더하고 pick을 CC 안에서 고른다. 판정에 규칙 (4)(pick − 표준 RRF)를
덧붙이고 정규화별 CV·α 곡선·질의별 차이를 리포트에 붙인다. 사전 등록은 docs/worklog/phase-4/2026-09-29_01_*.md.
--cache를 주지 않으면 eval/cache/에서 가장 최근 파일을 쓴다.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_retrieval import git_revision  # noqa: E402

from eval.candidate_cache import CacheError, cache_digest, gate, read_cache, recorded_fusion_config  # noqa: E402
from eval.fusion_sweep import (  # noqa: E402
    CONVEX,
    DEFAULT_FOLDS,
    DEFAULT_REPEATS,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    build_convex_specs,
    build_specs,
    convex_groups,
    render_convex_sections,
    render_markdown,
    run_sweep,
    write_csv,
)

CACHE_DIR = REPO_ROOT / "eval" / "cache"
EXIT_GATE = 1
EXIT_INPUT = 2
# 기준선(eval/results/20260928-034856.csv)에서 hybrid와 vector의 정답 순위가 갈린 질의. convex 리포트가 순위를 따로 싣는다.
BASELINE_SPLIT_QUERIES = (
    "ki-2609.06251-en",
    "ki-2609.20804-en",
    "lang-en",
    "ki-2609.27334-en",
    "lang-abbr-dpo",
    "mp-first-proposed-en",
    "qf-long-paragraph-en",
)
FOCUS_QUERY = "lang-abbr-dpo"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("gate", "sweep", "convex"))
    parser.add_argument("--cache", default=None, help="gzip JSONL 캐시. 기본: eval/cache/의 가장 최근 파일")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "eval" / "results"))
    args = parser.parse_args(argv)
    if args.folds < 2 or args.repeats < 1 or args.resamples < 1:
        parser.error("--folds must be >= 2, --repeats and --resamples >= 1")
    return args


def resolve_cache(value: str | None) -> Path | None:
    if value:
        return Path(value)
    candidates = sorted(CACHE_DIR.glob("*.jsonl.gz"))
    return candidates[-1] if candidates else None


def run_gate(queries, *, k: int, config=None) -> bool:
    mismatches = gate(queries, k=k, config=config)
    for item in mismatches:
        print(f"불일치 {item.query_id}", file=sys.stderr)
        print(f"  live    : {item.live}", file=sys.stderr)
        print(f"  replayed: {item.replayed}", file=sys.stderr)
    if mismatches:
        print(
            f"게이트 실패: 기록 시점 설정 재생이 live hybrid와 다른 질의 {len(mismatches)}/{len(queries)}개", file=sys.stderr
        )
        return False
    print(f"게이트 통과: 기록 시점 설정 재생이 live hybrid와 {len(queries)}개 질의 모두 같습니다.", file=sys.stderr)
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_path = resolve_cache(args.cache)
    if cache_path is None or not cache_path.exists():
        print(
            f"캐시가 없습니다: {cache_path or CACHE_DIR}. scripts/eval_dump_candidates.py로 먼저 만드세요.",
            file=sys.stderr,
        )
        return EXIT_INPUT
    try:
        header, queries = read_cache(cache_path)
    except (CacheError, OSError, ValueError) as exc:
        print(f"캐시를 읽을 수 없습니다: {exc}", file=sys.stderr)
        return EXIT_INPUT
    if not queries:
        print("캐시에 질의가 없습니다.", file=sys.stderr)
        return EXIT_INPUT
    k = int(header["k"])
    print(f"캐시: {cache_path} sha256:{cache_digest(cache_path)} — 질의 {len(queries)}개, k={k}", file=sys.stderr)

    try:
        recorded = recorded_fusion_config(header)
    except (CacheError, TypeError, ValueError) as exc:
        print(f"캐시 머리말의 융합 설정을 읽을 수 없습니다: {exc}", file=sys.stderr)
        return EXIT_INPUT
    print(f"게이트 설정: 기록 시점 제품 융합 weighting={recorded.weighting}", file=sys.stderr)
    if not run_gate(queries, k=k, config=recorded):
        return EXIT_GATE
    if args.command == "gate":
        return 0

    started_at = datetime.now()

    def progress(index: int, total: int, spec) -> None:
        if index == 1 or index % 100 == 0 or index == total:
            print(f"[{index}/{total}] {spec.name}", file=sys.stderr)

    options: dict = {}
    if args.command == "convex":
        specs = build_specs() + build_convex_specs()
        options = {
            "specs": specs,
            "selectable_families": (CONVEX,),
            "cv_groups": convex_groups(specs),
            "rrf_check": True,
        }
    report = run_sweep(
        queries,
        k=k,
        seed=args.seed,
        n_folds=args.folds,
        repeats=args.repeats,
        n_resamples=args.resamples,
        progress=progress,
        **options,
    )
    corpus = header.get("corpus") or {}
    query_info = header.get("queries") or {}
    meta = {
        "실행 시각": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "커밋": git_revision(),
        "캐시": f"`{cache_path.name}` sha256:{cache_digest(cache_path)} — 기록 {header.get('created_at')}, "
        f"기록 커밋 {header.get('git_revision')}",
        "질의셋": f"`{query_info.get('path')}` sha256:{query_info.get('digest')} — {len(queries)}개 "
        f"(lang {query_info.get('langs')}, source {query_info.get('sources')})",
        "설정": f"k={k}, 채널 후보 {header.get('branch_limit')}개(저장소 요청 {header.get('branch_fetch_limit')}개), "
        f"설정 {len(report.specs)}개",
        "코퍼스": f"papers {corpus.get('papers')} (청크 보유 {corpus.get('papers_with_chunks')}), "
        f"chunks {corpus.get('chunks')}, embeddings {corpus.get('embeddings')}, 본문 source {corpus.get('fulltext_sources')}",
        "임베딩 모델": str(header.get("embedding_model")),
        "게이트": f"기록 시점 설정(weighting={recorded.weighting}) 재생 = live hybrid, {len(queries)}/{len(queries)}",
    }
    if args.command == "convex":
        meta["사전 등록"] = "docs/worklog/phase-4/2026-09-29_01 (pick은 CC 가족 안에서 고른다)"
    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    title = f"{'Convex combination fusion' if args.command == 'convex' else 'Hybrid fusion'} sweep {stamp}"
    markdown = render_markdown(report, title=title, meta=meta)
    if args.command == "convex":
        markdown += render_convex_sections(
            report, queries, watch_queries=BASELINE_SPLIT_QUERIES, focus_query=FOCUS_QUERY
        )
    (out_dir / f"fusion_{stamp}.md").write_text(markdown, encoding="utf-8")
    write_csv(report, out_dir / f"fusion_{stamp}.csv")
    print(markdown)
    print(f"결과: {out_dir / f'fusion_{stamp}'}.{{md,csv}}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
