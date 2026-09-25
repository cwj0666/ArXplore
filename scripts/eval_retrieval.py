"""오프라인 검색 평가: 질의셋으로 lexical/vector/hybrid(+ablation)를 실행하고 지표를 기록한다.

    python scripts/eval_retrieval.py                                  # 기본 3방식, k=10
    python scripts/eval_retrieval.py --methods lexical --limit 5      # 키 없이 lexical만, 5개 질의
    python scripts/eval_retrieval.py --ablations all --keep-going     # ablation 포함, 개별 실패는 기록 후 계속

    python scripts/eval_retrieval.py --queries eval/queries.cases.jsonl --category language,query_form

PostgreSQL이 필요하고 vector/hybrid 계열은 OPENAI_API_KEY(질의 임베딩)가 필요하다.
--queries는 JSONL 파일 또는 *.jsonl이 모인 디렉터리. 거절·입력 거부 기대, 상세 챗 전용, 대화 이력 케이스는 검색 평가에서 뺀다.
DB에 연결할 수 없거나, 질의셋의 정답 arxiv_id/chunk_id가 DB에 없거나, 자리표시자 id가 남아 있으면
아무것도 쓰지 않고 종료 코드 2로 끝난다.
결과: eval/results/<timestamp>.csv(질의별), <timestamp>_summary.csv(집계), <timestamp>.md(README용 표).
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.dataset import DatasetError, EvalQuery, dataset_digest, find_placeholders, load_queries  # noqa: E402
from eval.report import render_markdown, write_query_csv, write_summary_csv  # noqa: E402
from eval.runner import (  # noqa: E402
    METHODS,
    aggregate,
    resolve_methods,
    run_evaluation,
    run_query,
    split_retrieval_queries,
)

EXIT_PRECONDITION = 2


class PreconditionError(RuntimeError):
    pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", default=str(REPO_ROOT / "eval" / "queries.jsonl"))
    parser.add_argument(
        "--methods", default="lexical,vector,hybrid", help="쉼표 구분. 사용 가능: " + ", ".join(METHODS)
    )
    parser.add_argument("--ablations", default="", help="쉼표 구분: nodiv, nofilter, norerank, plainrrf 또는 all")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--adjacency-window", type=int, default=1, help="제품 경로와 같은 문맥 창(지연 측정에 포함)")
    parser.add_argument("--limit", type=int, default=None, help="파일 순서대로 앞 N개 질의만 실행")
    parser.add_argument("--lang", choices=("ko", "en"), default=None)
    parser.add_argument("--category", default=None, help="쉼표 구분 케이스 카테고리만 실행(eval/cases 참고)")
    parser.add_argument("--warmup", type=int, default=1, help="방식별로 버리는 워밍업 질의 수")
    parser.add_argument("--keep-going", action="store_true", help="질의 단위 예외를 errors 열에 기록하고 계속")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "eval" / "results"))
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


def parse_categories(value: str | None) -> set[str] | None:
    if value is None:
        return None
    names = {name.strip() for name in value.split(",") if name.strip()}
    return names or None


def select_queries(
    queries: list[EvalQuery], *, lang: str | None, limit: int | None, categories: set[str] | None = None
) -> list[EvalQuery]:
    selected = [
        query
        for query in queries
        if (lang is None or query.lang == lang) and (categories is None or query.category in categories)
    ]
    return selected[:limit] if limit is not None else selected


def placeholder_problems(queries: list[EvalQuery]) -> list[str]:
    """자리표시자 arxiv_id(`0000.0000a` 형식)나 제목 템플릿이 남은 질의 id."""
    return [
        query.id
        for query in queries
        if find_placeholders([query.query, *query.relevant_arxiv_ids, *query.must_mention_arxiv_ids])
    ]


def _connect(settings: Any):
    import psycopg2

    from src.shared import build_postgres_connection_params

    params = build_postgres_connection_params(settings)
    return psycopg2.connect(connect_timeout=5, **params)


def preflight(settings: Any, queries: list[EvalQuery], *, needs_embeddings: bool) -> dict[str, Any]:
    """DB 연결, 코퍼스 규모, 정답 id 존재 여부를 확인한다. 조건이 맞지 않으면 PreconditionError."""
    try:
        connection = _connect(settings)
    except Exception as exc:
        raise PreconditionError(f"PostgreSQL에 연결할 수 없습니다: {type(exc).__name__}: {exc}") from exc

    arxiv_ids = sorted({value for query in queries for value in query.relevant_arxiv_ids})
    chunk_ids = sorted({value for query in queries for value in query.relevant_chunk_ids})
    try:
        with connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM papers),
                    (SELECT COUNT(DISTINCT arxiv_id) FROM paper_chunks),
                    (SELECT COUNT(*) FROM paper_chunks),
                    (SELECT COUNT(*) FROM paper_embeddings)
                """
            )
            papers, papers_with_chunks, chunks, embeddings = cursor.fetchone()
            cursor.execute("SELECT source, COUNT(*) FROM paper_fulltexts GROUP BY source ORDER BY source")
            fulltext_sources = {str(row[0]): int(row[1]) for row in cursor.fetchall()}
            cursor.execute("SELECT arxiv_id FROM papers WHERE arxiv_id = ANY(%s)", (arxiv_ids,))
            found_arxiv_ids = {row[0] for row in cursor.fetchall()}
            found_chunk_ids: set[int] = set()
            if chunk_ids:
                cursor.execute("SELECT id FROM paper_chunks WHERE id = ANY(%s)", (chunk_ids,))
                found_chunk_ids = {int(row[0]) for row in cursor.fetchall()}
    except Exception as exc:
        raise PreconditionError(f"사전 점검 쿼리가 실패했습니다: {type(exc).__name__}: {exc}") from exc
    finally:
        connection.close()

    problems: list[str] = []
    missing_arxiv_ids = [value for value in arxiv_ids if value not in found_arxiv_ids]
    missing_chunk_ids = [value for value in chunk_ids if value not in found_chunk_ids]
    if missing_arxiv_ids:
        problems.append(f"정답 arxiv_id {len(missing_arxiv_ids)}개가 papers에 없습니다: {missing_arxiv_ids[:10]}")
    if missing_chunk_ids:
        problems.append(
            f"정답 chunk_id {len(missing_chunk_ids)}개가 paper_chunks에 없습니다(재청킹·백필 후 id가 바뀌었을 수 있음): "
            f"{missing_chunk_ids[:10]}"
        )
    if not chunks:
        problems.append("paper_chunks가 비어 있습니다.")
    if needs_embeddings and not embeddings:
        problems.append(
            "paper_embeddings가 비어 있어 vector/hybrid 결과가 의미 없습니다. --methods lexical로 실행하세요."
        )
    if problems:
        raise PreconditionError("\n".join(problems))

    return {
        "papers": int(papers),
        "papers_with_chunks": int(papers_with_chunks),
        "chunks": int(chunks),
        "embeddings": int(embeddings),
        "fulltext_sources": fulltext_sources,
    }


def git_revision() -> str:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}{' (dirty)' if dirty else ''}"


def file_digest(path: Path) -> str:
    if path.is_dir():
        return dataset_digest(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def build_meta(
    *,
    started_at: datetime,
    queries_path: Path,
    queries: list[EvalQuery],
    methods: list[str],
    args: argparse.Namespace,
    corpus: dict[str, Any],
    settings: Any,
    skipped: list[EvalQuery] | None = None,
) -> dict[str, str]:
    langs = Counter(query.lang for query in queries)
    sources = Counter(query.source for query in queries)
    sources_text = ", ".join(f"{key} {value}" for key, value in sorted(sources.items()))
    fulltext_text = ", ".join(f"{key} {value}" for key, value in corpus["fulltext_sources"].items()) or "없음"
    meta = {
        "실행 시각": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "커밋": git_revision(),
        "질의셋": f"`{queries_path.name}` sha256:{file_digest(queries_path)} — {len(queries)}개 "
        f"(ko {langs.get('ko', 0)} / en {langs.get('en', 0)}; {sources_text})",
        "설정": f"k={args.k}, adjacency_window={args.adjacency_window}, warmup={args.warmup}, methods={', '.join(methods)}",
        "코퍼스": f"papers {corpus['papers']} (청크 보유 {corpus['papers_with_chunks']}), "
        f"chunks {corpus['chunks']}, embeddings {corpus['embeddings']}",
        "본문 source": fulltext_text,
        "임베딩 모델": f"{settings.openai_embedding_model} ({settings.openai_embedding_dimensions}d)",
    }
    categories = Counter(query.category for query in queries if query.category)
    if categories:
        meta["케이스 카테고리"] = ", ".join(f"{key} {value}" for key, value in sorted(categories.items()))
    if skipped:
        meta["검색 평가 제외"] = (
            f"{len(skipped)}개 (거절·입력 거부 기대, 정답 논문 없음, 상세 챗 전용, 대화 이력 케이스): "
            + ", ".join(query.id for query in skipped[:20])
            + (" ..." if len(skipped) > 20 else "")
        )
    return meta


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queries_path = Path(args.queries)

    try:
        selected = select_queries(
            load_queries(queries_path),
            lang=args.lang,
            limit=args.limit,
            categories=parse_categories(args.category),
        )
    except FileNotFoundError:
        print(f"질의셋이 없습니다: {queries_path}. scripts/eval_build_queries.py로 먼저 만드세요.", file=sys.stderr)
        return EXIT_PRECONDITION
    except DatasetError as exc:
        print(f"질의셋 형식 오류: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION
    queries, skipped = split_retrieval_queries(selected)
    if skipped:
        print(f"검색 평가 대상이 아닌 케이스 {len(skipped)}개를 건너뜁니다.", file=sys.stderr)
    if not queries:
        print("실행할 질의가 없습니다.", file=sys.stderr)
        return EXIT_PRECONDITION
    unresolved = placeholder_problems(queries)
    if unresolved:
        print(
            f"자리표시자 id가 남은 질의 {len(unresolved)}개: {unresolved[:10]}. "
            "scripts/eval_build_queries.py --attach-ids로 실제 id를 채운 파일을 쓰세요.",
            file=sys.stderr,
        )
        return EXIT_PRECONDITION

    try:
        methods = resolve_methods(args.methods.split(","), args.ablations.split(","))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_PRECONDITION
    needs_embeddings = any(METHODS[method].needs_embeddings for method in methods)

    try:
        from src.shared import get_settings

        settings = get_settings()
    except Exception as exc:
        print(f"설정을 불러올 수 없습니다(.env 확인): {exc}", file=sys.stderr)
        return EXIT_PRECONDITION
    if needs_embeddings and not settings.openai_api_key:
        print(
            "vector/hybrid 계열에는 OPENAI_API_KEY가 필요합니다. 키 없이 돌리려면 --methods lexical.", file=sys.stderr
        )
        return EXIT_PRECONDITION

    try:
        corpus = preflight(settings, queries, needs_embeddings=needs_embeddings)
    except PreconditionError as exc:
        print(f"평가를 시작하지 않습니다.\n{exc}", file=sys.stderr)
        return EXIT_PRECONDITION

    from src.integrations.paper_retriever import PaperRetriever

    retriever = PaperRetriever()
    started_at = datetime.now()

    for method in methods:
        for query in queries[: max(0, args.warmup)]:
            run_query(retriever, method, query, k=args.k, adjacency_window=args.adjacency_window, keep_going=True)

    def progress(done: int, total: int, method: str, query: EvalQuery) -> None:
        print(f"[{done}/{total}] {method:<16} {query.id}", file=sys.stderr)

    results = run_evaluation(
        retriever,
        queries,
        methods,
        k=args.k,
        adjacency_window=args.adjacency_window,
        keep_going=args.keep_going,
        progress=progress,
    )
    aggregates = aggregate(results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    meta = build_meta(
        started_at=started_at,
        queries_path=queries_path,
        queries=queries,
        methods=methods,
        args=args,
        corpus=corpus,
        settings=settings,
        skipped=skipped,
    )
    markdown = render_markdown(aggregates, title=f"Retrieval evaluation {stamp}", meta=meta)
    write_query_csv(results, out_dir / f"{stamp}.csv")
    write_summary_csv(aggregates, out_dir / f"{stamp}_summary.csv")
    (out_dir / f"{stamp}.md").write_text(markdown, encoding="utf-8")

    print(markdown)
    error_count = sum(1 for result in results if not result.ok)
    print(f"결과: {out_dir / stamp}.{{csv,md}}, {out_dir / stamp}_summary.csv", file=sys.stderr)
    if error_count:
        print(f"{error_count}건의 질의 실행이 실패했습니다(집계에서 제외, CSV error 열 참고).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
