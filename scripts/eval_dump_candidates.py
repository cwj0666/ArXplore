"""hybrid 융합 입력 캐시 기록: 질의마다 제품 hybrid 경로가 융합에 넘기는 lexical/vector 후보와 제품 결과를 저장한다.

    python scripts/eval_dump_candidates.py                                   # eval/queries.jsonl, k=10
    python scripts/eval_dump_candidates.py --out eval/cache/candidates_run1.jsonl.gz

PostgreSQL과 OPENAI_API_KEY(질의 임베딩)가 필요하다. 사전 점검은 scripts/eval_retrieval.py와 같다(정답 id가 DB에
없거나 임베딩이 비어 있으면 아무것도 쓰지 않고 종료 코드 2).
질의마다 `search_paper_chunks_by_hybrid`(live), `hybrid_fusion_inputs`(융합 입력), vector·lexical 단독(live)을
limit=k로 부른다. 임베딩은 질의별로 한 번만 요청해 네 호출이 같은 벡터를 쓴다.
기록 뒤 기본 설정 재생이 live hybrid와 같은지 바로 비교해, 다르면 파일은 남기고 종료 코드 1로 끝난다.
재생·설정 비교는 scripts/eval_fusion_sweep.py.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_retrieval import (  # noqa: E402
    EXIT_PRECONDITION,
    PreconditionError,
    file_digest,
    git_revision,
    placeholder_problems,
    preflight,
    select_queries,
)

from eval.candidate_cache import capture_query, compare_live, write_cache  # noqa: E402
from eval.dataset import DatasetError, load_queries  # noqa: E402
from eval.runner import split_retrieval_queries  # noqa: E402
from src.integrations.hybrid_fusion import DEFAULT_HYBRID_FUSION  # noqa: E402
from src.integrations.paper_retriever import candidate_fetch_limit, hybrid_branch_limit  # noqa: E402


class MemoEmbeddingClient:
    """같은 텍스트 목록의 임베딩을 한 번만 요청한다. live 호출과 융합 입력 기록이 같은 질의 벡터를 쓰게 한다."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.cache: dict[tuple[str, ...], list[list[float]]] = {}

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        key = tuple(texts)
        if key not in self.cache:
            self.cache[key] = self.inner.embed_texts(list(texts))
        return self.cache[key]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", default=str(REPO_ROOT / "eval" / "queries.jsonl"))
    parser.add_argument("--k", type=int, default=10, help="평가 k. 채널별 후보 수는 hybrid_branch_limit(k)")
    parser.add_argument("--limit", type=int, default=None, help="파일 순서대로 앞 N개 질의만 기록")
    parser.add_argument("--lang", choices=("ko", "en"), default=None)
    parser.add_argument("--category", default=None, help="쉼표 구분 케이스 카테고리만 기록")
    parser.add_argument("--out", default=None, help="기본 eval/cache/candidates_<timestamp>.jsonl.gz")
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queries_path = Path(args.queries)
    categories = {name.strip() for name in (args.category or "").split(",") if name.strip()} or None
    try:
        selected = select_queries(load_queries(queries_path), lang=args.lang, limit=args.limit, categories=categories)
    except FileNotFoundError:
        print(f"질의셋이 없습니다: {queries_path}", file=sys.stderr)
        return EXIT_PRECONDITION
    except DatasetError as exc:
        print(f"질의셋 형식 오류: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION
    queries, skipped = split_retrieval_queries(selected)
    if not queries:
        print("기록할 질의가 없습니다.", file=sys.stderr)
        return EXIT_PRECONDITION
    unresolved = placeholder_problems(queries)
    if unresolved:
        print(f"자리표시자 id가 남은 질의 {len(unresolved)}개: {unresolved[:10]}", file=sys.stderr)
        return EXIT_PRECONDITION

    try:
        from src.shared import get_settings

        settings = get_settings()
    except Exception as exc:
        print(f"설정을 불러올 수 없습니다(.env 확인): {exc}", file=sys.stderr)
        return EXIT_PRECONDITION
    if not settings.openai_api_key:
        print("hybrid 입력 기록에는 OPENAI_API_KEY가 필요합니다.", file=sys.stderr)
        return EXIT_PRECONDITION
    try:
        corpus = preflight(settings, queries, needs_embeddings=True)
    except PreconditionError as exc:
        print(f"기록을 시작하지 않습니다.\n{exc}", file=sys.stderr)
        return EXIT_PRECONDITION

    from src.integrations.embedding_client import EmbeddingClient
    from src.integrations.paper_retriever import PaperRetriever

    retriever = PaperRetriever(embedding_client=MemoEmbeddingClient(EmbeddingClient()))
    started_at = datetime.now()
    captured = []
    for index, query in enumerate(queries, start=1):
        print(f"[{index}/{len(queries)}] {query.id}", file=sys.stderr)
        captured.append(capture_query(retriever, query, k=args.k))

    branch_limit = hybrid_branch_limit(args.k)
    langs = Counter(query.lang for query in queries)
    sources = Counter(query.source for query in queries)
    header = {
        "created_at": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "git_revision": git_revision(),
        "k": args.k,
        "branch_limit": branch_limit,
        "branch_fetch_limit": candidate_fetch_limit(branch_limit),
        "queries": {
            "path": str(
                queries_path.relative_to(REPO_ROOT) if queries_path.is_relative_to(REPO_ROOT) else queries_path
            ),
            "digest": file_digest(queries_path),
            "n": len(queries),
            "langs": dict(sorted(langs.items())),
            "sources": dict(sorted(sources.items())),
            "skipped_ids": [query.id for query in skipped],
            "filters": {"lang": args.lang, "limit": args.limit, "category": args.category},
        },
        "corpus": corpus,
        "embedding_model": f"{settings.openai_embedding_model} ({settings.openai_embedding_dimensions}d)",
        "vector_min_similarity": settings.vector_min_similarity,
        "fusion_default": asdict(DEFAULT_HYBRID_FUSION),
    }
    out_path = (
        Path(args.out) if args.out else REPO_ROOT / "eval" / "cache" / f"candidates_{started_at:%Y%m%d-%H%M%S}.jsonl.gz"
    )
    write_cache(out_path, header, captured)
    print(f"캐시: {out_path} ({out_path.stat().st_size:,} bytes, 질의 {len(captured)}개)", file=sys.stderr)

    mismatches = compare_live(captured, k=args.k)
    for channel in ("vector", "lexical"):
        differing = compare_live(captured, k=args.k, channel=channel)
        print(
            f"참고: 캐시로 만든 {channel} 단독 순위가 live(limit={args.k})와 다른 질의 {len(differing)}개 "
            "(후보 수가 달라 생길 수 있음)",
            file=sys.stderr,
        )
    if mismatches:
        print(
            f"기본 설정 재생이 live hybrid와 다른 질의 {len(mismatches)}개: {[item.query_id for item in mismatches[:10]]}",
            file=sys.stderr,
        )
        return 1
    print(f"기본 설정 재생이 live hybrid와 {len(captured)}개 질의 모두 같습니다.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
