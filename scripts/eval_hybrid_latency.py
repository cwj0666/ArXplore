"""hybrid 검색 지연 A/B: lexical/vector 채널 순차(A, `parallel_channels=False`) vs 병렬(B, 기본) 실행.

    python scripts/eval_hybrid_latency.py                    # eval/queries.jsonl 검색 평가 대상 전체, 워밍업 1 + 반복 3
    python scripts/eval_hybrid_latency.py --limit 4 --repeats 1 --no-memo-check   # 연기 테스트

PostgreSQL과 OPENAI_API_KEY(질의 임베딩)가 필요하다. LLM 생성 호출은 없다.
질의 블록마다 eval 하니스 경로(`search_paper_contexts_by_hybrid`, k, window=1)와 제품 경로(`retrieve_contexts`, limit=5)를
A/B 쌍으로, lexical·vector 단독을 대조군으로 실행한다. 쌍 순서는 반복마다 A→B / B→A로 바꾼다. 임베딩은 캐시하지 않는다.
끝으로 임베딩을 질의당 한 번만 요청해 A와 B가 공유하는 패스로 결과 dict 전체가 같은지 확인한다(`--no-memo-check`로 끔).
결과: eval/results/latency_<timestamp>.{md,csv}. 절차는 eval/latency.py.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_dump_candidates import MemoEmbeddingClient  # noqa: E402
from eval_retrieval import EXIT_PRECONDITION, _connect, file_digest, git_revision, select_queries  # noqa: E402

from eval.dataset import DatasetError, load_queries  # noqa: E402
from eval.latency import (  # noqa: E402
    CONTROL_PATHS,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    EMBEDDING_PRICE_PER_MTOK,
    HYBRID_EVAL,
    LEXICAL_ONLY,
    PAIRED_PATHS,
    PARALLEL,
    PRODUCT,
    SEQUENTIAL,
    VECTOR_ONLY,
    BenchmarkPlan,
    StageRecorder,
    embedding_cost,
    equality_rows,
    instrument,
    paired_delta,
    render_markdown,
    run_benchmark,
    write_samples_csv,
)
from eval.runner import split_retrieval_queries  # noqa: E402


class CountingEmbeddingClient:
    """실제 임베딩 요청 횟수와 입력 토큰 수를 센다(tiktoken이 없으면 글자 수/4 추정)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls = 0
        self.texts = 0
        self.tokens = 0
        try:
            import tiktoken

            self._encoding = tiktoken.encoding_for_model("text-embedding-3-large")
            self.token_method = "tiktoken cl100k_base"
        except Exception:  # noqa: BLE001
            self._encoding = None
            self.token_method = "추정(글자 수 / 4)"

    def is_available(self) -> bool:
        return self.inner.is_available()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        self.texts += len(texts)
        for text in texts:
            self.tokens += len(self._encoding.encode(text)) if self._encoding else max(1, len(text) // 4)
        return self.inner.embed_texts(texts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", default=str(REPO_ROOT / "eval" / "queries.jsonl"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--adjacency-window", type=int, default=1)
    parser.add_argument("--product-limit", type=int, default=5, help="제품 경로 limit (에이전트 도구와 같음)")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="파일 순서대로 앞 N개 질의만")
    parser.add_argument("--no-product", action="store_true", help="제품 경로 쌍을 빼고 eval 경로만 잰다")
    parser.add_argument("--no-memo-check", action="store_true")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "eval" / "results"))
    args = parser.parse_args(argv)
    if args.repeats < 2 and not args.limit:
        parser.error("--repeats must be >= 2 for a full run (반복 간 기준선이 필요하다)")
    return args


def machine_info() -> dict[str, str]:
    cpu = "unknown"
    memory = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                memory = f"{int(line.split()[1]) / 1024 / 1024:.1f} GiB"
                break
    except OSError:
        pass
    release = platform.release()
    wsl = "WSL2" if "microsoft" in release.lower() else "아님"
    return {
        "머신": f"{cpu}, 논리 코어 {os.cpu_count()}, RAM {memory}",
        "OS": f"{platform.system()} {release} (WSL: {wsl}), Python {platform.python_version()}",
    }


def database_info(settings: Any) -> dict[str, str]:
    connection = _connect(settings)
    try:
        with connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    current_setting('server_version'),
                    (SELECT extversion FROM pg_extension WHERE extname = 'vector'),
                    (SELECT COUNT(*) FROM papers),
                    (SELECT COUNT(*) FROM paper_chunks),
                    (SELECT COUNT(*) FROM paper_embeddings),
                    current_setting('shared_buffers')
                """
            )
            version, vector_version, papers, chunks, embeddings, shared_buffers = cursor.fetchone()
    finally:
        connection.close()
    host = settings.postgres_host
    return {
        "DB": f"PostgreSQL {version} + pgvector {vector_version}, host {host} (포트 {settings.server_postgres_port}), "
        f"shared_buffers {shared_buffers}",
        "코퍼스": f"papers {papers}, chunks {chunks}, embeddings {embeddings}",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queries_path = Path(args.queries)
    try:
        selected = select_queries(load_queries(queries_path), lang=None, limit=None)
    except (FileNotFoundError, DatasetError) as exc:
        print(f"질의셋을 읽을 수 없습니다: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION
    queries, _ = split_retrieval_queries(selected)
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print("잴 질의가 없습니다.", file=sys.stderr)
        return EXIT_PRECONDITION

    from src.core.agent.retrieval import retrieve_contexts
    from src.integrations.db import resolve_pool_max
    from src.integrations.embedding_client import EmbeddingClient
    from src.integrations.paper_repository import PaperRepository
    from src.integrations.paper_retriever import PaperRetriever
    from src.integrations.vector_repository import VectorRepository
    from src.shared import get_settings

    settings = get_settings()
    if not settings.openai_api_key:
        print("질의 임베딩에 OPENAI_API_KEY가 필요합니다.", file=sys.stderr)
        return EXIT_PRECONDITION
    if settings.retrieval_mode != "hybrid" and not args.no_product:
        print("제품 경로를 재려면 RETRIEVAL_MODE=hybrid여야 합니다.", file=sys.stderr)
        return EXIT_PRECONDITION
    try:
        db_meta = database_info(settings)
    except Exception as exc:  # noqa: BLE001
        print(f"PostgreSQL에 연결할 수 없습니다: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_PRECONDITION

    repository = PaperRepository(settings=settings)
    vector_repository = VectorRepository(settings=settings)
    embeddings = CountingEmbeddingClient(EmbeddingClient(settings=settings))
    recorder = StageRecorder()
    retrievers = {
        variant: instrument(
            PaperRetriever(
                repository=repository,
                embedding_client=embeddings,
                vector_repository=vector_repository,
                parallel_channels=variant == PARALLEL,
            ),
            recorder,
        )
        for variant in (SEQUENTIAL, PARALLEL)
    }

    k, window, product_limit = args.k, args.adjacency_window, args.product_limit
    paired = {
        HYBRID_EVAL: lambda r, q: (r.search_paper_contexts_by_hybrid(q, limit=k, adjacency_window=window), "hybrid")
    }
    if not args.no_product:
        paired[PRODUCT] = lambda r, q: retrieve_contexts(q, retriever=r, limit=product_limit)
    controls = {
        LEXICAL_ONLY: lambda r, q: (r.search_paper_contexts(q, limit=k, adjacency_window=window), "lexical"),
        VECTOR_ONLY: lambda r, q: (r.search_paper_contexts_by_vector(q, limit=k, adjacency_window=window), "vector"),
    }
    plan = BenchmarkPlan(
        queries=queries,
        retrievers=retrievers,
        recorder=recorder,
        paired_paths=paired,
        control_paths=controls,
        repeats=args.repeats,
        warmup=args.warmup,
    )

    def progress(done: int, total: int, query: Any) -> None:
        if done % 20 == 0 or done == total:
            print(f"[{done}/{total}] {query.id}", file=sys.stderr)

    started_at = datetime.now()
    samples = run_benchmark(plan, progress=progress)
    timed_calls, timed_tokens = embeddings.calls, embeddings.tokens

    memo_equality: dict[str, tuple[int, int]] = {}
    if not args.no_memo_check:
        memo = MemoEmbeddingClient(embeddings)
        memo_retrievers = {
            variant: PaperRetriever(
                repository=repository,
                embedding_client=memo,
                vector_repository=vector_repository,
                parallel_channels=variant == PARALLEL,
            )
            for variant in (SEQUENTIAL, PARALLEL)
        }
        for path, fn in paired.items():
            same = sum(
                fn(memo_retrievers[SEQUENTIAL], query.query) == fn(memo_retrievers[PARALLEL], query.query)
                for query in queries
            )
            memo_equality[path] = (same, len(queries))
    finished_at = datetime.now()

    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"latency_{stamp}.csv"
    md_path = out_dir / f"latency_{stamp}.md"
    write_samples_csv(samples, csv_path)

    langs = Counter(query.lang for query in queries)
    paths = list(paired)
    deltas = [
        paired_delta(samples, queries, path, subset=subset, resamples=args.resamples, seed=args.seed)
        for path in paths
        for subset in ("all", "ko", "en")
    ]
    equality = [row for path in paths for row in equality_rows(samples, path, args.repeats)]
    meta = {
        "실행 시각": f"{started_at:%Y-%m-%d %H:%M:%S} ~ {finished_at:%H:%M:%S}",
        "커밋": git_revision(),
        **machine_info(),
        **db_meta,
        "연결 풀": f"POSTGRES_POOL_MAX={resolve_pool_max(settings)} (단일 클라이언트 스레드, 병렬 변형은 요청당 최대 2 연결)",
        "질의셋": f"`{queries_path.name}` sha256:{file_digest(queries_path)} 검색 평가 대상 {len(queries)}개 "
        f"(ko {langs.get('ko', 0)} / en {langs.get('en', 0)})",
        "설정": f"k={k}, adjacency_window={window}, 제품 limit={product_limit}, 워밍업 {args.warmup}, 반복 {args.repeats}, "
        f"RETRIEVAL_MODE={settings.retrieval_mode}, VECTOR_MIN_SIMILARITY={settings.vector_min_similarity}",
        "A / B": "A = `PaperRetriever(parallel_channels=False)`(병렬화 전 순차 경로), B = 기본(병렬). 같은 프로세스·저장소·"
        "임베딩 클라이언트를 공유",
        "API": f"질의 임베딩 {settings.openai_embedding_model} ({settings.openai_embedding_dimensions}d) 왕복 포함, 캐시 없음",
        "순서": "반복마다 쌍 순서 A→B / B→A 교대, 블록 = eval 경로 쌍 → 제품 경로 쌍 → lexical 단독 → vector 단독",
        "bootstrap": f"{args.resamples}회, seed {args.seed}, ko/en 쌍 클러스터 재표집",
    }
    total_tokens = embeddings.tokens
    usage = {
        "측정(워밍업 포함) 호출": f"{timed_calls}회, {timed_tokens} 토큰",
        "임베딩 고정 패스 호출": f"{embeddings.calls - timed_calls}회, {embeddings.tokens - timed_tokens} 토큰",
        "합계": f"{embeddings.calls}회 ({embeddings.texts}개 텍스트), {total_tokens} 토큰 ({embeddings.token_method})",
        "추정 비용": f"${embedding_cost(total_tokens):.5f} (${EMBEDDING_PRICE_PER_MTOK}/1M 토큰)",
    }
    notes = [
        "p50/p95는 질의 × 반복 표본 전체의 백분위(선형 보간). Δ는 질의별 반복 중앙값의 차이.",
        f"대조군({', '.join(CONTROL_PATHS)})은 이 변경과 무관한 경로로, 측정 중 DB·API 상태가 A/B에 공평했는지 보는 용도다.",
    ]
    md_path.write_text(
        render_markdown(
            samples,
            queries,
            title=f"hybrid 채널 병렬화 지연 측정 ({stamp})",
            meta=meta,
            deltas=deltas,
            equality=equality,
            memo_equality=memo_equality,
            embedding_usage=usage,
            notes=notes,
        ),
        encoding="utf-8",
    )
    print(f"결과: {md_path}\n      {csv_path}", file=sys.stderr)
    for path in PAIRED_PATHS:
        for row in deltas:
            if row.path == path and row.subset == "all":
                print(
                    f"{path}: A {row.median_a:.1f}ms → B {row.median_b:.1f}ms, Δ {row.delta_ms:+.1f}ms", file=sys.stderr
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
