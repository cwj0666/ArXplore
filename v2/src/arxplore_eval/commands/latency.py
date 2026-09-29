import asyncio
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from psycopg import Error as DatabaseError
from psycopg_pool import PoolTimeout

from arxplore.config import AppSettings, build_postgres_connection_params
from arxplore.retrieval.service import RetrievalConfig
from arxplore_eval.commands.common import (
    BASE_URL_ENV,
    BASE_URL_HELP,
    TIME_FORMAT,
    default_queries_path,
    default_results_dir,
    echo_err,
    embedding_client,
    eval_pool,
    fail,
    now,
    repo_root,
    settings_or_exit,
    stamp_of,
    usable_api_key,
)
from arxplore_eval.corpus import corpus_summary, database_info
from arxplore_eval.dataset import DatasetError, EvalQuery, load_queries
from arxplore_eval.embedders import CountingEmbedder, MemoEmbedder
from arxplore_eval.latency import (
    CONTROL_PATHS,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    EMBEDDING_PRICE_PER_MTOK,
    PAIRED_PATHS,
    PARALLEL,
    PRODUCT,
    SEQUENTIAL,
    SUBSETS,
    BenchmarkPlan,
    BenchRetriever,
    PairedDelta,
    Sample,
    StageRecorder,
    build_paths,
    embedding_cost,
    equality_rows,
    memo_equality,
    paired_delta,
    render_markdown,
    run_benchmark,
    write_samples_csv,
)
from arxplore_eval.provenance import file_digest, git_revision, machine_info
from arxplore_eval.runner import split_retrieval_queries

PROGRESS_EVERY = 20


@dataclass(frozen=True)
class LatencyOptions:
    k: int
    adjacency_window: int
    product_limit: int
    repeats: int
    warmup: int
    include_product: bool
    memo_check: bool
    base_url: str | None


@dataclass(frozen=True)
class LatencyRun:
    samples: list[Sample]
    memo: dict[str, tuple[int, int]]
    database: dict[str, str]
    timed_calls: int
    timed_tokens: int
    embeddings: CountingEmbedder
    finished_at: str


def _progress(done: int, total: int, query: EvalQuery) -> None:
    if done % PROGRESS_EVERY == 0 or done == total:
        echo_err(f"[{done}/{total}] {query.id}")


async def run_latency(settings: AppSettings, queries: list[EvalQuery], options: LatencyOptions) -> LatencyRun:
    config = RetrievalConfig.from_settings(settings)
    async with eval_pool(settings) as pool:
        async with pool.connection() as connection:
            info = await database_info(connection)
            corpus = await corpus_summary(connection)
        client = embedding_client(settings, options.base_url)
        embeddings = CountingEmbedder(client)
        try:
            recorder = StageRecorder()
            retrievers = {
                variant: BenchRetriever(pool, embeddings, config, sequential=variant == SEQUENTIAL, recorder=recorder)
                for variant in (SEQUENTIAL, PARALLEL)
            }
            paired, controls = build_paths(
                k=options.k,
                adjacency_window=options.adjacency_window,
                product_limit=options.product_limit,
                include_product=options.include_product,
            )
            plan = BenchmarkPlan(
                queries=queries,
                retrievers=retrievers,
                recorder=recorder,
                paired_paths=paired,
                control_paths=controls,
                repeats=options.repeats,
                warmup=options.warmup,
            )
            samples = await run_benchmark(plan, progress=_progress)
            timed_calls, timed_tokens = embeddings.calls, embeddings.tokens
            memo: dict[str, tuple[int, int]] = {}
            if options.memo_check:
                memo_embedder = MemoEmbedder(embeddings)
                memo_retrievers = {
                    variant: BenchRetriever(pool, memo_embedder, config, sequential=variant == SEQUENTIAL)
                    for variant in (SEQUENTIAL, PARALLEL)
                }
                memo = await memo_equality(paired, memo_retrievers, queries)
        finally:
            await client.aclose()
    params = build_postgres_connection_params(settings)
    database = {
        "DB": f"PostgreSQL {info['server_version']} + pgvector {info['vector_version']}, host {params['host']} "
        f"(포트 {params['port']}), shared_buffers {info['shared_buffers']}",
        "코퍼스": f"papers {corpus.papers}, chunks {corpus.chunks}, embeddings {corpus.embeddings}",
    }
    return LatencyRun(
        samples=samples,
        memo=memo,
        database=database,
        timed_calls=timed_calls,
        timed_tokens=timed_tokens,
        embeddings=embeddings,
        finished_at=now().strftime("%H:%M:%S"),
    )


def _usage(run: LatencyRun) -> dict[str, str]:
    embeddings = run.embeddings
    return {
        "측정(워밍업 포함) 호출": f"{run.timed_calls}회, {run.timed_tokens} 토큰",
        "임베딩 고정 패스 호출": f"{embeddings.calls - run.timed_calls}회, {embeddings.tokens - run.timed_tokens} 토큰",
        "합계": f"{embeddings.calls}회 ({embeddings.texts}개 텍스트), {embeddings.tokens} 토큰 ({embeddings.token_method})",
        "추정 비용": f"${embedding_cost(embeddings.tokens):.5f} (${EMBEDDING_PRICE_PER_MTOK}/1M 토큰)",
    }


def register(app: typer.Typer) -> None:
    @app.command("latency", help="hybrid 채널 순차(A)·병렬(B) 실행의 지연을 A/B로 잰다.")
    def latency_command(
        queries: Annotated[Path | None, typer.Option(help="JSONL 파일 또는 *.jsonl 디렉터리")] = None,
        k: Annotated[int, typer.Option(min=1)] = 10,
        adjacency_window: Annotated[int, typer.Option(min=0)] = 1,
        product_limit: Annotated[int, typer.Option(min=1, help="제품 경로 limit (에이전트 도구와 같음)")] = 5,
        repeats: Annotated[int, typer.Option(min=1)] = 3,
        warmup: Annotated[int, typer.Option(min=0)] = 1,
        limit: Annotated[int | None, typer.Option(min=1, help="검색 평가 대상 중 앞 N개 질의만")] = None,
        no_product: Annotated[bool, typer.Option(help="제품 경로 쌍을 빼고 eval 경로만 잰다")] = False,
        no_memo_check: Annotated[bool, typer.Option(help="임베딩 고정 패스의 결과 동일성 확인을 건너뛴다")] = False,
        resamples: Annotated[int, typer.Option(min=1)] = DEFAULT_RESAMPLES,
        seed: Annotated[int, typer.Option()] = DEFAULT_SEED,
        out_dir: Annotated[Path | None, typer.Option()] = None,
        openai_base_url: Annotated[str | None, typer.Option(envvar=BASE_URL_ENV, help=BASE_URL_HELP)] = None,
    ) -> None:
        if repeats < 2 and not limit:
            fail("전체 실행에는 --repeats 2 이상이 필요합니다(반복 간 기준선).")
        queries_path = queries or default_queries_path()
        try:
            eligible, _ = split_retrieval_queries(load_queries(queries_path))
        except (FileNotFoundError, DatasetError) as exc:
            fail(f"질의셋을 읽을 수 없습니다: {exc}")
        selected = eligible[:limit] if limit else eligible
        if not selected:
            fail("잴 질의가 없습니다.")
        settings = settings_or_exit()
        if usable_api_key(settings) is None:
            fail("질의 임베딩에 OPENAI_API_KEY가 필요합니다.")
        if settings.retrieval_mode != "hybrid" and not no_product:
            fail("제품 경로를 재려면 RETRIEVAL_MODE=hybrid여야 합니다.")
        options = LatencyOptions(
            k=k,
            adjacency_window=adjacency_window,
            product_limit=product_limit,
            repeats=repeats,
            warmup=warmup,
            include_product=not no_product,
            memo_check=not no_memo_check,
            base_url=openai_base_url,
        )
        started_at = now()
        try:
            run = asyncio.run(run_latency(settings, selected, options))
        except (PoolTimeout, DatabaseError, OSError) as exc:
            fail(f"PostgreSQL에 연결할 수 없습니다: {type(exc).__name__}: {exc}")

        paths = [path for path in PAIRED_PATHS if path != PRODUCT or not no_product]
        deltas: list[PairedDelta] = [
            paired_delta(run.samples, selected, path, subset=subset, resamples=resamples, seed=seed)
            for path in paths
            for subset in SUBSETS
        ]
        equality = [row for path in paths for row in equality_rows(run.samples, path, repeats)]
        langs = Counter(query.lang for query in selected)
        meta = {
            "실행 시각": f"{started_at.strftime(TIME_FORMAT)} ~ {run.finished_at}",
            "커밋": git_revision(repo_root()),
            **machine_info(),
            **run.database,
            "연결 풀": f"POSTGRES_POOL_MAX={settings.postgres_pool_max} (단일 이벤트 루프, 병렬 변형은 요청당 최대 2 연결)",
            "질의셋": f"`{queries_path.name}` sha256:{file_digest(queries_path)} 검색 평가 대상 {len(selected)}개 "
            f"(ko {langs.get('ko', 0)} / en {langs.get('en', 0)})",
            "설정": f"k={k}, adjacency_window={adjacency_window}, 제품 limit={product_limit}, 워밍업 {warmup}, "
            f"반복 {repeats}, RETRIEVAL_MODE={settings.retrieval_mode}, "
            f"VECTOR_MIN_SIMILARITY={settings.vector_min_similarity}",
            "A / B": "A = `BenchRetriever(sequential=True)`(두 채널을 차례로 기다림), B = 제품 `Retriever`와 같은 병렬 "
            "경로(asyncio.gather). 같은 풀·임베딩 클라이언트를 공유",
            "API": f"질의 임베딩 {settings.openai_embedding_model} ({settings.openai_embedding_dimensions}d) 왕복 포함, "
            "캐시 없음",
            "순서": "반복마다 쌍 순서 A→B / B→A 교대, 블록 = eval 경로 쌍 → 제품 경로 쌍 → lexical 단독 → vector 단독",
            "bootstrap": f"{resamples}회, seed {seed}, ko/en 쌍 클러스터 재표집",
        }
        notes = [
            "p50/p95는 질의 x 반복 표본 전체의 백분위(선형 보간). Δ는 질의별 반복 중앙값의 차이.",
            f"대조군({', '.join(CONTROL_PATHS)})은 이 변경과 무관한 경로로, 측정 중 DB·API 상태가 A/B에 공평했는지 보는 용도다.",
        ]
        stamp = stamp_of(started_at)
        directory = out_dir or default_results_dir()
        directory.mkdir(parents=True, exist_ok=True)
        csv_path = directory / f"latency_{stamp}.csv"
        md_path = directory / f"latency_{stamp}.md"
        write_samples_csv(run.samples, csv_path)
        md_path.write_text(
            render_markdown(
                run.samples,
                title=f"hybrid 채널 병렬화 지연 측정 ({stamp})",
                meta=meta,
                deltas=deltas,
                equality=equality,
                memo=run.memo,
                embedding_usage=_usage(run),
                notes=notes,
            ),
            encoding="utf-8",
        )
        echo_err(f"결과: {md_path}\n      {csv_path}")
        for row in deltas:
            if row.subset == "all":
                echo_err(f"{row.path}: A {row.median_a:.1f}ms → B {row.median_b:.1f}ms, Δ {row.delta_ms:+.1f}ms")
