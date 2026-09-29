import asyncio
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from psycopg import Error as DatabaseError
from psycopg_pool import PoolTimeout

from arxplore.config import AppSettings
from arxplore.retrieval.service import RetrievalConfig, Retriever
from arxplore_eval.commands.common import (
    BASE_URL_ENV,
    BASE_URL_HELP,
    EXIT_FAILURES,
    TIME_FORMAT,
    Lang,
    default_queries_path,
    default_results_dir,
    echo_err,
    embedding_client,
    eval_pool,
    fail,
    lang_value,
    now,
    repo_root,
    settings_or_exit,
    split_names,
    stamp_of,
    usable_api_key,
)
from arxplore_eval.corpus import CorpusSummary, PreconditionError, retrieval_preflight
from arxplore_eval.dataset import (
    DatasetError,
    EvalQuery,
    load_queries,
    parse_categories,
    placeholder_problems,
    select_queries,
)
from arxplore_eval.provenance import file_digest, git_revision
from arxplore_eval.report import render_markdown, write_query_csv, write_summary_csv
from arxplore_eval.runner import (
    METHODS,
    QueryResult,
    aggregate,
    needs_embeddings,
    resolve_methods,
    run_evaluation,
    run_query,
    split_retrieval_queries,
)

DEFAULT_METHODS = "lexical,vector,hybrid"
SKIPPED_PREVIEW = 20


@dataclass(frozen=True)
class RetrievalRun:
    results: list[QueryResult]
    corpus: CorpusSummary


def load_selected_queries(path: Path, *, lang: str | None, limit: int | None, category: str | None) -> list[EvalQuery]:
    try:
        return select_queries(load_queries(path), lang=lang, limit=limit, categories=parse_categories(category))
    except FileNotFoundError:
        fail(f"질의셋이 없습니다: {path}. arxplore-eval build-queries로 먼저 만드세요.")
    except DatasetError as exc:
        fail(f"질의셋 형식 오류: {exc}")


def eligible_queries(selected: list[EvalQuery]) -> tuple[list[EvalQuery], list[EvalQuery]]:
    queries, skipped = split_retrieval_queries(selected)
    if skipped:
        echo_err(f"검색 평가 대상이 아닌 케이스 {len(skipped)}개를 건너뜁니다.")
    if not queries:
        fail("실행할 질의가 없습니다.")
    unresolved = placeholder_problems(queries)
    if unresolved:
        fail(
            f"자리표시자 id가 남은 질의 {len(unresolved)}개: {unresolved[:10]}. "
            "arxplore-eval build-queries --attach-ids로 실제 id를 채운 파일을 쓰세요."
        )
    return queries, skipped


def build_meta(
    *,
    started_at: str,
    queries_path: Path,
    queries: list[EvalQuery],
    methods: list[str],
    k: int,
    adjacency_window: int,
    warmup: int,
    corpus: CorpusSummary,
    settings: AppSettings,
    skipped: list[EvalQuery],
) -> dict[str, str]:
    langs = Counter(query.lang for query in queries)
    sources = ", ".join(f"{key} {value}" for key, value in sorted(Counter(query.source for query in queries).items()))
    meta = {
        "실행 시각": started_at,
        "커밋": git_revision(repo_root()),
        "질의셋": f"`{queries_path.name}` sha256:{file_digest(queries_path)} — {len(queries)}개 "
        f"(ko {langs.get('ko', 0)} / en {langs.get('en', 0)}; {sources})",
        "설정": f"k={k}, adjacency_window={adjacency_window}, warmup={warmup}, methods={', '.join(methods)}",
        "코퍼스": corpus.describe(),
        "본문 source": corpus.describe_sources(),
        "임베딩 모델": f"{settings.openai_embedding_model} ({settings.openai_embedding_dimensions}d)",
        "구현": "arxplore_eval (v2 공개 인터페이스)",
    }
    categories = Counter(query.category for query in queries if query.category)
    if categories:
        meta["케이스 카테고리"] = ", ".join(f"{key} {value}" for key, value in sorted(categories.items()))
    if skipped:
        meta["검색 평가 제외"] = (
            f"{len(skipped)}개 (거절·입력 거부 기대, 정답 논문 없음, 상세 챗 전용, 대화 이력 케이스): "
            + ", ".join(query.id for query in skipped[:SKIPPED_PREVIEW])
            + (" ..." if len(skipped) > SKIPPED_PREVIEW else "")
        )
    return meta


async def run_retrieval_evaluation(
    settings: AppSettings,
    queries: list[EvalQuery],
    methods: list[str],
    *,
    k: int,
    adjacency_window: int,
    warmup: int,
    keep_going: bool,
    base_url: str | None,
) -> RetrievalRun:
    async with eval_pool(settings) as pool:
        async with pool.connection() as connection:
            corpus = await retrieval_preflight(connection, queries, needs_embeddings=needs_embeddings(methods))
        embedder = embedding_client(settings, base_url)
        try:
            retriever = Retriever(pool, embedder, RetrievalConfig.from_settings(settings))
            for method in methods:
                for query in queries[: max(0, warmup)]:
                    await run_query(retriever, method, query, k=k, adjacency_window=adjacency_window, keep_going=True)

            def progress(done: int, total: int, method: str, query: EvalQuery) -> None:
                echo_err(f"[{done}/{total}] {method:<16} {query.id}")

            results = await run_evaluation(
                retriever,
                queries,
                methods,
                k=k,
                adjacency_window=adjacency_window,
                keep_going=keep_going,
                progress=progress,
            )
        finally:
            await embedder.aclose()
    return RetrievalRun(results=results, corpus=corpus)


def register(app: typer.Typer) -> None:
    @app.command("retrieval", help="질의셋으로 lexical/vector/hybrid(+ablation) 검색을 돌려 지표를 기록한다.")
    def retrieval_command(
        queries: Annotated[Path | None, typer.Option(help="JSONL 파일 또는 *.jsonl 디렉터리")] = None,
        methods: Annotated[str, typer.Option(help="쉼표 구분. 사용 가능: " + ", ".join(METHODS))] = DEFAULT_METHODS,
        ablations: Annotated[
            str, typer.Option(help="쉼표 구분: nodiv, nofilter, norerank, plainrrf, rules 또는 all")
        ] = "",
        k: Annotated[int, typer.Option(min=1)] = 10,
        adjacency_window: Annotated[int, typer.Option(min=0, help="제품 경로와 같은 문맥 창")] = 1,
        limit: Annotated[int | None, typer.Option(min=1, help="파일 순서대로 앞 N개 질의만")] = None,
        lang: Annotated[Lang | None, typer.Option()] = None,
        category: Annotated[str | None, typer.Option(help="쉼표 구분 케이스 카테고리")] = None,
        warmup: Annotated[int, typer.Option(min=0, help="방식별로 버리는 워밍업 질의 수")] = 1,
        keep_going: Annotated[bool, typer.Option(help="질의 단위 예외를 error 열에 기록하고 계속")] = False,
        out_dir: Annotated[Path | None, typer.Option()] = None,
        openai_base_url: Annotated[str | None, typer.Option(envvar=BASE_URL_ENV, help=BASE_URL_HELP)] = None,
    ) -> None:
        queries_path = queries or default_queries_path()
        selected = load_selected_queries(queries_path, lang=lang_value(lang), limit=limit, category=category)
        eligible, skipped = eligible_queries(selected)
        try:
            resolved = resolve_methods(split_names(methods), split_names(ablations))
        except ValueError as exc:
            fail(str(exc))
        settings = settings_or_exit()
        if needs_embeddings(resolved) and usable_api_key(settings) is None:
            fail("vector/hybrid 계열에는 OPENAI_API_KEY가 필요합니다. 키 없이 돌리려면 --methods lexical.")

        started_at = now()
        try:
            run = asyncio.run(
                run_retrieval_evaluation(
                    settings,
                    eligible,
                    resolved,
                    k=k,
                    adjacency_window=adjacency_window,
                    warmup=warmup,
                    keep_going=keep_going,
                    base_url=openai_base_url,
                )
            )
        except PreconditionError as exc:
            fail(f"평가를 시작하지 않습니다.\n{exc}")
        except (PoolTimeout, DatabaseError, OSError) as exc:
            fail(f"PostgreSQL에 연결할 수 없습니다: {type(exc).__name__}: {exc}")

        aggregates = aggregate(run.results)
        directory = out_dir or default_results_dir()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = stamp_of(started_at)
        meta = build_meta(
            started_at=started_at.strftime(TIME_FORMAT),
            queries_path=queries_path,
            queries=eligible,
            methods=resolved,
            k=k,
            adjacency_window=adjacency_window,
            warmup=warmup,
            corpus=run.corpus,
            settings=settings,
            skipped=skipped,
        )
        markdown = render_markdown(aggregates, title=f"Retrieval evaluation {stamp}", meta=meta)
        write_query_csv(run.results, directory / f"{stamp}.csv")
        write_summary_csv(aggregates, directory / f"{stamp}_summary.csv")
        (directory / f"{stamp}.md").write_text(markdown, encoding="utf-8")

        typer.echo(markdown)
        echo_err(f"결과: {directory / stamp}.{{csv,md}}, {directory / stamp}_summary.csv")
        error_count = sum(1 for result in run.results if not result.ok)
        if error_count:
            echo_err(f"{error_count}건의 질의 실행이 실패했습니다(집계에서 제외, CSV error 열 참고).")
            raise typer.Exit(code=EXIT_FAILURES)
