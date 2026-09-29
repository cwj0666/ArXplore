import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from psycopg import Error as DatabaseError
from psycopg_pool import PoolTimeout

from arxplore.config import AppSettings
from arxplore.retrieval.query import candidate_fetch_limit, hybrid_branch_limit
from arxplore.retrieval.service import RetrievalConfig, Retriever
from arxplore_eval.candidate_cache import CachedQuery, capture_query, compare_live, product_check, write_cache
from arxplore_eval.commands.common import (
    BASE_URL_ENV,
    BASE_URL_HELP,
    EXIT_FAILURES,
    TIME_FORMAT,
    Lang,
    default_cache_dir,
    default_queries_path,
    echo_err,
    embedding_client,
    eval_pool,
    fail,
    lang_value,
    now,
    repo_root,
    settings_or_exit,
    stamp_of,
    usable_api_key,
)
from arxplore_eval.commands.retrieval import eligible_queries, load_selected_queries
from arxplore_eval.corpus import CorpusSummary, PreconditionError, retrieval_preflight
from arxplore_eval.dataset import EvalQuery
from arxplore_eval.embedders import MemoEmbedder
from arxplore_eval.provenance import display_path, file_digest, git_revision
from arxplore_eval.research_fusion import PRODUCT_FUSION


@dataclass(frozen=True)
class Capture:
    queries: list[CachedQuery]
    corpus: CorpusSummary


async def capture_candidates(
    settings: AppSettings, queries: list[EvalQuery], *, k: int, base_url: str | None
) -> Capture:
    async with eval_pool(settings) as pool:
        async with pool.connection() as connection:
            corpus = await retrieval_preflight(connection, queries, needs_embeddings=True)
        client = embedding_client(settings, base_url)
        try:
            retriever = Retriever(pool, MemoEmbedder(client), RetrievalConfig.from_settings(settings))
            captured = []
            for index, query in enumerate(queries, start=1):
                echo_err(f"[{index}/{len(queries)}] {query.id}")
                captured.append(await capture_query(retriever, query, k=k))
        finally:
            await client.aclose()
    return Capture(queries=captured, corpus=corpus)


def cache_header(
    *,
    started_at: str,
    k: int,
    queries_path: Path,
    queries: list[EvalQuery],
    skipped: list[EvalQuery],
    filters: dict[str, Any],
    corpus: CorpusSummary,
    settings: AppSettings,
) -> dict[str, Any]:
    branch_limit = hybrid_branch_limit(k)
    return {
        "created_at": started_at,
        "git_revision": git_revision(repo_root()),
        "k": k,
        "branch_limit": branch_limit,
        "branch_fetch_limit": candidate_fetch_limit(branch_limit),
        "queries": {
            "path": display_path(queries_path, repo_root()),
            "digest": file_digest(queries_path),
            "n": len(queries),
            "langs": dict(sorted(Counter(query.lang for query in queries).items())),
            "sources": dict(sorted(Counter(query.source for query in queries).items())),
            "skipped_ids": [query.id for query in skipped],
            "filters": filters,
        },
        "corpus": corpus.to_dict(),
        "embedding_model": f"{settings.openai_embedding_model} ({settings.openai_embedding_dimensions}d)",
        "vector_min_similarity": settings.vector_min_similarity,
        "fusion_default": asdict(PRODUCT_FUSION),
        "implementation": "arxplore_eval",
    }


def register(app: typer.Typer) -> None:
    @app.command("dump-candidates", help="질의마다 제품 hybrid의 융합 입력과 live 결과를 캐시에 기록한다.")
    def dump_candidates_command(
        queries: Annotated[Path | None, typer.Option(help="JSONL 파일 또는 *.jsonl 디렉터리")] = None,
        k: Annotated[int, typer.Option(min=1, help="평가 k. 채널별 후보 수는 hybrid_branch_limit(k)")] = 10,
        limit: Annotated[int | None, typer.Option(min=1, help="파일 순서대로 앞 N개 질의만")] = None,
        lang: Annotated[Lang | None, typer.Option()] = None,
        category: Annotated[str | None, typer.Option(help="쉼표 구분 케이스 카테고리")] = None,
        out: Annotated[Path | None, typer.Option(help="기본 eval/cache/candidates_<timestamp>.jsonl.gz")] = None,
        openai_base_url: Annotated[str | None, typer.Option(envvar=BASE_URL_ENV, help=BASE_URL_HELP)] = None,
    ) -> None:
        queries_path = queries or default_queries_path()
        selected = load_selected_queries(queries_path, lang=lang_value(lang), limit=limit, category=category)
        eligible, skipped = eligible_queries(selected)
        settings = settings_or_exit()
        if usable_api_key(settings) is None:
            fail("hybrid 입력 기록에는 OPENAI_API_KEY가 필요합니다.")
        started_at = now()
        try:
            capture = asyncio.run(capture_candidates(settings, eligible, k=k, base_url=openai_base_url))
        except PreconditionError as exc:
            fail(f"기록을 시작하지 않습니다.\n{exc}")
        except (PoolTimeout, DatabaseError, OSError) as exc:
            fail(f"PostgreSQL에 연결할 수 없습니다: {type(exc).__name__}: {exc}")

        header = cache_header(
            started_at=started_at.strftime(TIME_FORMAT),
            k=k,
            queries_path=queries_path,
            queries=eligible,
            skipped=skipped,
            filters={"lang": lang_value(lang), "limit": limit, "category": category},
            corpus=capture.corpus,
            settings=settings,
        )
        out_path = out or default_cache_dir() / f"candidates_{stamp_of(started_at)}.jsonl.gz"
        write_cache(out_path, header, capture.queries)
        echo_err(f"캐시: {out_path} ({out_path.stat().st_size:,} bytes, 질의 {len(capture.queries)}개)")

        for channel in ("vector", "lexical"):
            differing = compare_live(capture.queries, k=k, channel=channel)
            echo_err(
                f"참고: 캐시로 만든 {channel} 단독 순위가 live(limit={k})와 다른 질의 {len(differing)}개 "
                "(후보 수가 달라 생길 수 있음)"
            )
        mismatches = compare_live(capture.queries, k=k)
        product = product_check(capture.queries, k=k)
        if mismatches or product:
            echo_err(
                f"기본 설정 재생이 live hybrid와 다른 질의 {len(mismatches)}개: "
                f"{[item.query_id for item in mismatches[:10]]}, 제품 융합 대조 불일치 {len(product)}개"
            )
            raise typer.Exit(code=EXIT_FAILURES)
        echo_err(f"기본 설정 재생이 live hybrid와 {len(capture.queries)}개 질의 모두 같습니다.")
