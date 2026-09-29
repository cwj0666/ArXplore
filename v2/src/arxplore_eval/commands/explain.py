import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

import typer
from psycopg import Error as DatabaseError

from arxplore.config import AppSettings
from arxplore.db.migrate import fetch_search_indexes
from arxplore.db.pool import connect_from_settings
from arxplore.retrieval.query import normalize_search_query
from arxplore.retrieval.service import RetrievalConfig
from arxplore_eval.commands.common import (
    BASE_URL_ENV,
    BASE_URL_HELP,
    POOL_ROLE,
    embedding_client,
    fail,
    settings_or_exit,
    usable_api_key,
)
from arxplore_eval.corpus import corpus_summary, database_info
from arxplore_eval.explain import (
    LEXICAL,
    VECTOR,
    Connection,
    ExplainRun,
    ExplainTarget,
    channel_indexes,
    channel_limit,
    explain_target,
    lexical_target,
    report_lines,
    sql_fetch_limit,
    stored_query_embedding,
    vector_target,
)

DEFAULT_QUERY = "direct preference optimization"
DEFAULT_LIMIT = 5


class Channel(StrEnum):
    lexical = "lexical"
    vector = "vector"
    both = "both"


@dataclass(frozen=True)
class ExplainOptions:
    query: str
    sql_limit: int
    arxiv_id: str | None
    channels: tuple[str, ...]
    runs: int
    embed: bool
    text: bool
    base_url: str | None


async def _repeat(connection: Connection, target: ExplainTarget, *, runs: int, text: bool) -> list[ExplainRun]:
    results = [await explain_target(connection, target) for _ in range(runs - 1)]
    results.append(await explain_target(connection, target, text=text))
    return results


async def _query_embedding(
    settings: AppSettings, connection: Connection, options: ExplainOptions
) -> tuple[list[float], str] | None:
    if not options.embed:
        return await stored_query_embedding(connection)
    client = embedding_client(settings, options.base_url)
    try:
        return (await client.embed([options.query]))[0], f"OpenAI {client.model}"
    finally:
        await client.aclose()


def _title(target: ExplainTarget, arxiv_id: str | None) -> str:
    scope = " (논문 범위, 정확 검색)" if target.channel == VECTOR and arxiv_id else ""
    return f"{target.channel}{scope} — {target.describe()}"


async def run_explain(settings: AppSettings, options: ExplainOptions) -> list[str]:
    connection = await connect_from_settings(settings, role=POOL_ROLE, autocommit=True)
    lines: list[str] = []
    async with connection:
        info = await database_info(connection)
        corpus = await corpus_summary(connection)
        present = await fetch_search_indexes(connection)
        expected = await channel_indexes(connection)
        lines.append(
            f"database: PostgreSQL {info['server_version']}, pgvector {info['vector_version']}, "
            f"papers={corpus.papers}, paper_chunks={corpus.chunks}, paper_embeddings={corpus.embeddings}"
        )
        lines.append(
            "search indexes: "
            + ", ".join(f"{name} {'present' if found else 'MISSING'}" for name, found in present.items())
        )
        lines.append(f"query: {options.query!r} sql_limit={options.sql_limit} arxiv_id={options.arxiv_id}")
        if LEXICAL in options.channels:
            target = lexical_target(options.query, sql_limit=options.sql_limit, arxiv_id=options.arxiv_id)
            if target is not None:
                runs = await _repeat(connection, target, runs=options.runs, text=options.text)
                lines += report_lines(
                    _title(target, options.arxiv_id),
                    [run.summary for run in runs],
                    expected[LEXICAL],
                    runs[-1].text_plan,
                )
        if VECTOR in options.channels:
            embedding = await _query_embedding(settings, connection, options)
            if embedding is None:
                lines.append("\nvector: paper_embeddings가 비어 있어 건너뜁니다.")
                return lines
            vector, source = embedding
            lines.append(f"\nvector query embedding: {source}")
            target = vector_target(
                vector,
                sql_limit=options.sql_limit,
                config=RetrievalConfig.from_settings(settings),
                arxiv_id=options.arxiv_id,
            )
            runs = await _repeat(connection, target, runs=options.runs, text=options.text)
            lines += report_lines(
                _title(target, options.arxiv_id),
                [run.summary for run in runs],
                () if options.arxiv_id else expected[VECTOR],
                runs[-1].text_plan,
            )
    return lines


def register(app: typer.Typer) -> None:
    @app.command(
        "explain",
        help="제품 검색 SQL에 EXPLAIN (ANALYZE, BUFFERS)를 붙여 읽기 전용 트랜잭션에서 돌리고 롤백한다.",
    )
    def explain_command(
        query: Annotated[str, typer.Option(help="검색 질의")] = DEFAULT_QUERY,
        limit: Annotated[
            int, typer.Option(min=1, help="제품 검색 limit. SQL LIMIT은 제품 규칙으로 계산한다")
        ] = DEFAULT_LIMIT,
        hybrid: Annotated[
            bool,
            typer.Option("--hybrid/--no-hybrid", help="hybrid 경로처럼 채널마다 hybrid_branch_limit(limit)를 쓴다"),
        ] = True,
        sql_limit: Annotated[int | None, typer.Option(min=1, help="SQL LIMIT을 직접 지정한다")] = None,
        arxiv_id: Annotated[str | None, typer.Option(help="논문 범위 검색(상세 챗 경로)")] = None,
        channel: Annotated[Channel, typer.Option(help="EXPLAIN할 채널")] = Channel.both,
        runs: Annotated[int, typer.Option(min=1, help="같은 SQL을 반복 실행한 횟수(실행 시간 중앙값)")] = 1,
        embed: Annotated[
            bool, typer.Option(help="질의 임베딩을 OpenAI API로 만든다(유료). 기본은 저장된 임베딩 하나")
        ] = False,
        text: Annotated[bool, typer.Option(help="PostgreSQL 텍스트 계획도 출력한다")] = False,
        openai_base_url: Annotated[str | None, typer.Option(envvar=BASE_URL_ENV, help=BASE_URL_HELP)] = None,
    ) -> None:
        normalized = normalize_search_query(query)
        if not normalized:
            fail("빈 질의입니다.")
        settings = settings_or_exit()
        if embed and channel != Channel.lexical and usable_api_key(settings) is None:
            fail("--embed에는 OPENAI_API_KEY가 필요합니다.")
        chosen = (LEXICAL, VECTOR) if channel == Channel.both else (channel.value,)
        options = ExplainOptions(
            query=normalized,
            sql_limit=sql_limit or sql_fetch_limit(channel_limit(limit, hybrid=hybrid), arxiv_id=arxiv_id),
            arxiv_id=arxiv_id,
            channels=chosen,
            runs=runs,
            embed=embed,
            text=text,
            base_url=openai_base_url,
        )
        try:
            lines = asyncio.run(run_explain(settings, options))
        except (DatabaseError, OSError) as exc:
            fail(f"EXPLAIN을 실행할 수 없습니다: {type(exc).__name__}: {exc}")
        for line in lines:
            typer.echo(line)
