import asyncio
from pathlib import Path
from typing import Annotated

import typer
from psycopg import AsyncConnection
from psycopg import Error as DatabaseError

from arxplore.db.pool import settings_conninfo
from arxplore_eval.build_queries import (
    CASE_QUERIES_FILE,
    COMBINED_QUERIES_FILE,
    GENERATED_QUERIES_FILE,
    combine_query_files,
)
from arxplore_eval.commands.common import EXIT_FAILURES, eval_dir, fail, settings_or_exit
from arxplore_eval.dataset import DatasetError, EvalQuery, load_queries
from arxplore_eval.remap import (
    DEFAULT_MIN_OVERLAP,
    QueryRemap,
    fetch_old_chunks,
    fetch_paper_chunks,
    read_only,
    remap_queries,
    report_lines,
    rewrite_chunk_ids,
)

CONNECT_TIMEOUT_SECONDS = 10
APPLICATION_NAME = "arxplore-eval-remap"


async def _connect(conninfo: str) -> AsyncConnection[object]:
    connection: AsyncConnection[object] = await AsyncConnection.connect(
        conninfo, autocommit=True, connect_timeout=CONNECT_TIMEOUT_SECONDS, application_name=APPLICATION_NAME
    )
    await read_only(connection)
    return connection


async def compute_remaps(queries: list[EvalQuery], *, old_db: str, new_db: str, min_overlap: float) -> list[QueryRemap]:
    chunk_ids = sorted({chunk_id for query in queries for chunk_id in query.relevant_chunk_ids})
    async with await _connect(old_db) as old:
        old_chunks = await fetch_old_chunks(old, chunk_ids)
    arxiv_ids = sorted({chunk.arxiv_id for chunk in old_chunks.values()})
    async with await _connect(new_db) as new:
        new_chunks = await fetch_paper_chunks(new, arxiv_ids)
    return remap_queries(queries, old_chunks, new_chunks, min_overlap=min_overlap)


def register(app: typer.Typer) -> None:
    @app.command(
        "remap-chunks",
        help="코퍼스 전환 뒤 청크 정답(relevant_chunk_ids)을 새 DB의 청크 id로 옮긴다. 두 DB는 읽기만 한다.",
    )
    def remap_chunks_command(
        old_db: Annotated[str, typer.Option(help="질의셋 청크 id가 가리키는 옛 DB (conninfo 또는 URL)")],
        new_db: Annotated[str | None, typer.Option(help="새 DB (기본: 설정의 PostgreSQL)")] = None,
        queries: Annotated[Path | None, typer.Option(help="기본 eval/queries.generated.jsonl")] = None,
        min_overlap: Annotated[
            float, typer.Option(min=0.0, max=1.0, help="이보다 겹침이 낮으면 사람 검토가 필요하다고 표시한다")
        ] = DEFAULT_MIN_OVERLAP,
        apply: Annotated[
            bool, typer.Option(help="모두 기준을 넘으면 질의 파일을 고치고 queries.jsonl을 다시 결합한다")
        ] = False,
    ) -> None:
        path = queries or eval_dir() / GENERATED_QUERIES_FILE
        try:
            loaded = load_queries(path)
        except (FileNotFoundError, DatasetError) as exc:
            fail(f"질의셋을 읽을 수 없습니다: {exc}")
        target = new_db or settings_conninfo(settings_or_exit())
        try:
            remaps = asyncio.run(compute_remaps(loaded, old_db=old_db, new_db=target, min_overlap=min_overlap))
        except (DatabaseError, OSError) as exc:
            fail(f"PostgreSQL에 연결할 수 없습니다: {type(exc).__name__}: {exc}")
        for line in report_lines(remaps):
            typer.echo(line)
        review = [remap.query_id for remap in remaps if remap.needs_review]
        if review:
            typer.echo(f"검토가 필요한 질의가 있어 파일을 고치지 않습니다: {', '.join(review)}", err=True)
            raise typer.Exit(code=EXIT_FAILURES)
        if not apply:
            typer.echo("미리보기: 파일을 고치지 않았습니다. 반영하려면 --apply.")
            return
        changed = rewrite_chunk_ids(path, remaps)
        typer.echo(f"{path}의 질의 {changed}건을 고쳤습니다.")
        if path.name != GENERATED_QUERIES_FILE:
            return
        try:
            combined = combine_query_files(
                path, path.with_name(CASE_QUERIES_FILE), path.with_name(COMBINED_QUERIES_FILE)
            )
        except (FileNotFoundError, DatasetError) as exc:
            fail(f"queries.jsonl을 다시 결합하지 못했습니다: {exc}", EXIT_FAILURES)
        typer.echo(f"{path.with_name(COMBINED_QUERIES_FILE)}를 {len(combined)}개 질의로 다시 결합했습니다.")
