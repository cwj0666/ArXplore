import asyncio
import importlib.util
import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from typer.testing import CliRunner

from arxplore.db.migrate import migrate
from arxplore.db.pool import create_pool, open_pool
from arxplore.pipeline.corpus import rechunk_corpus
from arxplore_eval.cli import app
from arxplore_eval.dataset import EvalQuery, load_queries
from arxplore_eval.remap import fetch_old_chunks, fetch_paper_chunks, read_only, remap_queries

pytestmark = pytest.mark.integration

DPO = "2401.00001"
REPO_EVAL = Path(__file__).resolve().parents[3] / "eval"
INVARIANTS = Path(__file__).resolve().parents[2] / "unit" / "eval" / "test_eval_query_sets.py"
GOLDEN_URL_VARIABLE = "GOLDEN_DATABASE_URL"
TEMPLATE_VARIABLE = "GOLDEN_TEMPLATE_DATABASE"
DEFAULT_TEMPLATE = "arxplore_tpl"
COPY_NAME = "v2_test_eval_remap"


async def test_remap_after_an_in_place_resplit(empty_database: str, chunk_ids: dict[tuple[str, int], int]) -> None:
    query = EvalQuery(
        id="cs-dpo",
        query="q",
        lang="en",
        relevant_arxiv_ids=(DPO,),
        relevant_chunk_ids=(chunk_ids[(DPO, 1)], chunk_ids[(DPO, 3)]),
        source="llm_synth",
    )
    async with await psycopg.AsyncConnection.connect(empty_database, autocommit=True) as connection:
        await read_only(connection)
        old = await fetch_old_chunks(connection, query.relevant_chunk_ids)
    async with await psycopg.AsyncConnection.connect(empty_database, autocommit=True) as connection:
        texts = [chunk.text for chunk in sorted(old.values(), key=lambda chunk: chunk.chunk_index)]
        async with connection.cursor() as cursor:
            await cursor.execute("SELECT chunk_text FROM paper_chunks WHERE arxiv_id = %s ORDER BY chunk_index", (DPO,))
            originals = [row[0] for row in await cursor.fetchall()]
            await cursor.execute("DELETE FROM paper_chunks WHERE arxiv_id = %s", (DPO,))
            for index in range(len(originals) - 1):
                await cursor.execute(
                    "INSERT INTO paper_chunks (arxiv_id, chunk_index, chunk_text, section_title, token_count) "
                    "VALUES (%s, %s, %s, %s, 10)",
                    (DPO, index, f"{originals[index]}\n\n{originals[index + 1]}", "3 Method"),
                )
        await read_only(connection)
        new = await fetch_paper_chunks(connection, [DPO, "2499.99999"])

    (remap,) = remap_queries([query], old, new)

    assert new["2499.99999"] == []
    assert [chunk.chunk_index for chunk in new[DPO]] == list(range(len(originals) - 1))
    assert remap.overlap == 1.0 and not remap.needs_review
    by_index = {chunk.chunk_id: chunk.chunk_index for chunk in new[DPO]}
    assert [by_index[chunk_id] for chunk_id in remap.new_chunk_ids] == [0, 2]
    assert all(texts[position] in new[DPO][index].text for position, index in enumerate((0, 2)))


def _admin_url(url: str) -> str:
    params = conninfo_to_dict(url)
    params["dbname"] = "postgres"
    return make_conninfo(**{key: str(value) for key, value in params.items()})


def _with_database(url: str, name: str) -> str:
    params = conninfo_to_dict(url)
    params["dbname"] = name
    return make_conninfo(**{key: str(value) for key, value in params.items()})


async def _rechunk(url: str, arxiv_ids: list[str]) -> None:
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as connection:
        await migrate(connection)
    pool = create_pool(url, role="test", max_size=2, timeout=30, connect_timeout=10)
    async with open_pool(pool):
        result = await rechunk_corpus(pool, arxiv_ids=arxiv_ids, apply=True)
    assert len(result.changed) == len(arxiv_ids)


def _invariant_tests() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_query_set_invariants", INVARIANTS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def golden_copy() -> Iterator[tuple[str, str]]:
    golden = os.environ.get(GOLDEN_URL_VARIABLE, "").strip()
    if not golden:
        pytest.skip(f"set {GOLDEN_URL_VARIABLE} to the read-only golden corpus to test the remap after a rechunk")
    template = os.environ.get(TEMPLATE_VARIABLE, "").strip() or DEFAULT_TEMPLATE
    admin = _admin_url(golden)
    with psycopg.connect(admin, autocommit=True) as connection:
        connection.execute(f'DROP DATABASE IF EXISTS "{COPY_NAME}" WITH (FORCE)')
        connection.execute(f'CREATE DATABASE "{COPY_NAME}" TEMPLATE "{template}"')
    try:
        yield golden, _with_database(golden, COPY_NAME)
    finally:
        with psycopg.connect(admin, autocommit=True) as connection:
            connection.execute(f'DROP DATABASE IF EXISTS "{COPY_NAME}" WITH (FORCE)')


def test_remap_to_a_rechunked_copy_of_the_golden_corpus(golden_copy: tuple[str, str], tmp_path: Path) -> None:
    golden, copy = golden_copy
    for name in ("queries.generated.jsonl", "queries.cases.jsonl", "queries.jsonl"):
        shutil.copyfile(REPO_EVAL / name, tmp_path / name)
    generated = tmp_path / "queries.generated.jsonl"
    before = generated.read_text(encoding="utf-8").splitlines()
    chunk_queries = [query for query in load_queries(generated) if query.relevant_chunk_ids]
    asyncio.run(_rechunk(copy, sorted({query.relevant_arxiv_ids[0] for query in chunk_queries})))
    options = ["remap-chunks", "--old-db", golden, "--new-db", copy, "--queries", str(generated)]

    preview = CliRunner().invoke(app, options)

    assert preview.exit_code == 0, preview.output
    rows = [line.split("\t") for line in preview.output.splitlines() if line.count("\t") == 3]
    assert {row[0] for row in rows} == {query.id for query in chunk_queries}
    assert all(row[3] == "ok" and float(row[2].removeprefix("overlap ")) >= 0.99 for row in rows)
    assert f"청크 정답 질의 {len(chunk_queries)}건: 검토 필요 0건" in preview.output
    assert generated.read_text(encoding="utf-8").splitlines() == before

    applied = CliRunner().invoke(app, [*options, "--apply"])

    assert applied.exit_code == 0, applied.output
    invariants = _invariant_tests()
    for name in sorted(vars(invariants)):
        if name.startswith("test_"):
            getattr(invariants, name)(tmp_path)
    after = generated.read_text(encoding="utf-8").splitlines()
    changed = [index for index, (old, new) in enumerate(zip(before, after, strict=True)) if old != new]
    assert {json.loads(after[index])["id"] for index in changed} == {query.id for query in chunk_queries}
    remapped = {query.id: query for query in load_queries(generated) if query.relevant_chunk_ids}
    with psycopg.connect(copy) as connection:
        found = connection.execute(
            "SELECT id, arxiv_id FROM paper_chunks WHERE id = ANY(%s)",
            (sorted({chunk_id for query in remapped.values() for chunk_id in query.relevant_chunk_ids}),),
        ).fetchall()
    papers = dict(found)
    for original in chunk_queries:
        query = remapped[original.id]
        assert query.relevant_chunk_ids != original.relevant_chunk_ids
        assert all(papers[chunk_id] == original.relevant_arxiv_ids[0] for chunk_id in query.relevant_chunk_ids)
