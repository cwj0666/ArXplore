import asyncio
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from psycopg import Error as DatabaseError

from arxplore.config import AppSettings
from arxplore.db.pool import connect_from_settings
from arxplore.llm.models import ChatModelFactory
from arxplore_eval.build_queries import (
    CASE_QUERIES_FILE,
    CASES_DIR_NAME,
    COMBINED_QUERIES_FILE,
    DEFAULT_CANDIDATES,
    DEFAULT_CHUNKS,
    DEFAULT_MAX_SHARED_WORDS,
    DEFAULT_MIN_ABSTRACT_CHARS,
    DEFAULT_MIN_CHUNK_CHARS,
    DEFAULT_PAPERS,
    DEFAULT_SEED,
    GENERATED_QUERIES_FILE,
    PLACEHOLDER_REGISTRY_NAME,
    QUERY_TEMPERATURE,
    AttachCandidates,
    ChunkSample,
    OverrideError,
    PaperSample,
    all_match_placeholders,
    attach_catalogue,
    attach_preview_lines,
    choose_distinct,
    combine_query_files,
    dry_run_lines,
    fetch_chunk_samples,
    fetch_paper_samples,
    find_attach_candidates,
    generate_queries,
    load_placeholder_registry,
    merge_with_existing,
    parse_overrides,
    read_case_payloads,
)
from arxplore_eval.commands.common import (
    BASE_URL_ENV,
    BASE_URL_HELP,
    POOL_ROLE,
    echo_err,
    eval_dir,
    fail,
    settings_or_exit,
    usable_api_key,
)
from arxplore_eval.dataset import DatasetError, dump_queries, find_placeholders, load_queries


class BuildMode(StrEnum):
    known_item = "known_item"
    chunk_synth = "chunk_synth"
    both = "both"


class ChunkLang(StrEnum):
    alternate = "alternate"
    ko = "ko"
    en = "en"


async def fetch_samples(
    settings: AppSettings,
    *,
    papers: int,
    chunks: int,
    seed: int,
    min_abstract_chars: int,
    min_chunk_chars: int,
    chunk_lang: str,
) -> tuple[list[PaperSample], list[ChunkSample]]:
    connection = await connect_from_settings(settings, role=POOL_ROLE, autocommit=True)
    async with connection:
        paper_samples = await fetch_paper_samples(
            connection, count=papers, seed=seed, min_abstract_chars=min_abstract_chars
        )
        chunk_samples = await fetch_chunk_samples(
            connection, count=chunks, seed=seed, min_chunk_chars=min_chunk_chars, lang_policy=chunk_lang
        )
    return paper_samples, chunk_samples


async def attach_candidates(
    settings: AppSettings,
    *,
    needed: list[str],
    registry: dict[str, Any],
    overrides: dict[str, str],
    case_ids: set[str],
    limit: int,
) -> AttachCandidates:
    connection = await connect_from_settings(settings, role=POOL_ROLE, autocommit=True)
    async with connection:
        return await find_attach_candidates(
            connection, needed=needed, registry=registry, overrides=overrides, case_ids=case_ids, limit=limit
        )


def run_combine() -> None:
    directory = eval_dir()
    try:
        combined = combine_query_files(
            directory / GENERATED_QUERIES_FILE, directory / CASE_QUERIES_FILE, directory / COMBINED_QUERIES_FILE
        )
    except (FileNotFoundError, DatasetError) as exc:
        fail(f"질의셋을 합칠 수 없습니다: {exc}")
    typer.echo(
        f"{directory / COMBINED_QUERIES_FILE}에 {GENERATED_QUERIES_FILE} + {CASE_QUERIES_FILE} "
        f"{len(combined)}개 질의를 썼습니다."
    )


def run_attach(
    *,
    cases: Path,
    placeholders: Path | None,
    attach_out: Path,
    overrides_text: list[str],
    candidates: int,
    write: bool,
    overwrite: bool,
) -> None:
    registry_path = placeholders or cases / PLACEHOLDER_REGISTRY_NAME
    try:
        payloads = read_case_payloads(cases)
        registry = load_placeholder_registry(registry_path)
        overrides = parse_overrides(overrides_text)
    except (FileNotFoundError, DatasetError, json.JSONDecodeError) as exc:
        fail(f"카탈로그를 읽을 수 없습니다: {exc}")
    if write and attach_out.exists() and not overwrite:
        fail(f"{attach_out}가 이미 있습니다. --overwrite를 지정하세요.")

    needed = sorted(find_placeholders(payloads))
    unknown = [value for value in needed if value not in registry["placeholders"] and value not in overrides]
    if unknown:
        fail(f"조건 파일에 없는 자리표시자: {unknown}")
    settings = settings_or_exit()
    try:
        found = asyncio.run(
            attach_candidates(
                settings,
                needed=needed,
                registry=registry,
                overrides=overrides,
                case_ids={str(payload.get("id")) for payload in payloads},
                limit=candidates,
            )
        )
    except OverrideError as exc:
        fail(str(exc))
    except (DatabaseError, OSError) as exc:
        fail(f"PostgreSQL에 연결할 수 없습니다: {type(exc).__name__}: {exc}")

    papers = choose_distinct(found.candidates, all_match_placeholders(needed, registry, overrides))
    for line in attach_preview_lines(needed, registry, found.candidates, papers):
        typer.echo(line)
    attached, dropped = attach_catalogue(payloads, papers, absent_present=found.absent_present)
    typer.echo(f"\n케이스 {len(payloads)}개 중 {len(attached)}개를 채웠고 {len(dropped)}개를 뺐습니다.")
    for line in dropped:
        typer.echo(f"  뺌: {line}")
    if not write:
        typer.echo("\n미리보기: 파일을 쓰지 않았습니다. 쓰려면 --write.")
        return
    attach_out.parent.mkdir(parents=True, exist_ok=True)
    attach_out.write_text(
        "".join(json.dumps(payload, ensure_ascii=False) + "\n" for payload in attached), encoding="utf-8"
    )
    load_queries(attach_out)
    typer.echo(f"{attach_out}에 {len(attached)}개 케이스를 썼습니다. 답이 실제 논문과 맞는지 훑어보고 쓰세요.")


def register(app: typer.Typer) -> None:
    @app.command("build-queries", help="DB 표본과 LLM으로 검색 평가 질의셋을 만든다. 기본은 dry-run.")
    def build_queries_command(
        mode: Annotated[BuildMode, typer.Option()] = BuildMode.both,
        papers: Annotated[int, typer.Option(min=0, help="known_item 논문 수(논문당 ko/en 2개)")] = DEFAULT_PAPERS,
        chunks: Annotated[int, typer.Option(min=0, help="chunk_synth 청크 수")] = DEFAULT_CHUNKS,
        chunk_lang: Annotated[ChunkLang, typer.Option()] = ChunkLang.alternate,
        seed: Annotated[int, typer.Option()] = DEFAULT_SEED,
        min_abstract_chars: Annotated[int, typer.Option(min=0)] = DEFAULT_MIN_ABSTRACT_CHARS,
        min_chunk_chars: Annotated[int, typer.Option(min=0)] = DEFAULT_MIN_CHUNK_CHARS,
        max_shared_words: Annotated[
            int, typer.Option(min=1, help="입력과 이 단어 수 이상 연속 일치하면 버림")
        ] = DEFAULT_MAX_SHARED_WORDS,
        out: Annotated[Path | None, typer.Option(help="기본 eval/queries.generated.jsonl")] = None,
        generate: Annotated[
            bool, typer.Option("--generate/--dry-run", help="LLM으로 질의를 만들어 --out에 쓴다")
        ] = False,
        overwrite: Annotated[bool, typer.Option()] = False,
        append: Annotated[bool, typer.Option(help="기존 파일에 새 id만 추가")] = False,
        show_prompts: Annotated[bool, typer.Option(help="dry-run에서 모든 프롬프트를 출력")] = False,
        combine: Annotated[
            bool, typer.Option(help="queries.generated.jsonl + queries.cases.jsonl을 이어 queries.jsonl을 다시 쓴다")
        ] = False,
        attach_ids: Annotated[bool, typer.Option(help="카탈로그 자리표시자를 DB 논문 id로 채운다")] = False,
        cases: Annotated[Path | None, typer.Option(help="카탈로그 JSONL 파일 또는 디렉터리")] = None,
        placeholders: Annotated[Path | None, typer.Option(help="자리표시자 조건 파일")] = None,
        attach_out: Annotated[Path | None, typer.Option(help="기본 eval/queries.cases.jsonl")] = None,
        set_: Annotated[
            list[str] | None, typer.Option("--set", metavar="PLACEHOLDER=ARXIV_ID", help="직접 지정(반복 가능)")
        ] = None,
        candidates: Annotated[int, typer.Option(min=1, help="자리표시자마다 보여 줄 후보 수")] = DEFAULT_CANDIDATES,
        write: Annotated[bool, typer.Option(help="--attach-ids 결과를 --attach-out에 쓴다")] = False,
        openai_base_url: Annotated[str | None, typer.Option(envvar=BASE_URL_ENV, help=BASE_URL_HELP)] = None,
    ) -> None:
        if overwrite and append:
            fail("--overwrite와 --append는 함께 쓸 수 없습니다.")
        if write and not attach_ids:
            fail("--write는 --attach-ids와 함께만 씁니다.")
        if combine:
            run_combine()
            return
        directory = eval_dir()
        if attach_ids:
            run_attach(
                cases=cases or directory / CASES_DIR_NAME,
                placeholders=placeholders,
                attach_out=attach_out or directory / CASE_QUERIES_FILE,
                overrides_text=list(set_ or []),
                candidates=candidates,
                write=write,
                overwrite=overwrite,
            )
            return

        out_path = out or directory / GENERATED_QUERIES_FILE
        if generate and out_path.exists() and not (overwrite or append):
            fail(f"{out_path}가 이미 있습니다. --overwrite 또는 --append를 지정하세요.")
        paper_count = papers if mode in (BuildMode.known_item, BuildMode.both) else 0
        chunk_count = chunks if mode in (BuildMode.chunk_synth, BuildMode.both) else 0
        settings = settings_or_exit()
        try:
            paper_samples, chunk_samples = asyncio.run(
                fetch_samples(
                    settings,
                    papers=paper_count,
                    chunks=chunk_count,
                    seed=seed,
                    min_abstract_chars=min_abstract_chars,
                    min_chunk_chars=min_chunk_chars,
                    chunk_lang=chunk_lang.value,
                )
            )
        except (DatabaseError, OSError) as exc:
            fail(f"PostgreSQL에 연결할 수 없습니다: {type(exc).__name__}: {exc}")

        if paper_count and len(paper_samples) < paper_count:
            echo_err(f"경고: 조건을 만족하는 논문이 {len(paper_samples)}편뿐입니다(요청 {paper_count}).")
        if chunk_count and len(chunk_samples) < chunk_count:
            echo_err(f"경고: 조건을 만족하는 본문 청크가 {len(chunk_samples)}개뿐입니다(요청 {chunk_count}).")
        if not paper_samples and not chunk_samples:
            fail("표본이 비어 있습니다. 수집·prepare 상태를 확인하세요.")
        if not generate:
            for line in dry_run_lines(paper_samples, chunk_samples, show_prompts=show_prompts):
                typer.echo(line)
            return
        if usable_api_key(settings) is None:
            fail("OPENAI_API_KEY가 설정되지 않았습니다.")

        model = ChatModelFactory.from_settings(settings, base_url=openai_base_url).create(temperature=QUERY_TEMPERATURE)
        try:
            records, rejected = asyncio.run(
                generate_queries(
                    paper_samples, chunk_samples, model=model, max_shared_words=max_shared_words, log=echo_err
                )
            )
            merged, skipped = merge_with_existing(out_path, records, append=append)
        except DatasetError as exc:
            fail(str(exc))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(dump_queries(merged), encoding="utf-8")
        load_queries(out_path)
        typer.echo(
            f"{len(records) - skipped}개 질의를 {out_path}에 썼습니다(전체 {len(merged)}개, 중복 id {skipped}개 건너뜀)."
        )
        if rejected:
            typer.echo(f"버린 질의 {len(rejected)}개:")
            for line in rejected:
                typer.echo(f"  {line}")
        typer.echo(
            "측정 전에 파일을 직접 훑어 정답이 모호한 질의를 지우고, 손으로 고친 항목은 source를 manual로 바꾸세요. "
            "그다음 --combine으로 queries.jsonl을 다시 만드세요."
        )
