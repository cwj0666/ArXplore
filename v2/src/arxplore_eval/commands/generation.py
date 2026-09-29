import asyncio
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from psycopg import Error as DatabaseError
from psycopg_pool import PoolTimeout

from arxplore.api.services.store import PostgresPaperStore
from arxplore.config import AppSettings
from arxplore.context import override_openai_runtime
from arxplore.llm.models import ChatModelFactory
from arxplore.llm.tracing import Tracer
from arxplore.retrieval.service import RetrievalConfig, Retriever
from arxplore_eval.answers import (
    AnswerRecord,
    AnswersError,
    answer_key,
    append_answer_record,
    build_chat_service,
    build_generators,
    collect_answers,
    completed_keys,
    load_answer_records,
    load_reference_answers,
    resolve_modes,
)
from arxplore_eval.behavior import DEFAULT_REFUSAL_PHRASES, OUTCOMES, load_refusal_phrases
from arxplore_eval.commands.common import (
    BASE_URL_ENV,
    BASE_URL_HELP,
    EXIT_FAILURES,
    EXIT_PRECONDITION,
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
from arxplore_eval.corpus import CorpusSummary, PreconditionError, fetch_chunk_texts, generation_preflight
from arxplore_eval.dataset import (
    DatasetError,
    EvalQuery,
    load_queries,
    parse_categories,
    placeholder_problems,
    select_queries,
)
from arxplore_eval.generation import (
    DEFAULT_CONCURRENCY,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    METRICS,
    MetricRequest,
    NullScorer,
    RagasScorer,
    RagasUnavailableError,
    Scorer,
    aggregate_scores,
    render_markdown,
    require_ragas,
    resolve_metrics,
    score_records,
    write_sample_csv,
    write_summary_csv,
)
from arxplore_eval.provenance import file_digest, git_revision


@dataclass(frozen=True)
class CollectionOptions:
    queries: Path
    modes: list[str]
    lang: str | None
    limit: int | None
    category: str | None
    out: Path
    answer_model: str
    keep_going: bool
    base_url: str | None
    trace: bool
    refusal_phrases: Sequence[str]


@dataclass(frozen=True)
class Collection:
    path: Path
    records: list[AnswerRecord]
    corpus: CorpusSummary
    errors: int


def select_records(
    records: Sequence[AnswerRecord],
    *,
    modes: Sequence[str],
    lang: str | None,
    limit: int | None,
    categories: set[str] | None = None,
) -> list[AnswerRecord]:
    selected = [
        record
        for record in records
        if record["mode"] in modes
        and (lang is None or record.get("lang") == lang)
        and (categories is None or str(record.get("category") or "") in categories)
    ]
    return selected[:limit] if limit is not None else selected


def needed_chunk_ids(records: Sequence[Mapping[str, Any]], metrics: Sequence[str]) -> list[int]:
    if "context_precision" not in metrics:
        return []
    ids = {
        int(chunk_id)
        for record in records
        if not str(record.get("reference_answer") or "").strip()
        for chunk_id in record.get("relevant_chunk_ids") or []
    }
    return sorted(ids)


def _collection_queries(options: CollectionOptions) -> tuple[list[EvalQuery], dict[str, str]]:
    try:
        queries = select_queries(
            load_queries(options.queries),
            lang=options.lang,
            limit=options.limit,
            categories=parse_categories(options.category),
        )
        references = load_reference_answers(options.queries)
    except FileNotFoundError as exc:
        raise PreconditionError(
            f"질의셋이 없습니다: {options.queries}. arxplore-eval build-queries로 먼저 만드세요."
        ) from exc
    except (DatasetError, ValueError) as exc:
        raise PreconditionError(f"질의셋 형식 오류: {exc}") from exc
    if not queries:
        raise PreconditionError("실행할 질의가 없습니다.")
    unresolved = placeholder_problems(queries)
    if unresolved:
        raise PreconditionError(
            f"자리표시자 id가 남은 질의 {len(unresolved)}개: {unresolved[:10]}. "
            "arxplore-eval build-queries --attach-ids로 실제 id를 채운 파일을 쓰세요."
        )
    return queries, references


def _done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    try:
        done = completed_keys(load_answer_records(path))
    except AnswersError as exc:
        raise PreconditionError(f"이어 쓸 답변 파일 형식 오류: {exc}") from exc
    echo_err(f"{path}에서 이어서 수집합니다(완료 {len(done)}건 건너뜀).")
    return done


def _progress(done: int, total: int, mode: str, query: EvalQuery, record: AnswerRecord) -> None:
    status = "error" if record.get("error") else f"{record.get('outcome')}, {len(record['contexts'])} contexts"
    echo_err(f"[answer {done}/{total}] {mode:<10} {query.id} ({status})")


async def run_collection(settings: AppSettings, api_key: str, options: CollectionOptions) -> Collection:
    queries, references = _collection_queries(options)
    async with eval_pool(settings) as pool:
        async with pool.connection() as connection:
            corpus = await generation_preflight(connection, queries, modes=options.modes)
        done = _done_keys(options.out)
        embedder = embedding_client(settings, options.base_url)
        try:
            service = build_chat_service(
                settings,
                store=PostgresPaperStore(pool),
                retriever=Retriever(pool, embedder, RetrievalConfig.from_settings(settings)),
                models=ChatModelFactory.from_settings(settings, base_url=options.base_url),
                tracer=Tracer.from_settings(settings) if options.trace else Tracer.disabled(),
            )
            with override_openai_runtime(api_key=api_key, model=options.answer_model):
                await collect_answers(
                    queries,
                    options.modes,
                    build_generators(service, options.modes),
                    done=done,
                    keep_going=options.keep_going,
                    reference_answers=references,
                    model=options.answer_model,
                    sink=lambda record: append_answer_record(options.out, record),
                    progress=_progress,
                    refusal_phrases=options.refusal_phrases,
                )
        finally:
            await embedder.aclose()
    wanted = {(query.id, mode) for query in queries for mode in options.modes if query.supports_mode(mode)}
    records = [record for record in load_answer_records(options.out) if answer_key(record) in wanted]
    errors = sum(1 for record in records if record.get("error"))
    return Collection(path=options.out, records=records, corpus=corpus, errors=errors)


async def load_chunk_texts(settings: AppSettings, chunk_ids: Sequence[int]) -> dict[int, str]:
    if not chunk_ids:
        return {}
    async with eval_pool(settings) as pool, pool.connection() as connection:
        return await fetch_chunk_texts(connection, chunk_ids)


def _judge_progress(done: int, total: int, request: MetricRequest) -> None:
    echo_err(f"[judge {done}/{total}] {request.metric}")


@dataclass(frozen=True)
class JudgeOptions:
    judge_model: str
    embedding_model: str
    judge_max_tokens: int | None
    concurrency: int
    adapt_language: bool
    base_url: str | None


def build_scorer(api_key: str | None, metrics: Sequence[str], options: JudgeOptions) -> Scorer:
    if not metrics or api_key is None:
        return NullScorer()
    return RagasScorer(
        api_key=api_key,
        judge_model=options.judge_model,
        embedding_model=options.embedding_model,
        metrics=metrics,
        adapt_korean=options.adapt_language,
        concurrency=options.concurrency,
        max_tokens=options.judge_max_tokens,
        base_url=options.base_url,
        progress=_judge_progress,
    )


def build_meta(
    *,
    started_at: str,
    answers_path: Path,
    records: Sequence[AnswerRecord],
    metrics: Sequence[str],
    behavior_only: bool,
    judge: JudgeOptions,
    refusal_phrases_path: Path | None,
    corpus: CorpusSummary | None,
) -> dict[str, str]:
    langs = Counter(str(record.get("lang")) for record in records)
    modes = Counter(str(record.get("mode")) for record in records)
    sources = Counter(str(record.get("source")) for record in records)
    answer_models = sorted({str(record.get("model")) for record in records if record.get("model")}) or ["unknown"]
    references = sum(1 for record in records if str(record.get("reference_answer") or "").strip())
    if behavior_only:
        judging = "RAGAS 생략(--behavior-only), 행동 지표만"
    else:
        judging = (
            f"ragas, judge={judge.judge_model}, embeddings={judge.embedding_model}, metrics={', '.join(metrics)}, "
            f"한국어 예시 번역={'on' if judge.adapt_language else 'off'}"
        )
    meta = {
        "실행 시각": started_at,
        "커밋": git_revision(repo_root()),
        "답변 파일": f"`{answers_path.name}` sha256:{file_digest(answers_path)} — {len(records)}건 "
        f"({', '.join(f'{key} {value}' for key, value in sorted(modes.items()))}; "
        f"ko {langs.get('ko', 0)} / en {langs.get('en', 0)}; "
        f"{', '.join(f'{key} {value}' for key, value in sorted(sources.items()))}; reference_answer {references})",
        "답변 모델": ", ".join(answer_models),
        "판정": judging,
        "거절 문구": f"`{refusal_phrases_path.name}`" if refusal_phrases_path else "내장 목록",
    }
    categories = Counter(str(record.get("category")) for record in records if record.get("category"))
    if categories:
        meta["케이스 카테고리"] = ", ".join(f"{key} {value}" for key, value in sorted(categories.items()))
    if corpus is not None:
        meta["코퍼스"] = f"papers {corpus.papers}, chunks {corpus.chunks}, embeddings {corpus.embeddings}"
    return meta


def _answer_records(
    path: Path, *, modes: Sequence[str], lang: str | None, limit: int | None, category: str | None
) -> list[AnswerRecord]:
    try:
        return select_records(
            load_answer_records(path), modes=modes, lang=lang, limit=limit, categories=parse_categories(category)
        )
    except FileNotFoundError as exc:
        raise PreconditionError(f"답변 파일이 없습니다: {path}") from exc
    except AnswersError as exc:
        raise PreconditionError(f"답변 파일 형식 오류: {exc}") from exc


def register(app: typer.Typer) -> None:
    @app.command("generation", help="에이전트·상세 챗 답변을 수집하고 RAGAS·행동 지표로 채점한다.")
    def generation_command(
        collect: Annotated[bool, typer.Option(help="질의셋으로 답변을 수집한 뒤 채점")] = False,
        answers: Annotated[Path | None, typer.Option(help="기존 답변 JSONL을 채점만 한다")] = None,
        queries: Annotated[Path | None, typer.Option(help="JSONL 파일 또는 *.jsonl 디렉터리")] = None,
        mode: Annotated[str, typer.Option(help="agent, paper_chat 또는 both(쉼표 구분 가능)")] = "both",
        metrics: Annotated[str, typer.Option(help="쉼표 구분. 사용 가능: " + ", ".join(METRICS) + ", all")] = "all",
        limit: Annotated[int | None, typer.Option(min=1, help="앞 N개 질의(또는 답변 레코드)만")] = None,
        lang: Annotated[Lang | None, typer.Option()] = None,
        category: Annotated[str | None, typer.Option(help="쉼표 구분 케이스 카테고리")] = None,
        behavior_only: Annotated[bool, typer.Option(help="RAGAS 판정 없이 행동 지표만 계산")] = False,
        refusal_phrases: Annotated[Path | None, typer.Option(help="거절 문구 파일(한 줄에 하나)")] = None,
        out: Annotated[
            Path | None, typer.Option(help="답변 JSONL 경로. 있으면 끝난 (id, mode)는 건너뛰고 이어 쓴다")
        ] = None,
        no_score: Annotated[bool, typer.Option(help="--collect와 함께: 수집만 하고 채점하지 않는다")] = False,
        answer_model: Annotated[str | None, typer.Option(help="답변 생성 모델. 기본은 OPENAI_MODEL")] = None,
        judge_model: Annotated[str, typer.Option()] = DEFAULT_JUDGE_MODEL,
        embedding_model: Annotated[str, typer.Option(help="answer_relevancy용 임베딩 모델")] = DEFAULT_EMBEDDING_MODEL,
        judge_max_tokens: Annotated[int | None, typer.Option(min=1, help="판정 LLM 출력 한도")] = None,
        concurrency: Annotated[int, typer.Option(min=1, help="동시에 보내는 판정 요청 수")] = DEFAULT_CONCURRENCY,
        no_adapt_language: Annotated[bool, typer.Option(help="한국어 질의에도 영어 few-shot 예시 그대로 판정")] = False,
        keep_going: Annotated[bool, typer.Option(help="수집 중 질의 단위 예외를 error 필드에 남기고 계속")] = False,
        out_dir: Annotated[Path | None, typer.Option()] = None,
        openai_base_url: Annotated[str | None, typer.Option(envvar=BASE_URL_ENV, help=BASE_URL_HELP)] = None,
        trace: Annotated[bool, typer.Option(help="LangSmith 추적을 설정대로 켠다(기본 끔)")] = False,
    ) -> None:
        if collect == (answers is not None):
            fail("--collect와 --answers 중 하나만 지정하세요.")
        if no_score and not collect:
            fail("--no-score는 --collect와 함께만 씁니다.")
        try:
            modes = resolve_modes(split_names(mode))
            chosen_metrics = [] if behavior_only else resolve_metrics(split_names(metrics))
            phrases = load_refusal_phrases(refusal_phrases) if refusal_phrases else list(DEFAULT_REFUSAL_PHRASES)
        except (ValueError, OSError) as exc:
            fail(str(exc))
        settings = settings_or_exit()
        api_key = usable_api_key(settings)
        if api_key is None and (collect or not behavior_only):
            fail("OPENAI_API_KEY가 없습니다. 답변 생성과 RAGAS 판정 모두 서버 키가 필요합니다.")
        if chosen_metrics and not no_score:
            try:
                require_ragas()
            except RagasUnavailableError as exc:
                fail(f"{exc} 행동 지표만 보려면 --behavior-only를 쓰세요.")

        started_at = now()
        stamp = stamp_of(started_at)
        directory = out_dir or default_results_dir()
        judge = JudgeOptions(
            judge_model=judge_model,
            embedding_model=embedding_model,
            judge_max_tokens=judge_max_tokens,
            concurrency=concurrency,
            adapt_language=not no_adapt_language,
            base_url=openai_base_url,
        )
        corpus: CorpusSummary | None = None
        try:
            if collect:
                if api_key is None:
                    raise PreconditionError("OPENAI_API_KEY가 없습니다.")
                options = CollectionOptions(
                    queries=queries or default_queries_path(),
                    modes=modes,
                    lang=lang_value(lang),
                    limit=limit,
                    category=category,
                    out=out or directory / f"answers_{stamp}.jsonl",
                    answer_model=answer_model or settings.openai_model,
                    keep_going=keep_going,
                    base_url=openai_base_url,
                    trace=trace,
                    refusal_phrases=phrases,
                )
                collection = asyncio.run(run_collection(settings, api_key, options))
                echo_err(f"답변: {collection.path} ({len(collection.records)}건, 실패 {collection.errors}건)")
                if no_score:
                    raise typer.Exit(code=EXIT_FAILURES if collection.errors else 0)
                answers_path, records, corpus = collection.path, collection.records, collection.corpus
            else:
                if answers is None:
                    raise PreconditionError("--answers가 필요합니다.")
                answers_path = answers
                records = _answer_records(
                    answers_path, modes=modes, lang=lang_value(lang), limit=limit, category=category
                )
            if not records:
                raise PreconditionError("채점할 답변 레코드가 없습니다.")
            chunk_texts = asyncio.run(load_chunk_texts(settings, needed_chunk_ids(records, chosen_metrics)))
        except PreconditionError as exc:
            fail(f"평가를 시작하지 않습니다.\n{exc}")
        except (PoolTimeout, DatabaseError, OSError) as exc:
            fail(f"PostgreSQL에 연결할 수 없습니다: {type(exc).__name__}: {exc}", EXIT_PRECONDITION)

        scorer = build_scorer(api_key, chosen_metrics, judge)
        samples = asyncio.run(
            score_records(records, chosen_metrics, scorer, chunk_texts=chunk_texts, refusal_phrases=phrases)
        )
        aggregates = aggregate_scores(samples, chosen_metrics)

        directory.mkdir(parents=True, exist_ok=True)
        meta = build_meta(
            started_at=started_at.strftime(TIME_FORMAT),
            answers_path=answers_path,
            records=records,
            metrics=chosen_metrics,
            behavior_only=behavior_only,
            judge=judge,
            refusal_phrases_path=refusal_phrases,
            corpus=corpus,
        )
        markdown = render_markdown(
            aggregates, samples, chosen_metrics, title=f"Generation evaluation {stamp}", meta=meta
        )
        base = directory / f"generation_{stamp}"
        write_sample_csv(samples, chosen_metrics, base.with_suffix(".csv"))
        write_summary_csv(aggregates, chosen_metrics, directory / f"generation_{stamp}_summary.csv")
        base.with_suffix(".md").write_text(markdown, encoding="utf-8")

        typer.echo(markdown)
        echo_err(f"결과: {base}.{{csv,md}}, {base}_summary.csv")
        outcome_counts = Counter(sample.outcome for sample in samples)
        echo_err("결과 종류: " + ", ".join(f"{outcome} {outcome_counts.get(outcome, 0)}" for outcome in OUTCOMES))
        generation_errors = sum(1 for sample in samples if sample.generation_error)
        metric_errors = sum(len(sample.errors) for sample in samples)
        if generation_errors or metric_errors:
            echo_err(f"답변 생성 실패 {generation_errors}건, 판정 실패 {metric_errors}건(집계에서 제외, CSV 참고).")
            raise typer.Exit(code=EXIT_FAILURES)
