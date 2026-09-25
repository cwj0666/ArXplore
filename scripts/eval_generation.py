"""생성 품질 평가(RAGAS): 에이전트·상세 챗 답변을 모으고 LLM 판정으로 채점한다.

    python scripts/eval_generation.py --collect --limit 5                      # 수집 후 채점 (에이전트 + 상세 챗)
    python scripts/eval_generation.py --collect --mode agent --no-score        # 에이전트 답변만 수집
    python scripts/eval_generation.py --answers eval/results/answers_X.jsonl   # 기존 답변 파일만 채점
    python scripts/eval_generation.py --answers FILE --metrics faithfulness,answer_relevancy
    python scripts/eval_generation.py --answers FILE --behavior-only           # 행동 지표만(키·DB 불필요)
    python scripts/eval_generation.py --collect --queries eval/queries.cases.jsonl --category safety --keep-going

OPENAI_API_KEY(서버 키)가 답변 생성과 판정에 모두 필요하다(--behavior-only로 기존 답변만 채점할 때는 불필요).
--queries는 JSONL 파일 또는 *.jsonl이 모인 디렉터리. 질의의 mode 힌트가 허용하지 않는 (질의, 모드) 조합은 건너뛰고,
제품 API가 거부할 입력(빈 문자열, 4,000자 초과)은 생성하지 않고 rejected 결과로 기록한다. --collect는 PostgreSQL이 필요하고, 기존 답변 파일 채점은
context_precision 기준으로 정답 청크 본문을 읽어야 할 때만 DB에 붙는다. 키가 없거나 DB에 연결할 수 없거나 질의셋이
DB와 맞지 않으면 아무것도 쓰지 않고 종료 코드 2로 끝난다.
결과: eval/results/answers_<timestamp>.jsonl(답변), generation_<timestamp>.csv(질의별),
generation_<timestamp>_summary.csv(집계), generation_<timestamp>.md(README용 표).
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.answers import (  # noqa: E402
    AnswersError,
    answer_key,
    append_answer_record,
    build_product_generators,
    collect_answers,
    completed_keys,
    load_answer_records,
    load_reference_answers,
    resolve_modes,
)
from eval.behavior import DEFAULT_REFUSAL_PHRASES, OUTCOMES, load_refusal_phrases  # noqa: E402
from eval.dataset import DatasetError, EvalQuery, dataset_digest, find_placeholders, load_queries  # noqa: E402
from eval.generation import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    METRICS,
    MetricRequest,
    RagasScorer,
    Scorer,
    aggregate_scores,
    render_markdown,
    resolve_metrics,
    score_records,
    write_sample_csv,
    write_summary_csv,
)

EXIT_PRECONDITION = 2
PLACEHOLDER_KEY_PREFIX = "change-me"


class PreconditionError(RuntimeError):
    pass


class NullScorer:
    """판정 요청을 받지 않는 점수기(--behavior-only). 요청이 오면 오류."""

    def score(self, requests: list[MetricRequest]) -> list[float | None | BaseException]:
        if requests:
            raise RuntimeError("--behavior-only에서는 RAGAS 판정 요청을 만들지 않습니다.")
        return []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--collect", action="store_true", help="질의셋으로 답변을 수집한 뒤 채점")
    source.add_argument("--answers", default=None, help="기존 답변 JSONL을 채점만 한다(수집 생략)")
    parser.add_argument("--queries", default=str(REPO_ROOT / "eval" / "queries.jsonl"))
    parser.add_argument("--mode", default="both", help="agent, paper_chat 또는 both(쉼표 구분 가능)")
    parser.add_argument("--metrics", default="all", help="쉼표 구분. 사용 가능: " + ", ".join(METRICS) + ", all")
    parser.add_argument("--limit", type=int, default=None, help="파일 순서대로 앞 N개 질의(또는 답변 레코드)만")
    parser.add_argument("--lang", choices=("ko", "en"), default=None)
    parser.add_argument("--category", default=None, help="쉼표 구분 케이스 카테고리만(eval/cases 참고)")
    parser.add_argument(
        "--behavior-only", action="store_true", help="RAGAS 판정 없이 행동 지표만 계산(기존 답변 채점은 키 불필요)"
    )
    parser.add_argument("--refusal-phrases", default=None, help="거절 문구 파일(한 줄에 하나). 기본은 내장 목록")
    parser.add_argument("--out", default=None, help="답변 JSONL 경로. 이미 있으면 끝난 (id, mode)는 건너뛰고 이어 쓴다")
    parser.add_argument("--no-score", action="store_true", help="--collect와 함께: 수집만 하고 채점하지 않는다")
    parser.add_argument("--answer-model", default=None, help="답변 생성 모델. 기본은 OPENAI_MODEL")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="answer_relevancy용 임베딩 모델")
    parser.add_argument(
        "--judge-max-tokens", type=int, default=None, help="판정 LLM 출력 한도(기본: gpt-5·o 계열 8192)"
    )
    parser.add_argument("--concurrency", type=int, default=4, help="동시에 보내는 판정 요청 수")
    parser.add_argument(
        "--no-adapt-language", action="store_true", help="한국어 질의에도 영어 few-shot 예시 그대로 판정"
    )
    parser.add_argument("--keep-going", action="store_true", help="수집 중 질의 단위 예외를 error 필드에 남기고 계속")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "eval" / "results"))
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.no_score and not args.collect:
        parser.error("--no-score는 --collect와 함께만 씁니다")
    return args


def parse_categories(value: str | None) -> set[str] | None:
    if value is None:
        return None
    names = {name.strip() for name in value.split(",") if name.strip()}
    return names or None


def select_queries(
    queries: list[EvalQuery], *, lang: str | None, limit: int | None, categories: set[str] | None = None
) -> list[EvalQuery]:
    selected = [
        query
        for query in queries
        if (lang is None or query.lang == lang) and (categories is None or query.category in categories)
    ]
    return selected[:limit] if limit is not None else selected


def select_records(
    records: list[dict[str, Any]],
    *,
    modes: list[str],
    lang: str | None,
    limit: int | None,
    categories: set[str] | None = None,
) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if record["mode"] in modes
        and (lang is None or record.get("lang") == lang)
        and (categories is None or str(record.get("category") or "") in categories)
    ]
    return selected[:limit] if limit is not None else selected


def paper_chat_targets(queries: list[EvalQuery], modes: list[str]) -> list[str]:
    """상세 챗으로 실행할 질의가 열어 둘 논문 id(입력 거부 기대 케이스 제외)."""
    if "paper_chat" not in modes:
        return []
    return sorted(
        {
            query.paper_chat_arxiv_id
            for query in queries
            if query.supports_mode("paper_chat") and not query.expect_rejected and query.paper_chat_arxiv_id
        }
    )


def placeholder_problems(queries: list[EvalQuery]) -> list[str]:
    """자리표시자 arxiv_id(`0000.0000a` 형식)나 제목 템플릿이 남은 질의 id."""
    return [
        query.id
        for query in queries
        if find_placeholders(
            [
                query.query,
                query.open_arxiv_id,
                *query.relevant_arxiv_ids,
                *query.must_mention_arxiv_ids,
                *(content for _, content in query.history),
            ]
        )
    ]


def usable_api_key(settings: Any) -> str | None:
    key = str(getattr(settings, "openai_api_key", None) or "").strip()
    if not key or key.startswith(PLACEHOLDER_KEY_PREFIX):
        return None
    return key


def _connect(settings: Any):
    import psycopg2

    from src.shared import build_postgres_connection_params

    params = build_postgres_connection_params(settings)
    return psycopg2.connect(connect_timeout=5, **params)


def _open(settings: Any):
    try:
        return _connect(settings)
    except Exception as exc:
        raise PreconditionError(f"PostgreSQL에 연결할 수 없습니다: {type(exc).__name__}: {exc}") from exc


def preflight(settings: Any, queries: list[EvalQuery], *, modes: list[str]) -> dict[str, Any]:
    """DB 연결, 코퍼스 규모, 상세 챗 대상 논문 존재 여부를 확인한다. 조건이 맞지 않으면 PreconditionError."""
    connection = _open(settings)
    paper_ids = paper_chat_targets(queries, modes)
    try:
        with connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM papers),
                    (SELECT COUNT(*) FROM paper_chunks),
                    (SELECT COUNT(*) FROM paper_embeddings)
                """
            )
            papers, chunks, embeddings = cursor.fetchone()
            found: set[str] = set()
            if paper_ids:
                cursor.execute("SELECT arxiv_id FROM papers WHERE arxiv_id = ANY(%s)", (paper_ids,))
                found = {row[0] for row in cursor.fetchall()}
    except Exception as exc:
        raise PreconditionError(f"사전 점검 쿼리가 실패했습니다: {type(exc).__name__}: {exc}") from exc
    finally:
        connection.close()

    problems: list[str] = []
    missing = [value for value in paper_ids if value not in found]
    if missing:
        problems.append(f"상세 챗 대상 arxiv_id {len(missing)}개가 papers에 없습니다: {missing[:10]}")
    if not chunks:
        problems.append("paper_chunks가 비어 있습니다.")
    if problems:
        raise PreconditionError("\n".join(problems))
    return {"papers": int(papers), "chunks": int(chunks), "embeddings": int(embeddings)}


def needed_chunk_ids(records: list[dict[str, Any]], metrics: list[str]) -> list[int]:
    """context_precision 기준으로 정답 청크 본문이 필요한 chunk_id(reference_answer가 없는 레코드만)."""
    if "context_precision" not in metrics:
        return []
    ids = {
        int(chunk_id)
        for record in records
        if not str(record.get("reference_answer") or "").strip()
        for chunk_id in record.get("relevant_chunk_ids") or []
    }
    return sorted(ids)


def fetch_chunk_texts(settings: Any, chunk_ids: list[int]) -> dict[int, str]:
    if not chunk_ids:
        return {}
    connection = _open(settings)
    try:
        with connection, connection.cursor() as cursor:
            cursor.execute("SELECT id, chunk_text FROM paper_chunks WHERE id = ANY(%s)", (chunk_ids,))
            return {int(row[0]): str(row[1] or "") for row in cursor.fetchall()}
    except Exception as exc:
        raise PreconditionError(f"정답 청크 조회가 실패했습니다: {type(exc).__name__}: {exc}") from exc
    finally:
        connection.close()


def build_scorer(args: argparse.Namespace, api_key: str, metrics: list[str]) -> Scorer:
    def progress(done: int, total: int, request: MetricRequest) -> None:
        print(f"[judge {done}/{total}] {request.metric}", file=sys.stderr)

    return RagasScorer(
        api_key=api_key,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        metrics=metrics,
        adapt_korean=not args.no_adapt_language,
        concurrency=args.concurrency,
        max_tokens=args.judge_max_tokens,
        progress=progress,
    )


def git_revision() -> str:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}{' (dirty)' if dirty else ''}"


def file_digest(path: Path) -> str:
    if path.is_dir():
        return dataset_digest(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def build_meta(
    *,
    started_at: datetime,
    answers_path: Path,
    records: list[dict[str, Any]],
    metrics: list[str],
    args: argparse.Namespace,
    corpus: dict[str, Any] | None,
) -> dict[str, str]:
    langs = Counter(str(record.get("lang")) for record in records)
    modes = Counter(str(record.get("mode")) for record in records)
    sources = Counter(str(record.get("source")) for record in records)
    answer_models = sorted({str(record.get("model")) for record in records if record.get("model")}) or ["unknown"]
    references = sum(1 for record in records if str(record.get("reference_answer") or "").strip())
    if args.behavior_only:
        judge = "RAGAS 생략(--behavior-only), 행동 지표만"
    else:
        judge = (
            f"ragas, judge={args.judge_model}, embeddings={args.embedding_model}, metrics={', '.join(metrics)}, "
            f"한국어 예시 번역={'off' if args.no_adapt_language else 'on'}"
        )
    meta = {
        "실행 시각": started_at.strftime("%Y-%m-%d %H:%M:%S"),
        "커밋": git_revision(),
        "답변 파일": f"`{answers_path.name}` sha256:{file_digest(answers_path)} — {len(records)}건 "
        f"({', '.join(f'{key} {value}' for key, value in sorted(modes.items()))}; "
        f"ko {langs.get('ko', 0)} / en {langs.get('en', 0)}; "
        f"{', '.join(f'{key} {value}' for key, value in sorted(sources.items()))}; reference_answer {references})",
        "답변 모델": ", ".join(answer_models),
        "판정": judge,
        "거절 문구": f"`{Path(args.refusal_phrases).name}`" if args.refusal_phrases else "내장 목록",
    }
    categories = Counter(str(record.get("category")) for record in records if record.get("category"))
    if categories:
        meta["케이스 카테고리"] = ", ".join(f"{key} {value}" for key, value in sorted(categories.items()))
    if corpus is not None:
        meta["코퍼스"] = f"papers {corpus['papers']}, chunks {corpus['chunks']}, embeddings {corpus['embeddings']}"
    return meta


def run_collection(
    args: argparse.Namespace,
    settings: Any,
    api_key: str,
    modes: list[str],
    stamp: str,
    refusal_phrases: list[str] | tuple[str, ...] = DEFAULT_REFUSAL_PHRASES,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any], int]:
    queries_path = Path(args.queries)
    try:
        queries = select_queries(
            load_queries(queries_path),
            lang=args.lang,
            limit=args.limit,
            categories=parse_categories(args.category),
        )
        reference_answers = load_reference_answers(queries_path)
    except FileNotFoundError as exc:
        raise PreconditionError(
            f"질의셋이 없습니다: {queries_path}. scripts/eval_build_queries.py로 먼저 만드세요."
        ) from exc
    except (DatasetError, ValueError) as exc:
        raise PreconditionError(f"질의셋 형식 오류: {exc}") from exc
    if not queries:
        raise PreconditionError("실행할 질의가 없습니다.")
    unresolved = placeholder_problems(queries)
    if unresolved:
        raise PreconditionError(
            f"자리표시자 id가 남은 질의 {len(unresolved)}개: {unresolved[:10]}. "
            "scripts/eval_build_queries.py --attach-ids로 실제 id를 채운 파일을 쓰세요."
        )

    corpus = preflight(settings, queries, modes=modes)

    out_path = Path(args.out) if args.out else Path(args.out_dir) / f"answers_{stamp}.jsonl"
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        try:
            done = completed_keys(load_answer_records(out_path))
        except AnswersError as exc:
            raise PreconditionError(f"이어 쓸 답변 파일 형식 오류: {exc}") from exc
        print(f"{out_path}에서 이어서 수집합니다(완료 {len(done)}건 건너뜀).", file=sys.stderr)

    from src.shared import override_openai_runtime

    answer_model = args.answer_model or settings.openai_model

    def progress(done_count: int, total: int, mode: str, query: EvalQuery, record: dict[str, Any]) -> None:
        if record.get("error"):
            status = "error"
        else:
            status = f"{record.get('outcome')}, {len(record['contexts'])} contexts"
        print(f"[answer {done_count}/{total}] {mode:<10} {query.id} ({status})", file=sys.stderr)

    with override_openai_runtime(api_key=api_key, model=answer_model):
        collect_answers(
            queries,
            modes,
            build_product_generators(modes),
            done=done,
            keep_going=args.keep_going,
            reference_answers=reference_answers,
            model=answer_model,
            sink=lambda record: append_answer_record(out_path, record),
            progress=progress,
            refusal_phrases=refusal_phrases,
        )

    wanted = {(query.id, mode) for query in queries for mode in modes if query.supports_mode(mode)}
    records = [record for record in load_answer_records(out_path) if answer_key(record) in wanted]
    errors = sum(1 for record in records if record.get("error"))
    return out_path, records, corpus, errors


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        modes = resolve_modes(args.mode.split(","))
        metrics = [] if args.behavior_only else resolve_metrics(args.metrics.split(","))
        refusal_phrases = (
            load_refusal_phrases(args.refusal_phrases) if args.refusal_phrases else list(DEFAULT_REFUSAL_PHRASES)
        )
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_PRECONDITION

    try:
        from src.shared import get_settings

        settings = get_settings()
    except Exception as exc:
        print(f"설정을 불러올 수 없습니다(.env 확인): {exc}", file=sys.stderr)
        return EXIT_PRECONDITION
    api_key = usable_api_key(settings)
    if api_key is None and (args.collect or not args.behavior_only):
        print("OPENAI_API_KEY가 없습니다. 답변 생성과 RAGAS 판정 모두 서버 키가 필요합니다.", file=sys.stderr)
        return EXIT_PRECONDITION

    started_at = datetime.now()
    stamp = started_at.strftime("%Y%m%d-%H%M%S")
    corpus: dict[str, Any] | None = None
    collection_errors = 0
    try:
        if args.collect:
            answers_path, records, corpus, collection_errors = run_collection(
                args, settings, str(api_key), modes, stamp, refusal_phrases
            )
            print(f"답변: {answers_path} ({len(records)}건, 실패 {collection_errors}건)", file=sys.stderr)
            if args.no_score:
                return 1 if collection_errors else 0
        else:
            answers_path = Path(args.answers)
            try:
                records = select_records(
                    load_answer_records(answers_path),
                    modes=modes,
                    lang=args.lang,
                    limit=args.limit,
                    categories=parse_categories(args.category),
                )
            except FileNotFoundError as exc:
                raise PreconditionError(f"답변 파일이 없습니다: {answers_path}") from exc
            except AnswersError as exc:
                raise PreconditionError(f"답변 파일 형식 오류: {exc}") from exc
        if not records:
            raise PreconditionError("채점할 답변 레코드가 없습니다.")
        chunk_texts = fetch_chunk_texts(settings, needed_chunk_ids(records, metrics))
    except PreconditionError as exc:
        print(f"평가를 시작하지 않습니다.\n{exc}", file=sys.stderr)
        return EXIT_PRECONDITION

    scorer: Scorer = NullScorer() if not metrics else build_scorer(args, str(api_key), metrics)
    samples = score_records(records, metrics, scorer, chunk_texts=chunk_texts, refusal_phrases=refusal_phrases)
    aggregates = aggregate_scores(samples, metrics)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = build_meta(
        started_at=started_at, answers_path=answers_path, records=records, metrics=metrics, args=args, corpus=corpus
    )
    markdown = render_markdown(aggregates, samples, metrics, title=f"Generation evaluation {stamp}", meta=meta)
    base = out_dir / f"generation_{stamp}"
    write_sample_csv(samples, metrics, base.with_suffix(".csv"))
    write_summary_csv(aggregates, metrics, out_dir / f"generation_{stamp}_summary.csv")
    base.with_suffix(".md").write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"결과: {base}.{{csv,md}}, {base}_summary.csv", file=sys.stderr)
    outcome_counts = Counter(sample.outcome for sample in samples)
    print(
        "결과 종류: " + ", ".join(f"{outcome} {outcome_counts.get(outcome, 0)}" for outcome in OUTCOMES),
        file=sys.stderr,
    )
    generation_errors = sum(1 for sample in samples if sample.generation_error)
    metric_errors = sum(len(sample.errors) for sample in samples)
    if generation_errors or metric_errors:
        print(
            f"답변 생성 실패 {generation_errors}건, 판정 실패 {metric_errors}건(집계에서 제외, CSV 참고).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
