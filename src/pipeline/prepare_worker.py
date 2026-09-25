from __future__ import annotations

import argparse
import json
import time
from typing import Any

from src.integrations.paper_repository import PaperRepository
from src.integrations.prepare_job_repository import PrepareJobRepository
from src.pipeline import run_backfill_prepare_papers, run_consume_prepare_queue, run_embed_papers


def _normalize_optional_positive_int(value: int | str | None, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value if value > 0 else default
    normalized = str(value).strip()
    if not normalized:
        return default
    parsed = int(normalized)
    return parsed if parsed > 0 else default


def _collect_prepared_arxiv_ids(prepare_result: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    collected: list[str] = []
    # 일부 논문이 실패해 재시도로 넘어간 날짜도 이미 적재된 논문은 바로 임베딩한다.
    for success in [*prepare_result.get("successes", []), *prepare_result.get("failures", [])]:
        if not isinstance(success, dict):
            continue
        for value in success.get("prepared_arxiv_ids", []):
            arxiv_id = str(value or "").strip()
            if not arxiv_id or arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            collected.append(arxiv_id)
    return collected


def _run_embed_after_prepare(
    *,
    prepare_result: dict[str, Any],
    embed_max_chunks: int,
    embed_backlog_max_chunks: int,
) -> dict[str, Any]:
    arxiv_ids = _collect_prepared_arxiv_ids(prepare_result)

    failures: list[dict[str, str]] = []
    per_arxiv: list[dict[str, Any]] = []
    total_selected = 0
    total_embedded = 0
    embedded_arxiv_count = 0
    backlog_selected_total = 0
    backlog_embedded_total = 0

    for arxiv_id in arxiv_ids:
        selected_sum = 0
        embedded_sum = 0
        try:
            # 한 번에 처리량이 제한되므로 남은 청크가 있으면 같은 논문을 반복 임베딩한다.
            for _ in range(100):
                embed_result = run_embed_papers(
                    runtime="local",
                    user="local_prepare_worker",
                    max_chunks=embed_max_chunks,
                    arxiv_id=arxiv_id,
                )
                status = str(embed_result.get("status") or "")
                selected = int(embed_result.get("selected_chunk_count", 0) or 0)
                embedded = int(embed_result.get("embedded_chunk_count", 0) or 0)
                selected_sum += selected
                embedded_sum += embedded

                if status == "no_op" or selected < embed_max_chunks:
                    break
            else:
                failures.append({"arxiv_id": arxiv_id, "error": "embed_loop_limit_exceeded"})
                continue
        except Exception as exc:
            failures.append({"arxiv_id": arxiv_id, "error": str(exc)})
            continue

        total_selected += selected_sum
        total_embedded += embedded_sum
        if embedded_sum > 0:
            embedded_arxiv_count += 1
        per_arxiv.append(
            {
                "arxiv_id": arxiv_id,
                "selected_chunk_count": selected_sum,
                "embedded_chunk_count": embedded_sum,
            }
        )

    backlog_error: str | None = None
    normalized_backlog_max_chunks = max(0, int(embed_backlog_max_chunks))
    if normalized_backlog_max_chunks > 0:
        remaining_backlog_budget = normalized_backlog_max_chunks
        try:
            while remaining_backlog_budget > 0:
                current_limit = min(embed_max_chunks, remaining_backlog_budget)
                backlog_result = run_embed_papers(
                    runtime="local",
                    user="local_prepare_worker",
                    max_chunks=current_limit,
                    arxiv_id=None,
                )
                backlog_status = str(backlog_result.get("status") or "")
                backlog_selected = int(backlog_result.get("selected_chunk_count", 0) or 0)
                backlog_embedded = int(backlog_result.get("embedded_chunk_count", 0) or 0)
                backlog_selected_total += backlog_selected
                backlog_embedded_total += backlog_embedded

                if backlog_status == "no_op" or backlog_selected <= 0 or backlog_embedded <= 0:
                    break
                if backlog_selected < current_limit:
                    break
                remaining_backlog_budget -= backlog_selected
        except Exception as exc:
            backlog_error = f"{type(exc).__name__}: {exc}"

    if failures and embedded_arxiv_count == 0:
        status = "failed"
    elif failures or backlog_error:
        status = "partial_failed"
    elif total_embedded == 0 and backlog_embedded_total == 0:
        status = "no_op"
    else:
        status = "success"

    return {
        "stage": "embed_after_prepare",
        "status": status,
        "target_arxiv_count": len(arxiv_ids),
        "embedded_arxiv_count": embedded_arxiv_count,
        "selected_chunk_count": total_selected,
        "embedded_chunk_count": total_embedded,
        "backlog_selected_chunk_count": backlog_selected_total,
        "backlog_embedded_chunk_count": backlog_embedded_total,
        "backlog_error": backlog_error,
        "per_arxiv": per_arxiv,
        "failures": failures,
    }


def _run_once(args: argparse.Namespace) -> dict[str, Any]:
    normalized_worker_id = (args.worker_id or args.state_name or "").strip() or "default"
    normalized_max_papers = (args.max_papers or "").strip() or None
    normalized_embed_max_chunks = _normalize_optional_positive_int(args.embed_max_chunks, default=200)
    normalized_embed_backlog_max_chunks = _normalize_optional_positive_int(args.embed_backlog_max_chunks, default=0)

    if args.mode == "auto":
        prepare_result = run_consume_prepare_queue(
            runtime="local",
            user="local_prepare_worker",
            mode="auto",
            worker_id=normalized_worker_id,
            max_jobs_per_run=max(1, int(args.max_jobs_per_run)),
            max_papers=normalized_max_papers,
            force=bool(args.force),
        )
        if args.skip_embed:
            prepare_result["embed"] = {
                "stage": "embed_after_prepare",
                "status": "skipped",
                "reason": "skip_embed_enabled",
            }
            return prepare_result

        # prepare 성공이 없어도 backlog 임베딩은 예산만큼 진행한다.
        try:
            embed_result = _run_embed_after_prepare(
                prepare_result=prepare_result,
                embed_max_chunks=normalized_embed_max_chunks,
                embed_backlog_max_chunks=normalized_embed_backlog_max_chunks,
            )
        except Exception as exc:
            embed_result = {
                "stage": "embed_after_prepare",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        prepare_result["embed"] = embed_result

        embed_status = str(embed_result.get("status") or "")
        prepare_status = str(prepare_result.get("status") or "")
        if embed_status == "failed" and prepare_status not in {"failed"}:
            prepare_result["status"] = "failed"
        elif embed_status == "partial_failed" and prepare_status == "success":
            prepare_result["status"] = "partial_failed"
        return prepare_result

    return run_backfill_prepare_papers(
        runtime="local",
        user="local_prepare_worker",
        cursor_date=(args.cursor_date or "").strip() or None,
        oldest_date=(args.oldest_date or "").strip() or None,
        batch_days=max(1, int(args.batch_days)),
        state_name=normalized_worker_id,
        max_papers=normalized_max_papers,
        force=bool(args.force),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="로컬 GPU에서 prepare 워커를 실행한다.")
    parser.add_argument(
        "--mode",
        choices=["auto", "backfill"],
        default="auto",
        help="auto는 신규 수집분 자동 추적, backfill은 과거 날짜 수동 배치 처리다.",
    )
    parser.add_argument("--max-jobs-per-run", type=int, default=1, help="auto 모드에서 한 run에 소비할 큐 작업 수.")
    parser.add_argument("--cursor-date", default="", help="시작 cursor 날짜(YYYY-MM-DD). 비우면 저장된 state를 사용한다.")
    parser.add_argument("--oldest-date", default="", help="종료 기준 날짜(YYYY-MM-DD). 비우면 state 또는 기본값을 사용한다.")
    parser.add_argument("--batch-days", type=int, default=3, help="한 번에 처리할 날짜 수. 기본값은 3이다.")
    parser.add_argument("--max-papers", default="", help="날짜당 최대 논문 수. 비우면 해당 날짜 전체를 처리한다.")
    parser.add_argument(
        "--embed-max-chunks",
        type=int,
        default=200,
        help="auto 모드에서 arXiv별 임베딩 배치 크기. 기본값은 200이다.",
    )
    parser.add_argument(
        "--embed-backlog-max-chunks",
        type=int,
        default=400,
        help="auto 모드에서 run마다 추가로 태울 backlog 임베딩 최대 청크 수(신규 prepare 성공 여부와 무관). 기본값은 400이고 0이면 끈다.",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="auto 모드에서 prepare 성공 후 자동 임베딩을 비활성화한다.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="본문 source 순위·content_hash 검사를 건너뛰고 파싱 결과로 본문·청크를 교체한다.",
    )
    parser.add_argument("--worker-id", default="default", help="큐 작업 선점에 사용하는 워커 식별자.")
    parser.add_argument("--state-name", default="", help="deprecated: --worker-id를 사용한다.")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="활성화하면 completed 상태가 될 때까지 반복 실행한다.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=60.0,
        help="backfill loop 실행 시 run 사이 대기 시간(초). 기본값은 60초다.",
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=120.0,
        help="auto loop 실행 시 새 작업 알림을 기다릴 최대 시간(초). 기본값은 120초다.",
    )
    args = parser.parse_args()
    PaperRepository().ensure_schema()
    prepare_job_repository = PrepareJobRepository()
    prepare_job_repository.ensure_schema()

    while True:
        result = _run_once(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))

        if not args.loop:
            return 0 if result.get("status") != "failed" else 1

        status = str(result.get("status") or "")
        if args.mode == "auto":
            if status == "no_op":
                prepare_job_repository.wait_for_prepare_job(timeout_seconds=float(args.wait_timeout_seconds))
                continue
            if status == "failed":
                prepare_job_repository.wait_for_prepare_job(timeout_seconds=min(float(args.wait_timeout_seconds), 30.0))
                continue
            continue

        if status in {"completed", "no_op"}:
            return 0
        if status == "failed":
            return 1
        time.sleep(max(0.0, float(args.sleep_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
