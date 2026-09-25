"""max_attempts를 소진해 failed가 된 prepare_jobs를 pending으로 되돌린다. 기본은 dry-run.

    python scripts/requeue_failed_prepare_jobs.py --since 2026-04-07 --mode auto          # 대상만 출력
    python scripts/requeue_failed_prepare_jobs.py --since 2026-04-07 --mode auto --apply  # 실제 전환

--apply는 attempt_count를 0, next_attempt_at을 NULL로 초기화하므로 즉시 max_attempts만큼 다시 시도된다.
--force를 함께 주면 job payload에 force를 넣어, 재처리 때 기존 본문보다 낮은 순위의 파싱 결과나
같은 내용이어도 본문·청크를 다시 쓴다(내용이 바뀐 청크의 임베딩은 다시 만들어진다).
출력의 attempt_count는 전환 전 값이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integrations.prepare_job_repository import PrepareJobRepository  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="failed prepare_jobs를 pending으로 재등록한다 (기본 dry-run).")
    parser.add_argument("--since", default=None, help="이 날짜(YYYY-MM-DD) 이상인 target_date만 대상으로 한다.")
    parser.add_argument("--mode", default="auto", help="prepare_jobs.mode 값. 기본 auto.")
    parser.add_argument(
        "--apply", action="store_true", help="지정하면 실제로 pending으로 전환하고 attempt_count를 초기화한다."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="--apply와 함께 쓰면 재처리 시 본문 source 순위·content_hash 검사를 건너뛴다.",
    )
    return parser


def main(argv: list[str] | None = None, *, repository: PrepareJobRepository | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = repository or PrepareJobRepository()
    rows = repository.requeue_failed_prepare_jobs(
        mode=args.mode,
        since_date=args.since,
        dry_run=not args.apply,
        force=args.force,
    )

    header = "[APPLY] pending으로 전환됨" if args.apply else "[DRY-RUN] 전환 대상"
    print(f"{header}: {len(rows)}건 (mode={args.mode}, since={args.since or '-'}, force={args.force})")
    for row in rows:
        print(f"  id={row['id']}\ttarget_date={row['target_date']}\tattempt_count={row['attempt_count']}")
    if not args.apply and rows:
        print("실제로 전환하려면 --apply를 붙여 다시 실행한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
