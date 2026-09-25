---
title: prepare_jobs 선점 펜싱과 heartbeat 기반 stale 회수, 재시도 backoff
date: 2026-09-25
area: [pipeline]
decision: prepare_jobs 선점에 claim_generation·worker_id 토큰을 발급해 완료·실패·heartbeat를 토큰으로 제한하고, stale 판정은 heartbeat_at 기준 900초, 재시도는 최대 3회·60초×2^(n-1)·상한 1시간 backoff로 한다
---

**결정과 근거** — prepare 큐는 PostgreSQL 테이블 `prepare_jobs`에 `SKIP LOCKED` 선점과 `LISTEN/NOTIFY`로 만든 큐다. 기준선은
선점 시각(`claimed_at`)만으로 오래된 잡을 되돌렸기 때문에, 오래 걸리는 정상 잡이 다른 워커에게 다시 선점될 수 있었고
원래 워커가 뒤늦게 완료를 기록하면 새 선점을 덮을 수 있었다. 선점할 때 `claim_generation`을 1 올려 `(job_id, worker_id,
claim_generation)` 토큰을 발급하고, 완료·실패·heartbeat는 토큰이 일치할 때만 반영한다. 워커는 논문마다 `heartbeat_at`을
갱신하고, stale 회수는 `heartbeat_at`(없으면 `claimed_at`)이 `PREPARE_JOB_STALE_SECONDS`(기본 900초)보다 오래된 잡만 되돌린다.
실패는 `PREPARE_JOB_MAX_ATTEMPTS`(기본 3)까지 `60초 × 2^(n-1)`(상한 3600초) 뒤 재시도하고, 시도를 다 쓰면 `failed`로 닫는다
(8d20944).

**트레이드오프** — 별도 브로커(Redis, RabbitMQ) 대신 PostgreSQL 한 곳에 큐를 두어 raw 저장과 잡 등록을 한 트랜잭션으로
묶을 수 있다(phase-4). 대신 선점·펜싱·backoff 규칙을 SQL로 직접 유지해야 하고, 토큰 비교가 모든 상태 전이 문장에 들어가
SQL이 길어졌다. 900초는 논문 한 건 처리 시간보다 충분히 길게 잡은 값이라, 워커가 죽은 뒤 잡이 회수되기까지 최대 15분이
걸린다.

**eval 영향** — 단위 `tests/unit/test_prepare_jobs.py`(31개: `test_compute_retry_backoff_seconds`,
`test_heartbeat_is_fenced_by_claim_token`, `test_complete_with_lost_claim_changes_nothing` 등)와 통합
`tests/integration/test_prepare_jobs_pg.py`의 `test_stale_reclaim_fences_out_zombie_worker`,
`test_wrong_claim_token_cannot_complete_or_fail`, `test_stale_reset_uses_heartbeat_not_claimed_at`,
`test_failures_back_off_then_fail_and_requeue_resets`가 실제 PostgreSQL에서 상태 전이를 확인한다.

**알려진 한계** — 토큰을 잃은 워커는 이미 수행한 부수 효과(청크 저장, 임베딩 호출)를 되돌리지 못한다. 재처리 멱등성
(phase-0, phase-3 항목)이 이를 흡수한다. 여러 워커를 동시에 띄운 부하 시험은 하지 않았다.

**트러블슈팅** — 해당 없음
