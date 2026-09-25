---
title: raw payload 해시로 revision을 올리고 처리 중 재수집은 pending_refresh로
date: 2026-09-25
area: [pipeline]
decision: 같은 날짜를 다시 수집해도 payload 해시가 같으면 revision을 올리지 않고, 처리 중인 잡에 새 revision이 오면 pending_refresh로 표시해 완료 뒤 한 번 더 처리한다
---

**결정과 근거** — 수집 DAG는 같은 날짜를 여러 번 가져올 수 있다. 매 수집을 새 입력으로 보면 불필요한 재처리가 생기고, 무시하면
HF Daily Papers가 나중에 바꾼 목록을 놓친다. raw payload를 키 순서와 무관하게 직렬화해 해시하고, 해시가 달라질 때만
`revision`을 1 올린다. 잡 등록은 revision이 기존 잡보다 클 때만 `done` 잡을 `pending`으로 되돌리고, `processing` 중인 잡은
`pending_refresh`로 표시해 완료 시 다시 `pending`으로, 실패 시 시도 횟수를 0으로 초기화해 즉시 재시도한다(8d20944). raw
저장소는 phase-4에서 MongoDB에서 PostgreSQL `raw_daily_papers`로 옮겼고, 같은 규칙을 한 번의 `INSERT ... ON CONFLICT`로
계산한다.

**트레이드오프** — 해시 비교는 payload 전체 기준이라 순위·upvote처럼 처리에 영향이 없는 필드가 바뀌어도 revision이 오른다.
필드를 골라 해시하는 대안은 어떤 필드가 처리에 영향을 주는지 계속 관리해야 해서 택하지 않았다. 재처리 비용은 phase-3의
content_hash 비교가 줄인다(본문이 같으면 청크 교체 생략).

**eval 영향** — `tests/unit/test_prepare_refresh.py`의 `test_save_hash_ignores_key_order_so_reordered_payload_is_not_a_new_revision`,
`test_payload_hash_ignores_key_order_but_not_values`, 통합 `test_prepare_jobs_pg.py`의
`test_recollect_refreshes_done_job_only_for_newer_revision`, `test_recollect_during_processing_requeues_once_after_complete`,
`test_recollect_during_processing_gives_fresh_attempts_on_failure`, `test_raw_store_pg.py`의
`test_concurrent_distinct_payloads_get_distinct_revisions`.

**알려진 한계** — revision은 날짜 단위라 한 날짜 안의 논문 하나만 바뀌어도 그 날짜 전체가 재처리 대상이 된다(본문이 같은
논문은 content_hash로 건너뛴다).

**트러블슈팅** — 해당 없음
