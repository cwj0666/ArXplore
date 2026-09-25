---
title: PostgreSQL 단일 저장소로 MongoDB 제거
date: 2026-09-25
area: [pipeline, infra]
decision: raw payload와 파이프라인 상태를 MongoDB에서 PostgreSQL raw_daily_papers·pipeline_state로 옮기고, raw 저장과 prepare 잡 등록을 한 트랜잭션으로 묶어 MongoDB 서비스·설정·의존성을 없앤다
---

**결정과 근거** — MongoDB는 수집한 raw payload와 파이프라인 커서만 담고 있었고, 나머지(정제 데이터, 청크, 벡터, 작업 큐,
AI 결과 캐시)는 모두 PostgreSQL에 있었다. 두 저장소에 나뉘어 있어 "raw는 저장됐는데 잡은 등록되지 않은" 상태가 가능했다.
raw는 `raw_daily_papers(source, date, payload JSONB, payload_hash, revision, fetched_count, collected_at)`(기본키
`(source, date)`), 커서는 `pipeline_state(key, value JSONB)`로 옮겼다. 수집 태스크는 raw 저장과 `prepare_jobs` 등록을 한
연결·한 트랜잭션에서 하고, 커밋 뒤에 NOTIFY가 전달된다(d60bbbc). revision 규칙(phase-1)은 한 번의
`INSERT ... ON CONFLICT`로 계산한다. JSONB가 받지 않는 문자(NUL)는 해시 전에 지운다.

**트레이드오프** — 문서형 저장소의 스키마 유연성을 잃지만 raw는 날짜 단위 JSON 한 덩어리라 JSONB 컬럼 하나로 충분했다. 운영할
서비스가 하나 줄고 compose·설정·의존성에서 MongoDB가 빠졌다. 대신 raw와 정제 데이터가 같은 DB 부하를 나눠 쓰고, 규모가 커지면
raw 보관을 분리할지 다시 봐야 한다. 기존 MongoDB 데이터를 옮기는 스크립트는 만들지 않았다(로컬 코퍼스는 새로 수집했다).

**eval 영향** — 통합 `tests/integration/test_raw_store_pg.py`(9개)의 `test_collect_commits_raw_and_job_together_and_notifies`,
`test_collect_enqueue_failure_rolls_back_raw_and_sends_no_notification`, `test_collect_enqueue_database_error_rolls_back_raw`,
`test_nul_characters_are_stored_without_error`, 단위 `test_prepare_refresh.py`의
`test_collect_saves_raw_and_enqueues_in_one_transaction`, `test_save_strips_characters_jsonb_rejects_before_hashing`. 잡 등록이
실패하면 raw 저장도 롤백되고 NOTIFY가 나가지 않는 것을 실제 PostgreSQL에서 확인한다.

**알려진 한계** — 팀 시절 MongoDB에 쌓인 raw 이력은 이 저장소로 이전되지 않는다.

**트러블슈팅** — 해당 없음
