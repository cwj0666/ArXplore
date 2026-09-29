---
title: psycopg 3 async 저장소, 번호 마이그레이션과 상시 LISTEN
date: 2026-09-29
area: [backend, pipeline]
decision: DB 접근을 psycopg 3 async 풀과 테이블 묶음별 함수 모듈로 옮기되 옛 SQL을 그대로 쓴다. 스키마는 `arxplore migrate`의 번호 SQL(0001 = 옛 ensure_schema, 0002~0006)로만 바꾸고 API·worker는 시작할 때 버전과 벡터 차원을 확인한다(F15). worker 알림은 계속 열려 있는 LISTEN 전용 연결 하나로 받는다(F10)
---

**결정과 근거** — 옛 저장소는 psycopg2 동기 연결을 쓰고, 스키마는 `scripts/migrate_schema.py`와 worker 시작 때의
`ensure_schema`가 만들었다. 새 구현은 API 요청 경로와 DB를 async로 쓰기로 했으므로(계획 1절) DB 계층부터 옮겼다.
- 풀(ec568c1): `AsyncConnectionPool(max_size=POSTGRES_POOL_MAX, timeout=POSTGRES_POOL_TIMEOUT, check=check_connection)`.
  연결마다 `connect_timeout`, `application_name=arxplore-<역할>`, dict 행을 쓴다. `open_pool`은 DB에 닿지 않으면 바로
  실패한다.
- 저장소(edd5024): `arxplore.db.repositories`의 papers, fulltexts, ai_cache, raw, pipeline_state, jobs, embeddings 모듈이다.
  - 함수마다 `ConnectionSource`를 받는다. 풀을 넘기면 호출 하나가 트랜잭션 하나(옛 `get_connection` 블록)이고, 호출자
    연결을 넘기면 그 트랜잭션 안에서 돌고 커밋하지 않는다. 그래서 수집이 raw 저장과 잡 등록을 한 트랜잭션으로 묶고
    NOTIFY는 커밋 때 나간다.
  - SQL은 옛 SQL 그대로다: save_paper의 COALESCE 규칙, 멱등 청크 저장(chunk_index와 md5 목록이 같으면 메타데이터만
    갱신), revision과 changed를 한 문장으로 정하는 raw upsert, 펜싱 토큰·stale reset·backoff·pending_refresh·force 제거를
    가진 큐.
  - 벡터는 옛 구현과 같은 `%.12f` 텍스트 리터럴로 보낸다. `execute_values`는 pipeline 모드의 `executemany`가 됐고, 행은
    `arxplore.domain`의 frozen dataclass로 바꾼다. `target_date`는 `datetime.date`다.
- 마이그레이션 실행기(9d68734): `schema_migrations(version, checksum, applied_at)`에 파일마다 기록하고, 세션 advisory
  lock으로 동시 실행을 막고, 파일 하나를 트랜잭션 하나로 돌린다. 적용 뒤 바뀐 파일, 빌드가 모르는 버전, 이미 적용된
  버전보다 낮은 대기 파일이 있으면 아무것도 적용하기 전에 멈춘다.
  - 0001: 옛 `ensure_schema` 세 개의 DDL을 순서대로 옮겼다. 옛 스키마가 있는 DB에서는 아무것도 바꾸지 않고 버전만
    기록한다. HNSW 생성 실패는 옛 savepoint처럼 경고만 남기고, `arxplore db-status`가 인덱스를 MISSING으로 보여 준다.
  - 0002: 레거시 topics 테이블 3개를 CASCADE 없이 지운다(F16, d3d6007). 옛 `ensure_schema`가 매번 빈 테이블로 다시 만들던
    것이다.
  - 0003: `prepare_jobs.target_date`를 TEXT에서 DATE로 바꾸고 status CHECK를 더한다(F17, 92a3265). YYYY-MM-DD가 아닌 값이나
    모르는 status가 있으면 파일 전체를 되돌린다.
  - 0004 AI 캐시 `prompt_version`(F22, 93aca17), 0005 `scheduler_runs`(57324fa), 0006 `papers.arxiv_lookup_*`(F14, 0ccb57a)
- 스키마 비교(54a3a37, `v2/scripts/compare_schema.py`): 빈 DB에 옛 `scripts/migrate_schema.py`로 만든 스키마와
  `arxplore migrate --target 1`로 만든 스키마의 `pg_dump --schema-only`와 카탈로그가 같다. 0002·0003 뒤의 차이는 F16·F17뿐이다.
  골든 DB의 스키마 덤프를 복원한 DB에서 0001은 아무것도 바꾸지 않았고, 옛 큐 코드는 DATE 컬럼에서도 돌았다.
- F15(cc99987): DDL은 `arxplore migrate`만 실행한다. API와 worker는 시작할 때 `verify_schema`로 대기 마이그레이션, 바뀐
  파일, 모르는 버전, `paper_embeddings.embedding`이 `vector(OPENAI_EMBEDDING_DIMENSIONS)`가 아닌 경우를 확인하고 하나라도
  있으면 뜨지 않는다. 옛 코드는 `VECTOR(1536)`을 박아 두고 설정과 비교하지 않았다.
- F10(cc81248): 옛 `wait_for_prepare_job`은 기다릴 때마다 새 연결을 열어, worker가 일하는 동안 온 알림을 잃었다. claim을
  먼저 하는 루프가 이를 가리고 있었다. `ChannelListener`는 autocommit 전용 연결로 구독을 유지하고, `wait(timeout)`은 이미
  쌓였거나 시간 안에 온 알림을 한 번에 돌려준다. 연결이 끊기면 다시 구독하고 바로 돌아와 호출자가 큐를 다시 보게 한다.
  wait가 취소돼도 같은 연결로 다음 알림을 받는다(803a5ad). worker 루프는 이 연결 하나를 실행 사이에 계속 쓴다(2d626df).
- 차등 비교(8ba9ec5, `v2/scripts/diff_repositories.py`): 옛 저장소(루트 `.venv`, 하위 프로세스)와 새 저장소에 같은
  229단계 시나리오를 돌리고, 단계마다 반환값, 잡 채널의 pg_notify payload, 테이블 9개의 스냅샷을 비교했다.
  - 시나리오는 raw 저장과 revision, pipeline state, 수집 트랜잭션의 커밋·롤백·DB 오류, 큐 전이 전부(펜싱, stale reset,
    backoff, requeue, refresh, force), 동률이 있는 papers upsert와 목록, 멱등 청크 저장, 임베딩, AI 캐시를 다룬다.
  - `NOW()` 시각은 있는지만, backoff는 분 단위로, target_date는 문자열로 비교해 옛 TEXT 컬럼과 새 DATE 컬럼을 맞췄다.
  - 두 번째 패스는 골든 DB를 두 구현이 읽기 전용으로 읽어 결과를 비교했다.
- 옛 통합 테스트 37개(`tests/integration`)가 다루던 큐·raw·청크 동작은 같은 이름의 v2 통합 테스트로 옮겼다(예:
  `test_stale_reclaim_fences_out_zombie_worker`, `test_collect_enqueue_failure_rolls_back_raw_and_sends_no_notification`).

**트레이드오프** — ORM을 쓰지 않고 SQL을 저장소 모듈의 상수와 빌더로 둔다. 옛 SQL과 문장 단위로 대조할 수 있고
검색 SQL의 손 튜닝을 지킬 수 있는 대신, 스키마 변경은 SQL 문자열과 마이그레이션을 함께 고쳐야 한다.
- `ConnectionSource` 하나로 풀과 호출자 연결을 받으면 트랜잭션 경계를 호출자가 정할 수 있다. 대신 함수가 커밋하는지는
  넘긴 인자에 달려 있다.
- 벡터를 이진이 아니라 `%.12f` 텍스트로 보낸다. 요청이 커지지만 pgvector가 옛 구현과 같은 문자열을 해석하므로 G1이
  351/351로 같다.
- 마이그레이션을 시작할 때 자동으로 돌리지 않는다. 배포에 `migrate` 단계가 하나 늘지만 요청 경로와 worker에서 DDL이
  사라지고, 스키마보다 오래된 코드나 차원이 다른 설정으로 뜨는 것을 막는다.
- 번호 SQL은 여러 작업자가 동시에 마이그레이션을 더할 때 번호를 미리 나눠야 한다(사례 2).
- 상시 LISTEN은 worker마다 연결을 하나 더 쓴다(`application_name=arxplore-worker-listener`).

**eval 영향** — 새 측정 지점은 다음과 같다.
- 통합 테스트: `v2/tests/integration/repo/`(저장소, LISTEN), `v2/tests/integration/test_migrate_pg.py`,
  `test_schema_pg.py`(0001 no-op, 0002·0003 변환, `verify_schema`), `test_migration_lock_pg.py`, `test_pool_pg.py`
- 단위 테스트: `v2/tests/unit/test_migrate_plan.py`(번호 규칙, 체크섬, 순서 거부)
- 차등 도구: `v2/scripts/diff_repositories.py`, `v2/scripts/compare_schema.py`
- `arxplore db-status`: 적용 버전, 검색 인덱스 존재, 임베딩 차원, `ready` 줄. CI 통합 잡은 새 DB에 dry-run, 적용, 재실행
  no-op을 거쳐 `ready: yes`인지 본다.

**알려진 한계** — 차등 시나리오는 한 연결로 차례로 도는 조작이라 동시 claim 같은 병행성은 통합 테스트
(`test_concurrent_claims_skip_locked_jobs`, `test_concurrent_distinct_payloads_get_distinct_revisions`)에만 기댄다.
- 차등 비교에서 `NOW()`로 쓰는 시각은 있는지만, backoff는 분 단위로만 맞췄다. 이 값들의 초 단위 이하 차이는 잡지 못한다.
- HNSW 생성 실패는 경고뿐이라 운영자가 `db-status`를 봐야 안다.
- 통합 테스트는 테스트마다 public 스키마를 지우므로 전용 `v2_test*` DB가 있어야 돈다(사례 1).

**트러블슈팅** — 통합 테스트가 다른 작업자의 공용 DB를 지움

- 증상: `V2_TEST_DATABASE_URL` 없이 v2 통합 테스트를 돌리자, 다른 작업자들이 쓰던 공용 DB `v2_test`의 public 스키마가
  지워졌다.
- 원인: 통합 테스트 fixture는 테스트마다 public 스키마를 지우고 다시 만든다. 변수가 없으면 기본값
  `postgresql://arxplore@localhost:55432/v2_test`(ec568c1)로 떨어졌다. 이름이 `v2_test`로 시작하는지는 검사했지만, 기본
  DB 자체가 여러 작업자의 공용이었다.
- 확인 방법: `v2/tests/integration/conftest.py`의 `_configured_url`이 빈 변수에서 기본 URL을 돌려주는 것을 확인했다.
- 해결: 0d7e5d0에서 기본 URL을 없앴다. 변수가 없으면 통합 테스트를 건너뛰고, `V2_TEST_REQUIRE_DB`가 있으면 실패한다.
  DB 이름이 `v2_test`로 시작해야 한다는 검사는 그대로다.
- 재발 방지: 작업자마다 `v2_test_repo_legacy`, `v2_test_api_smoke`처럼 전용 `v2_test*` DB를 명시한다. CI 통합 잡은 변수와
  `V2_TEST_REQUIRE_DB=1`을 함께 줘 DB가 없으면 건너뛰지 않고 실패한다.

사례 2 — 병렬 이식이 마이그레이션을 더하면서 생긴 번호 문제

- 증상: API·LLM 이식(P4·P5)과 파이프라인 이식(P3)이 각자 마이그레이션을 더하자, 패키지 마이그레이션 검사
  (`test_versions_and_names`)가 새 파일마다 실패했다.
- 원인: 검사가 패키지의 마이그레이션 목록 전체를 0001~0003으로 고정하고 있었다. 또 로더는 같은 번호가 둘이면 거부하고,
  실행기는 이미 적용된 버전보다 낮은 대기 파일을 `MigrationOrderError`로 거부한다(9d68734). 두 작업이 번호를 나누지 않고
  각자 다음 번호를 쓰면 한쪽 파일은 적용할 수 없다.
- 확인 방법: 실패한 검사의 기대 목록과, 두 거부를 재현하는 단위 테스트 `test_rejects_duplicate_versions`,
  `test_pending_migration_older_than_applied_one_is_rejected`를 확인했다.
- 해결: af5acfd에서 검사를 앞 세 파일 고정 + 번호 유일·증가·파일 이름 접두어 일치로 바꿨다. 번호는 미리 나눴다. 0004는
  P4·P5(F22, 93aca17), 0005부터는 P3(0005 `scheduler_runs` 57324fa, 0006 F14 0ccb57a)가 쓰고, 0005는 0004가 커밋된 뒤에
  커밋했다.
- 재발 방지: `test_versions_and_names`가 번호 규칙을 검사하고, 실행기는 순서가 어긋난 대기 파일을 적용 전에 거부한다.
