---
title: Airflow를 cron 스케줄러로 바꾸고 prepare 파이프라인 수정
date: 2026-09-29
area: [pipeline, infra]
decision: Airflow DAG 3개를 `arxplore scheduler`의 cron 작업(실행 이력 테이블, advisory lock, 직전 슬롯 1회 catch-up)으로 바꾸고, 수집·prepare·임베딩·worker를 옛 동작 그대로 옮긴 뒤 F07~F09·F11~F14·F30과 논문 처리 중 heartbeat를 얹는다. layout 본문이 확정된 논문은 파싱 전에 건너뛰고(F08), 코퍼스 전환은 dry-run이 기본인 rechunk·reparse 도구로 한다
---

**결정과 근거** — Airflow는 DAG 3개, 하루 몇 번의 HTTP 호출에 서버 컨테이너 4개와 시크릿 3종을 쓰고 있었다(01번 항목).
작업은 그대로 두고 실행기만 가벼운 것으로 바꿨다.

이식(4f6b447, 0aab1db, 2d626df, facbe2d)
- `prepare_papers.py`를 mapping(HF raw 항목 → 후보), policy(source 순위, content_hash, 초록 보정, 저장 판단 순수 함수),
  paper(저장, 다운로드, 파싱, 청크, 해시 순서), run(날짜 실행과 큐 소비)으로 나눴다. 옛 worker와 같은 raw 날짜(G3 PDF
  10편과 기록된 HURIDOCS 응답, 초록 폴백 2편, 중복 id)를 첫 실행, 변화 없는 재실행, 강제 재실행으로 돌려 papers, fulltexts,
  청크 736개와 id, 임베딩 623개, prepare_jobs, worker 출력이 같았다.
- 임베딩 루프는 옛 한도에 이름을 붙였다: 라운드당 200청크(references 제외), 논문당 최대 100라운드, backlog 예산 400청크.
  worker는 no_op 뒤 대기, 실패 뒤 최대 30초 대기, 그 밖에는 바로 다시 claim하는 옛 루프 규칙을 따르고, 시작할 때 DDL 대신
  스키마 버전과 벡터 차원을 확인하며(F15), LISTEN 연결 하나를 실행 사이에 계속 쓴다(F10). 임베딩 요청 15회까지 옛 worker와
  같았다.
- 수집, 백필, enrich, LangSmith 정리는 옛 기본값 그대로 옮겼다. 수집은 HTTP 요청을 트랜잭션 전에 끝내고, raw upsert와
  prepare 잡 등록을 한 트랜잭션으로 묶어 NOTIFY가 커밋 때 나간다.

스케줄러(57324fa): `arxplore scheduler`가 `SCHEDULER_TIMEZONE`(기본 Asia/Seoul)에서 작업 세 개를 돌린다.
- collect `0 18 * * *`, maintenance `0 */3 * * *`(백필 뒤 enrich, 백필이 실패해도 enrich는 돌고 실행은 실패로 남는다),
  langsmith_cleanup `0 3 * * *`
- 작업마다 다음 croniter 시각까지 최대 60초씩 잔다. 시작할 때 직전 슬롯의 실행 기록이 없으면 그 슬롯을 한 번 돌린다
  (Airflow `catchup=False`와 같다). 예약 collect는 자기 슬롯의 날짜를 수집한다.
- 마이그레이션 0005의 `scheduler_runs`에 작업, trigger(scheduled·manual), 예정 시각, 상태, 파라미터, 결과, 오류, 시도 횟수,
  시작·끝 시각을 남긴다. 예약 실행은 (job, scheduled_for) 유일 인덱스로 한 슬롯을 두 번 돌리지 않고, 세션 advisory lock으로
  스케줄러 하나만 돈다. 멈춘 스케줄러가 남긴 running 행은 시작할 때 abandoned로 바꾼다.
- `arxplore run collect|maintenance|langsmith-cleanup`이 옛 DAG params를 옵션으로 받고 manual 실행으로 기록된다.
  `arxplore scheduler history`가 최근 실행을 보여 준다.

수정

| ID | 커밋 | 내용 |
| --- | --- | --- |
| F07 | 0a3451b | prepare가 파싱 전에 HF 메타데이터를 저장하면서 enrich가 arXiv에서 채운 title, authors, abstract, pdf_url, published_at, updated_at, source를 덮었다. 행이 `hf_daily_papers_raw`로 돌아가 enrich가 3시간마다 다시 집었다. 이제 `hf_daily_papers+arxiv` 행의 이 필드들은 HF upsert가 바꾸지 않고, upvotes·GitHub 필드는 HF가 계속 갱신한다. 논문 저장은 파싱 뒤, 본문 저장 직전에 한다 |
| F08 | 32e087a | 저장된 본문이 final(source layout_pdf, 청크 있음, content_hash 있음)이면 force가 아닐 때 다운로드·파싱 없이 papers 행(HF 신호)만 갱신하고 `parse_skipped`로 보고한다. pdf·초록 폴백 본문은 매번 다시 파싱해 나중의 layout 결과로 바뀔 수 있게 한다 |
| F09 | e4c26cc | 백필이 수집한 날짜를 raw 행과 같은 트랜잭션에서 mode=backfill 잡으로 등록한다. worker는 auto 잡을 먼저, `PREPARE_BACKFILL_ENABLED=true`일 때만 backfill 잡을 가져가 같은 heartbeat·펜싱·임베딩을 거친다. 옛 `--mode backfill`은 자기 커서로 raw를 돌며 임베딩을 하지 않았다 |
| F11 | 1b0b198 | HF 페이지 HTML의 data-props 대신 `/api/daily_papers?date=` JSON을 User-Agent `arxplore/2.0`와 재시도(최대 3회, 2·4초 또는 Retry-After 60초까지)로 받는다. 골든 raw의 2026-09-11·15·22(26·33·34편)와 비교해 prepare 매핑 결과(id, 제목, 저자, 초록, publishedAt, PDF·GitHub URL)가 같았고, 토요일 09-12는 빈 목록이었다 |
| F12 | 3330ec4 | collect가 실패하면 10분 간격으로 3번 더 시도한다. 실행 행에 시도 횟수와 마지막 오류가 남는다 |
| F13 | c5a00a0 | `LANGSMITH_API_KEY`가 없으면 LangSmith 정리를 skipped로 끝낸다. `--limit`은 살펴본 run 수가 아니라 삭제 대상 수를 제한한다 |
| F14 | 0ccb57a | arXiv가 돌려주지 않은 논문은 1·2·4·8·16·30일(최대 30일) 뒤에만 다시 조회한다(마이그레이션 0006). rate limit·타임아웃으로 실패한 조회는 기록하지 않는다 |
| F30 | 9dfa9b4 | 수집 날짜를 컨테이너 시간대 대신 `SCHEDULER_TIMEZONE`으로 계산한다. UTC 컨테이너의 Airflow는 00:00~09:00 KST 실행에서 전날 목록을 수집했다 |

- 논문 처리 중 heartbeat(ab8c2c6): 논문 하나를 처리하는 동안 `min(60초, stale 임계 / 3)`마다 heartbeat를 보낸다(05번 항목
  사례 2).
- 코퍼스 전환 도구(5aa3237): F01~F03은 청크를 바꾸지만 content_hash는 그대로고, F08 뒤로 final인 layout 논문은 다시 파싱하지
  않는다. 그래서 dry-run이 기본인 명령 두 개를 만들었다.
  - `arxplore run rechunk [--arxiv-id ...] [--apply]`: 저장된 sections로 청크를 다시 만들어 바뀌는 논문 수, 청크 전후,
    지워질 임베딩, 재임베딩 토큰 추정(references 제외 청크의 len // 4)을 보고한다. `--apply`는 바뀐 청크 묶음만 바꾸고 그
    임베딩은 cascade로 지워 worker backlog가 다시 만든다. 텍스트가 같은 청크는 id와 임베딩을 지킨다.
  - `arxplore run reparse [--since] [--until] [--mode] [--no-force] [--apply]`: 수집된 raw 날짜를 force 잡으로 다시 연다.
    처리 중인 잡은 완료가 force를 지우므로 건드리지 않는다.

**트레이드오프** — Airflow UI, 태스크 로그, params 폼을 잃는다. 대신 `scheduler_runs`, `arxplore scheduler history`,
`arxplore run <작업> --옵션`(trigger=manual)을 쓴다. 서버의 Airflow 컨테이너와 시크릿이 빠지는 것은 10번 항목에 적었다.
- catch-up은 직전 슬롯 한 번뿐이다. 스케줄러가 여러 슬롯 동안 멈춰 있었어도 한 번만 돈다.
- F08 뒤로 파서가 바뀌어도 layout_pdf 논문에는 force(잡 payload, worker `--force`, `run reparse`)로만 반영된다.
- F09 기본값이 false라 백필 날짜는 설정을 켜기 전까지 처리되지 않는다. 켜면 처리량이 늘어난다.
- F11: 이미 수집한 날짜는 다시 받지 않으므로 새 payload 형식은 새로 수집하는 날짜부터다. API 항목에는 뷰어별 `upvoted`
  키가 없다.
- `run rechunk --apply`는 바뀐 청크의 임베딩을 지우므로 재임베딩 비용이 든다. 그래서 두 도구 모두 dry-run이 기본이다.
- croniter는 타입 정보가 없어 `scheduler.cron`이 importlib으로 불러온다.

**eval 영향** — 새 측정 지점은 다음과 같다.
- 스케줄러: `v2/tests/integration/pipeline/test_scheduler_pg.py`(catch-up 한 번, 슬롯당 한 번, lock, abandoned, 재시도,
  시간대)
- prepare: `test_pipeline_prepare_pg.py`(F08 `test_a_final_layout_fulltext_skips_download_and_parse` 등),
  `test_pipeline_enrich_pg.py`(F07 `test_prepare_does_not_overwrite_what_enrich_filled`, F14), `test_pipeline_queue_pg.py`,
  `test_pipeline_worker_pg.py`, `test_pipeline_collect_pg.py`, `test_pipeline_corpus_pg.py`(rechunk·reparse)
- 단위: `v2/tests/unit/pipeline/`(F11 `test_pipeline_sources.py` 등)
- 보고 필드: prepare 결과의 `parse_skipped`, worker 출력의 `modes`, `run rechunk`의 재임베딩 토큰 추정

**알려진 한계** — 코퍼스 전환(재청킹, 재파싱, 재임베딩)과 전후 검색 재측정 결과는 아직 저장소에 없다.
- 청커가 바뀌어도 content_hash가 그대로라 자동 재청킹이 없다. 전환은 `run rechunk`로 사람이 한다.
- prepare 차등 실행은 일회성이라 스크립트가 커밋되지 않았다.
- 스케줄러는 advisory lock으로 한 인스턴스만 돈다. 서버 한 대 운영을 전제로 한다.
- 수동 실행(`arxplore run collect`)은 `--retries`를 주지 않으면 재시도하지 않는다.

**트러블슈팅** — F08의 "완료" 조건에 content_hash를 넣은 이유

- 증상: 운영에서 관찰한 것이 아니라 F08을 설계하며 확인한 위험이다. 파싱 전 건너뛰기는 한 번 잘못 판정하면 force 전까지
  그 논문을 다시 파싱하지 않는다. 저장이 본문과 청크 교체 사이에서 멈춘 논문은 source가 layout_pdf이고 옛 청크도 있어서,
  source 순위와 청크 유무만 보면 완료로 판정된다.
- 원인: prepare는 본문을 해시 없이 저장하고(해시 컬럼이 비워진다), 청크를 바꾼 뒤, 마지막에 content_hash를 기록한다
  (0aab1db, phase-3 2026-09-25_05). 해시가 비어 있으면 마지막 저장이 청크 교체 전에 멈췄다는 뜻이고, 그 청크는 새 본문과
  맞지 않을 수 있다.
- 확인 방법: `pipeline/prepare/paper.py`에서 저장 순서(`save_fulltext(..., content_hash=None)` → `save_chunks` →
  `update_content_hash`)를, fulltexts 저장소 upsert에서 해시가 덮어써지는 것(`content_hash = EXCLUDED.content_hash`)을
  확인했다. 해시를 지운 layout_pdf 본문이 다시 파싱되는지는 통합 테스트로 봤다.
- 해결: 32e087a에서 final 조건을 "source가 최상위(layout_pdf) + 청크 있음 + content_hash 있음 + force 아님"으로 정했다
  (`policy.fulltext_is_final`). 해시가 없으면 파싱을 건너뛰지 않고 기존 source 순위·해시 규칙으로 다시 판정한다.
- 재발 방지: `v2/tests/unit/pipeline/test_pipeline_prepare_policy.py::test_only_a_complete_layout_fulltext_is_final`(해시 없음,
  청크 0, force, 낮은 source는 final 아님)과 `v2/tests/integration/pipeline/test_pipeline_prepare_pg.py`의
  `test_lower_ranked_source_keeps_the_stored_fulltext`(해시를 지운 layout_pdf 본문을 다시 파싱한다)

사례 2 — 주말 반복 목록과 이미 처리한 논문의 중복 파싱

- 증상: phase-4 코퍼스 재구축에서 고유 논문 275편을 위해 raw 375행을 모두 파싱했고, 09-12·09-13 날짜 잡(각 26편)은 약
  530초씩 파싱만 하고 저장할 것이 없었다(phase-4 2026-09-26_05 사례 3).
- 원인: prepare는 PDF를 받아 파싱한 뒤에야 source 순위와 content_hash로 unchanged를 판정했다. HF Daily Papers 페이지는
  주말에 금요일 목록을 되풀이했다.
- 확인 방법: F11을 확인하며 JSON API가 토요일 2026-09-12에 빈 목록을 돌려주는 것을 봤다. F08 뒤 차등 날짜(layout 10편,
  초록 폴백 2편)를 변화 없이 다시 돌려 받은 PDF 수를 셌다.
- 해결: F11(1b0b198)로 수집을 JSON API로 바꿔 주말은 빈 목록이 된다. F08(32e087a)로 final인 layout 논문은 다운로드와 파싱을
  건너뛴다. 차등 날짜의 변화 없는 재실행이 PDF를 12개 대신 2개(layout이 아닌 2편)만 받고, 저장된 행과 결과는 같다.
- 재발 방지: `v2/tests/integration/pipeline/test_pipeline_prepare_pg.py::test_a_final_layout_fulltext_skips_download_and_parse`,
  `v2/tests/unit/pipeline/test_pipeline_sources.py::test_a_weekend_day_is_an_empty_list`. prepare 결과의 `parse_skipped`로
  건너뛴 논문을 셀 수 있다.
