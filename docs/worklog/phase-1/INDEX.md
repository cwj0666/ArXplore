# Phase 1 작업 로그 색인

`python scripts/worklog.py index`가 생성한다. 손으로 고치지 않는다.
항목을 찾을 때는 이 표만 읽고 필요한 파일만 연다.

| 날짜 | 제목 | 영역 | 결정 | 파일 |
|---|---|---|---|---|
| 2026-09-25 | README를 코드 기준으로 다시 쓰고 환경 변수와 로컬 단독 실행 경로 정리 | docs, infra | README·아키텍처 문서를 실제 코드 동작에 맞추고, .env.example에 런타임 변수 57개를 필수·선택으로 나눠 적으며, 원격 서버 없이 웹을 띄우는 local-db compose 프로필을 더한다 | [2026-09-25_04_README를-코드-기준으로-다시-쓰고-환경-변수와-로컬-단독-실행-경로-정리.md](2026-09-25_04_README를-코드-기준으로-다시-쓰고-환경-변수와-로컬-단독-실행-경로-정리.md) |
| 2026-09-25 | 단위 테스트 151개 추가와 CI 5잡, gitleaks 사용자 규칙 | tooling, infra | RRF 계산·가중치·필터·다양성·요약 그래프 호출 예산·설정 파싱을 단위 테스트로 고정하고, GitHub Actions에 python·integration·frontend·infra·security 5잡을 둔다 | [2026-09-25_03_단위-테스트-151개-추가와-CI-5잡,-gitleaks-사용자-규칙.md](2026-09-25_03_단위-테스트-151개-추가와-CI-5잡,-gitleaks-사용자-규칙.md) |
| 2026-09-25 | raw payload 해시로 revision을 올리고 처리 중 재수집은 pending_refresh로 | pipeline | 같은 날짜를 다시 수집해도 payload 해시가 같으면 revision을 올리지 않고, 처리 중인 잡에 새 revision이 오면 pending_refresh로 표시해 완료 뒤 한 번 더 처리한다 | [2026-09-25_02_raw-payload-해시로-revision을-올리고-처리-중-재수집은-pending_refresh로.md](2026-09-25_02_raw-payload-해시로-revision을-올리고-처리-중-재수집은-pending_refresh로.md) |
| 2026-09-25 | prepare_jobs 선점 펜싱과 heartbeat 기반 stale 회수, 재시도 backoff | pipeline | prepare_jobs 선점에 claim_generation·worker_id 토큰을 발급해 완료·실패·heartbeat를 토큰으로 제한하고, stale 판정은 heartbeat_at 기준 900초, 재시도는 최대 3회·60초×2^(n-1)·상한 1시간 backoff로 한다 | [2026-09-25_01_prepare_jobs-선점-펜싱과-heartbeat-기반-stale-회수,-재시도-backoff.md](2026-09-25_01_prepare_jobs-선점-펜싱과-heartbeat-기반-stale-회수,-재시도-backoff.md) |
