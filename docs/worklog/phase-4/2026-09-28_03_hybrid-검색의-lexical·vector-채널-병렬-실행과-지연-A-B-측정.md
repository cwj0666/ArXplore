---
title: hybrid 검색의 lexical·vector 채널 병렬 실행과 지연 A-B 측정
date: 2026-09-28
area: [retrieval, eval]
decision: hybrid 검색은 vector 채널(질의 임베딩 → 벡터 SQL)을 호출마다 만드는 작업 스레드 1개에서, lexical 채널을 호출 스레드에서 동시에 실행한다. 같은 조건 A/B에서 eval 경로 질의별 지연이 평균 139ms(24%) 줄었고 결과는 같다
---

**결정과 근거** — `PaperRetriever.hybrid_fusion_inputs`는 lexical 경로를 끝낸 뒤 vector 경로(임베딩 API 왕복 → 벡터 SQL)를
실행했다. 두 경로는 서로의 결과를 쓰지 않으므로 전체 지연은 두 채널의 합이 아니라 둘 중 긴 쪽이면 된다. vector를 작업 스레드에
먼저 띄우고 lexical을 호출 스레드에서 실행한 뒤 vector 결과를 기다린다(커밋 ba2e6a3).

- 연결: 두 저장소 메서드는 호출 안에서 `get_connection()`으로 풀 연결을 빌리고 반납하므로 스레드마다 자기 연결을 쓴다. 요청당
  동시 연결은 최대 2개이고, 한 스레드가 연결을 쥔 채 다른 연결을 기다리는 경우가 없어 풀 교착은 생기지 않는다.
- 문맥: 요청 범위 OpenAI 키(`override_openai_runtime`)는 ContextVar라 작업 스레드를 `contextvars.copy_context()`로 실행한다.
  LangSmith 추적 문맥도 같은 방법으로 넘어간다.
- 예외: 순차 실행과 같은 우선순위를 지킨다. lexical이 실패하면 vector를 기다리지 않고 lexical 예외를 낸다. lexical이 성공하면
  vector 예외를 그대로 다시 낸다. 임베딩 `OpenAIError`는 이전과 같이 `retrieve_contexts`의 lexical 폴백으로 이어진다.
- 순차 경로는 `PaperRetriever(parallel_channels=False)`로 남겼다. 같은 프로세스에서 A/B를 재기 위한 스위치이고 코드는 7b1c36a의
  순차 호출 그대로다.

측정(`eval/results/latency_20260928-151847.md`, `scripts/eval_hybrid_latency.py`, 커밋 ba2e6a3 위에서 실행): 로컬 WSL2
(i7-13650HX 20 논리 코어, RAM 11.5 GiB), 같은 머신의 PostgreSQL 16.15 + pgvector 0.8.6에 트랙 A 덤프(275편, 청크 16,010,
임베딩 13,328)를 복원했다. 검색 평가 대상 111개 질의(ko 58 / en 53), 워밍업 1회 + 반복 3회, 질의 블록마다 A(순차)/B(병렬)
쌍을 반복마다 A→B / B→A로 번갈아 실행했고 임베딩은 캐시하지 않았다(API 왕복 포함). POSTGRES_POOL_MAX=8.

| 경로 | A p50 / p95 | B p50 / p95 | 질의별 Δ(B−A) 평균, 95% CI | 상대 |
| --- | --- | --- | --- | --- |
| eval `search_paper_contexts_by_hybrid` (k=10) | 489 / 831 | 282 / 582 | −139 ms [−156, −122] | −23.9% [−26.8, −21.1] |
| 제품 `retrieve_contexts` (limit=5) | 390 / 669 | 235 / 448 | −120 ms [−133, −107] | −25.6% [−28.3, −23.0] |

언어별로는 en이 eval 경로 −217 ms(−35.2%), ko가 −68 ms(−13.7%)다. 한국어 질의는 lexical SQL이 대부분 10 ms 안에 끝나
(대조군 lexical 단독 ko p50 10 ms, en p50 281 ms) 숨길 시간이 적다. 단계 중앙값(eval 경로)은 임베딩 API 149 ms, lexical SQL
231~237 ms, vector SQL 57 ms, 융합 0.1 ms, 문맥 창 쿼리 1.2 ms로 A와 B가 같고, 전체만 489 → 282 ms로 줄었다. 대조군 lexical 단독
(p50 160 ms)과 vector 단독(p50 198 ms)은 이 변경과 무관한 경로로 측정 기간의 DB·API 상태를 보여 준다.

결과 동일성: 반복마다 A와 B의 순위 (chunk_id, arxiv_id)를 비교해 eval 경로 0/1/1건, 제품 경로 0/1/0건이 달랐다. 같은 코드를
다시 실행한 기준선(A vs A, B vs B)도 반복 간 0~1건이 다르고 불일치 질의가 겹친다(`sf-4000-chars-padded`,
`ki-2609.23169-ko`, `ki-2609.15478-ko`). 임베딩을 질의당 한 번만 요청해 A와 B가 같은 벡터를 쓰게 한 패스에서는 두 경로 모두
111/111건이 반환 dict 전체(점수·score_breakdown·문맥 포함)까지 같았다. 따라서 불일치는 임베딩 API가 호출마다 조금 다른 벡터를
돌려주는 데서 오고 병렬화와는 무관하다.

**트레이드오프** — 작업 스레드는 호출마다 `ThreadPoolExecutor(max_workers=1)`로 만든다. 모듈 공용 실행기는 스레드 생성 비용을
아끼지만 gunicorn gthread 워커(4 × 8스레드)의 동시 요청이 공용 실행기의 스레드 수에 막혀 줄을 서고, fork 뒤 상태도 따져야 한다.
스레드 생성 비용은 100 µs 단위로 임베딩 왕복(약 150 ms)에 비해 무시할 만하다. 대가는 두 가지다. 요청당 DB 연결이 최대 2개로
늘어 동시 요청이 많으면 풀 대기가 빨리 온다(풀 상한은 그대로). lexical이 실패하면 순차 실행에서는 부르지 않던 임베딩 호출이 이미
나가 있을 수 있다(결과는 버린다). `asyncio` 전환은 저장소·임베딩 클라이언트가 모두 동기라 범위 밖이다. 융합·다른 최적화는 이
변경에 넣지 않았다. 영어 lexical SQL(약 230 ms)이 이제 병렬 경로의 긴 쪽이 되는 경우가 많다.

**eval 영향** — `eval/latency.py`(계측 래퍼, 쌍 교대 실행, 질의별 중앙값 차이의 ko/en 쌍 클러스터 bootstrap, 반복 간 동일성
기준선)와 `scripts/eval_hybrid_latency.py`가 생겼다. 단위 테스트 `tests/unit/test_hybrid_parallel.py`(병렬 == 순차, 채널 동시
실행, ContextVar 전달, 채널별 예외 전파와 우선순위, 스레드별 풀 연결, `retrieve_contexts` 폴백)와 `tests/unit/test_eval_latency.py`.
임베딩 호출은 2,330회, 103,988 토큰(tiktoken), 추정 $0.0135였다.

**알려진 한계** — 단일 클라이언트 스레드로 잰 지연이라 동시 요청 부하에서의 처리량·풀 대기는 모른다. DB가 같은 머신이라 네트워크
왕복이 없다. 원격 DB에서는 SQL 쪽 지연이 커져 병렬 효과가 달라질 수 있다. README Evaluation 표의 지연(hybrid 786 / 1149 ms)은
다른 장비(파드)에서 순차 코드로 잰 값이라 이 측정과 비교하지 않았고 고치지 않았다. 쌍 안에서 두 번째로 실행한 쪽이 중앙값
35~60 ms 빨랐다(같은 질의 재실행의 캐시 효과). 반복마다 순서를 바꿔 상쇄했고, 반복별 질의 차이 중앙값은 A→B와 B→A에서 eval 경로
−200 / −201 / −205 ms로 순서와 무관했다.

**트러블슈팅** — 해당 없음
