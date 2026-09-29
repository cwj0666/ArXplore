---
title: LLM 계층 create_agent 이식과 FastAPI 전환
date: 2026-09-29
area: [agent, backend]
decision: 체인·요약 그래프·상세 챗·에이전트를 프롬프트 바이트 그대로 `arxplore.llm`으로 옮기고, 에이전트는 create_agent와 미들웨어 두 개로 옛 단계 제한 동작을 지킨다. Django 뷰는 경로·문구·상태 코드·검사 순서가 같은 FastAPI로 바꾸고(G5 60/60, G4 60/60), F18·F20~F26·F32와 AI 캐시 프롬프트 버전 2를 수정 커밋으로 얹는다
---

**결정과 근거** — LLM 계층과 API는 G4·G5가 HTTP 시나리오 단위로 함께 캡처됐기 때문에 한 작업으로 만들고 한 번에
검증했다.

LLM(4ebcb40)
- 모델 팩토리는 하나다. 키는 요청 범위 키, 서버 키 순으로, 모델은 인자, 요청 범위, `OPENAI_MODEL` 순으로 고른다. gpt-5
  계열에는 temperature를 넘기지 않는다. `stream_usage=True`를 명시해 base URL과 무관하게 운영과 같은 요청을 보낸다(02번
  항목 사례 2).
- 개요·핵심 포인트(섹션 8개 선택, 2,200/7,000/1,600자 한도), 요약 그래프(버킷별 2/3/3/3과 최대 12섹션,
  3,200/900/12,000/22,000자 한도, 네 버킷 병렬), 상세 챗(초록이 1번 출처, chunk_id 중복 제거, 첫 청크 폴백, `[n]` 인용
  확인)은 옛 규칙과 상수 그대로다. 알려진 버그(F20)도 이식 단계에서는 그대로 뒀다.
- 에이전트는 `langchain.agents.create_agent`로 옮겼다. LangGraph 1.0에서 `create_react_agent`가 deprecated됐기
  때문이다(01번 항목).
  - `RequestModelMiddleware`가 호출마다 요청 범위 모델과 키를 고른다.
  - `StepLimitMiddleware`가 옛 남은 단계 검사(남은 단계가 둘보다 적을 때의 도구 호출은 단계 제한 문구가 된다)를 지켜,
    한도 근처의 호출이 `create_react_agent`와 같다. `GraphRecursionError`는 옛 안내 문구로 바꾼다.
  - 120자 보류 버퍼는 노드 이름 `model`을 따른다. 도구 이름과 설명은 옛 docstring 문구를 명시적 문자열로 둔다. 도구
    hit은 ContextVar 리스트로 공유한다.
- 추적은 환경변수 대신 langsmith `tracing_context`를 명시적으로 넘긴다. stage 이름과 태그 규칙은 같다.

API(603e24e)
- 경로 9개의 메서드, 끝 슬래시, 문구, 상태 코드, 검사 순서가 같다. 오류는 모두 `{"error": ...}`이고, FastAPI 422는 400
  "잘못된 요청입니다.", 405는 빈 text/html 본문과 Allow, 모르는 경로는 JSON 404다.
- CSRF: `/bootstrap.json`이 `csrftoken` 쿠키(1년, Path=/, SameSite=Lax, HttpOnly 아님, `SECURE_COOKIES`면 Secure)를 심는다.
  변경 요청은 `X-CSRFToken`이 쿠키와 같아야 하고 Origin은 같은 출처이거나 신뢰 목록에 있어야 한다. 토큰은 서명 없는 무작위
  32자라 `DJANGO_SECRET_KEY`가 사라졌다(0e5ebfc).
- rate limit: 벽시계 분 단위 창, 키 `ratelimit:{scope}:ip:{ip}:{start}`, TTL 65초, 저장소는 메모리 또는 Redis(`REDIS_URL`).
  LLM 경로 6개가 한 버킷을, detail이 따로 한 버킷을 쓰고, 메서드·CSRF 검사 뒤에 센다. 클라이언트 IP는
  `RATE_LIMIT_IP_HEADER`의 첫 값, 소켓 주소, `unknown` 순이며 `none`이면 헤더를 쓰지 않는다.
- SSE: `data: {json, ensure_ascii}` 프레임, 빈 조각 생략, citations 이벤트 1회, 실패하면 error 뒤 `[DONE]`.
- 동등성: 0530057 시점에 G5 60/60, G4 60/60(121 passed)이었다. 어댑터는 FastAPI 앱을 `httpx.ASGITransport`로 부르고, 옛
  스키마로 만든 번들 DB 사본에 v2 마이그레이션을 먼저 적용한다.
- 부속: `arxplore openapi`가 경로 9개와 요청·SSE 이벤트 모델을 담은 스키마를 내보낸다(b1f618a). `SERVE_FRONTEND_DIST`로
  프론트를 서빙할 때 없는 `/assets/*`와 해석할 수 없는 경로는 404다(b1f618a, a662e34). 채팅 서비스 조립은 공개 함수
  `build_chat_service`로 냈다(4299ea0).

수정과 골든 차이(`g5_http.v2.diff.md`)

| ID | 커밋 | 내용 | G5·G4 차이 |
| --- | --- | --- | --- |
| F20 | 16d2ce0 | 핵심 포인트 번호 제거 `^\d+[.)]`가 `3.5%`의 `3.`까지 지웠다. 뒤에 숫자가 오지 않을 때만 지운다 | 1개 시나리오(`5% …` → `3.5% …`) |
| F21 | dc4958e, 252bdf1, 6c90df3 | 요약 그래프가 고른 other 섹션을 요약 전에 버렸다. other를 다섯 번째 버킷으로 요약해 병합 입력 끝(`[기타 해설]`, `[기타 원문 근거]`)에 붙이고, front_matter·references·toc와 감사의 글은 other 후보에서 뺀다. 요약당 호출이 (other 섹션 수 + 1)회 는다 | 3개 시나리오(호출 16 → 18회, 실패 시나리오 10 → 13회) |
| F22 | 93aca17, 424bab6 | 마이그레이션 0004로 AI 캐시에 `prompt_version`(기본 1)을 두고 버전이 다르면 캐시로 보지 않는다. 424bab6에서 두 버전을 2로 올려 F20·F21 이전 캐시를 다시 만든다 | 0004 자체는 없음. 버전 2는 4개 시나리오 |
| F23 | 6e528ef | 개요·요약 추적이 운영에서도 `runtime:dev`였다. `RUNTIME` 설정값을 쓴다. 보내지 않던 stage 4개는 옮기지 않았다 | 없음 |
| F24 | 803ee00 | 관련 논문의 arXiv 검색(이식 단계부터 httpx 비동기)을 (질의, max_results)별 1시간, 최대 512개 LRU로 캐시한다. 동시에 들어온 miss는 검색 1회를 함께 기다린다 | 없음 |
| F25 | 3980fe4 | SSE는 응답 중 `http.disconnect`를 따로 기다려 끊기면 스트림(에이전트 그래프, OpenAI 스트림)을 취소한다. 비스트리밍 LLM 경로는 생성을 작업으로 돌려 끊기면 취소하고 캐시하지 않으며 빈 499로 끝낸다 | 없음 |
| F26 | 64435d0 | CSRF 실패를 Django 403 HTML 대신 403 JSON 문구로 답한다 | 1개 시나리오의 응답 4개 |
| F32 | d00ef58 | 목록·관련 논문·트렌딩 도구의 정렬에 arxiv_id 동률 규칙을 더했다. `published_at` 값이 275편에 26종뿐이라 같은 날짜 안 순서가 힙 순서에 달려 있었다 | G5 7개 시나리오, G4는 트렌딩 도구가 받는 최근 10편 중 2편(1위 논문과 답변은 같음) |
| F18 | caa7e59 | 최근 1,500편을 파이썬에서 거르던 목록 검색을 SQL 토큰 AND(제목·초록 ILIKE, 순서 무관, 최대 20토큰)와 SQL `COUNT`·`LIMIT/OFFSET`으로 바꿨다. 오래된 논문도 검색되고 `language model`은 82편에서 114편이 된다 | 2개 시나리오(5 → 12편, 0 → 12편) |

F18은 F32 뒤에 올렸다. 동률 규칙 없이 SQL로 쪽을 나누면 목록이 겹치고 빠졌기 때문이다(사례).

**트레이드오프** — Django가 해 주던 CSRF, `Paginator.get_page`의 쪽 번호 규칙, 405 순서를 직접 재현했다. 코드가 늘었지만
G5 60개 시나리오가 이 의미를 고정한다.
- 옛 `create_react_agent`의 한도 동작을 지키려고 미들웨어 하나(`StepLimitMiddleware`)를 더 유지한다.
- F21은 상세 요약의 LLM 비용을 늘린다. 비본문 섹션을 빼서 12섹션 예산을 본문에 쓰게 했다.
- F22 버전 2: 버전 1 캐시가 있는 논문은 처음 요청할 때 다시 생성한다. 개요·핵심 포인트는 LLM 2회이고, 상세 요약은 모델마다
  섹션 수만큼이다(골든 논문 2609.28470은 13회). 서버 키가 없으면 옛 캐시를 주는 대신 503이다.
- F25: 끊긴 요청의 생성 결과는 버린다. 사용자가 다시 요청하면 처음부터 생성한다.
- F18: 토큰마다 부분 문자열 ILIKE라 관련도 순위는 없고, 정렬은 날짜·추천수 그대로다. 인덱스 없이 papers를 훑는다.

**eval 영향** — 새 측정 지점은 다음과 같다.
- G5·G4 변형 `tests/golden/data/g5_http.v2.json.gz`(F20에서 처음 생김)와 `g5_http.v2.diff.md`. 최종 변형으로 G5가 7회 연속
  121개 통과했다(a8323a1).
- 단위: `v2/tests/unit/llm/`, `v2/tests/unit/api/`(끊김은 원시 ASGI로 도는 `test_api_disconnect.py`)
- 통합: `v2/tests/integration/api/test_api_listing_pg.py`(F18·F32), `test_api_generation_cache_pg.py`(F22)
- OpenAPI 스키마 `v2/frontend/openapi.json`과 `arxplore openapi` 출력의 대조(08·10번 항목)

**알려진 한계** — G5·G4는 요청과 응답 형식만 본다. 답변 품질은 보지 않는다.
- F25의 끊김 처리는 G5로 확인하지 못한다. `httpx.ASGITransport`는 응답이 끝난 뒤에만 `http.disconnect`를 보내기 때문이다.
  원시 ASGI 테스트만 있다.
- analyze·summary는 논문을 정규화한 id로 찾지만 캐시와 본문은 URL의 원래 id로 조회한다(`api/services/generation.py`, 옛
  `backend/papers/services.py`도 같다). `…v3`처럼 버전이 붙은 경로에서는 캐시와 본문을 찾지 못한다. 고칠지는 아직 정하지
  않았다.
- rate limit 저장소는 `REDIS_URL`이 없으면 프로세스 메모리라 API 워커 수만큼 허용량이 늘어난다. compose의 `API_WORKERS`
  기본값은 1이다.
- F24 결과 캐시도 프로세스별이다.

**트러블슈팅** — 동률 규칙 없이 SQL로 쪽을 나누면 목록이 겹치고 빠짐

- 증상: F18 초안(SQL `LIMIT/OFFSET`)으로 최신순 목록 14쪽을 끝까지 넘기며 arxiv_id를 모으니 275행 중 고유 논문이 259편이었다.
  16편은 두 번 나왔고 16편은 한 번도 나오지 않았다.
- 원인: 정렬 키 `COALESCE(published_at, updated_at_utc)`가 275편에 26종뿐이라 같은 값 안의 순서가 정해져 있지 않았다. SQL은
  동률 구간의 순서를 보장하지 않으므로, 쪽마다 따로 도는 `LIMIT/OFFSET` 질의는 동률 구간을 서로 다른 순서로 돌려줄 수 있고
  그러면 쪽 경계에서 겹치고 빠진다. 옛 구현은 요청마다 최근 1,500편을 같은 질의로 가져와 파이썬에서 쪽을 나눴고, 같은 날짜
  안 순서는 힙 순서에 달려 있었다(d00ef58).
- 확인 방법: 골든 DB와 같은 275편이 든 `v2_test_api_smoke`에서 `/papers/list.json`을 마지막 쪽까지 넘기며 중복과 누락을 셌다
  (`g5_http.v2.diff.md`의 F18 절).
- 해결: d00ef58(F32)에서 최근 논문 조회를 `db/repositories/listing.py`로 옮기고 정렬에 `arxiv_id DESC`를 더했다. caa7e59(F18)는
  같은 정렬로 SQL에서 세고 쪽을 나눈다. 정렬 2가지(최신순, 추천순) × 검색어 3가지(없음, `diffusion`, `language model`)에서
  중복 0, 누락 0이고, 쪽을 이어 붙인 결과가 한 번에 정렬한 결과와 순서까지 같다.
- 재발 방지: `v2/tests/integration/api/test_api_listing_pg.py`의 `test_pages_over_tied_dates_neither_repeat_nor_skip_papers`,
  `test_recent_papers_break_date_ties_by_arxiv_id_descending`
