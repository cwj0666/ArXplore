# G5·G4 HTTP 골든 v2 변형: 원본과 다른 점

변형 파일은 `g5_http.v2.json.gz`이고 v2 구현(`GOLDEN_IMPL` 기본값)과 비교할 때 쓰인다. 원본 `g5_http.json.gz`는 옛 Django 구현 기준으로 그대로 둔다.

다시 만드는 명령(저장소 루트): `uv run python -m scripts.goldens.capture_g5_http`(기본으로 v2 변형을 쓴다)

- 응답은 `compare_responses_v2` 규칙으로 비교한다. 대상은 상태 코드, content-type 미디어 타입, allow, set-cookie 속성, 허용 목록 헤더, SSE 바이트, JSON 값, 그 밖의 본문 바이트다. 가짜 OpenAI·arXiv 요청 기록(G4)은 그대로 비교한다.
- 이식 동등성: 0530057 시점에 v2는 원본 골든과 G5 60/60 시나리오, G4 60/60 시나리오가 같았다(`tests/golden/test_g5_http.py` 121 passed).
- 변형 파일은 차이가 처음 생긴 F20에서 만들었다. 그 전 수정(F22)까지 v2는 원본과 비교했다.

## F22: AI 캐시 키에 프롬프트 버전

- 마이그레이션 `0004_ai_cache_prompt_version`이 `paper_ai_overviews`와 `paper_ai_detailed_summaries`에 `prompt_version INTEGER NOT NULL DEFAULT 1`(양수 CHECK)을 더한다. 기존 행은 모두 버전 1이 된다.
- 조회는 `prompt_version`이 현재 버전과 같은 행만 캐시로 본다. 다른 버전의 행은 없는 것으로 보고 새로 만들며, 저장할 때 같은 행을 덮어쓰고 버전을 바꾼다. 행 키(`arxiv_id`, 요약은 `arxiv_id, model`)는 그대로다.
- 이 커밋의 버전은 개요·핵심 포인트 `ANALYSIS_PROMPT_VERSION = 1`, 상세 요약 `SUMMARY_PROMPT_VERSION = 1`이다. 옛 프롬프트와 같으므로 기존 캐시가 그대로 쓰인다. 두 버전은 뒤의 "F22 후속" 커밋에서 2로 올렸다.
- G5·G4 차이: 없음. 골든 DB 복사본에 0004를 적용한 뒤에도 60/60, 60/60이 같다. 캐시가 있는 `analyze_cached`, `summary_cached`, `*_without_key`의 캐시 응답이 그대로 나온다.

## F20: 핵심 포인트 번호 제거가 소수를 자르지 않게

- 핵심 포인트 줄 앞의 번호 표식은 `\d+[.)]` 뒤에 숫자가 오지 않을 때만 지운다(`^(?:[-*•]|\d+[.)](?!\d))\s*`). `3.5% ...`는 그대로 두고, `1. 3.5% ...`와 `2) ...`는 예전처럼 번호만 지운다.
- 번호와 숫자가 붙어 있는 `2)4.2 ...` 같은 줄은 이제 번호를 지우지 않는다. LLM이 번호 뒤에 공백을 두는 한 영향이 없다.
- 새로 생성하는 결과부터 적용된다. 캐시된 핵심 포인트는 다시 만들지 않는다(F22 프롬프트 버전은 1 그대로).
- G5 차이는 1개 시나리오 `analyze:analyze_generate_then_cached`의 두 응답이다. `key_findings[1]`이 `5% 정확도 향상이 이 논문의 핵심 결과입니다.`에서 `3.5% 정확도 향상이 이 논문의 핵심 결과입니다.`로 바뀐다. 두 번째 응답은 첫 응답이 저장한 캐시라 같은 값이다. 중복 제거 키가 서로 달라서 6개 목록의 나머지 항목은 그대로다.
- G4 차이: 없음.

## F21: 상세 요약에서 other 섹션도 요약

- 섹션 선택 규칙(버킷별 2/3/3/3, 나머지를 other로 채워 최대 12개)은 그대로다. 달라진 점은 선택된 other 섹션을 버리지 않는다는 것이다. other도 다섯 번째 버킷으로 섹션 요약(3,200자)과 버킷 요약을 거친다. 그 결과를 `[기타 해설]`, `[기타 원문 근거]`(앞 2섹션, 900자) 라벨로 병합 입력의 맨 뒤에 붙인다.
- 후속 커밋에서 other 후보에서 비본문 섹션을 뺐다. `arxplore.parsing.roles.section_role`이 front_matter, references, toc로 보는 제목과 `acknowledg`가 들어간 제목이 대상이다. 네 키워드 버킷에 든 섹션은 제목이 무엇이든 그대로다. 빠진 섹션은 12개 예산도 쓰지 않는다. 옛 구현도 other를 요약하지 않았으므로, 목적은 키워드에 걸리지 않아 버려지던 본문 섹션만 살리는 것이다.
- 네 버킷이 모두 비어도 본문 other가 있으면 fallback 본문 대신 병합 입력으로 최종 요약을 만든다. 요약할 섹션이 하나도 없을 때만 fallback 본문을 쓴다.
- LLM 호출은 선택된 본문 other 섹션 수 + 1회 늘어난다(없으면 0회).
- 새로 생성하는 요약부터 적용된다. 캐시된 요약은 다시 만들지 않는다(F22 프롬프트 버전은 1 그대로).
- G5·G4 차이(3개 시나리오)
  - `summary:summary_generate_gpt5_mini`, `summary:summary_generate_gpt5`(2608.30597)
    - other 섹션 `3 Posterior Label Correction DPO`의 섹션 호출과 other 버킷 호출이 더해져 LLM 호출이 16회에서 18회로 는다. `Front Matter`는 빠진다.
    - 최종 호출의 본문 끝에 `[기타 해설]`과 `[기타 원문 근거] [근거 1] 3 Posterior Label Correction DPO`가 붙는다.
    - 가짜 서버 답은 마지막 사용자 메시지의 해시라 요약 문자열이 `Fake reply e9402cd9982b: ...`에서 `Fake reply cf8f3efa6657: ...`로 바뀐다. gpt5_mini의 두 번째 응답은 캐시라 같은 값이다.
  - `summary:summary_model_failure`(2609.26796)
    - other 섹션 `2.2 Flash-Cache: IO-aware key-value caching`, `2.3 Flash-Verify: KV-cache-driven draft-and-verify Parallel Decoding`의 섹션 호출 2회와 버킷 호출 1회가 더해진다. LLM 호출은 10회에서 13회로 늘고, 최종 호출은 여전히 400이다. 응답(500)은 같다.
    - 선택에서 빠진 other 섹션은 `Front Matter`, `Acknowledgments`, `References`다.

## F23: 추적 runtime 라벨

- 옛 구현은 개요·핵심 포인트·상세 요약 체인에 runtime을 넘기지 않아 운영에서도 `runtime:dev`로 기록했다. 이제 다섯 stage(`paper_overview`, `paper_key_findings`, `summary`, `paper_chat`, `agent_chat`) 모두 `RUNTIME` 설정을 해석한 값(`development`→`dev`, `prod`→`production`, 모르는 값→`dev`)을 태그 `runtime:`과 메타데이터 `runtime`에 쓴다.
- v2 추적 모듈에는 제품 경로의 다섯 stage만 있다. 옛 허용 목록에만 있던 `translation`, `rag_answer`, `paper_detail_document`, `analyze_paper_detail`은 옮기지 않았다. 파이프라인 stage는 P3 파이프라인 이식에서 다룬다.
- G5·G4 차이: 없음. 추적은 가짜 서버 요청 본문에 들어가지 않고 골든 실행에서는 꺼져 있다.

## F24: 외부 arXiv 검색 async와 결과 캐시

- 상세 요청 안의 arXiv 관련 논문 검색은 이식 단계부터 `httpx.AsyncClient`로 돈다(`arxplore.sources.arxiv`, 3793658). 그래서 옛 구현과 달리 요청 스레드를 막지 않는다. URL, 파라미터, 8초 타임아웃, 1회 시도는 옛 값 그대로다.
- F24는 결과 캐시를 더한다. 키는 (질의, max_results)이고 성공한 결과만 1시간 보관한다. 최대 512개이며 오래 안 쓴 것부터 버린다. 실패는 캐시하지 않아 다음 요청이 다시 시도한다.
- 같은 키로 동시에 들어온 요청은 검색 1회를 함께 기다린다. 한 요청이 끊겨도 공유 검색은 끝까지 돈다.
- 골든 하네스의 `reset_state`가 시나리오마다 캐시를 비운다.
- G5·G4 차이: 없음. 외부 검색을 하는 두 시나리오(`detail_external_related`, `detail_external_failure`)는 요청이 1번씩이다.

## F25: 클라이언트가 끊으면 생성 중단

- SSE 두 경로(`/papers/<id>/chat/stream/`, `/papers/assistant/stream/`)는 응답을 보내는 동안 `http.disconnect`를 따로 기다린다. 연결이 끊기면 스트림 작업을 취소하고, 에이전트 그래프와 OpenAI 스트림도 함께 닫힌다. ASGI spec 2.4 서버처럼 Starlette 기본 응답이 끊김을 다음 쓰기 때에만 알아채는 경우에도 같다. 옛 WSGI 구현은 다음 청크를 쓸 때까지 생성을 계속했다.
- 스트림이 아닌 LLM 경로(`analyze/`, `summary/`, `/papers/<id>/chat/`, `/papers/assistant/chat/`)는 생성을 별도 작업으로 돌리면서 끊김을 기다린다. 끊기면 생성을 취소하고 캐시에 저장하지 않으며 빈 499 응답으로 끝낸다. 이 응답은 이미 떠난 클라이언트에게 가지 않는다.
- 검증 순서(400·404·503)와 캐시 응답은 그대로다. 끊김 감시는 생성 단계에만 붙는다.
- G5·G4 차이: 없음. 골든 하네스의 ASGITransport는 응답이 끝난 뒤에만 `http.disconnect`를 보낸다.

## F26: CSRF 실패를 403 JSON 문구로

- CSRF 검증 실패(쿠키 없음, 형식이 틀린 쿠키, 헤더 없음, 토큰 불일치, 신뢰하지 않는 Origin)는 이제 403 `{"error": "요청을 확인하지 못했습니다(CSRF 검증 실패). 페이지를 새로고침한 뒤 다시 시도해 주세요."}`다. 실패 종류는 응답이 아니라 INFO 로그에만 남긴다.
- 이식 단계는 Django 기본 403 HTML(쿠키가 없을 때만 쿠키 안내 문단 두 개가 붙는 두 변형)을 바이트까지 재현했다. 그 페이지 상수 모듈 `arxplore.api.csrf_page`는 이 수정에서 지웠다.
- 검사 조건과 순서는 그대로다. 405가 먼저이고 rate limit 카운트보다 앞이다. 보안 헤더 4종도 그대로 붙는다.
- G5 차이는 1개 시나리오 `pages:csrf_failures`의 앞 네 응답이다. 상태 403은 같고, content-type이 `text/html`에서 `application/json`으로, 본문이 HTML에서 위 JSON으로 바뀐다. 다섯 번째 응답(신뢰 Origin, 404)은 같다.
- G4 차이: 없음.

## F32: 목록·관련 논문 정렬에 arxiv_id 동률 규칙

- 최근 논문 조회를 새 저장소 모듈 `db/repositories/listing.py`로 옮겼다. 정렬은 `COALESCE(published_at, updated_at_utc) DESC, arxiv_id DESC`다. 목록(최근 1,500편), 관련 논문 후보(300편), 트렌딩 도구(10편)가 모두 이 조회를 쓴다.
- 추천순 목록과 트렌딩 도구의 추천수 정렬은 안정 정렬이다. 그래서 추천수가 같으면 위 순서(날짜, 다음 arxiv_id 내림차순)를 따른다.
- 관련 논문은 정렬 키 `(점수, 추천수, 발행일)`에 `arxiv_id`를 더했다. 모두 같으면 arxiv_id 내림차순이다.
- 이전에는 `published_at` 값이 275편에 26종뿐이어서 같은 날짜 안의 순서가 힙 순서에 달려 있었다.
- G5 차이(이전 v2 변형 대비 7개 시나리오, 목록 응답 9개)
  - 기본 목록 1쪽(`list_default`, `list_page_invalid`의 세 응답, `list_unknown_sort_and_mode`): 같은 날짜 안에서 arxiv_id 내림차순으로 바뀌어 18자리가 움직인다. 경계에서 `2609.25636`, `2609.26355`, `2609.26457`이 2쪽으로 가고 `2609.26780`, `2609.26793`, `2609.26796`이 1쪽으로 온다.
  - 2쪽(`list_page_two`, `detail_bucket_is_separate`의 목록 응답): 10편이 바뀐다.
  - `list_query`(diffusion 23건): 1쪽 구성은 같고 순서만 11자리 바뀐다.
  - 추천순 `list_query_upvotes_mode`와 상세의 관련 논문은 달라지지 않는다.
- G4 차이: `agent_trending_then_link`의 트렌딩 도구 결과. 최근 10편에서 `2609.27321`, `2609.27284`가 빠지고 `2609.27980`, `2609.28470`이 들어온다. 1위 논문 `2609.28466`과 답변은 같다.

## F18: 목록 검색을 SQL 토큰 AND로

- `/papers/list.json`은 이제 최근 1,500편을 파이썬으로 거르지 않는다. `db/repositories/listing.py`의 `count_papers`와 `list_papers_page`가 SQL에서 거르고, 세고, 정렬하고, 쪽을 나눈다(`COUNT(*)`, `LIMIT/OFFSET`). 1,500편 제한이 없어져 오래된 논문도 검색된다.
- 검색어는 공백으로 나눈 토큰이다. 대소문자를 무시해 중복 토큰을 한 번만 쓰고, 그 앞 20개까지만 쓴다. 토큰마다 제목이나 초록 중 한 곳에 있어야 하고 모든 토큰이 맞아야 한다. 순서는 상관없다. 비교는 `ILIKE ... ESCAPE '\'`라 `%`, `_`, `\`는 글자 그대로 찾는다.
- 정렬은 F32와 같다. 최신순은 `COALESCE(published_at, updated_at_utc) DESC, arxiv_id DESC`이고, 추천순은 그 앞에 `upvotes DESC`를 둔다. 쪽 번호 처리, 쪽당 21편, 응답 필드는 그대로다. 응답의 `query`는 여전히 앞뒤 공백만 지운 입력 문자열이다.
- 관련 논문 후보(300편)와 트렌딩 도구(10편)는 계속 최근 논문 조회 `list_recent_papers`를 쓴다.
- 쪽 넘김 측정: 골든 DB와 같은 275편(`published_at` 26종)이 든 `v2_test_api_smoke`에서 `/papers/list.json`을 마지막 쪽까지 넘기며 arxiv_id를 모았다.

| 정렬 | 검색어 | 쪽 | total_items | 모은 행 | 중복 | 누락 |
|---|---|---|---|---|---|---|
| latest | (없음) | 14 | 275 | 275 | 0 | 0 |
| latest | `diffusion` | 2 | 23 | 23 | 0 | 0 |
| latest | `language model` | 6 | 114 | 114 | 0 | 0 |
| upvotes | (없음) | 14 | 275 | 275 | 0 | 0 |
| upvotes | `diffusion` | 2 | 23 | 23 | 0 | 0 |
| upvotes | `language model` | 6 | 114 | 114 | 0 | 0 |

  - 여섯 경우 모두 쪽을 이어 붙인 결과가 한 번에 정렬한 전체 결과와 순서까지 같다.
  - F32 동률 규칙 없이 이 수정만 올렸을 때는 최신순 14쪽에서 275행 중 고유 논문이 259편이었다(16편 중복, 16편 누락).
  - `language model`은 예전 규칙(입력 문자열 전체의 부분 일치)으로 82편이고 지금은 114편이다.
- G5 차이(이전 v2 변형 대비 2개 시나리오)
  - `list_query_case_and_spaces`(`  Diffusion Models `): 5편에서 12편이 된다. 예전에는 `diffusion models`가 붙어 있는 논문만 찾았다. 새로 들어온 7편은 `2609.12945`, `2609.13287`, `2609.16034`, `2609.16372`, `2609.24657`, `2609.24788`, `2609.26796`이다.
  - `list_query_word_order`(`models diffusion`): 0편에서 12편이 된다. 위 시나리오와 같은 12편이 같은 순서로 나온다.
  - 한 단어 검색 `list_query`(`diffusion` 23편), 기본 목록, 쪽 넘김, 추천순은 그대로다.
- G4 차이: 없음.

## F22 후속: 프롬프트 버전 2

- `ANALYSIS_PROMPT_VERSION`(개요·핵심 포인트)과 `SUMMARY_PROMPT_VERSION`(상세 요약)을 1에서 2로 올렸다. 이미 캐시된 결과도 F20·F21이 반영된 결과로 다시 만들기 위해서다. 프롬프트 문자열과 생성 과정은 그대로다.
- 버전 1 행은 캐시로 보지 않는다. 0004 전에 쓰인 행과 그 뒤 이 커밋 전까지 저장된 행이 모두 버전 1이다. 키가 있으면 첫 요청 때 새로 만들고 같은 행을 버전 2로 덮어쓴다. 키가 없으면 캐시 응답 대신 503이다. F20·F21 절에 적은, 캐시된 결과를 다시 만들지 않는다는 설명은 이 커밋부터 맞지 않는다.
- 비용: 버전 1 캐시가 있는 논문은 처음 요청할 때 다시 생성한다. 개요·핵심 포인트는 LLM 2회다. 상세 요약은 모델마다 섹션 수에 따라 여러 번이고, 골든 논문 `2609.28470`은 13회다.
- G5·G4 차이(이전 v2 변형 대비 4개 시나리오, 모두 골든 DB에 캐시가 있는 `2609.28470`)
  - `analyze:analyze_cached`: 버전 1 캐시를 쓰지 않고 새로 만든다. `cached`가 true에서 false가 되고, `overview`와 `key_findings`(5개에서 1개)가 가짜 서버 답으로 바뀐다. G4에 gpt-5-mini 호출 2회가 생긴다.
  - `summary:summary_cached`: 같은 이유로 새로 만든다. `cached`가 false가 되고 `summary`가 가짜 서버 답으로 바뀐다. G4에 gpt-5-mini 호출 13회가 생긴다.
  - `analyze:analyze_without_key`(두 번째 응답), `summary:summary_without_key`(첫 번째 응답): 여전히 200 `cached: true`다. 다만 값은 같은 번들에서 먼저 도는 `analyze_cached`, `summary_cached`가 번들 DB 사본에 버전 2로 저장한 결과다. 두 시나리오의 나머지 응답(404, 503, 400)은 같다.
  - 그 밖의 시나리오는 그대로다. 생성 시나리오(`analyze_generate_then_cached`, `summary_generate_*`)는 캐시가 없던 논문이라 달라지지 않는다.
- G4 비교 방식: `summary_cached`도 이제 버킷을 병렬로 도는 요약 그래프를 타므로, 버킷 사이 요청 순서가 실행마다 바뀐다. 처음에는 이 시나리오의 LLM 기록 방식이 기본값 `sequence`(도착 순서 그대로)라서 G4 비교가 가끔 실패했다(5회 중 3회). 하네스 4f1473b가 이 시나리오를 `llm_layout="summary_graph"`로 바꿨다. 이제 호출을 섹션 분류별·버킷별로 묶어 비교한다. 버킷 안의 섹션 호출은 차례로 돌아 순서가 고정된다. 이 기록 방식으로 변형을 다시 잡은 뒤 G5를 7회 연속 돌려 매번 121개가 통과했다. 다시 잡은 변형에서 달라진 것은 이 시나리오의 `llm_requests` 형식뿐이다(섹션 호출 limitations 1·other 9, 버킷 호출 limitations 1·other 1, 최종 호출 1).

## F38: analyze·summary가 찾은 논문의 id로 캐시와 본문을 읽고 쓴다

- `find_paper`는 경로 id(`2609.28470v3`)로 논문이 없으면 정규화한 id로 다시 찾는다. 그런데 analyze·summary는 캐시 조회, fulltext 조회, 캐시 저장에 경로 id를 그대로 썼다. 이제 셋 다 찾은 논문의 `arxiv_id`를 쓴다.
- 수정 전 버전 붙은 id의 동작(가짜 서버로 재현)
  - 키가 있으면 analyze는 500이었다. fulltext를 못 찾아, 본문 없이는 개요를 만들지 않는 검사(`MissingPaperContextError`)에 걸렸다. summary는 LLM을 부르지 않고 200 빈 요약을 돌려줬다.
  - 키가 없으면 같은 논문의 캐시가 있어도 503이었다.
- 이제 버전 붙은 id도 버전 없는 id와 같다. 키가 없을 때는 캐시가 있으면 `cached: true`, 없으면 503이다. 생성 결과는 정규화한 id의 행에 저장된다.
- G5·G4 차이: 없음. analyze·summary를 버전 붙은 id로 부르는 시나리오가 없다(버전 id는 `detail.json`과 상세 챗 스트림에만 있고, 두 경로는 원래 찾은 논문의 id를 썼다).
