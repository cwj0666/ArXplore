# G5·G4 HTTP 골든 v2 변형: 원본과 다른 점

변형 파일은 `g5_http.v2.json.gz`이고 `GOLDEN_IMPL=v2`일 때만 쓰인다. 원본 `g5_http.json.gz`는 옛 Django 구현 기준으로 그대로 둔다.

다시 만드는 명령(저장소 루트): `GOLDEN_IMPL=v2 v2/.venv/bin/python -m scripts.goldens.capture_g5_http --variant v2`

- 응답은 `compare_responses_v2` 규칙으로 비교한다. 대상은 상태 코드, content-type 미디어 타입, allow, set-cookie 속성, 허용 목록 헤더, SSE 바이트, JSON 값, 그 밖의 본문 바이트다. 가짜 OpenAI·arXiv 요청 기록(G4)은 그대로 비교한다.
- 이식 동등성: 0530057 시점에 v2는 원본 골든과 G5 60/60 시나리오, G4 60/60 시나리오가 같았다(`tests/golden/test_g5_http.py` 121 passed).
- 변형 파일은 차이가 처음 생긴 F20에서 만들었다. 그 전 수정(F22)까지 v2는 원본과 비교했다.

## F22: AI 캐시 키에 프롬프트 버전

- 마이그레이션 `0004_ai_cache_prompt_version`이 `paper_ai_overviews`와 `paper_ai_detailed_summaries`에 `prompt_version INTEGER NOT NULL DEFAULT 1`(양수 CHECK)을 더한다. 기존 행은 모두 버전 1이 된다.
- 조회는 `prompt_version`이 현재 버전과 같은 행만 캐시로 본다. 다른 버전의 행은 없는 것으로 보고 새로 만들며, 저장할 때 같은 행을 덮어쓰고 버전을 바꾼다. 행 키(`arxiv_id`, 요약은 `arxiv_id, model`)는 그대로다.
- 현재 버전은 개요·핵심 포인트 `ANALYSIS_PROMPT_VERSION = 1`, 상세 요약 `SUMMARY_PROMPT_VERSION = 1`이다. 옛 프롬프트와 같으므로 기존 캐시가 그대로 쓰인다.
- G5·G4 차이: 없음. 골든 DB 복사본에 0004를 적용한 뒤에도 60/60, 60/60이 같다. 캐시가 있는 `analyze_cached`, `summary_cached`, `*_without_key`의 캐시 응답이 그대로 나온다.

## F20: 핵심 포인트 번호 제거가 소수를 자르지 않게

- 핵심 포인트 줄 앞의 번호 표식은 `\d+[.)]` 뒤에 숫자가 오지 않을 때만 지운다(`^(?:[-*•]|\d+[.)](?!\d))\s*`). `3.5% ...`는 그대로 두고, `1. 3.5% ...`와 `2) ...`는 예전처럼 번호만 지운다.
- 번호와 숫자가 붙어 있는 `2)4.2 ...` 같은 줄은 이제 번호를 지우지 않는다. LLM이 번호 뒤에 공백을 두는 한 영향이 없다.
- 새로 생성하는 결과부터 적용된다. 캐시된 핵심 포인트는 다시 만들지 않는다(F22 프롬프트 버전은 1 그대로).
- G5 차이는 1개 시나리오 `analyze:analyze_generate_then_cached`의 두 응답이다. `key_findings[1]`이 `5% 정확도 향상이 이 논문의 핵심 결과입니다.`에서 `3.5% 정확도 향상이 이 논문의 핵심 결과입니다.`로 바뀐다. 두 번째 응답은 첫 응답이 저장한 캐시라 같은 값이다. 중복 제거 키가 서로 달라서 6개 목록의 나머지 항목은 그대로다.
- G4 차이: 없음.

## F21: 상세 요약에서 other 섹션도 요약

- 섹션 선택 규칙(버킷별 2/3/3/3, 나머지를 other로 채워 최대 12개)은 그대로다. 달라진 점은 선택된 other 섹션을 버리지 않는다는 것이다. other도 다섯 번째 버킷으로 섹션 요약(3,200자)과 버킷 요약을 거친다. 그 결과를 `[기타 해설]`, `[기타 원문 근거]`(앞 2섹션, 900자) 라벨로 병합 입력의 맨 뒤에 붙인다.
- 네 버킷이 모두 비어도 other가 있으면 fallback 본문 대신 병합 입력으로 최종 요약을 만든다. 텍스트가 있는 섹션이 하나도 없을 때만 fallback 본문을 쓴다.
- LLM 호출은 선택된 other 섹션 수 + 1회 늘어난다. 네 버킷이 다 찬 논문은 other가 1개라 2회, 버킷이 빈 논문은 더 늘어난다(최대 13회).
- 새로 생성하는 요약부터 적용된다. 캐시된 요약은 다시 만들지 않는다(F22 프롬프트 버전은 1 그대로).
- G5·G4 차이(3개 시나리오)
  - `summary:summary_generate_gpt5_mini`, `summary:summary_generate_gpt5`(2608.30597)
    - other 섹션 `Front Matter` 1개의 섹션 호출과 other 버킷 호출이 더해진다. LLM 호출은 16회에서 18회로 는다.
    - 최종 호출의 본문 끝에 `[기타 해설]`과 `[기타 원문 근거] [근거 1] Front Matter`가 붙는다.
    - 가짜 서버 답은 마지막 사용자 메시지의 해시라 요약 문자열이 `Fake reply e9402cd9982b: ...`에서 `Fake reply 72c2cb03cd6c: ...`로 바뀐다. gpt5_mini의 두 번째 응답은 캐시라 같은 값이다.
  - `summary:summary_model_failure`(2609.26796): other 섹션 5개의 섹션 호출 5회와 버킷 호출 1회가 더해진다. LLM 호출은 10회에서 16회로 늘고, 최종 호출은 여전히 400이다. 응답(500)은 같다.
- 참고: 두 논문 모두 첫 other 섹션이 `Front Matter`(제목·저자·소속)다. 이런 섹션까지 요약에 넣을지는 별도 판단이 필요하다.

## F23: 추적 runtime 라벨

- 옛 구현은 개요·핵심 포인트·상세 요약 체인에 runtime을 넘기지 않아 운영에서도 `runtime:dev`로 기록했다. 이제 다섯 stage(`paper_overview`, `paper_key_findings`, `summary`, `paper_chat`, `agent_chat`) 모두 `RUNTIME` 설정을 해석한 값(`development`→`dev`, `prod`→`production`, 모르는 값→`dev`)을 태그 `runtime:`과 메타데이터 `runtime`에 쓴다.
- v2 추적 모듈에는 제품 경로의 다섯 stage만 있다. 옛 허용 목록에만 있던 `translation`, `rag_answer`, `paper_detail_document`, `analyze_paper_detail`은 옮기지 않았다. 파이프라인 stage는 P3 파이프라인 이식에서 다룬다.
- G5·G4 차이: 없음. 추적은 가짜 서버 요청 본문에 들어가지 않고 골든 실행에서는 꺼져 있다.
