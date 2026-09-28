# G5·G4 HTTP 골든 v2 변형: 원본과 다른 점

변형 파일은 `g5_http.v2.json.gz`이고 `GOLDEN_IMPL=v2`일 때만 쓰인다. 원본 `g5_http.json.gz`는 옛 Django 구현 기준으로 그대로 둔다.

다시 만드는 명령(저장소 루트): `GOLDEN_IMPL=v2 v2/.venv/bin/python -m scripts.goldens.capture_g5_http --variant v2`

- 응답은 `compare_responses_v2` 규칙으로 비교한다. 대상은 상태 코드, content-type 미디어 타입, allow, set-cookie 속성, 허용 목록 헤더, SSE 바이트, JSON 값, 그 밖의 본문 바이트다. 가짜 OpenAI·arXiv 요청 기록(G4)은 그대로 비교한다.
- 이식 동등성: 0530057 시점에 v2는 원본 골든과 G5 60/60 시나리오, G4 60/60 시나리오가 같았다(`tests/golden/test_g5_http.py` 121 passed).
- 변형 파일은 차이가 처음 생긴 수정에서 만든다. 그 전까지 v2는 원본을 쓴다.

## F22: AI 캐시 키에 프롬프트 버전

- 마이그레이션 `0004_ai_cache_prompt_version`이 `paper_ai_overviews`와 `paper_ai_detailed_summaries`에 `prompt_version INTEGER NOT NULL DEFAULT 1`(양수 CHECK)을 더한다. 기존 행은 모두 버전 1이 된다.
- 조회는 `prompt_version`이 현재 버전과 같은 행만 캐시로 본다. 다른 버전의 행은 없는 것으로 보고 새로 만들며, 저장할 때 같은 행을 덮어쓰고 버전을 바꾼다. 행 키(`arxiv_id`, 요약은 `arxiv_id, model`)는 그대로다.
- 현재 버전은 개요·핵심 포인트 `ANALYSIS_PROMPT_VERSION = 1`, 상세 요약 `SUMMARY_PROMPT_VERSION = 1`이다. 옛 프롬프트와 같으므로 기존 캐시가 그대로 쓰인다.
- G5·G4 차이: 없음. 골든 DB 복사본에 0004를 적용한 뒤에도 60/60, 60/60이 같다. 캐시가 있는 `analyze_cached`, `summary_cached`, `*_without_key`의 캐시 응답이 그대로 나온다.
