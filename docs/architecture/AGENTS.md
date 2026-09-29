# ArXplore AI 작업 규칙

이 문서는 ArXplore 저장소에서 AI 도구가 작업할 때 따르는 규칙이다. 수집, 파싱, 적재, 임베딩, 검색, 생성, API, UI가 모두
갖춰진 상태이고, 작업은 이 구조와 공용 계약 위에서 한다.

## 1. 먼저 읽을 문서

1. `README.md`: 기능, 구조, 평가 요약
2. `docs/architecture/ARCHITECTURE.md`: 런타임, 모듈, 스키마와 마이그레이션, 큐, 검색, API
3. `docs/SETUP.md`: 실행, 테스트, 골든 테스트, CI
4. `docs/EVALUATION.md`, `eval/README.md`: 평가 수치, 측정 조건, 한계, 하네스 사용법
5. `docs/worklog/README.md`와 페이즈별 `INDEX.md`: 결정 기록

문서와 코드가 다르면 코드를 바로 고치지 않는다. 현재 동작을 먼저 확인해 설명하고, 문서를 맞출지 코드를 고칠지 구분한 뒤 진행한다.

## 2. 코드 규칙

- 코드와 설정 파일에 주석과 docstring을 쓰지 않는다. 설명이 필요한 곳은 이름과 구조로 드러낸다. LLM에 넘어가는 도구 설명처럼 기능인 문자열은 `@tool(description=...)` 같은 명시적 인자로 둔다. `.env.example`의 변수 설명만 예외다.
- Python 3.14와 uv를 쓴다. 의존성은 `pyproject.toml`과 `uv.lock`이 기준이고, 바꾸면 `uv lock`으로 lock을 함께 갱신한다. 평가 전용 의존성은 `eval` 그룹이다.
- 커밋 전에 `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy`(strict, `src`), `uv run pytest tests/unit`을 통과시킨다. 프론트는 `npm run lint`, `npm run typecheck`, `npm test`, `npm run build`다.
- API, DB, OpenAI, LangGraph는 async로 쓴다. 파싱·정제·청킹 같은 CPU 작업은 동기 함수로 두고 `asyncio.to_thread`로 부른다.
- ORM을 쓰지 않는다. SQL은 `db/repositories`와 검색 모듈의 상수·빌더로 둔다. 검색 SQL은 손으로 튜닝한 결과라 문장 구조를 바꾸면 골든으로 확인한다.
- 요청 경로, worker, 스케줄러에 DDL을 넣지 않는다. 스키마는 새 번호 마이그레이션으로만 바꾼다.
- 외부 요청의 헤더와 본문에 사용자 이메일 같은 개인정보를 넣지 않는다.

## 3. 공용 계약

아래는 여러 계층이 함께 쓰는 계약이다. 바꿔야 하면 이유와 영향 받는 계층을 먼저 정리하고, 관련 문서와 골든을 같은 작업에서 고친다.

### HTTP API

- 경로, 메서드, 끝 슬래시, 상태 코드, 오류 문구, 검사 순서가 계약이다. 모든 오류는 `{"error": 문구}`이고, 요청 검증 실패는 400 "잘못된 요청입니다.", 405는 빈 본문과 `Allow`다.
- 요청·응답 모델과 SSE 이벤트는 `arxplore openapi`가 내보내는 스키마이고, 그 출력이 `frontend/openapi.json`에 커밋되어 있다. 프론트 타입은 이 파일에서 `npm run generate:api`로 만든다. 스키마를 바꾸면 두 파일을 같이 갱신한다(`uv run arxplore openapi --output frontend/openapi.json`, `cd frontend && npm run generate:api`). CI와 `npm run check:api`가 어긋남을 잡는다.
- 새 API 경로는 `docker/web/nginx.conf`의 location과 `frontend/vite.config.ts`의 프록시 목록에 같이 넣는다. SSE 경로는 버퍼링을 끈 location을 쓴다.
- LLM을 부르는 6개 경로(`analyze/`, `summary/`, 상세 챗과 어시스턴트의 JSON·SSE)가 llm rate limit 버킷 하나를 공유하고, `detail.json`은 detail 버킷이다.

### SSE

엔드포인트는 둘 다 `POST`이고 본문은 `{"message": str, "history": [{"role": "user"|"assistant", "content": str}]}`다.

- 에이전트: `/papers/assistant/stream/`
- 상세 챗: `/papers/<arxiv_id>/chat/stream/`

이벤트 순서는 `chunk`* → `citations`(정확히 1회, 빈 배열 가능) → `[DONE]`이다. 스트리밍 도중 오류가 나면 `error` → `[DONE]`으로 끝나고 `citations`는 보내지 않는다.

```text
data: {"chunk": "..."}
data: {"citations": [{"arxiv_id": str, "title": str, "url": str, "section_title": str|null, "chunk_id": int|null, "in_answer": bool}]}
data: {"error": "..."}
data: [DONE]
```

- `in_answer`: 답변이 이 출처를 실제로 인용했으면 `true`다. 에이전트는 답변 속 `[제목](URL)` 링크를 도구 결과와 대조하고(URL 또는 arXiv ID 일치), 상세 챗은 `[1]`, `[2]` 같은 발췌문 번호를 대조한다. 답변에 인용이 하나도 없으면 사용한 출처 전체를 `in_answer: false`로 보낸다. 도구 결과에 없는 링크는 서버 로그에 경고로 남기고 답변은 고치지 않는다.
- 상세 챗의 1번 출처는 논문 초록이다(`section_title: "Abstract"`, `chunk_id: null`). 트렌딩 도구에서 온 에이전트 출처도 `chunk_id: null`이다.
- 스트리밍 전의 검증 실패는 JSON으로 응답한다: 빈 메시지·4,000자 초과 메시지·잘못된 `history`는 400, 없는 논문은 404(상세 챗), 서버 `OPENAI_API_KEY`가 없으면 503이다. `history`는 서버에서 user/assistant만 남기고 최근 20개, 메시지당 4,000자로 자른다.
- `error` 문구는 사용자용 고정 문구다. 예외 메시지는 응답에 싣지 않고 서버 로그에 남긴다.
- 비스트리밍 경로의 응답: 에이전트 `/papers/assistant/chat/`는 `{"answer", "citations"}`, 상세 챗 `/papers/<arxiv_id>/chat/`은 `{"answer", "citations", "retrieval_mode": "hybrid"|"lexical"|"first_chunks"}`다.

에이전트 검색 도구(`search_paper_chunks_tool`)가 LLM에 넘기는 문자열도 계약이다. 도구 이름과 설명 문자열은 바꾸지 않는다.

```text
[1] 제목: {paper_title} | arxiv_id: {arxiv_id} | 섹션: {section_title 또는 -} | chunk_id: {chunk_id 또는 -}
출처(URL): {pdf_url 또는 https://arxiv.org/abs/{arxiv_id}}
내용: {context_text, 없으면 chunk_text}
```

결과가 없으면 "검색된 관련 논문이 없습니다. …지어내지 말고…" 문구를 돌려준다. 도구는 돌려준 hit을 요청 범위 레지스트리에 기록하고, 스트리밍 계층이 이를 `citations`로 만든다.

### 검색 결과

검색 결과는 `arxplore.domain.retrieval`의 frozen dataclass(`RetrievalHit`, `RetrievalContext`)다. `chunk_id`, `arxiv_id`, `chunk_text`, `section_title`, `content_role`, `score`는 에이전트 도구, 상세 챗, 평가 하네스, 골든 G1이 함께 쓰는 핵심 필드라 이름과 의미를 바꾸지 않는다. 평가 하네스는 검색의 공개 단계 함수만 쓰므로, 공개 함수의 시그니처를 바꾸면 `arxplore_eval`도 같이 고친다.

### 마이그레이션

- 스키마는 `src/arxplore/db/migrations/NNNN_<이름>.sql`로만 바꾸고, 번호는 하나씩 늘린다. 파일 하나가 트랜잭션 하나다.
- 적용된 파일은 고치지 않는다. 실행기가 체크섬으로 바뀐 파일을 찾아 멈추므로, 고칠 내용은 새 번호로 더한다.
- API, worker, 스케줄러는 시작할 때 대기 마이그레이션과 임베딩 차원을 확인한다. 임베딩 차원을 바꾸는 것은 스키마 변경이다.

### AI 결과 캐시

개요·핵심 포인트와 상세 요약은 `prompt_version`과 함께 캐시된다. 결과를 바꾸는 프롬프트나 후처리를 고치면 `ANALYSIS_PROMPT_VERSION`(`llm/detail.py`) 또는 `SUMMARY_PROMPT_VERSION`(`llm/summary_graph.py`)을 올린다. 이전 버전 행은 첫 요청 때 다시 생성되므로 LLM 비용이 든다.

## 4. 골든과 수정 ID

`tests/golden`의 골든 6종은 리팩토링 전 구현의 출력을 고정한 것이다. 모든 LLM·임베딩·arXiv 호출은 `tests/fakes`의 가짜 서버가 받는다.

| ID | 대상 |
| --- | --- |
| G1 | 검색 순위(평가 경로, 제품 경로, 상세 챗) |
| G2 | 저장된 섹션에서 만든 청크 |
| G3 | pypdf 파싱, HURIDOCS 응답 재생 |
| G4 | 가짜 서버가 받은 LLM 요청 본문(G5와 함께 캡처) |
| G5 | HTTP·SSE 응답 |
| G6 | 화면(PNG, innerText, ARIA) |

- 원본 골든 파일(`<이름>.json.gz`, `g6_ui/`)은 옛 구현의 기준선이라 고치지 않는다. 현재 구현과 다른 결과는 변형(`<이름>.v2.json.gz`, `g6_ui.v2/`)에 두고, 비교는 변형이 있으면 변형과, 없으면 원본과 한다.
- 동작을 바꾸지 않는 변경은 골든이 그대로 통과해야 한다.
- 동작을 바꾸는 수정은 수정 ID(F 번호)를 단 별도 커밋으로 하고, 골든 차이는 그 커밋에서만 허용한다.
  1. 수정 커밋에서 해당 골든의 변형을 다시 캡처한다(`uv run python -m scripts.goldens.capture_<종류>`, 기본으로 v2 변형을 쓴다).
  2. `tests/golden/data/<이름>.v2.diff.md`에 수정 ID 절을 더해 무엇이 몇 개 바뀌었고 왜 그런지 적는다.
  3. 커밋 메시지에 수정 ID와 사유를 쓰고, 결정은 worklog에 남긴다.
- G1 질의 임베딩 표를 다시 받는 `capture_g1_retrieval embed`는 실제 OpenAI API를 부른다(유료). 먼저 `embed --dry-run`으로 필요한 입력이 표에 모두 있는지 확인하고, 실제 호출은 사용자 승인 뒤에만 한다.

## 5. 테스트 DB와 비용

- 통합 테스트는 테스트마다 public 스키마를 지운다. `V2_TEST_DATABASE_URL`로 작업 전용 DB를 명시하고, 이름은 `v2_test`로 시작해야 한다. 여러 작업자가 쓰는 DB를 가리키지 않고, 끝나면 만든 DB를 지운다.
- 골든 DB의 `arxplore`(원본)와 `arxplore_tpl`(복사용 템플릿)에는 쓰지 않는다. 쓰기가 필요하면 `CREATE DATABASE ... TEMPLATE arxplore_tpl`로 만든 복사본을 쓴다.
- 유료 API(OpenAI LLM·임베딩) 호출은 사용자 승인 뒤에만 한다. 테스트, 골든, 스모크는 가짜 OpenAI 서버(`tests/fakes/openai_server.py`)에 `OPENAI_BASE_URL`을 붙여 돌린다.
- `.env`에는 실제 키가 있다. 골든 하네스와 스모크처럼 가짜 서버를 쓰는 실행은 `.env`가 없는 위치에서 돌리거나 설정을 명시적으로 넘긴다.

## 6. 저장소 구조 규칙

- `src/arxplore`의 패키지 경계를 지킨다. `domain`은 I/O가 없는 모델, `db`는 SQL, `sources`는 외부 HTTP, `parsing`은 순수 CPU 작업, `retrieval`과 `llm`은 도메인 로직, `pipeline`과 `scheduler`는 작업 조립, `api`는 HTTP 계층이다. 명령은 패키지의 `commands.register`가 `cli.py`에 붙인다.
- API 라우트에 SQL을 직접 쓰지 않고, 외부 연동을 `llm`이나 `api`에 넣지 않는다.
- 평가 하네스(`src/arxplore_eval`)는 제품의 공개 함수만 쓰고, 연구용 설정은 `research_fusion`에 둔다.
- 프론트는 `pages`가 조립만 하고 기능은 `features`, 공용 코드는 `shared`에 둔다. 서버 상태는 TanStack Query로 다루고 스타일은 CSS Modules와 `shared/styles/tokens.css`의 토큰을 쓴다.
- 서버는 수집·저장과 스케줄러만 맡고, GPU 파싱과 임베딩은 로컬 worker가 한다. 이 분리를 깨지 않는다.

## 7. 용어 규칙

| 표준 표현 | 설명 |
|---|---|
| `논문 개요` | 단일 논문의 목적, 접근, 결과를 설명하는 overview |
| `핵심 포인트` | 단일 논문의 기여와 결과를 정리한 key findings |
| `상세 요약` | 논문 본문을 한국어로 구조화한 detailed summary |
| `논문 청크` | 검색과 근거 제시의 입력 단위 |
| `lexical retrieval` | 전문 검색 |
| `vector retrieval` | 임베딩 기반 검색 |
| `hybrid retrieval` | lexical과 vector 점수를 합친 검색 |
| `RAG 응답` | 검색 결과를 근거로 만든 답변 |
| `prepare job` | 날짜 단위 prepare 작업 |
| `골든` / `변형 골든` | 옛 구현 출력의 기준선 / 수정으로 달라진 현재 구현의 기준 |
| `수정 ID` | 동작을 바꾸는 수정의 F 번호 |

문맥에 맞는다는 이유로 다른 용어를 섞어 쓰지 않는다.
