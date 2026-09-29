# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### 컨테이너 실행

```bash
# 로컬 단독: 로컬 PostgreSQL(pgvector) 포함. PROD_POSTGRES_HOST=postgres-local:5432
docker compose --profile local-db up -d postgres-local
docker compose --profile local-db run --rm api migrate
docker compose --profile local-db up -d --build

# Vite HMR 개발 서버 (dev 프로필, 다른 프로필과 겹쳐 쓴다)
docker compose --profile local-db --profile dev up -d vite

# 원격 서버 DB 사용 (api + web). db-status가 ready가 아니면 멈추고 migrate 명령을 안내한다
bash scripts/setup.sh

# 내리기
docker compose down

# GPU parser + worker 포함
docker compose --profile parser up -d --build

# 서버 인프라 (PostgreSQL, scheduler) — 원격 서버에서 실행
bash scripts/setup-server.sh

# SSH 포트 포워딩 (원격 서버 PostgreSQL → localhost)
bash scripts/setup.sh forward [start|stop|status|restart]
```

### 개발

```bash
# Python 3.14 환경 + dev 그룹 (평가 의존성은 uv sync --group eval)
uv sync

# 스키마 적용(멱등). 호스트에서는 POSTGRES_HOST, 컨테이너에서는 docker compose run --rm api migrate
uv run arxplore migrate --dry-run
uv run arxplore migrate
uv run arxplore db-status

# API 개발 서버 (127.0.0.1:8001). frontend/dist까지 서빙하려면 SERVE_FRONTEND_DIST=true
uv run arxplore api --reload

# 단위 테스트 (DB·API 키 불필요)
uv run pytest tests/unit

# 통합 테스트 (작업 전용 v2_test* pgvector DB 필요. 테스트마다 public 스키마를 지운다)
V2_TEST_DATABASE_URL=postgresql://arxplore@localhost:55432/v2_test_<작업> V2_TEST_REQUIRE_DB=1 uv run pytest tests/integration

# 가짜 서버·골든 테스트 (골든 DB·PDF·Playwright가 없으면 해당 테스트는 건너뜀)
uv run pytest tests/fakes tests/golden

# 린트·타입
uv run ruff check . && uv run ruff format --check . && uv run mypy

# 프론트엔드
cd frontend && npm ci && npm run lint && npm run typecheck && npm test && npm run build
cd frontend && npm run check:api          # 백엔드 OpenAPI 스키마와 openapi.json·생성 타입 대조

# API 스키마를 바꾼 뒤 프론트 타입 갱신
uv run arxplore openapi --output frontend/openapi.json && (cd frontend && npm run generate:api)

# 파이프라인 작업 수동 실행과 스케줄러 기록
uv run arxplore run collect --target-date YYYY-MM-DD
uv run arxplore run rechunk              # 기본 dry-run, --apply로 반영
uv run arxplore run reparse --since YYYY-MM-DD --until YYYY-MM-DD   # 기본 dry-run
uv run arxplore scheduler history

# 검색 평가 (eval/README.md). lexical만은 API 키 불필요
uv run arxplore-eval retrieval --methods lexical

# 검색 쿼리 실행 계획 확인 (EXPLAIN ANALYZE, 읽기 전용 트랜잭션)
uv run arxplore-eval explain --query "..."

# worklog 항목 만들기와 색인
python scripts/worklog.py new "제목" --phase 5 --area backend
python scripts/worklog.py index && python scripts/worklog.py check

# 컨테이너 셸과 로그
docker compose exec api bash
docker compose logs -f [api|web|vite|worker]
```

### Key Ports

| Service | Host Port |
|---------|-----------|
| Web (nginx) | 80 (`PROD_HTTP_PORT`) |
| API | 8001 (compose에서는 내부에만 공개, 호스트 실행 `arxplore api` 기본값) |
| Vite (dev) | 5173 (`FRONTEND_PORT`) |
| Layout Parser | 5060 (`LAYOUT_PARSER_PORT`) |
| PostgreSQL | 15432 (`SERVER_POSTGRES_PORT`, local-db 프로필은 127.0.0.1에만 공개) |
| 골든 DB | 55432 (로컬 Docker `arxplore-golden-pg`, 127.0.0.1) |

## Architecture

ArXplore는 HuggingFace Daily Papers + arXiv 논문을 수집·처리해 RAG 기반 채팅 인터페이스로 제공하는 플랫폼입니다.

### Data Flow

```
HF Daily Papers / arXiv
  → arxplore scheduler (collect 매일 18:00, maintenance 3시간마다, langsmith_cleanup 매일 03:00)  [서버]
  → PostgreSQL raw_daily_papers (raw payload JSONB) + prepare_jobs queue, 한 트랜잭션  [서버]
  → prepare_jobs LISTEN/NOTIFY (commit 뒤 전달)
  → arxplore worker (PDF parse → OpenAI 임베딩 API)  [--profile parser]
  → PostgreSQL: papers, paper_fulltexts, paper_chunks, paper_embeddings (pgvector)
  → Retrieval (RETRIEVAL_MODE=hybrid 기본: 질의 임베딩 키가 있으면 hybrid, 없거나 실패하면 lexical)
  → 개요·핵심 포인트·상세 요약(결과 캐시) + LangGraph 에이전트 + 상세 챗
  → FastAPI (공개 접근, IP당 rate limit, SSE) → React UI
```

### Docker Compose 구조

로컬 서비스는 `compose.yaml`(프로젝트 `arxplore`), 서버 인프라는 `compose.server.yaml`(프로젝트 `arxplore_server`)로 나눠 관리합니다. `api`·`worker`·`scheduler`는 같은 app 이미지(`docker/app/Dockerfile`, python:3.14-slim, uv, 비root UID 1000, 진입점 `arxplore`)를 명령만 바꿔 씁니다.

| 서비스 | 프로필 | 설명 |
|--------|--------|------|
| `api` | (기본) | `arxplore api` (uvicorn, `API_WORKERS` 기본 1). `/bootstrap.json` HTTP healthcheck |
| `web` | (기본) | React 빌드 서빙 + API 프록시 + 보안 헤더·gzip(nginx stable-alpine). api가 healthy일 때 시작 |
| `vite` | `dev` | 프론트엔드 HMR 개발 서버 (`npm ci`, `API_PROXY_TARGET=http://api:8001`) |
| `worker` | `parser` | prepare queue 소비(`arxplore worker --loop`). `LAYOUT_PARSER_BASE_URL` 기본 `http://layout-parser:5060`, layout-parser가 healthy일 때 시작 |
| `layout-parser` | `parser` | HURIDOCS GPU PDF 파서(v0.0.31) |
| `postgres-local` | `local-db` | 로컬 단독 실행용 PostgreSQL 16 + pgvector (`127.0.0.1:${SERVER_POSTGRES_PORT:-15432}`) |

서버 compose는 `postgres`(포트는 `TAILSCALE_SERVER_IP`에만 바인딩, 외부 볼륨)와 `scheduler` 두 서비스입니다.

nginx(`docker/web/nginx.conf`)는 SPA 경로를 `index.html`로 돌리고 API만 FastAPI로 프록시한다. 새 API 경로를 추가하면 nginx location과 `frontend/vite.config.ts` 프록시에 같이 넣어야 한다. SSE 경로(`/papers/assistant/stream/`, `/papers/<id>/chat/stream/`)는 버퍼링을 끈 location을 쓴다.

의존성: `pyproject.toml` + `uv.lock`(dependency group `dev`, `eval`), `frontend/package.json` + `package-lock.json`.

### Key Architectural Split

**Server-side** (`compose.server.yaml`): PostgreSQL, scheduler — 항상 켜져 있는 원격 서버에서 실행. PostgreSQL이 유일한 저장소다(raw payload·파이프라인 상태·정제 데이터·벡터·작업 큐·AI 캐시·스케줄러 기록).

**Local** (`compose.yaml`): api + web은 로컬에서 실행하고, vite는 `dev` 프로필로 필요할 때만 띄운다. parser 프로필은 GPU 보유 시에만 추가. `local-db` 프로필은 원격 서버 없이 웹만 띄울 때 쓴다(수집이 없으므로 빈 DB).

**worker는 서버가 아닌 로컬에서 실행** — GPU가 필요한 HURIDOCS 파싱을 로컬에서 처리하고 결과를 서버 DB에 직접 적재한다. 임베딩은 GPU가 아니라 OpenAI API(`text-embedding-3-large`, 1536차원)로 만든다. 이 분리를 깨지 말 것.

**스키마는 `arxplore migrate`만 바꾼다.** `src/arxplore/db/migrations`의 번호 SQL(0001~0006)을 파일 하나당 트랜잭션 하나로 적용하고 `schema_migrations`에 체크섬을 기록한다. api·worker·scheduler는 시작할 때 대기 마이그레이션과 임베딩 차원을 확인하고 맞지 않으면 뜨지 않는다. 요청 경로와 worker에는 DDL이 없다.

### Module Responsibilities

- **`src/arxplore/`** — 제품 패키지 (`arxplore` 명령, `cli.py`가 각 패키지의 `commands.register`를 붙인다)
  - `config.py` — Pydantic `AppSettings`(실행 위치의 `.env` 또는 `ARXPLORE_ENV_FILE`). 옛 `DJANGO_*`, `OPENAI_API_BASE` 이름도 별칭으로 받는다
  - `context.py` — 요청 범위 OpenAI 키·모델(`override_openai_runtime`), 검색 관찰 훅(`observe_retrieval`)
  - `domain/` — frozen dataclass 모델
  - `db/` — psycopg 3 async 풀, 마이그레이션 실행기, 상시 LISTEN 연결, 테이블별 저장소 함수(`repositories/`)
  - `sources/` — HF Daily Papers JSON API, arXiv 검색·메타데이터
  - `parsing/` — HURIDOCS 클라이언트, pypdf, 정제, 섹션, `content_role`, 청커, 3단계 폴백
  - `retrieval/` — 질의, lexical, vector, 융합, 다양성, 문맥 창, 모드 선택(`service.Retriever`)
  - `llm/` — 모델 팩토리, 프롬프트, 개요·핵심 포인트, 상세 요약 그래프, 상세 챗, 인용, 에이전트(`agent/`), 추적
  - `pipeline/` — 수집, 백필, 보강, prepare, 임베딩, worker, LangSmith 정리, 코퍼스 도구(rechunk, reparse)
  - `scheduler/` — cron 작업, 실행 기록(`scheduler_runs`), advisory lock
  - `api/` — FastAPI 앱, CSRF·보안 헤더, rate limit, SSE, 서비스(listing, detail, generation, chat), 라우트
- **`src/arxplore_eval/`** — 검색·생성 평가 하네스(`arxplore-eval` 명령). 제품 공개 함수만 쓰고 연구용 융합은 `research_fusion`
- **`frontend/`** — React 19 + React Router 7(데이터 라우터, lazy 라우트) + TanStack Query + TypeScript + CSS Modules. `app/`(라우터·provider·오류 경계), `pages/`(조립), `features/`(papers-list, paper-detail, chat, paper-chat, assistant), `shared/`(api·ui·lib·styles). 목록 상태는 URL 쿼리가 원본(`features/papers-list/listParams.ts`), 모달은 `shared/ui/useModalDialog.ts`(Esc·포커스 복귀), 색상은 `shared/styles/tokens.css`의 토큰을 쓴다. API 타입은 `openapi.json`에서 생성한다
- **`tests/`** — `unit`, `integration`(pgvector), `golden`(골든 6종과 하네스), `fakes`(가짜 OpenAI·arXiv 서버)
- **`scripts/`** — 셋업 스크립트, `goldens/`(골든 캡처), `worklog.py`
- **`eval/`** — 평가 데이터(질의셋, 케이스, 결과, 캐시)

### PDF Parsing Strategy

3단계 폴백:
1. HURIDOCS Layout Parser (Docker, `LAYOUT_PARSER_BASE_URL`. 코드에는 기본값·자동 감지가 없고, compose의 worker만 비어 있을 때 `http://layout-parser:5060`을 넣는다. 비어 있으면 이 단계를 건너뛴다. 422·5xx·연결 오류는 재시도하고 타임아웃은 재시도하지 않는다)
2. pypdf
3. abstract only

청크에 `content_role`, `section_title`, `parser_metadata`, `quality_metrics` 저장. 청킹은 섹션 단위 글자 수 기준(최대 1800자, 이웃 청크 겹침 200자).

재처리는 멱등하다. source 순위 `layout_pdf > pdf > fallback_abstract`에서 기존보다 낮은 순위 결과는 저장하지 않고, 같은 source에 `content_hash`까지 같으면 본문·청크 교체를 건너뛴다. 청크 텍스트가 같으면 메타데이터만 갱신해 청크 id와 임베딩을 보존한다. 확정된 layout 본문(청크·`content_hash` 있음)은 force가 아니면 다운로드·파싱 없이 건너뛴다. 강제 재처리는 worker `--force`, job payload `force: true`, `arxplore run reparse`. 청커만 바뀌었으면 `arxplore run rechunk`로 저장된 섹션에서 다시 청킹한다.

### Retrieval

제품 경로 선택: `arxplore.retrieval.service.Retriever.retrieve_contexts`. 에이전트 도구와 상세 챗이 공유한다.
- `RETRIEVAL_MODE=hybrid`(기본)이고 질의 임베딩 키(요청 범위 `override_openai_runtime` 키 > 서버 `OPENAI_API_KEY`)가 있으면 hybrid, 키가 없거나 임베딩 호출이 `OpenAIError`로 실패하면 lexical. `RETRIEVAL_MODE=lexical`이면 항상 lexical
- **Lexical** — PostgreSQL 전문 검색(`english` 설정). 제목·초록·청크 tsvector 생성 컬럼 + GIN 인덱스. 한국어 질의는 거의 맞지 않는다
- **Vector** — pgvector 코사인 거리 (text-embedding-3-large, 1536 dims), HNSW 인덱스, 2단계(후보 → rerank), `VECTOR_MIN_SIMILARITY`로 하한 설정. `hnsw.ef_search` 설정문은 `retrieval.vector.hnsw_ef_search_statement`
- **Hybrid** — 두 채널을 각자 풀 연결로 `asyncio.gather`에서 동시에 돌리고, 채널 점수 per-query min-max 정규화의 convex combination(0.35·lexical + 0.65·vector, `arxplore.retrieval.fusion`)으로 합친다. vector 결과가 있으면 lexical 부분 일치 행 제외. 예전 가중 RRF 규칙은 `arxplore_eval.research_fusion.LEGACY_RULES_FUSION`(ablation `hybrid_rules`)
- DB 연결은 `arxplore.db.pool`(프로세스당 `POSTGRES_POOL_MAX`, 기본 8). 검색 SQL은 `prepare=False`로 보낸다

오프라인 평가 하네스: `src/arxplore_eval`, `arxplore-eval`, 데이터는 `eval/`.

### Agent

`arxplore.llm.agent` — `langchain.agents.create_agent` 기반 에이전트. `RequestModelMiddleware`(요청 범위 모델·키), `StepLimitMiddleware`(남은 단계가 부족할 때 도구 호출을 단계 제한 안내로), `recursion_limit=AGENT_RECURSION_LIMIT` 기본 12(초과 시 안내 문구로 종료). AGENT_STREAM_BUFFER_CHARS=120(도구 호출 전 서두 누출 방지 버퍼; 120자 넘는 서두 뒤 도구 호출은 막지 못함)
- `search_paper_chunks_tool` — `retrieve_contexts`(hybrid → lexical) 기반 청크 검색 5개. hit을 요청 범위 레지스트리에 기록
- `get_trending_papers_tool` — 최근 논문 10편을 추천수 순으로
- `llm/citations.py` — 답변 속 링크(에이전트)·`[n]` 번호(상세 챗)를 도구 hit과 대조해 `citations` 이벤트(`in_answer`)를 만든다

`arxplore.llm.paper_chat` — 상세 페이지 챗. 질문으로 해당 논문 안을 검색해 초록(1번) + 청크 최대 5개로 답하고 SSE로 스트리밍. 검색 결과가 없으면 앞 청크(`retrieval_mode: "first_chunks"`).

SSE 이벤트 계약(`docs/architecture/AGENTS.md` 3절): `chunk`* → `citations` 1회 → `[DONE]`, 오류 시 `error` → `[DONE]`. 클라이언트가 연결을 끊으면 생성을 취소한다.

### Web access model

- 계정·로그인·세션·즐겨찾기가 없다. 모든 엔드포인트가 공개이고, LLM·임베딩 호출은 서버 `OPENAI_API_KEY`를 쓴다
- `analyze/`·`summary/`는 현재 프롬프트 버전의 캐시가 있으면 `{"cached": true, ...}`, 없으면 생성해 캐시한 뒤 `{"cached": false, ...}`. 개요는 gpt-5-mini, 상세 요약 모델은 요청 body의 `model`(gpt-5-mini 또는 gpt-5, 목록 밖이면 400). 서버 키가 없으면 생성·챗·에이전트는 503
- bootstrap(`/bootstrap.json`)은 CSRF 쿠키를 심고 `default_summary_model`, `available_summary_models`만 돌려준다. POST는 `X-CSRFToken` 헤더가 쿠키와 같아야 하고, Origin이 있으면 같은 출처이거나 `CSRF_TRUSTED_ORIGINS`에 있어야 한다. 실패는 403 JSON
- 오류는 모두 `{"error": 문구}`. 요청 검증 실패 400, 405는 빈 본문 + `Allow`
- rate limit은 클라이언트 IP(`RATE_LIMIT_IP_HEADER`, `none`이면 소켓 주소) 단위: LLM을 부르는 6개 경로(analyze, summary, 상세 챗 JSON·SSE, 어시스턴트 JSON·SSE)가 `RATE_LIMIT_LLM_PER_MINUTE`을 공유, `detail.json`은 `RATE_LIMIT_DETAIL_PER_MINUTE`. `REDIS_URL`이 없으면 프로세스 메모리라 워커 수만큼 허용량이 늘어난다

### Architectural Contracts (do not break)

- HTTP API: 경로·메서드·끝 슬래시·상태 코드·오류 문구·검사 순서. 요청·응답·SSE 모델은 `arxplore openapi` 스키마(`frontend/openapi.json`에 커밋)와 프론트 생성 타입이 기준이다.
- SSE 이벤트 순서와 `citations` 필드: `arxiv_id`, `title`, `url`, `section_title`, `chunk_id`, `in_answer`.
- Retrieval 결과 핵심 필드: `chunk_id`, `arxiv_id`, `chunk_text`, `section_title`, `content_role`, `score`(`arxplore.domain.retrieval.RetrievalHit`).
- 마이그레이션: 적용된 파일은 고치지 않고 새 번호를 더한다.
- 에이전트 도구 이름·설명 문자열과 도구 출력 형식.
- AI 캐시: 결과를 바꾸는 프롬프트·후처리 변경은 `ANALYSIS_PROMPT_VERSION`·`SUMMARY_PROMPT_VERSION`을 올린다.

## Rules

- **주석 금지**: 코드와 설정 파일에 주석과 docstring을 쓰지 않는다. 설명이 필요하면 이름과 구조로 드러낸다. LLM에 넘어가는 도구 설명처럼 기능인 문자열은 `@tool(description=...)` 같은 명시적 인자로 둔다. `.env.example`의 변수 설명만 예외다. 프론트는 ESLint `local/no-comments`가 잡는다.
- **골든 절차**: `tests/golden`의 원본 골든은 옛 구현의 출력이라 고치지 않는다. 동작을 바꾸지 않는 변경은 골든이 그대로 통과해야 하고, 동작을 바꾸는 수정은 수정 ID(F 번호)를 단 별도 커밋에서만 골든 차이를 허용한다. 그 커밋에서 변형 골든을 다시 캡처하고(`uv run python -m scripts.goldens.capture_<종류>`, 기본으로 `.v2` 변형을 쓴다) `tests/golden/data/<이름>.v2.diff.md`에 원인과 수치를 적는다. 자세한 절차는 `docs/architecture/AGENTS.md` 4절.
- **골든 DB**: Docker 컨테이너 `arxplore-golden-pg`(localhost:55432, user `arxplore`, trust). DB `arxplore`는 골든 원본이라 쓰지 않고, `arxplore_tpl`은 복사용 템플릿이라 접속하지 않는다. 쓰기가 필요하면 `CREATE DATABASE ... TEMPLATE arxplore_tpl` 복사본을 쓴다.
- **테스트 DB**: 통합 테스트는 테스트마다 public 스키마를 지운다. `V2_TEST_DATABASE_URL`로 작업 전용 `v2_test*` DB를 명시하고(공용 `v2_test`를 가리키지 않는다) 끝나면 지운다.
- **비용**: 유료 API(OpenAI LLM·임베딩) 호출은 사용자 승인 뒤에만 한다. 테스트·골든·스모크는 가짜 OpenAI 서버(`tests/fakes/openai_server.py`)에 `OPENAI_BASE_URL`을 붙여 돌리고, 실제 키가 든 `.env`가 없는 위치에서 실행한다. G1 `capture_g1_retrieval embed`는 실제 API를 부르므로 `--dry-run`으로 먼저 확인한다.
- **개인정보**: 외부 요청의 헤더·본문에 사용자 이메일 등 개인정보를 넣지 않는다.

## Configuration

모든 런타임 설정은 실행 위치의 `.env`(compose는 루트 `.env`). `src/arxplore/config.py`가 Pydantic `BaseSettings`로 로드하고, 빈 값은 설정하지 않은 것으로 본다. 전체 목록과 필수/선택 구분은 **`.env.example`** 이 기준이다(`cp .env.example .env`).

로컬 실행 필수:

```
PROD_POSTGRES_HOST          # api/worker 컨테이너의 POSTGRES_HOST. host 또는 host:port
POSTGRES_HOST               # 호스트에서 실행하는 arxplore 명령용 (예: localhost)
SERVER_POSTGRES_PORT        # 서버가 공개하는 PostgreSQL 포트. compose 기본 15432, config.py 기본 5432
POSTGRES_DB / APP_POSTGRES_DB / POSTGRES_USER / POSTGRES_PASSWORD
OPENAI_API_KEY              # 웹 AI 기능(개요·요약 생성, 챗, 에이전트, 질의 임베딩)과 worker 임베딩. 없으면 생성·챗 503
```

서버·worker 필수:

```
LAYOUT_PARSER_BASE_URL      # 코드 기본값 없음. compose worker는 비면 http://layout-parser:5060
TAILSCALE_SERVER_IP         # 서버 compose 포트 바인딩 + setup.sh forward
```

선택(기본값은 `.env.example` 주석 참고):

```
OPENAI_BASE_URL=            # OpenAI 호환 주소(가짜 서버). 옛 이름 OPENAI_API_BASE
DEBUG=false                 # "true"일 때만 켜짐(/docs, ALLOWED_HOSTS *). 옛 이름 DJANGO_DEBUG
ALLOWED_HOSTS / CSRF_TRUSTED_ORIGINS / SECURE_COOKIES=false   # 옛 이름 DJANGO_*
SERVE_FRONTEND_DIST=false   # api가 frontend/dist를 SPA로 서빙
RUNTIME=development         # LangSmith runtime 태그
API_WORKERS=1               # compose api의 uvicorn 워커 수
REDIS_URL=                  # rate limit 공유 저장소. 비우면 프로세스 메모리
RATE_LIMIT_ENABLED=true / RATE_LIMIT_LLM_PER_MINUTE=30 / RATE_LIMIT_DETAIL_PER_MINUTE=60 / RATE_LIMIT_IP_HEADER=X-Real-IP
RETRIEVAL_MODE=hybrid       # hybrid | lexical
VECTOR_MIN_SIMILARITY=0
AGENT_RECURSION_LIMIT=12    # 최소 4
POSTGRES_POOL_MAX=8 / POSTGRES_POOL_TIMEOUT=30 / POSTGRES_CONNECT_TIMEOUT=10
SCHEDULER_TIMEZONE=Asia/Seoul
PREPARE_BACKFILL_ENABLED=false   # true면 worker가 백필 잡도 처리
PREPARE_JOB_STALE_SECONDS=900 / PREPARE_JOB_MAX_ATTEMPTS=3
```

주소 규칙: `POSTGRES_HOST`/`PROD_POSTGRES_HOST`는 `host` 또는 `host:port`. 포트를 생략하면 `SERVER_POSTGRES_PORT`(서버가 호스트에 공개하는 포트)를 쓴다. 같은 compose 네트워크 안에서 컨테이너 이름으로 붙을 때는 `postgres-local:5432`, `postgres:5432`처럼 내부 포트를 명시한다.

Vite 프록시는 `frontend/vite.config.ts`에서 `API_PROXY_TARGET`(기본 `http://127.0.0.1:8001`, compose의 vite는 `http://api:8001`)으로 API 경로를 포워딩하고 `X-Real-IP`를 붙인다.

## Reference Docs

1. `README.md` — 소개 화면, Features, Architecture, Tech Stack, Evaluation 요약, Getting Started, Project Structure, Documentation. 현재 상태만 쓰고, 로드맵·단서·측정 조건·한계·날짜·근거 링크는 README에 쓰지 않고 `docs/EVALUATION.md`와 worklog에 둔다
2. `docs/architecture/ARCHITECTURE.md` — 런타임, 모듈, 스키마와 마이그레이션, 큐, 스케줄러, 검색, LLM, API, 기술적 결정과 트레이드오프
   - `docs/EVALUATION.md` — 검색·생성 평가 상세, 지연, 측정 조건, 알려진 한계와 로드맵 전체
   - `docs/SETUP.md` — 실행 방법, compose 프로필, 운영 기본값, 코퍼스 도구, 테스트·골든 절차와 CI
3. `docs/architecture/AGENTS.md` — AI 작업 규칙, 공용 계약(API, SSE, 검색 결과, 마이그레이션), 골든과 수정 ID 절차, 용어
4. `eval/README.md` — 평가 하네스(`arxplore-eval`) 사용법과 지표 정의
5. `.env.example` — 환경 변수 전체 목록
6. `docs/worklog/` — 결정 이유는 docs/worklog. 페이즈별 `INDEX.md`와 트러블슈팅 모음 `TROUBLESHOOTING.md`는 `python scripts/worklog.py index`가 생성하고 `check`가 CI·pre-commit에서 검사한다
