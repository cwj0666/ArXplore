# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### 컨테이너 실행

```bash
# 로컬 단독: 로컬 PostgreSQL(pgvector) 포함. PROD_POSTGRES_HOST=postgres-local:5432
docker compose --profile local-db up -d postgres-local
docker compose --profile local-db up -d --build

# Vite HMR 개발 서버 (dev 프로필, 다른 프로필과 겹쳐 쓴다)
docker compose --profile local-db --profile dev up -d vite

# 원격 서버 DB 사용 (django + nginx)
bash scripts/setup.sh

# 내리기
docker compose down

# GPU parser + prepare-worker 포함
docker compose --profile parser up -d --build

# 서버 인프라 (PostgreSQL, MongoDB, Airflow) — 원격 서버에서 실행
bash scripts/setup-server.sh

# SSH 포트 포워딩 (원격 서버 → localhost)
bash scripts/setup.sh forward [start|stop|status|restart]
```

### 개발

```bash
# 스키마 생성/갱신 (멱등). 새 DB를 쓰거나 스키마가 바뀐 뒤 1회 실행
python scripts/migrate_schema.py

# 개발 의존성 (런타임 requirements.txt + pytest·ruff·jupyter)
pip install -r requirements-dev.txt

# 백엔드 단위 테스트 (DB·API 키 불필요)
pytest tests/unit

# 통합 테스트 (일회용 pgvector DB 필요)
TEST_DATABASE_URL=postgresql://arxplore:arxplore@localhost:5432/arxplore_test pytest tests/integration -m integration

# 린트
ruff check .

# 프론트엔드 타입체크 / 빌드
cd frontend && npm run typecheck && npm run build

# 검색 평가 (eval/README.md). lexical만은 API 키 불필요
python scripts/eval_retrieval.py --methods lexical

# 검색 쿼리 실행 계획 확인 (EXPLAIN ANALYZE, 실제 DB 필요)
python scripts/explain_retrieval.py --query "..."

# Django 컨테이너 셸
docker compose exec django bash

# vite 이미지 재빌드 (package.json 변경 후)
docker compose --profile dev build vite

# 로그 확인
docker compose logs -f [django|nginx|vite]
```

### Key Ports

| Service | Host Port |
|---------|-----------|
| Web (nginx) | 80 (`PROD_HTTP_PORT`) |
| Vite (dev) | 5173 (`FRONTEND_PORT`) |
| Layout Parser | 5060 (`LAYOUT_PARSER_PORT`) |
| Airflow | 18080 |
| MongoDB | 17017 |
| PostgreSQL | 15432 (`SERVER_POSTGRES_PORT`, local-db 프로필은 127.0.0.1에만 공개) |

## Architecture

ArXplore는 HuggingFace Daily Papers + arXiv 논문을 수집·처리해 RAG 기반 채팅 인터페이스로 제공하는 플랫폼입니다.

### Data Flow

```
HF Daily Papers / arXiv
  → Airflow DAGs (daily_collect, maintenance, langsmith_maintenance)  [서버]
  → MongoDB (raw payload)                       [서버]
  → PostgreSQL prepare_jobs queue (LISTEN/NOTIFY)
  → prepare-worker (PDF parse → OpenAI 임베딩 API)  [--profile parser]
  → PostgreSQL: papers, paper_fulltexts, paper_chunks, paper_embeddings (pgvector)
  → Retrieval (RETRIEVAL_MODE=hybrid 기본: 질의 임베딩 키가 있으면 hybrid, 없거나 실패하면 lexical)
  → LangChain chains (개요·핵심 포인트·상세 요약, 결과 캐시) + LangGraph ReAct Agent + 상세 챗
  → Django API (데모 모드, rate limit, SSE) → React UI
```

### Docker Compose 구조

로컬 서비스는 `docker-compose.yml`, 서버 인프라는 `docker-compose.server.yml`로 나눠 관리합니다. `docker-compose.yml`의 서비스는 다음과 같습니다.

| 서비스 | 프로필 | 설명 |
|--------|--------|------|
| `django` | (기본) | gunicorn WSGI 서버. 비root(`app`, UID 1000), TCP healthcheck |
| `nginx` | (기본) | React 빌드 서빙 + API 프록시 + 보안 헤더·gzip. django가 healthy일 때 시작 |
| `vite` | `dev` | 프론트엔드 HMR 개발 서버 (`npm ci`) |
| `prepare-worker` | `parser` | prepare queue 소비 worker. `LAYOUT_PARSER_BASE_URL` 기본 `http://layout-parser:5060`, layout-parser가 healthy일 때 시작 |
| `layout-parser` | `parser` | HURIDOCS GPU PDF 파서 |
| `postgres-local` | `local-db` | 로컬 단독 실행용 PostgreSQL 16 + pgvector (`127.0.0.1:${SERVER_POSTGRES_PORT:-15432}`) |

서버 인프라(PostgreSQL, MongoDB, Airflow)는 `docker-compose.server.yml`로 별도 운영합니다. Airflow 서비스 4개는 `x-airflow-common` 앵커로 이미지·환경 변수·볼륨을 공유합니다.

nginx(`docker/nginx/nginx.conf`)는 SPA 경로를 `index.html`로 돌리고 API만 Django로 프록시한다. 새 API 경로를 추가하면 nginx location과 `frontend/vite.config.ts` 프록시에 같이 넣어야 한다. SSE 경로(`/papers/assistant/stream/`, `/papers/<id>/chat/stream/`)는 버퍼링을 끈 location을 쓴다. admin 경로는 프록시하지 않는다.

의존성 파일: `requirements.txt`(런타임, django 이미지), `requirements-dev.txt`(테스트·린트·노트북), `requirements-airflow.txt`(Airflow 이미지, 공식 constraints와 함께 설치).

### Key Architectural Split

**Server-side** (`docker-compose.server.yml`): PostgreSQL, MongoDB, Airflow — 항상 켜져있는 원격 서버에서 실행.

**Local** (`docker-compose.yml`): Django(gunicorn) + nginx는 로컬에서 실행하고, vite는 `dev` 프로필로 필요할 때만 띄운다. parser 프로필은 GPU 보유 시에만 추가. `local-db` 프로필은 원격 서버 없이 웹만 띄울 때 쓴다(수집이 없으므로 빈 DB).

**prepare-worker는 Airflow가 아닌 로컬에서 실행** — GPU가 필요한 HURIDOCS 파싱을 로컬에서 처리하고 결과를 서버 DB에 직접 적재한다. 임베딩은 GPU가 아니라 OpenAI API(`text-embedding-3-large`, 1536차원)로 만든다. 이 분리를 깨지 말 것.

**스키마는 `scripts/migrate_schema.py`가 만든다.** `PaperRepository()`·`PrepareJobRepository()` 생성자는 DDL을 실행하지 않는다(요청 경로에서 DDL 금지). prepare-worker는 시작할 때 `ensure_schema()`를 1회 호출한다. DAG 태스크는 스키마가 이미 있다고 가정한다.

### Module Responsibilities

- **`backend/`** — Django 프로젝트 루트 (`manage.py`, `arxplore_web/` 설정, `papers/` 앱)
  - `papers/api_views.py` — REST 엔드포인트 (인증, 논문 조회·분석·채팅, 즐겨찾기, `assistant/stream/`·`<id>/chat/stream/` SSE). 캐시 없음 + 미로그인은 401 `login_required`, 키 없음은 400 `api_key_required`
  - `papers/services.py` — 비즈니스 로직 계층 (LLM 체인 호출, AI 요약 캐싱, 로컬 + arXiv 외부 검색을 결합한 관련 논문 합성, 권한, 데모 모드 판단)
  - `papers/ratelimit.py` — Django cache 기반 분당 고정 윈도 rate limit (429 + `Retry-After`)
  - `papers/secret_box.py` — 세션에 저장하는 개인 OpenAI 키 암호화(Fernet)
  - `papers/models.py` — `UserSettings`, `FavoritePaper` (Django ORM)
  - AI overview/요약 결과는 모델이 아니라 `src/integrations/paper_repository.py`가 PostgreSQL `paper_ai_overviews`, `paper_ai_detailed_summaries` 테이블에 직접 캐싱한다
- **`src/core/`** — LLM 체인, 프롬프트, 상세 요약 그래프, LangGraph 에이전트
- **`src/integrations/`** — 외부 I/O: MongoDB, PostgreSQL 리포지토리, HURIDOCS 클라이언트, OpenAI 임베딩, hybrid retriever
- **`src/pipeline/`** — Airflow DAG 및 prepare-worker가 호출하는 진입점 스크립트
- **`src/shared/`** — Pydantic `AppSettings` (`.env` 로드), LangSmith 트레이싱
- **`dags/`** — Airflow DAG 3개 (TaskFlow `@dag`/`@task`로 `src/pipeline/` 호출)
- **`frontend/`** — React 18 + Vite + TanStack Query + TypeScript. 목록 상태는 URL 쿼리가 원본(`pages/list/listParams.ts`), 화면 이동은 react-router `Link`/`navigate`, 계정 메뉴는 `components/account/AccountMenu.tsx` 공용, 모달은 `helpers/useModalDialog.ts`(Esc·포커스 복귀). 색상은 `styles/global.css`의 `:root` 토큰을 쓴다

### PDF Parsing Strategy

3단계 폴백:
1. HURIDOCS Layout Parser (Docker, `LAYOUT_PARSER_BASE_URL`. 코드에는 기본값·자동 감지가 없고, compose의 prepare-worker만 비어 있을 때 `http://layout-parser:5060`을 넣는다. 비어 있으면 이 단계를 건너뛴다)
2. pypdf
3. abstract only

청크에 `content_role`, `section_title`, `parser_metadata`, `quality_metrics` 저장. 청킹은 글자 수 기준(1800자, 겹침 200자).

재처리는 멱등하다. source 순위 `layout_pdf > pdf > fallback_abstract`에서 기존보다 낮은 순위 결과는 저장하지 않고, 같은 source에 `content_hash`(정규화 본문 + 섹션 제목의 sha256)까지 같으면 본문·청크 교체를 건너뛴다. 청크 텍스트가 같으면 DELETE/INSERT 없이 메타데이터만 갱신해 청크 id와 임베딩을 보존한다. 강제 재처리는 `--force`(worker CLI, requeue 스크립트 `--apply --force`, job payload `force: true`). 날짜 잡에서 논문 1건이라도 실패하면 잡은 backoff 후 재시도되고, 성공했던 논문은 unchanged로 건너뛴다.

### Retrieval

제품 경로 선택: `src/core/agent/retrieval.py`의 `retrieve_contexts`. 에이전트 도구와 상세 챗이 공유한다.
- `RETRIEVAL_MODE=hybrid`(기본)이고 질의 임베딩 키(사용자 세션 키 > 서버 `OPENAI_API_KEY`)가 있으면 hybrid, 키가 없거나 임베딩 호출이 `OpenAIError`로 실패하면 lexical. `RETRIEVAL_MODE=lexical`이면 항상 lexical

구현: `src/integrations/paper_retriever.py`
- **Lexical** — PostgreSQL 전문 검색(`english` 설정). 제목·초록·청크 tsvector 생성 컬럼 + GIN 인덱스. 한국어 질의는 거의 맞지 않는다
- **Vector** — pgvector 코사인 거리 (text-embedding-3-large, 1536 dims), HNSW 인덱스, `VECTOR_MIN_SIMILARITY`로 하한 설정
- **Hybrid** — reciprocal rank fusion(k=60) + content-role reranking
- DB 연결은 `src/integrations/db.py` 풀(프로세스당 `POSTGRES_POOL_MAX`, 기본 8)

오프라인 평가 하니스: `eval/`, `scripts/eval_build_queries.py`, `scripts/eval_retrieval.py`.

### Agent

`src/core/agent/chatbot.py` — LangGraph ReAct Agent (`stream_mode=["messages", "updates"]`, `recursion_limit=AGENT_RECURSION_LIMIT` 기본 12, 초과 시 안내 문구로 종료) / AGENT_STREAM_BUFFER_CHARS=120(도구 호출 전 서두 누출 방지 버퍼; 120자 넘는 서두 뒤 도구 호출은 막지 못함)
- `search_paper_chunks_tool` — `retrieve_contexts`(hybrid → lexical) 기반 청크 검색 5개. hit을 요청 범위 레지스트리에 기록
- `get_trending_papers_tool` — 트렌딩 논문 통계
- `citations.py` — 답변 속 링크(에이전트)·`[n]` 번호(상세 챗)를 도구 hit과 대조해 `citations` 이벤트(`in_answer`)를 만든다

`src/core/agent/paper_chat.py` — 상세 페이지 챗. 질문으로 해당 논문 안을 검색해 초록(1번) + 청크 최대 5개로 답하고 SSE로 스트리밍. 검색 결과가 없으면 앞 청크(`retrieval_mode: "first_chunks"`).

SSE 이벤트 계약(`docs/architecture/AGENTS.md` 3절): `chunk`* → `citations` 1회 → `[DONE]`, 오류 시 `error` → `[DONE]`.

### Web access model

- `DEMO_MODE=true`(기본): 목록·`detail.json`은 익명 허용. `analyze/`·`summary/`는 캐시가 있으면 누구에게나 `{"cached": true, ...}`, 없으면 미로그인 401 `login_required`, 키 없음 400 `api_key_required`. 챗은 항상 로그인 + 키. bootstrap에 `demo_mode`, `login_required_for`
- 프론트는 이 응답으로 카드 안에 로그인/설정 안내를 띄운다. 생성 요청은 AbortController로 취소하지만 서버 쪽 생성은 계속될 수 있다
- rate limit: 인증 `RATE_LIMIT_AUTH_PER_MINUTE`(IP당), LLM `RATE_LIMIT_LLM_PER_MINUTE`(사용자당, 비로그인은 IP). `REDIS_URL`이 없으면 프로세스별 LocMem이라 워커 수만큼 허용량이 늘어난다

### Architectural Contracts (do not break)

`PaperDetailDocument` 필드: `arxiv_id`, `title`, `overview`, `key_findings` — 체인·API·UI 공용 계약.

Retrieval 결과 shape: `chunk_id`, `arxiv_id`, `chunk_text`, `section_title`, `content_role`, `score`.

## Configuration

모든 런타임 설정은 루트 `.env`. `src/shared/settings.py`가 Pydantic `BaseSettings`로 로드. 전체 목록과 필수/선택 구분은 **`.env.example`** 이 기준이다(`cp .env.example .env`).

로컬 실행 필수:

```
DJANGO_SECRET_KEY           # 비어 있으면 Django가 시작하지 않음(DJANGO_DEBUG=true 제외). change-me*면 setup.sh가 거부
PROD_POSTGRES_HOST          # django/worker 컨테이너의 POSTGRES_HOST. host 또는 host:port
POSTGRES_HOST               # 호스트에서 실행하는 스크립트용 (예: localhost)
SERVER_POSTGRES_PORT        # 서버가 공개하는 PostgreSQL 포트. compose 기본 15432, settings.py 기본 5432
POSTGRES_DB / APP_POSTGRES_DB / POSTGRES_USER / POSTGRES_PASSWORD
```

서버·worker 필수:

```
OPENAI_API_KEY              # prepare-worker 임베딩. 웹 AI 기능은 사용자 개인 키(세션)를 쓴다
MONGO_HOST / SERVER_MONGO_PORT / MONGO_INITDB_ROOT_USERNAME / MONGO_INITDB_ROOT_PASSWORD
LAYOUT_PARSER_BASE_URL      # 코드 기본값 없음. compose prepare-worker는 비면 http://layout-parser:5060
TAILSCALE_SERVER_IP         # 서버 compose 포트 바인딩 + setup.sh forward
AIRFLOW_ADMIN_USER          # Airflow SimpleAuthManager admin 사용자
AIRFLOW_FERNET_KEY          # 비어 있으면 서버 compose가 실행을 거부
```

선택(기본값은 `.env.example` 주석 참고):

```
DJANGO_DEBUG                # 비어 있으면 False. "true"일 때만 켜짐
DJANGO_SECURE_COOKIES=false # HTTPS 뒤에서만 true
DJANGO_ADMIN_ENABLED=false / DJANGO_ADMIN_PATH=admin/
DEMO_MODE=true
SESSION_KEY_ENCRYPTION_KEY= # 비우면 DJANGO_SECRET_KEY에서 유도. 바꾸면 저장된 세션 키 폐기
REDIS_URL=                  # rate limit 공유 캐시. 비우면 프로세스별 LocMem
RATE_LIMIT_ENABLED=true / RATE_LIMIT_AUTH_PER_MINUTE=10 / RATE_LIMIT_LLM_PER_MINUTE=30 / RATE_LIMIT_DETAIL_PER_MINUTE=60 / RATE_LIMIT_IP_HEADER=X-Real-IP
RETRIEVAL_MODE=hybrid       # hybrid | lexical
VECTOR_MIN_SIMILARITY=0
AGENT_RECURSION_LIMIT=12    # 최소 4
POSTGRES_POOL_MAX=8         # 프로세스당 연결 풀 상한
POSTGRES_POOL_TIMEOUT=30    # 풀 고갈 시 대기(초)
```

주소 규칙: `POSTGRES_HOST`/`PROD_POSTGRES_HOST`/`MONGO_HOST`는 `host` 또는 `host:port`. 포트를 생략하면 `SERVER_POSTGRES_PORT`/`SERVER_MONGO_PORT`(서버가 호스트에 공개하는 포트)를 쓴다. 같은 compose 네트워크 안에서 컨테이너 이름으로 붙을 때는 `postgres-local:5432`, `arxplore-postgres:5432`처럼 내부 포트를 명시한다.

Django 설정: `backend/arxplore_web/settings.py`. 언어 `ko-kr`, 타임존 `Asia/Seoul`.

Vite 프록시는 `frontend/vite.config.ts`에서 `arxplore-django:8001`로 포워딩 (Host 헤더 `localhost` 고정).

## Reference Docs

1. `README.md` — 기능, 담당 범위, 검색 계층 현황, Quick Start, 한계와 로드맵
2. `docs/architecture/ARCHITECTURE.md` — 시스템 구조, DB 스키마, 큐 동작
3. `docs/architecture/AGENTS.md` — AI 작업 규칙, 모듈 계약, 용어
4. `docs/management/WORKFLOW.md` — 운영 절차, 작업 모드
5. `docs/management/ROLES.md` — 5인 팀 역할 분담, 공용 계약 (팀 프로젝트 당시 문서)
6. `docs/management/TEAM_SETUP.md` — 원격 서버 모드 환경 설정 절차 (팀 프로젝트 당시 문서)
7. `.env.example` — 환경 변수 전체 목록
