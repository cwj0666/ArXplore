# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### 컨테이너 실행

```bash
# 로컬 단독: 로컬 PostgreSQL(pgvector) 포함. PROD_POSTGRES_HOST=postgres-local:5432
docker compose --profile local-db up -d postgres-local
docker compose --profile local-db up -d --build

# 원격 서버 DB 사용 (django + nginx + vite)
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

# 백엔드 단위 테스트 (DB·API 키 불필요)
pytest tests/unit

# 프론트엔드 타입체크
cd frontend && npm run typecheck

# Django 컨테이너 셸
docker compose exec django bash

# vite 이미지 재빌드 (package.json 변경 후)
docker compose build vite

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
  → Retrieval (제품 경로는 lexical만. vector / hybrid는 구현만 되어 있고 미연결)
  → LangChain chains + LangGraph React Agent
  → Django REST API → React UI
```

### Docker Compose 구조

단일 `docker-compose.yml`로 모든 서비스를 관리합니다.

| 서비스 | 프로필 | 설명 |
|--------|--------|------|
| `django` | (기본) | gunicorn WSGI 서버 |
| `nginx` | (기본) | React 빌드 서빙 + API 프록시 |
| `vite` | (기본) | 프론트엔드 HMR 개발 서버 |
| `prepare-worker` | `parser` | prepare queue 소비 worker |
| `layout-parser` | `parser` | HURIDOCS GPU PDF 파서 |
| `postgres-local` | `local-db` | 로컬 단독 실행용 PostgreSQL 16 + pgvector (`127.0.0.1:${SERVER_POSTGRES_PORT:-15432}`) |

서버 인프라(PostgreSQL, MongoDB, Airflow)는 `docker-compose.server.yml`로 별도 운영합니다.

### Key Architectural Split

**Server-side** (`docker-compose.server.yml`): PostgreSQL, MongoDB, Airflow — 항상 켜져있는 원격 서버에서 실행.

**Local** (`docker-compose.yml`): Django(gunicorn) + nginx + vite는 로컬에서 실행. parser 프로필은 GPU 보유 시에만 추가. `local-db` 프로필은 원격 서버 없이 웹만 띄울 때 쓴다(수집이 없으므로 빈 DB).

**prepare-worker는 Airflow가 아닌 로컬에서 실행** — GPU가 필요한 HURIDOCS 파싱을 로컬에서 처리하고 결과를 서버 DB에 직접 적재한다. 임베딩은 GPU가 아니라 OpenAI API(`text-embedding-3-large`, 1536차원)로 만든다. 이 분리를 깨지 말 것.

**스키마는 `scripts/migrate_schema.py`가 만든다.** `PaperRepository()`·`PrepareJobRepository()` 생성자는 DDL을 실행하지 않는다(요청 경로에서 DDL 금지). prepare-worker는 시작할 때 `ensure_schema()`를 1회 호출한다.

### Module Responsibilities

- **`backend/`** — Django 프로젝트 루트 (`manage.py`, `arxplore_web/` 설정, `papers/` 앱)
  - `papers/api_views.py` — REST 엔드포인트 (인증, 논문 조회·분석·채팅, 즐겨찾기, `assistant/stream/` SSE)
  - `papers/services.py` — 비즈니스 로직 계층 (LLM 체인 호출, AI 요약 캐싱, 로컬 + arXiv 외부 검색을 결합한 관련 논문 합성, 권한)
  - `papers/models.py` — `UserSettings`, `FavoritePaper` (Django ORM)
  - AI overview/요약 결과는 모델이 아니라 `src/integrations/paper_repository.py`가 PostgreSQL `paper_ai_overviews`, `paper_ai_detailed_summaries` 테이블에 직접 캐싱한다
- **`src/core/`** — LLM 체인, 프롬프트, 상세 요약 그래프, LangGraph 에이전트
- **`src/integrations/`** — 외부 I/O: MongoDB, PostgreSQL 리포지토리, HURIDOCS 클라이언트, OpenAI 임베딩, hybrid retriever
- **`src/pipeline/`** — Airflow DAG 및 prepare-worker가 호출하는 진입점 스크립트
- **`src/shared/`** — Pydantic `AppSettings` (`.env` 로드), LangSmith 트레이싱
- **`dags/`** — Airflow DAG 3개 (TaskFlow `@dag`/`@task`로 `src/pipeline/` 호출)
- **`frontend/`** — React 18 + Vite + TanStack Query + TypeScript

### PDF Parsing Strategy

3단계 폴백:
1. HURIDOCS Layout Parser (Docker, `LAYOUT_PARSER_BASE_URL`. 기본값·자동 감지 없음. `.env`에 직접 넣는다. compose 네트워크 안에서는 `http://layout-parser:5060`. 비어 있으면 이 단계를 건너뛴다)
2. pypdf
3. abstract only

청크에 `content_role`, `section_title`, `parser_metadata`, `quality_metrics` 저장. 청킹은 글자 수 기준(1800자, 겹침 200자).

새 결과가 `fallback_abstract`이고 기존 본문 source가 `layout_pdf`/`pdf`이면 본문·청크·임베딩을 교체하지 않는다(청크 DELETE가 임베딩을 CASCADE 삭제하기 때문).

### Retrieval

`src/integrations/paper_retriever.py`:
- **Lexical** — PostgreSQL 전문 검색(`english` 설정). 에이전트 도구가 쓰는 유일한 제품 경로
- **Vector** — pgvector 코사인 거리 (text-embedding-3-large, 1536 dims). 벡터 인덱스 없음, 제품 경로 미연결
- **Hybrid** — reciprocal rank fusion + content-role reranking. 제품 경로 미연결(평가 후 연결 예정)

### Agent

`src/core/agent/chatbot.py` — LangGraph ReAct Agent (`stream_mode="messages"`)
- `search_paper_chunks_tool` — PostgreSQL 전문 검색(lexical) 기반 청크 검색
- `get_trending_papers_tool` — 트렌딩 논문 통계

### Architectural Contracts (do not break)

`PaperDetailDocument` 필드: `arxiv_id`, `title`, `overview`, `key_findings` — 체인·API·UI 공용 계약.

Retrieval 결과 shape: `chunk_id`, `arxiv_id`, `chunk_text`, `section_title`, `content_role`, `score`.

## Configuration

모든 런타임 설정은 루트 `.env`. `src/shared/settings.py`가 Pydantic `BaseSettings`로 로드. 전체 목록과 필수/선택 구분은 **`.env.example`** 이 기준이다(`cp .env.example .env`).

로컬 실행 필수:

```
DJANGO_SECRET_KEY           # 비어 있거나 change-me*면 setup.sh가 거부
PROD_POSTGRES_HOST          # django/worker 컨테이너의 POSTGRES_HOST. host 또는 host:port
POSTGRES_HOST               # 호스트에서 실행하는 스크립트용 (예: localhost)
SERVER_POSTGRES_PORT        # 서버가 공개하는 PostgreSQL 포트. compose 기본 15432, settings.py 기본 5432
POSTGRES_DB / APP_POSTGRES_DB / POSTGRES_USER / POSTGRES_PASSWORD
```

서버·worker 필수:

```
OPENAI_API_KEY              # prepare-worker 임베딩. 웹 AI 기능은 사용자 개인 키(세션)를 쓴다
MONGO_HOST / SERVER_MONGO_PORT / MONGO_INITDB_ROOT_USERNAME / MONGO_INITDB_ROOT_PASSWORD
LAYOUT_PARSER_BASE_URL      # 자동 감지 없음. compose 네트워크 안: http://layout-parser:5060
TAILSCALE_SERVER_IP         # 서버 compose 포트 바인딩 + setup.sh forward
AIRFLOW_ADMIN_USER          # Airflow SimpleAuthManager admin 사용자
AIRFLOW_FERNET_KEY          # 비어 있으면 서버 compose가 실행을 거부
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
