# ArXplore

[![CI](https://github.com/SKNETWORKS-FAMILY-AICAMP/ArXplore/actions/workflows/ci.yml/badge.svg)](https://github.com/SKNETWORKS-FAMILY-AICAMP/ArXplore/actions/workflows/ci.yml)

Hugging Face Daily Papers에 올라오는 AI 논문을 매일 수집하고, PDF를 파싱·청킹해 PostgreSQL에 적재한 뒤, 논문 목록 탐색 · 한국어 개요와 상세 요약 · LangGraph 에이전트 채팅으로 읽을 수 있게 만든 논문 탐색 서비스입니다.

<img width="1900" height="915" alt="논문 목록 화면. 상단에 키워드 검색과 AI 어시스턴트 탭이 있는 검색창, 아래에 논문 카드 그리드(제목, 초록 미리보기, 게시일, 추천수, 즐겨찾기 버튼)와 최신순 정렬 선택이 있다." src="https://github.com/user-attachments/assets/97fee07e-b0ea-407a-b3cc-37d751cb42f0" />

<img width="1901" height="939" alt="논문 상세 화면. 왼쪽에 PDF 분할 보기, 오른쪽에 제목·저자·게시일과 AI가 생성한 한국어 개요 카드가 있고, 상단에 상세요약 생성 버튼, 오른쪽 아래에 논문 챗 버튼이 있다." src="https://github.com/user-attachments/assets/e484d29b-379b-4efb-9b8e-0c9f397b7c16" />

## 프로젝트 성격과 담당 범위

- **SK네트웍스 AI 캠프 팀 프로젝트**입니다(2026년 3~4월). 역할 분담 문서는 5인 팀 기준으로 작성되어 있습니다([ROLES.md](./docs/management/ROLES.md)).
- **본인 담당**: 검색 데이터 계층(수집 → PDF 파싱 → 청킹 → 임베딩 → 검색)을 맡아 시작했고, 이후 범위를 넓혀 프로젝트 전체를 주도했습니다. Streamlit 화면을 Django + React로 옮긴 작업(c64a957)과 그 이후 기능(에이전트 스트리밍 채팅과 중지, 관련 논문 카드, compose 통합 등)은 본인 작업입니다.
- **팀원 기여(git 기록 기준)**: 논문 상세 문서 기반(`yeseung-Yang`), 한국어 번역·요약 프롬프트(`lucky`).
- **저장소 기록**: 팀 프로젝트 종료 시점(143cc72) 커밋 41개. git 작성자 이름별로 `cwj0666` 33, `최원준` 3(두 이름 모두 본인), `lucky` 2, `yeseung-Yang` 2, `SKNETWORKS-AICAMP-ADMIN` 1입니다. 작성자 수는 팀원 수와 같지 않고, 커밋으로 남지 않은 기여는 이 숫자에 드러나지 않습니다.
- 2026년 9월 이후 커밋은 본인이 포트폴리오 정리를 위해 진행한 코드 점검과 수정입니다. AI 코딩 도구를 함께 사용했고 커밋 트레일러에 표기했습니다.

## 주요 기능

현재 코드가 실제로 하는 일만 적었습니다.

- **수집(서버 Airflow)**: `arxplore_daily_collect`가 매일 18:00(KST) HF Daily Papers 원본을 PostgreSQL(`raw_daily_papers`, JSONB)에 저장하고, 같은 트랜잭션에서 날짜 단위 prepare 작업을 등록합니다. `arxplore_maintenance`는 3시간마다 과거 raw를 backfill하고 arXiv 메타데이터를 보강합니다. `arxplore_langsmith_maintenance`는 매일 03:00에 오래된 LangSmith trace를 정리합니다.
- **Prepare(로컬 worker)**: PDF를 3단계 폴백(HURIDOCS → pypdf → 초록)으로 파싱하고, 섹션과 `content_role`을 붙여 글자 수 기준(1,800자, 겹침 200자)으로 청킹한 뒤 OpenAI API로 임베딩합니다.
- **논문 목록**: 최신순·추천순 정렬, 제목·초록 부분 문자열 검색(최근 1,500편 대상), 페이지네이션, 즐겨찾기. 페이지·정렬·검색어가 URL에 남아 뒤로/앞으로 가기로 그대로 돌아옵니다.
- **데모 모드**(`DEMO_MODE=true`, 기본값): 로그인 없이 목록과 상세 페이지를 열고, 이미 캐시된 개요·핵심 포인트·상세 요약을 볼 수 있습니다. 캐시가 없는 결과를 새로 만들거나 챗을 쓰려면 로그인과 개인 OpenAI 키가 필요하고, 화면은 그 자리에서 로그인·키 등록 안내를 보여 줍니다. `DEMO_MODE=false`면 상세 페이지부터 로그인이 필요합니다.
- **논문 상세**
  - PDF 분할 보기
  - 개요와 핵심 포인트: gpt-5-mini로 생성하고 논문 단위로 캐시합니다. 생성 중에는 카드 안에 진행 상태와 취소 버튼이 나오고 초록은 계속 읽을 수 있습니다. 취소는 화면의 대기만 멈추며, 서버에서 이미 시작된 생성은 끝까지 진행되어 캐시됩니다.
  - 상세 요약: 사용자가 gpt-5-mini / gpt-5 중에서 고르고, LangGraph 요약 그래프가 섹션을 배경·방법·실험·한계로 묶어 요약합니다. 논문×모델 단위로 캐시합니다.
  - 관련 논문: 로컬 DB 후보를 카테고리·키워드 겹침으로 점수화하고, 부족하면 arXiv 검색으로 채웁니다.
  - 논문 챗: 질문으로 **그 논문 안**을 검색(hybrid, 실패 시 lexical)해 초록 + 발췌 청크 최대 5개를 근거로 답하고, SSE로 스트리밍합니다. 답변의 `[1]`, `[2]` 번호를 발췌문과 대조해 출처 칩(섹션 이름 포함)으로 보여 주고, 중지 버튼으로 끊을 수 있습니다.
- **AI 어시스턴트**: LangGraph ReAct 에이전트가 SSE로 응답을 스트리밍하고, 사용자는 중지 버튼으로 끊을 수 있습니다. 도구는 `search_paper_chunks_tool`(hybrid 검색, 임베딩 불가 시 lexical)과 `get_trending_papers_tool`(최근 논문 추천수 순) 두 개입니다. 답변 속 링크를 도구 결과와 대조한 구조화된 citation을 함께 보내고, 단계 수 제한(`AGENT_RECURSION_LIMIT`, 기본 12)에 걸리면 안내 문구로 마무리합니다.
- **계정과 보호 장치**: 회원가입(Django 비밀번호 검증)·로그인. 개인 OpenAI API 키는 암호화해 세션에 저장합니다. 로그인·회원가입과 LLM 호출 엔드포인트에 분당 rate limit이 걸려 있고(초과 시 429와 `Retry-After`), 화면은 대기 시간을 안내합니다.
- **미구현**: 근거 청크 번역 UI(`translate_chunk` 체인만 있고 엔드포인트 없음).

## 아키텍처

```mermaid
flowchart TD
    A[HF Daily Papers] --> C[arxplore_daily_collect]
    A --> D[arxplore_maintenance<br/>backfill + enrich]
    C -->|한 트랜잭션| B[PostgreSQL raw_daily_papers<br/>JSONB payload]
    C -->|한 트랜잭션| E[PostgreSQL prepare_jobs]
    D --> B
    E -->|LISTEN/NOTIFY| F[prepare-worker]
    F --> G[HURIDOCS Layout Parser]
    F --> H[pypdf / abstract fallback]
    B --> I[prepare_papers]
    G --> I
    H --> I
    I --> J[(PostgreSQL + pgvector<br/>papers / fulltexts / chunks / embeddings)]
    D --> J
    J --> K[Retrieval<br/>lexical / vector / hybrid + rerank]
    J --> L[Paper Detail Chains<br/>overview / key findings / summary]
    K --> M[LangGraph React Agent<br/>Agentic RAG]
    L --> N[React UI]
    M --> N
```

- **서버 스택**(`docker-compose.server.yml`): PostgreSQL(pgvector), Airflow. 항상 켜 두는 수집·저장 계층입니다. 저장소는 PostgreSQL 하나로, raw payload·파이프라인 상태·정제 데이터·벡터·작업 큐를 모두 담습니다.
- **로컬 스택**(`docker-compose.yml`): Django(gunicorn) + nginx(React 빌드). `dev` 프로필은 Vite HMR 서버, `parser` 프로필은 HURIDOCS 파서와 prepare-worker, `local-db` 프로필은 로컬 PostgreSQL을 더합니다.
- **GPU는 HURIDOCS 파서에만 씁니다.** 임베딩은 OpenAI API(`text-embedding-3-large`를 `dimensions=1536`으로 줄여 요청)로 만듭니다.
- **도메인 범위**: HF Daily Papers 큐레이션 피드 전체입니다. arXiv 카테고리로 따로 거르지 않습니다.
- **기술 스택**: Python 3.12, Django 5, React 18 + TypeScript + Vite, LangChain / LangGraph / LangSmith, PostgreSQL 16 + pgvector, Airflow 3.

세부 구조와 테이블 스키마는 [ARCHITECTURE.md](./docs/architecture/ARCHITECTURE.md)에 있습니다.

## 검색 계층

제품 경로는 `src/core/agent/retrieval.py`의 `retrieve_contexts` 하나로 모입니다. 에이전트 검색 도구와 상세 챗이 같은 규칙을 씁니다.

| 경로 | 상태 | 내용 |
| --- | --- | --- |
| hybrid | **제품 경로** (`RETRIEVAL_MODE=hybrid`, 기본값) | lexical과 vector 결과를 RRF(k=60)와 방법별 가중치로 합칩니다. 질의 임베딩 키(사용자 세션 키, 없으면 서버 `OPENAI_API_KEY`)가 있을 때만 씁니다. |
| lexical | **폴백 경로** (키가 없거나 임베딩 호출 실패, 또는 `RETRIEVAL_MODE=lexical`) | 제목(A)·초록(B)·청크(C) 가중 tsvector에 `websearch_to_tsquery` + `plainto_tsquery`로 `ts_rank_cd` 점수를 매기고, ILIKE 보너스, 섹션·`content_role` 가중, 질의 토큰 겹침 rerank, 참고문헌처럼 보이는 텍스트 필터, 논문 다양성 보정, 인접 청크 병합을 거칩니다. |
| vector | hybrid의 구성 요소 | `paper_embeddings` 코사인 거리(`<=>`)와 섹션·`content_role` 감점. `VECTOR_MIN_SIMILARITY`(기본 0)로 낮은 유사도를 거를 수 있습니다. |

- 상세 챗은 같은 경로를 `arxiv_id`로 한정해 호출하고, 결과가 비면 논문의 앞 청크로 대신합니다(응답의 `retrieval_mode`가 `hybrid` / `lexical` / `first_chunks` 중 하나).
- 인덱스: `scripts/migrate_schema.py`가 제목·초록과 청크의 tsvector 생성 컬럼에 GIN 인덱스를, `paper_embeddings`에 HNSW 인덱스(pgvector 0.5.0 이상)를 만듭니다. 기존 DB에 처음 적용할 때는 테이블을 다시 쓰므로 prepare-worker를 멈추고 실행합니다.
- 알려진 제약: FTS 설정이 `english`라서 한국어 질의는 lexical에서 거의 맞지 않습니다. 키가 없어 lexical로 내려가면 한국어 질문의 검색 품질이 크게 떨어집니다.

## 데이터 파이프라인

```text
HF Daily Papers → raw_daily_papers(JSONB) + prepare_jobs → prepare-worker → papers / paper_fulltexts / paper_chunks → paper_embeddings
```

모든 단계가 같은 PostgreSQL을 씁니다.

**원본 저장** (`src/integrations/raw_store.py`)

- HF 응답은 `raw_daily_papers`에 `(source, date)`당 1행으로 JSONB 그대로 저장합니다. 키 순서와 무관한 `payload_hash`가 저장된 값과 다를 때만 `revision`이 오르고, 같으면 `collected_at`만 갱신합니다. revision 계산은 `INSERT ... ON CONFLICT` 한 문장 안에서 끝납니다.
- 수집 태스크는 raw upsert와 `prepare_jobs` 등록을 한 트랜잭션으로 실행합니다. 등록이 실패하면 raw 저장도 롤백되고, `pg_notify`는 commit 뒤에 전달되므로 worker는 raw가 보이는 시점에만 깨어납니다.
- backfill 커서는 `pipeline_state`(key → JSONB) 테이블에 둡니다.

**작업 큐** (`src/integrations/prepare_job_repository.py`)

- 큐는 PostgreSQL 테이블 하나(`prepare_jobs`, `(mode, target_date)` 유일)입니다. 등록할 때 `pg_notify`를 보내고, worker는 `LISTEN`으로 기다리다가 타임아웃이 나면 다시 확인합니다.
- claim은 `FOR UPDATE SKIP LOCKED`로 대기 중인 잡 1건을 잡고 `(job_id, worker_id, claim_generation)` 토큰을 발급합니다. 완료·실패 기록은 이 토큰이 맞을 때만 반영되므로, 다른 worker가 다시 가져간 잡을 이전 worker가 덮어쓰지 못합니다.
- stale 판정을 따로 돌리는 감시 프로세스는 없습니다. worker가 claim할 때 마지막 heartbeat(없으면 claim 시각)가 `PREPARE_JOB_STALE_SECONDS`(기본 900초)보다 오래된 `processing` 잡을 되돌립니다. heartbeat는 논문 한 편을 처리하기 전마다 갱신합니다.
- 실패한 잡은 `PREPARE_JOB_MAX_ATTEMPTS`(기본 3회)까지 지수 backoff(60초부터 두 배씩, 최대 1시간) 뒤에 다시 시도하고, 횟수를 다 쓰면 `failed`로 닫습니다. 같은 날짜를 다시 수집해 raw revision이 올라가면 완료된 잡도 다시 처리합니다.

**데이터 보호 장치**

- 논문 단위 격리: 한 논문의 예외는 기록하고 나머지 논문을 계속 처리합니다. 큐 잡은 실패한 논문이 하나라도 있으면 backoff 후 재시도되고(최대 `PREPARE_JOB_MAX_ATTEMPTS`), 이미 저장된 논문은 재시도에서 건너뜁니다. 날짜 backfill은 실패가 있는 날짜에서 커서를 진행하지 않습니다.
- 멱등 재처리: 본문 source 순위(`layout_pdf` > `pdf` > `fallback_abstract`)에서 낮은 순위 결과로는 덮어쓰지 않고, 같은 source에 내용 해시까지 같으면 저장을 건너뜁니다. 청크 텍스트가 같으면 청크 id와 임베딩을 보존합니다. 일시적인 다운로드·파서 실패가 기존 임베딩을 CASCADE로 지우던 문제를 막고, 강제 재처리는 `--force`로 합니다. 논문 1건이라도 실패한 날짜 잡은 backoff 후 재시도됩니다.
- 임베딩 backlog: prepare 성공 여부와 상관없이 매 루프에서 누락된 임베딩을 `EMBED_BACKLOG_MAX_CHUNKS`(기본 400)까지 채우고, backlog 오류가 worker를 멈추지 않습니다.
- 참고문헌 판정: 섹션 제목이 참고문헌 제목과 정확히 맞을 때만 `references`로 분류합니다. "Direct Preference Optimization" 같은 본문 섹션이 검색에서 빠지던 문제를 고쳤습니다.
- 운영 스크립트는 모두 dry-run이 기본입니다: `scripts/requeue_failed_prepare_jobs.py --since YYYY-MM-DD [--apply]`, `scripts/backfill_content_roles.py [--apply]`.

## Quick Start

### compose 프로필

| 프로필 | 서비스 | 용도 |
| --- | --- | --- |
| (기본) | `django`, `nginx` | 웹 앱. nginx는 django healthcheck가 통과한 뒤 뜹니다. |
| `dev` | `vite` | 프론트엔드 HMR 개발 서버(`http://localhost:5173`, 호스트 127.0.0.1에만 바인딩) |
| `parser` | `layout-parser`, `prepare-worker` | GPU PDF 파서와 prepare 큐 worker. worker는 파서 healthcheck 통과 뒤 뜹니다. |
| `local-db` | `postgres-local` | 원격 서버 없이 쓰는 로컬 PostgreSQL(pgvector) |

프로필은 겹쳐 쓸 수 있습니다. 예: `docker compose --profile local-db --profile dev up -d --build`.

### (a) 로컬 단독 실행

원격 서버 없이 로컬 PostgreSQL 하나로 웹 앱을 띄웁니다. 수집(Airflow)은 돌지 않으므로 **논문 목록은 빈 상태로 시작합니다.**

```bash
cp .env.example .env
# DJANGO_SECRET_KEY, POSTGRES_PASSWORD 등 change-me로 시작하는 값을 실제 값으로 바꾼다

docker compose --profile local-db up -d postgres-local   # pgvector/pgvector:pg16, 127.0.0.1:15432

python scripts/migrate_schema.py   # 호스트 Python 3.12 + requirements.txt 필요
# 호스트에 Python 환경이 없으면: docker compose run --rm django python /workspace/scripts/migrate_schema.py

docker compose --profile local-db up -d --build             # 웹: http://localhost
docker compose --profile local-db --profile dev up -d vite  # (선택) Vite HMR: http://localhost:5173
```

- 데모 모드가 기본이라 로그인 없이 목록과 상세를 볼 수 있습니다. 회원가입 → 설정에서 개인 OpenAI API 키를 등록하면 생성과 챗이 열립니다.
- `DJANGO_CSRF_TRUSTED_ORIGINS`에는 앱에 접속하는 모든 origin이 들어가야 합니다. 기본값은 `http://localhost`, `http://127.0.0.1`과 Vite dev 서버(`:5173`)이며, `PROD_HTTP_PORT`·`FRONTEND_PORT`를 바꾸거나 다른 호스트명으로 접속하면 해당 origin(포트 포함)을 추가하지 않는 한 로그인 등 POST 요청이 403으로 막힙니다.
- `DJANGO_SECRET_KEY`가 비어 있으면 Django가 시작하지 않습니다. `DJANGO_DEBUG`는 기본으로 꺼져 있고 값이 `true`일 때만 켜집니다(compose는 항상 끕니다).
- 접속 주소: `postgres-local`은 django·worker와 같은 compose 기본 네트워크에 있으므로 컨테이너는 `PROD_POSTGRES_HOST=postgres-local:5432`로 접속합니다(`host.docker.internal`이나 `extra_hosts`가 필요 없습니다). 호스트에서 돌리는 스크립트는 `POSTGRES_HOST=localhost`와 `SERVER_POSTGRES_PORT=15432`를 씁니다. `.env.example`의 기본값이 이 구성입니다.
- `PROD_POSTGRES_HOST`와 `POSTGRES_HOST`는 `host` 또는 `host:port` 형식입니다. 포트를 생략하면 `SERVER_POSTGRES_PORT`(`.env.example` 값 15432)를 씁니다.
- django 이미지는 root가 아닌 `app`(UID 1000) 사용자로 돕니다. 이전 이미지로 만든 `django_static` 볼륨은 root 소유라 `collectstatic`이 실패하므로 한 번 지우고 다시 올립니다: `docker compose down && docker volume rm arxplore_django_static`.

### (b) 원격 서버 모드

서버(PostgreSQL · Airflow)를 Tailscale로 공유하고(참고: 팀 시절 커밋 c7b2c34에 포함됐던 Tailscale 인증 키는 폐기되었고 현재 문서는 플레이스홀더만 담습니다), 로컬에서 웹과 GPU 파서·prepare-worker를 돌리는 원래 팀 구성입니다. 절차는 [TEAM_SETUP.md](./docs/management/TEAM_SETUP.md)를 따릅니다.

```bash
bash scripts/setup-server.sh                  # 서버: PostgreSQL / Airflow
bash scripts/setup.sh                         # 로컬: django + nginx
docker compose --profile parser up -d --build # 로컬 GPU: layout-parser + prepare-worker
```

`parser` 프로필의 prepare-worker는 `LAYOUT_PARSER_BASE_URL`이 비어 있으면 `http://layout-parser:5060`을 씁니다. Airflow 이미지는 DAG에 필요한 패키지(`requirements-airflow.txt`)만 Airflow 공식 constraints 파일에 맞춰 설치합니다.

### 운영 기본값

- nginx: `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy` 헤더와 gzip을 켭니다. Django admin 경로는 프록시하지 않고, admin 자체도 `DJANGO_ADMIN_ENABLED=false`가 기본입니다.
- HTTPS 뒤에 둘 때는 `DJANGO_SECURE_COOKIES=true`로 세션·CSRF 쿠키에 Secure를 붙입니다. nginx는 HTTP만 제공합니다.
- rate limit 카운터는 기본으로 프로세스별 메모리 캐시라서 gunicorn 워커 4개가 따로 셉니다. 한도를 정확히 공유하려면 `REDIS_URL`을 지정합니다.
- 세션에 저장한 개인 키는 `SESSION_KEY_ENCRYPTION_KEY`(비우면 `DJANGO_SECRET_KEY`에서 유도)로 암호화합니다. 이 값이나 `DJANGO_SECRET_KEY`를 바꾸면 사용자는 키를 다시 등록해야 합니다.

## 테스트와 CI

```bash
pip install -r requirements-dev.txt   # requirements.txt(런타임) + pytest·ruff·jupyter 등 개발 도구
pytest tests/unit                     # DB·API 키 없이 도는 단위 테스트

# 통합 테스트(큐·raw 저장소·검색): 일회용 PostgreSQL(pgvector) 필요
TEST_DATABASE_URL=postgresql://arxplore:arxplore@localhost:5432/arxplore_test \
  pytest tests/integration -m integration

cd frontend && npm ci
npm run typecheck
npm run build
```

GitHub Actions(`.github/workflows/ci.yml`) 잡 구성:

| 잡 | 내용 |
| --- | --- |
| python | ruff, `pytest tests/unit`, `manage.py check`, `makemigrations --check` |
| integration | pgvector 서비스 컨테이너로 `pytest tests/integration` |
| frontend | `npm run typecheck`, `npm run build` |
| infra | 더미 `.env`로 두 compose 파일 `docker compose config`, hadolint |
| security | gitleaks로 git 히스토리 시크릿 스캔 |

## Evaluation

검색 품질은 아직 측정하지 않았습니다. 아래 표는 측정할 항목의 자리이고, 값은 실제 실행 결과로만 채웁니다.

| 검색 방식 | hit@1 | hit@5 | hit@10 | MRR@10 | 지연 p50 / p95 (ms) |
| --- | --- | --- | --- | --- | --- |
| lexical | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 |
| vector | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 |
| hybrid | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 | 측정 예정 |

평가 하니스는 [`eval/`](./eval/README.md)에 있습니다. 한국어·영어 질의 30~50개(알려진 논문을 초록으로 찾는 known-item 질의 + 본문 청크 하나로만 답할 수 있는 LLM 합성 질의)로 세 경로와 ablation(논문 다양성, lexical 필터, vector rerank, 표준 RRF)을 비교하고, 논문 단위·청크 단위 hit@k·MRR·recall, 상위 10개 중 참고문헌·목차·앞부분 청크 비율, 지연을 기록합니다. 파서·`content_role` 수정 후 재처리와 백필(`scripts/backfill_content_roles.py`), 임베딩 backlog 소진을 마친 DB에서 측정합니다.

```bash
python scripts/eval_build_queries.py                     # 표본·프롬프트 확인 (dry-run, LLM 미호출)
python scripts/eval_build_queries.py --generate          # eval/queries.jsonl 생성 → 사람이 검토
python scripts/eval_retrieval.py --methods lexical       # API 키 없이 lexical만
python scripts/eval_retrieval.py --ablations all         # 3방식 + ablation (OPENAI_API_KEY 필요)
```

결과는 `eval/results/<timestamp>.md`(위 표와 같은 모양의 붙여넣기용 표 포함)와 질의별 CSV로 남습니다. DB에 연결할 수 없거나 질의셋의 정답 id가 DB에 없으면 결과를 쓰지 않고 실패합니다.

## 기술적 결정과 트레이드오프

- **PostgreSQL 단일 저장소**: raw payload(JSONB)·파이프라인 상태, 정제 데이터, 벡터, 작업 큐, AI 결과 캐시, Django 테이블을 한 DB에 둡니다. raw 저장과 prepare 작업 등록을 한 트랜잭션으로 묶을 수 있고, 별도 메시지 브로커 없이 `SKIP LOCKED`와 `LISTEN/NOTIFY`로 큐를 만들 수 있어 운영할 대상이 줄어듭니다. 대신 벡터 인덱스와 FTS 튜닝을 직접 챙겨야 하고, 규모가 커지면 분리를 검토해야 합니다.
- **서버/로컬 worker 분리**: 서버는 항상 켜진 수집·저장만 맡고, GPU가 필요한 파싱은 로컬 worker가 서버 DB에 직접 적재합니다. 서버에 GPU가 없어도 되지만, 로컬 worker가 꺼져 있으면 수집분이 처리되지 않고 Tailscale 연결에 의존합니다.
- **3단 파서 폴백**: HURIDOCS 레이아웃 분석이 섹션 구조를 가장 잘 살리지만 GPU 컨테이너와 긴 처리 시간이 필요합니다. 실패하면 pypdf, 그것도 실패하면 초록으로 내려가 최소한의 청크는 남깁니다. 폴백 결과가 기존 PDF 본문을 덮지 않도록 막아 두었습니다.
- **AI 결과 캐시 키**: 개요는 `arxiv_id` 단독 기본키(모델은 기록용), 상세 요약은 `(arxiv_id, model)` 유일 제약입니다. 모든 사용자가 캐시를 공유하므로 같은 논문을 다시 열 때 LLM을 부르지 않습니다. 대신 프롬프트를 바꿔도 기존 캐시를 무효화할 버전 정보가 없고, 한 사용자가 만든 결과를 모두가 봅니다.

## 알려진 한계와 로드맵

- **검색 평가 수치**: 위 Evaluation 표는 아직 비어 있습니다. hybrid를 제품 경로로 먼저 연결했고, 재처리·백필을 마친 DB에서 lexical / vector / hybrid와 ablation을 측정해 채울 계획입니다. 인덱스 도입 전후 지연도 `EXPLAIN ANALYZE`로 함께 기록합니다.
- **한국어 lexical**: FTS 설정이 `english`라 임베딩 키가 없는 lexical 폴백에서는 한국어 질문이 거의 맞지 않습니다.
- **ASGI 전환**: 지금은 gunicorn gthread(워커 4 × 스레드 8)라 SSE 스트림 하나가 스레드 하나를 오래 점유합니다.
- **배포**: nginx가 HTTP만 제공합니다. 외부에 공개하기 전에 TLS(또는 Tailscale 전용 접근)와 `DJANGO_SECURE_COOKIES=true`, 공유 rate limit용 Redis가 필요합니다.
- **캐시 무효화**: AI 결과 캐시에 프롬프트 버전이 없어서 프롬프트를 바꿔도 기존 결과가 그대로 보입니다.

## 프로젝트 구조

```text
backend/            Django 프로젝트 (arxplore_web 설정, papers 앱 API)
frontend/           React + TypeScript + Vite
src/core/           모델, 프롬프트, 요약 그래프, LangGraph 에이전트
src/integrations/   PostgreSQL 저장소(raw·정제·벡터·큐), PDF 파서, 임베딩, 검색
src/pipeline/       수집·prepare·임베딩 진입점과 prepare-worker
src/shared/         설정(Pydantic AppSettings)과 LangSmith 트레이싱
dags/               Airflow DAG 3개
docker/             이미지와 런타임 설정
scripts/            실행 스크립트, 스키마 마이그레이션, 운영 스크립트
tests/              단위 테스트(unit), PostgreSQL 통합 테스트(integration)
docs/               아키텍처와 팀 운영 문서
```

## 문서

- [Architecture](./docs/architecture/ARCHITECTURE.md): 런타임 구성, 모듈 경계, 테이블 스키마, 큐 동작
- [AI Rules](./docs/architecture/AGENTS.md): AI 도구 작업 규칙과 공용 계약
- [Workflow](./docs/management/WORKFLOW.md), [Roles](./docs/management/ROLES.md), [Team Setup](./docs/management/TEAM_SETUP.md): 팀 프로젝트 당시 운영 문서
- [`.env.example`](./.env.example): 전체 환경 변수와 필수·선택 구분

## License

팀 프로젝트라서 라이선스는 팀원 동의를 받은 뒤 정합니다. 제안안은 MIT입니다. LICENSE 파일이 추가되기 전까지 저작권은 각 기여자에게 있습니다.
