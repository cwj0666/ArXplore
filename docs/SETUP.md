# 실행·운영·테스트

README의 Getting Started를 자세히 풀어 쓴 문서입니다. 환경 변수 전체 목록과 기본값은 [`.env.example`](../.env.example)이 기준입니다.

## 구성 요소

API, worker, 스케줄러는 같은 패키지(`src/arxplore`)와 같은 app 이미지를 쓰고, 명령만 다릅니다.

| 명령 | 하는 일 |
| --- | --- |
| `arxplore migrate` | `src/arxplore/db/migrations`의 번호 SQL을 차례로 적용한다. `--dry-run`, `--target` |
| `arxplore db-status` | 적용한 마이그레이션, 임베딩 차원, 검색 인덱스를 보여 주고 마지막 줄에 `ready: yes|no`를 쓴다 |
| `arxplore api` | FastAPI 앱을 uvicorn으로 띄운다(기본 `127.0.0.1:8001`) |
| `arxplore worker --loop` | prepare 잡을 받아 PDF 파싱, 청크 저장, 임베딩을 한다 |
| `arxplore scheduler` | collect, maintenance, langsmith_cleanup을 cron 일정으로 돌린다. `scheduler history`는 최근 실행 기록 |
| `arxplore run <작업>` | collect, maintenance, langsmith-cleanup을 한 번 돌리고 수동 실행으로 기록한다. rechunk, reparse는 코퍼스 도구다 |
| `arxplore openapi` | API OpenAPI 스키마를 JSON으로 쓴다(`frontend/openapi.json`의 원본) |

API와 worker는 시작할 때 스키마 버전과 임베딩 차원을 확인하고, 적용하지 않은 마이그레이션이 있거나 차원이 설정과 다르면 뜨지 않습니다. 스키마는 `arxplore migrate`만 바꿉니다.

## compose 프로필

로컬 스택은 `compose.yaml`(프로젝트 이름 `arxplore`), 서버 스택은 `compose.server.yaml`(프로젝트 이름 `arxplore_server`)입니다.

| 프로필 | 서비스 | 용도 |
| --- | --- | --- |
| (기본) | `api`, `web` | 웹 앱. `web`(nginx)은 `api` healthcheck가 통과한 뒤 뜨고 `PROD_HTTP_PORT`(기본 80)에 공개됩니다 |
| `dev` | `vite` | 프론트엔드 HMR 개발 서버(`127.0.0.1:${FRONTEND_PORT:-5173}`), API는 `api:8001`로 프록시 |
| `parser` | `layout-parser`, `worker` | GPU PDF 파서와 prepare 큐 worker. worker는 파서 healthcheck가 통과한 뒤 뜹니다 |
| `local-db` | `postgres-local` | 원격 서버 없이 쓰는 로컬 PostgreSQL 16 + pgvector(`127.0.0.1:${SERVER_POSTGRES_PORT:-15432}`) |

프로필은 겹쳐 쓸 수 있습니다. 예: `docker compose --profile local-db --profile dev up -d --build`. `.env`에 `COMPOSE_PROFILES=local-db`를 두면 `--profile`을 붙이지 않아도 됩니다.

## (a) 로컬 단독 실행

원격 서버 없이 로컬 PostgreSQL 하나로 웹 앱을 띄웁니다. 수집 스케줄러가 돌지 않으므로 논문 목록은 빈 상태로 시작합니다.

```bash
cp .env.example .env
# POSTGRES_PASSWORD, OPENAI_API_KEY 등 change-me로 시작하는 값을 실제 값으로 바꾼다

docker compose --profile local-db up -d postgres-local        # pgvector/pgvector:pg16, 127.0.0.1:15432
docker compose --profile local-db run --rm api migrate         # 스키마 적용
docker compose --profile local-db up -d --build                # 웹: http://localhost
docker compose --profile local-db --profile dev up -d vite     # (선택) Vite HMR: http://localhost:5173
```

- 로그인 없이 모든 화면을 씁니다. 개요·상세 요약 생성, 챗, 어시스턴트는 `.env`의 `OPENAI_API_KEY`를 씁니다. 키가 없으면 캐시된 결과만 보이고 새 생성과 챗은 503입니다.
- 논문을 채우려면 수집과 worker가 필요합니다. 로컬에서 한 날짜만 받으려면 `docker compose --profile local-db run --rm api run collect --target-date YYYY-MM-DD`를 돌리고, GPU가 있으면 `parser` 프로필로 worker를 띄웁니다.
- 컨테이너는 `PROD_POSTGRES_HOST`(local-db는 `postgres-local:5432`)로, 호스트에서 돌리는 `uv run arxplore ...`는 `POSTGRES_HOST=localhost`와 `SERVER_POSTGRES_PORT`로 접속합니다. 두 값 모두 `host` 또는 `host:port`이고, 포트를 생략하면 `SERVER_POSTGRES_PORT`를 씁니다.
- `CSRF_TRUSTED_ORIGINS` 기본값은 `http://localhost`, `http://127.0.0.1`과 두 호스트의 `FRONTEND_PORT` origin입니다. 같은 origin 요청은 목록과 관계없이 통과하므로, 다른 origin에서 여는 화면이 있을 때만 추가합니다.
- 기존 DB에 새 버전을 올릴 때는 먼저 백업하고 `migrate --dry-run`으로 적용할 파일을 확인합니다. 0002는 옛 topics 테이블을 지우고 0003은 `prepare_jobs.target_date`를 DATE로 바꾸므로 되돌리려면 백업이 필요합니다.

`scripts/setup.sh`는 `.env`와 `PROD_POSTGRES_HOST`를 확인하고 이미지를 빌드한 뒤, `db-status`가 `ready: yes`일 때만 스택을 올립니다. 준비되지 않았으면 migrate 명령을 안내하고 멈춥니다.

## (b) 원격 서버 모드

서버(PostgreSQL, 스케줄러)를 Tailscale로 공유하고, 로컬에서 웹과 GPU 파서·worker를 돌리는 구성입니다.

```bash
bash scripts/setup-server.sh                   # 서버: postgres 기동, arxplore migrate, scheduler 기동
bash scripts/setup.sh                          # 로컬: api + web
docker compose --profile parser up -d --build  # 로컬 GPU: layout-parser + worker
bash scripts/setup.sh forward                  # (선택) 서버 PostgreSQL을 localhost로 포워딩
```

- `compose.server.yaml`은 `postgres`와 `scheduler` 두 서비스입니다. PostgreSQL 포트는 `TAILSCALE_SERVER_IP`에만 바인딩하고, 데이터 볼륨은 외부 볼륨 `arxplore_server_arxplore_postgres_data`입니다. `setup-server.sh`는 볼륨이 없으면 만들고, 같은 볼륨을 쓰는 옛 Airflow 스택 컨테이너가 남아 있으면 멈춥니다.
- 스케줄러는 `SCHEDULER_TIMEZONE`(기본 Asia/Seoul)으로 collect(매일 18:00), maintenance(3시간마다), langsmith_cleanup(매일 03:00)을 돌립니다. 시작할 때 직전 슬롯의 실행 기록이 없으면 그 슬롯을 한 번 돌리고, advisory lock으로 인스턴스 하나만 돕니다. 실행 기록은 `docker compose -f compose.server.yaml exec scheduler arxplore scheduler history`로 봅니다.
- worker는 `LAYOUT_PARSER_BASE_URL`이 비어 있으면 compose가 `http://layout-parser:5060`을 넣습니다. compose 밖에서 worker를 돌리면 직접 적고, 끝내 비어 있으면 layout 단계를 건너뛰고 pypdf로 파싱합니다.
- 백필 수집분(mode=backfill 잡)은 `PREPARE_BACKFILL_ENABLED=true`일 때만 worker가 처리합니다.

## 운영 기본값

- nginx(`docker/web/nginx.conf`): API 경로만 `api`로 프록시하고 나머지는 SPA로 돌립니다. `/assets/`는 1년 immutable 캐시, `index.html`은 no-cache입니다. SSE 경로는 버퍼링·gzip을 끄고 읽기 제한을 300초로 둡니다. 보안 헤더 4종(`X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Cross-Origin-Opener-Policy`)과 gzip을 붙입니다.
- 새 API 경로를 추가하면 nginx location과 `frontend/vite.config.ts`의 프록시 목록에 같이 넣습니다.
- HTTPS 뒤에 둘 때는 `SECURE_COOKIES=true`로 CSRF 쿠키에 Secure를 붙입니다. nginx는 HTTP만 제공합니다.
- rate limit은 클라이언트 IP 단위입니다. IP는 `RATE_LIMIT_IP_HEADER`(기본 `X-Real-IP`)의 첫 값이고, `none`이면 소켓 주소를 씁니다. LLM을 부르는 6개 경로가 `RATE_LIMIT_LLM_PER_MINUTE`(기본 30)을 공유하고 `detail.json`은 `RATE_LIMIT_DETAIL_PER_MINUTE`(기본 60)입니다. 카운터는 기본으로 프로세스 메모리에 있으므로 `API_WORKERS`를 2 이상으로 두면 `REDIS_URL`을 함께 지정합니다.
- 공개 배포에서는 모든 방문자의 LLM 호출이 서버 키 비용으로 잡힙니다. IP당 한도 외에 전체 사용량 상한은 없습니다.
- 도커 없이 API 하나로 프론트까지 띄우려면 `cd frontend && npm run build` 뒤 `SERVE_FRONTEND_DIST=true uv run arxplore api`를 저장소 루트에서 실행합니다.

## 코퍼스 도구

청커나 파서를 바꾼 뒤 기존 코퍼스에 반영할 때 씁니다. 둘 다 기본은 dry-run입니다.

```bash
uv run arxplore run rechunk                     # 저장된 섹션으로 다시 청킹했을 때 바뀌는 논문·청크·재임베딩 토큰 추정
uv run arxplore run rechunk --apply             # 바뀐 청크만 교체. 지워진 임베딩은 worker backlog가 다시 만든다
uv run arxplore run reparse --since 2026-09-11 --until 2026-09-24            # 다시 파싱할 날짜 잡 확인
uv run arxplore run reparse --since 2026-09-11 --until 2026-09-24 --apply    # force 잡으로 다시 연다
```

실패한 날짜 잡을 다시 돌릴 때도 `run reparse --no-force --apply`를 씁니다. 처리 중인 잡은 건드리지 않습니다.

## 테스트와 CI

```bash
uv sync                                  # Python 3.14 환경과 dev 그룹(pytest, ruff, mypy 등)
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest tests/unit                 # DB·API 키 없이 도는 단위 테스트

# 통합 테스트: 전용 pgvector DB가 필요하다. 테스트마다 public 스키마를 지우므로 이름이 v2_test로 시작해야 한다
V2_TEST_DATABASE_URL=postgresql://arxplore:arxplore@localhost:5432/v2_test V2_TEST_REQUIRE_DB=1 \
  uv run pytest tests/integration

cd frontend && npm ci
npm run lint && npm run typecheck && npm test && npm run build
npm run check:api                        # 백엔드 OpenAPI 스키마와 frontend/openapi.json, 생성 타입이 같은지
```

`V2_TEST_DATABASE_URL`이 없으면 통합 테스트는 건너뛰고, `V2_TEST_REQUIRE_DB`가 있으면 대신 실패합니다. API 스키마를 바꾸면 `uv run arxplore openapi --output frontend/openapi.json`과 `cd frontend && npm run generate:api`로 생성 타입을 함께 갱신합니다.

### 골든 테스트

`tests/golden`은 리팩토링 전 구현의 출력을 고정한 골든 6종으로 현재 구현을 검사합니다. `tests/fakes`의 가짜 OpenAI·arXiv 서버가 모든 LLM·임베딩·arXiv 호출을 받고, 하네스는 루프백 밖으로 연결하지 않습니다.

| ID | 대상 | 필요한 것 |
| --- | --- | --- |
| G1 | 검색 순위(평가 111, 제품 142, 상세 챗 97) | 골든 DB |
| G2 | 저장된 섹션에서 만든 청크 | 골든 DB |
| G3 | pypdf 파싱(PDF 10편), HURIDOCS 응답 재생 | PDF(`GOLDEN_PDF_DIR`). 재생은 입력 없이 돈다 |
| G5 | HTTP·SSE 응답과 가짜 서버가 받은 LLM 요청(G4) | 골든 DB |
| G6 | 화면 11개의 PNG, innerText, ARIA | 골든 DB, Playwright, `frontend/dist` |

```bash
uv run pytest tests/fakes tests/golden
```

- 골든 DB는 평가 코퍼스를 담은 PostgreSQL 사본으로 저장소 밖에 있습니다. `GOLDEN_DATABASE_URL`(기본 `postgresql://arxplore@localhost:55432/arxplore`)을 읽기 전용으로 쓰고, 쓰기가 필요한 G5·G6은 템플릿 DB `arxplore_tpl`에서 만든 복사본에 마이그레이션을 적용해 씁니다. DB에 닿지 못하면 해당 테스트는 건너뜁니다.
- G3 PDF 10편은 `uv run python -m scripts.goldens.capture_g3_pypdf fetch`로 `GOLDEN_PDF_DIR`에 받습니다. HURIDOCS 응답 재생은 저장소의 기록만으로 돌고, 실제 파서와의 비교는 `GOLDEN_LAYOUT_PARSER_URL`(기본 `http://127.0.0.1:5060`)에 파서가 떠 있을 때만 돕니다.
- G6은 `tests/golden/ui`에서 `npm ci`와 `npx playwright install chromium`으로 Playwright를 준비하고 `frontend/dist`를 빌드한 뒤 돕니다. 하네스가 DB 사본에 `arxplore migrate`를 돌리고 `SERVE_FRONTEND_DIST=true`로 `arxplore api`를 띄워 캡처합니다.
- 원본 골든 파일은 옛 구현의 출력이고 그대로 둡니다. 수정으로 달라진 결과는 `<이름>.v2.json.gz`(G6은 `g6_ui.v2/`) 변형으로 두고, 비교는 변형이 있으면 변형과, 없으면 원본과 합니다. 차이와 원인은 `tests/golden/data/*.v2.diff.md`에 적습니다.
- 캡처는 `uv run python -m scripts.goldens.capture_<g1_retrieval|g2_chunks|g3_pypdf|g3_layout|g5_http|g6_ui>`이고 기본으로 v2 변형을 씁니다.

GitHub Actions(`.github/workflows/ci.yml`) 잡 구성:

| 잡 | 내용 |
| --- | --- |
| python | `uv lock --check`, ruff check·format, mypy, `arxplore openapi`와 `frontend/openapi.json` 대조, 단위 테스트, 골든 DB 없이 도는 가짜 서버·골든 하네스 테스트 |
| integration | pgvector 서비스 컨테이너에서 새 DB에 `migrate --dry-run`, 적용, 재실행 뒤 `db-status`가 `ready: yes`인지 보고 통합 테스트 |
| frontend | Node 24에서 `npm ci`, lint, typecheck, test, build |
| infra | 더미 `.env`로 두 compose 파일 `docker compose config`, hadolint, shellcheck, `nginx -t` |
| security | gitleaks로 git 히스토리 시크릿 스캔 |
| worklog | `scripts/worklog.py check` |
