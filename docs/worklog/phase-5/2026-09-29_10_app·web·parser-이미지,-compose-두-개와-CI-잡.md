---
title: app·web·parser 이미지, compose 두 개와 CI 잡
date: 2026-09-29
area: [infra, tooling]
decision: api·worker·scheduler가 함께 쓰는 app 이미지(python:3.14-slim, uv, 비root)와 nginx web 이미지를 만들고, 로컬 compose(api·web과 dev·parser·local-db 프로필)와 서버 compose(postgres·scheduler)로 나눈다. 서버에서 Airflow 컨테이너 4개와 시크릿 3종이 빠진다. CI는 v2를 루트로 올린 뒤의 배치를 기준으로 python·integration·frontend·infra·security·worklog 잡을 둔다
---

**결정과 근거** — 새 스택(FastAPI, 경량 스케줄러, uv)에 맞춰 이미지와 compose를 다시 만들었다. 옛 compose와 스크립트는 P8
전환까지 그대로 둔다.

이미지(e3170a7)
- app: `python:3.14-slim`에서 `uv sync --frozen --no-dev`로 의존성 층을 먼저 만들고 프로젝트를 editable이 아닌 형태로 설치한다.
  UID 1000 비root로 돌고 진입점은 `arxplore`다. api(`arxplore api`), worker(`worker --loop`), scheduler(`scheduler`)가 같은
  이미지를 명령만 바꿔 쓴다. 베이스 이미지에 `SCHEDULER_TIMEZONE`에 필요한 tzdata가 있다.
- web: Node 24에서 프론트를 빌드하고 nginx stable-alpine으로 서빙한다.
- parser: 옛 구현과 같은 HURIDOCS v0.0.31이다.
- postgres init 스크립트는 `APP_POSTGRES_DB`를 `format %I`로 인용해 만든다. `.dockerignore`는 두 빌드가 읽는 경로만 허용한다.

nginx(`v2/docker/web/nginx.conf`)
- `/assets/`는 1년 immutable 캐시를 주고 없는 파일은 404다(`try_files $uri =404`). `index.html`은 no-cache, 그 밖의 경로는
  SPA로 돌린다. FastAPI의 `SERVE_FRONTEND_DIST` 서빙도 같은 규칙을 따른다(b1f618a).
- 옛 nginx.conf의 API 경로를 모두 프록시한다. 프론트가 더 이상 부르지 않는 채팅 JSON 경로도 다른 클라이언트를 위해 둔다.
- SSE 경로는 버퍼링·캐시·gzip을 끄고 읽기 타임아웃을 300초로 둔다.
- Host는 포트를 붙여 넘긴다(`$http_host`, 옛 설정은 `$host`). 기본 포트가 아닐 때 같은 출처 POST가 Origin 검사를
  통과하게 하기 위해서다.
- upstream은 Docker DNS(`resolver 127.0.0.11`)로 다시 해석해, api 컨테이너를 다시 만들어도 nginx가 옛 주소에 머물지 않는다.

compose(780ac27)
- `compose.yaml`(프로젝트 `arxplore`): api(8001, `/bootstrap.json` HTTP healthcheck)와, api가 healthy일 때 뜨는 web
  (`PROD_HTTP_PORT`). 프로필은 dev(Vite, `API_PROXY_TARGET=http://api:8001`), parser(GPU layout-parser와 worker,
  `LAYOUT_PARSER_BASE_URL` 기본 `http://layout-parser:5060`, `PREPARE_BACKFILL_ENABLED` 전달), local-db(pgvector pg16,
  `127.0.0.1:SERVER_POSTGRES_PORT`)다. api와 worker는 local-db가 켜졌을 때만 postgres-local을 기다린다. 앱 서비스는 init
  프로세스로 돌아 `docker stop`이 kill 타임아웃을 기다리지 않는다.
- 마이그레이션은 별도 단계(`docker compose run --rm api migrate`)다. API는 스키마가 준비되지 않으면 뜨지 않는다(F15, 03번 항목).
- `compose.server.yaml`(프로젝트 `arxplore_server`): postgres와 scheduler 두 서비스다. 옛 서버 compose의 Airflow 서비스 4개
  (init, web, scheduler, dag-processor)와 시크릿 3종(`AIRFLOW_FERNET_KEY`, `AIRFLOW_API_SECRET_KEY`,
  `AIRFLOW_API_AUTH_JWT_SECRET`)이 빠졌다. `TAILSCALE_SERVER_IP` 바인딩과 외부 데이터 볼륨 이름은 그대로다.
- `.env.example`(fd20d96)은 필수·서버·선택으로 나누고 옛 `DJANGO_*`·`OPENAI_API_BASE` 별칭을 표시했다. Django·Airflow
  변수는 빼고 `OPENAI_BASE_URL`, `POSTGRES_CONNECT_TIMEOUT`, `PREPARE_BACKFILL_ENABLED`, `SCHEDULER_TIMEZONE`,
  `SERVE_FRONTEND_DIST`, `RUNTIME`, `API_WORKERS`, `RATE_LIMIT_IP_HEADER=none`을 더했다.
- 셋업 스크립트(c7fe994): `setup.sh`는 `db-status`가 ready가 아니면 스택을 올리지 않고 migrate 명령을 안내한다.
  `forward`는 PostgreSQL만 터널링한다. `setup-server.sh`는 옛 Airflow 스택 컨테이너가 남아 있으면 멈추고, 외부 볼륨이
  없으면 만든 뒤 postgres, `arxplore migrate`, scheduler 순으로 올린다.

CI(9bc84e4, fcde8cb, `v2/.github/workflows/ci.yml`)
- python: `uv lock --check`, `uv sync --frozen`, ruff check·format, mypy, `arxplore openapi`와 `frontend/openapi.json` 대조,
  단위 테스트
- integration: pgvector pg16 서비스에서 새 DB를 `arxplore migrate --dry-run`, 적용, 재실행(no-op) 뒤 `db-status`가
  `ready: yes`인지 보고, `V2_TEST_REQUIRE_DB`로 통합 테스트를 돌린다.
- frontend: Node 24에서 `npm ci`, lint, typecheck, test, build
- infra: 두 compose 파일의 `docker compose config -q`, hadolint, shellcheck, `nginx -t`
- security: 전체 이력 gitleaks. worklog: `scripts/worklog.py check`
- 골든 테스트는 골든 DB와 Playwright가 필요해 CI 밖에서 돈다.

**트레이드오프** — 서버에서 Airflow UI와 태스크 로그를 잃는다. 실행 이력은 `scheduler_runs`와 `arxplore scheduler
history`로, 수동 실행은 `arxplore run`으로 대신한다(06번 항목).
- app 이미지 하나를 세 역할이 나눠 쓴다. 빌드는 한 번이지만 각 컨테이너에 다른 역할의 의존성도 들어간다.
- 마이그레이션을 시작 명령에 넣지 않아 배포 절차가 한 단계 늘었다. 대신 요청 경로와 worker에서 DDL이 사라졌다.
- `API_WORKERS` 기본값이 1이다. `REDIS_URL` 없이 늘리면 rate limit 허용량이 워커 수만큼 는다(07번 항목).
- CI 워크플로는 v2를 루트로 올린 뒤의 경로로 썼다. 전환 전에는 같은 명령을 로컬에서 돌려 확인한다.

**eval 영향** — 새 측정 지점은 CI 잡 여섯 개다. 그중 `arxplore openapi | diff -u frontend/openapi.json -`가 백엔드
스키마와 프론트 생성 타입의 어긋남을, integration 잡의 migrate 세 번과 `db-status`가 빈 DB에서의 마이그레이션 멱등성을,
infra 잡이 compose·Dockerfile·셸 스크립트·nginx 설정의 문법을 잡는다.

**알려진 한계** — 전환 전에는 compose 프로젝트 이름이 부딪힌다. 옛 `docker-compose.yml`과 `v2/compose.yaml`이 모두
`name: arxplore`이고, 볼륨 이름도 같다(`postgres_local_data`, `vite_node_modules`). `-p` 없이 v2 스택을 올리면 옛
컨테이너(django, nginx, prepare-worker)를 orphan으로 보고 같은 이름의 서비스(vite, layout-parser, postgres-local)를 새
설정으로 바꾼다. 그 상태에서 `docker compose run --rm api migrate`를 돌리면 옛 로컬 DB(`arxplore_postgres_local_data`)에
0002~0006이 적용된다. 0002는 테이블을 지우고 0003은 컬럼 타입을 바꾸므로 백업 없이는 되돌릴 수 없다. 전환 전까지는 다른
프로젝트 이름(`-p`)으로 올려야 한다.
- CI는 이미지를 빌드하지 않는다. infra 잡은 Dockerfile 린트(hadolint)와 compose 문법만 보므로, 빌드 실패나 컨테이너 기동
  실패는 로컬에서만 드러난다. compose 전체 기동(healthcheck, nginx 경로, SSE)을 확인하는 CI 잡과 결과 파일도 없다.
- 워크플로는 `v2/.github`에 있어 지금은 돌지 않는다. 브랜치를 push하지 않으므로 P8 전환 전까지 CI 결과가 없다.
- 골든 테스트(G1~G6)는 CI에 없다.

**트러블슈팅** — 해당 없음
