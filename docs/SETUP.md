# 실행·운영·테스트

README의 Quick Start를 자세히 풀어 쓴 문서입니다. 환경 변수 전체 목록은 [`.env.example`](../.env.example)이 기준입니다.

## compose 프로필

| 프로필 | 서비스 | 용도 |
| --- | --- | --- |
| (기본) | `django`, `nginx` | 웹 앱. nginx는 django healthcheck가 통과한 뒤 뜹니다. |
| `dev` | `vite` | 프론트엔드 HMR 개발 서버(`http://localhost:5173`, 호스트 127.0.0.1에만 바인딩) |
| `parser` | `layout-parser`, `prepare-worker` | GPU PDF 파서와 prepare 큐 worker. worker는 파서 healthcheck 통과 뒤 뜹니다. |
| `local-db` | `postgres-local` | 원격 서버 없이 쓰는 로컬 PostgreSQL(pgvector) |

프로필은 겹쳐 쓸 수 있습니다. 예: `docker compose --profile local-db --profile dev up -d --build`.

## (a) 로컬 단독 실행

원격 서버 없이 로컬 PostgreSQL 하나로 웹 앱을 띄웁니다. 수집(Airflow)은 돌지 않으므로 **논문 목록은 빈 상태로 시작합니다.**

```bash
cp .env.example .env
# DJANGO_SECRET_KEY, POSTGRES_PASSWORD, OPENAI_API_KEY 등 change-me로 시작하는 값을 실제 값으로 바꾼다

docker compose --profile local-db up -d postgres-local   # pgvector/pgvector:pg16, 127.0.0.1:15432

python scripts/migrate_schema.py   # 호스트 Python 3.12 + requirements.txt 필요
# 호스트에 Python 환경이 없으면: docker compose run --rm django python /workspace/scripts/migrate_schema.py

docker compose --profile local-db up -d --build             # 웹: http://localhost
docker compose --profile local-db --profile dev up -d vite  # (선택) Vite HMR: http://localhost:5173
```

- 로그인 없이 모든 화면을 씁니다. 개요·상세 요약 생성, 챗, 어시스턴트는 `.env`의 `OPENAI_API_KEY`를 씁니다.
- `DJANGO_CSRF_TRUSTED_ORIGINS`에는 앱에 접속하는 모든 origin이 들어가야 합니다. 기본값은 `http://localhost`, `http://127.0.0.1`과 Vite dev 서버(`:5173`)이며, `PROD_HTTP_PORT`·`FRONTEND_PORT`를 바꾸거나 다른 호스트명으로 접속하면 해당 origin(포트 포함)을 추가하지 않는 한 개요·요약·챗 등 POST 요청이 403으로 막힙니다.
- `DJANGO_SECRET_KEY`가 비어 있으면 Django가 시작하지 않습니다. `DJANGO_DEBUG`는 기본으로 꺼져 있고 값이 `true`일 때만 켜집니다(compose는 항상 끕니다).
- 접속 주소: `postgres-local`은 django·worker와 같은 compose 기본 네트워크에 있으므로 컨테이너는 `PROD_POSTGRES_HOST=postgres-local:5432`로 접속합니다(`host.docker.internal`이나 `extra_hosts`가 필요 없습니다). 호스트에서 돌리는 스크립트는 `POSTGRES_HOST=localhost`와 `SERVER_POSTGRES_PORT=15432`를 씁니다. `.env.example`의 기본값이 이 구성입니다.
- `PROD_POSTGRES_HOST`와 `POSTGRES_HOST`는 `host` 또는 `host:port` 형식입니다. 포트를 생략하면 `SERVER_POSTGRES_PORT`(`.env.example` 값 15432)를 씁니다.
- django 이미지는 root가 아닌 `app`(UID 1000) 사용자로 돕니다. 이전 이미지로 만든 `django_static` 볼륨은 root 소유라 `collectstatic`이 실패하므로 한 번 지우고 다시 올립니다: `docker compose down && docker volume rm arxplore_django_static`.

## (b) 원격 서버 모드

서버(PostgreSQL · Airflow)를 Tailscale로 공유하고(참고: 팀 시절 커밋 c7b2c34에 포함됐던 Tailscale 인증 키는 폐기되었고 현재 문서는 플레이스홀더만 담습니다), 로컬에서 웹과 GPU 파서·prepare-worker를 돌리는 원래 팀 구성입니다. 절차는 [TEAM_SETUP.md](./management/TEAM_SETUP.md)를 따릅니다.

```bash
bash scripts/setup-server.sh                  # 서버: PostgreSQL / Airflow
bash scripts/setup.sh                         # 로컬: django + nginx
docker compose --profile parser up -d --build # 로컬 GPU: layout-parser + prepare-worker
```

`parser` 프로필의 prepare-worker는 `LAYOUT_PARSER_BASE_URL`이 비어 있으면 `http://layout-parser:5060`을 씁니다. Airflow 이미지는 DAG에 필요한 패키지(`requirements-airflow.txt`)만 Airflow 공식 constraints 파일에 맞춰 설치합니다.

## 운영 기본값

- nginx: `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy` 헤더와 gzip을 켭니다. Django admin은 설치하지 않습니다.
- HTTPS 뒤에 둘 때는 `DJANGO_SECURE_COOKIES=true`로 CSRF 쿠키에 Secure를 붙입니다. nginx는 HTTP만 제공합니다.
- rate limit은 클라이언트 IP(`RATE_LIMIT_IP_HEADER`, 기본 `X-Real-IP`) 단위입니다. LLM 호출은 `RATE_LIMIT_LLM_PER_MINUTE`(기본 30), `detail.json`은 `RATE_LIMIT_DETAIL_PER_MINUTE`(기본 60)입니다. 카운터는 기본으로 프로세스별 메모리 캐시라서 gunicorn 워커 4개가 따로 셉니다. 한도를 정확히 공유하려면 `REDIS_URL`을 지정합니다.
- 공개 배포에서는 모든 방문자의 LLM 호출이 서버 키 비용으로 잡힙니다. IP당 한도 외에 전체 사용량 상한은 없습니다.

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
| python | ruff, worklog 검사, `pytest tests/unit`, `manage.py check`, `makemigrations --check` |
| integration | pgvector 서비스 컨테이너로 `pytest tests/integration` |
| frontend | `npm run typecheck`, `npm run build` |
| infra | 더미 `.env`로 두 compose 파일 `docker compose config`, hadolint |
| security | gitleaks로 git 히스토리 시크릿 스캔 |
