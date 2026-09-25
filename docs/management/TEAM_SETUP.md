# ArXplore 개발 환경 설정 가이드

> SK네트웍스 AI 캠프 팀 프로젝트 당시 문서입니다. 역할·작업 방식은 그때 기준이며, 현재 코드 구조는 [README](../../README.md)와 [ARCHITECTURE](../architecture/ARCHITECTURE.md)를 따릅니다.

## 1. 문서 목적

이 문서는 ArXplore 프로젝트 구성원이 로컬 개발 환경과 서버 연결 환경을 준비할 때 따라야 하는 절차를 정리한다. 현재 운영 구조는 `서버 DB + 서버 Airflow + 로컬 dev 컨테이너 + 로컬 parser + 로컬 prepare-worker`를 기준으로 한다. 즉, 서버는 수집 자동화와 저장소를 담당하고, 무거운 파싱과 임베딩은 팀원 각자의 개발용 PC에서 수행한다.

## 2. 전체 흐름

```text
1. WSL 또는 로컬 개발 환경 준비
2. Tailscale 연결
3. 저장소 clone
4. .env 배치
5. 기본 컨테이너 실행 (django + nginx, 필요하면 dev 프로필로 vite) 후 스키마 생성 (migrate_schema.py)
6. parser 프로필 실행 (layout-parser + prepare-worker가 함께 올라옴)
7. 필요 시 서버 Airflow와 DB 접근을 위한 포트 포워딩
```

## 3. 준비 사항

- Git
- Docker Desktop 또는 Docker Engine
- 전달받은 `.env` (변수 목록과 필수/선택 구분은 루트 `.env.example` 참고)
- 전달받은 Tailscale Auth Key

설치 확인:

```bash
docker --version
docker compose version
```

## 4. Tailscale 설치

개발 컨테이너에서 서버의 MongoDB, PostgreSQL, Airflow에 접근하려면 Tailscale 연결이 필요하다.

### WSL 사용자

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --auth-key=<TAILSCALE_AUTH_KEY>
tailscale status
```

### macOS 사용자

```bash
brew install --cask tailscale
sudo tailscale up --auth-key=<전달받은 Auth Key>
```

### 연결 확인

```bash
tailscale ping <TAILSCALE_SERVER_IP>
```

`pong`이 오면 서버 연결이 정상이다.

Auth Key와 서버 Tailscale IP는 저장소에 기록하지 않고 팀 내부 채널로 전달받는다. 서버 IP는 `.env`의 `TAILSCALE_SERVER_IP`에 넣는다.

## 5. 저장소 가져오기

```bash
git clone -b dev <repository-url>
cd ArXplore
```

## 6. `.env` 배치

전달받은 `.env`를 프로젝트 루트에 둔다. 현재 `.env`에는 MongoDB, PostgreSQL, LangSmith, parser, worker가 사용할 접속 정보가 포함된다. 새로 만들 때는 `cp .env.example .env`에서 시작한다.

환경 변수 변경을 반영해야 할 때는 컨테이너를 재생성하는 편이 안전하다.

```bash
docker compose up -d --force-recreate django nginx
```

현재 구조에서 중요한 값은 다음과 같다.

- `MONGO_DB=arxplore_source`
- `POSTGRES_DB=arxplore_meta`
- `APP_POSTGRES_DB=arxplore_app`
- `SERVER_MONGO_PORT`
- `SERVER_POSTGRES_PORT` (서버 compose가 publish하는 호스트 포트, 기본 `15432`)
- `PROD_POSTGRES_HOST` (django·prepare-worker 컨테이너가 접속할 서버 PostgreSQL. 보통 `<TAILSCALE_SERVER_IP>`)
- `LAYOUT_PARSER_BASE_URL` (기본값·자동 감지 없음. prepare-worker와 같은 compose 네트워크이므로 `http://layout-parser:5060`)
- `TAILSCALE_SERVER_IP` (`scripts/setup.sh forward`, 서버 compose 포트 바인딩에 사용)

접속 값은 아래 규칙을 유지한다.

- `MONGO_HOST`, `POSTGRES_HOST`, `PROD_POSTGRES_HOST`는 `host` 또는 `host:port` 형식이다
- 포트를 생략하면 `SERVER_MONGO_PORT`, `SERVER_POSTGRES_PORT`(서버가 공개하는 포트)를 쓴다
- 서버 compose 안의 Airflow처럼 같은 네트워크에서 컨테이너 이름으로 붙을 때는 `arxplore-postgres:5432`, `arxplore-mongo:27017`처럼 내부 포트를 적는다

개인별로 바꿔야 하는 값은 최소한 아래다.

| 항목 | 설명 |
|------|------|
| `LANGSMITH_TRACE_USER` | 개인 trace 식별자 |

## 7. 시연/수정 컨테이너 실행

```bash
bash scripts/setup.sh
docker compose ps
```

기본 컨테이너:

- `arxplore-django`
- `arxplore-nginx`
- `arxplore-vite` (`dev` 프로필: `docker compose --profile dev up -d vite`)

기본 접속:

- Web: `http://127.0.0.1`
- Vite (프론트엔드 수정 실시간 확인): `http://127.0.0.1:5173`

새 DB를 쓰거나 스키마가 바뀐 뒤에는 스키마를 1회 만든다. 리포지토리 생성자는 DDL을 실행하지 않으며, prepare-worker는 시작할 때 스스로 스키마를 확인한다.

```bash
docker compose exec django python /workspace/scripts/migrate_schema.py
```

`arxplore-vite`는 `dev` 프로필이라 `setup.sh`로는 올라오지 않는다. 프론트엔드를 수정할 때 `docker compose --profile dev up -d vite`로 따로 띄운다.

이 환경에서 할 수 있는 작업:

- Django API 점검
- React 프론트엔드 개발
- 간단한 DB 점검

## 8. 로컬 parser 및 prepare-worker 실행

PDF 파싱 품질 검증과 실제 prepare는 HURIDOCS 기반 parser 컨테이너와 prepare-worker를 함께 쓰는 것을 기준으로 한다. 둘은 같은 `parser` 프로필에 묶여 있어 한 번에 올라온다.

```bash
docker compose --profile parser up -d --build
docker logs -f arxplore-layout-parser
docker logs -f arxplore-prepare-worker
```

parser 컨테이너 원칙:

- 공용 서버 스택에 올리지 않는다
- 로컬 개발용 PC에서 실행한다
- 가능하면 GPU를 사용하고, 없으면 CPU fallback을 허용한다
- parser가 없어도 `pypdf` fallback은 동작하지만 품질 기준은 parser 실행 상태를 기본으로 본다

prepare-worker 동작 방식:

- `arxplore_daily_collect`가 `prepare_jobs`에 날짜를 넣는다
- `prepare-worker`가 새 job을 기다린다
- job을 claim하면 `prepare -> embed`를 수행한다
- 결과는 서버 PostgreSQL에 직접 적재된다

`prepare-worker`는 auto 모드에서 `LISTEN/NOTIFY`로 대기하며, 새 작업이 없을 때는 polling보다 가벼운 형태로 쉬고 있다가 job이 생기면 거의 즉시 반응한다.

### 1회 실행 또는 backfill이 필요할 때

상시 worker가 떠 있는 동안에도 점검/백필 목적으로 같은 컨테이너 안에서 직접 호출할 수 있다.

```bash
# 1회 실행 점검
docker compose --profile parser exec prepare-worker \
  python3 -m src.pipeline.prepare_worker --mode auto --max-jobs-per-run 1

# 과거 raw 백필
docker compose --profile parser exec prepare-worker \
  python3 -m src.pipeline.prepare_worker --mode backfill --batch-days 3 --loop --sleep-seconds 120
```

## 9. 서버 스택 실행

서버 스택은 수집 자동화와 DB 운영을 담당한다.

```bash
bash scripts/setup-server.sh
docker compose -p arxplore_server -f docker-compose.server.yml ps
```

서버 `.env`에는 아래 값이 반드시 있어야 한다. 비어 있으면 `docker compose`가 실행을 거부한다.

| 항목 | 설명 |
|------|------|
| `TAILSCALE_SERVER_IP` | PostgreSQL(15432), MongoDB(17017), Airflow(18080)를 이 IP에만 바인딩한다. 로컬 worker가 Tailscale로 접속하므로 `127.0.0.1`로 바꾸지 않는다 |
| `AIRFLOW_FERNET_KEY` | `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`로 생성 |
| `AIRFLOW_ADMIN_USER` | Airflow SimpleAuthManager의 admin 사용자 이름 |

Airflow 로그인 비밀번호는 SimpleAuthManager가 최초 기동 시 생성한다. `docker logs arxplore-airflow-web`에서 확인하거나 컨테이너 안의 `simple_auth_manager_passwords.json.generated`를 확인한다.

핵심 컨테이너:

- `arxplore-postgres`
- `arxplore-mongo`
- `arxplore-airflow-init`
- `arxplore-airflow-web`
- `arxplore-airflow-scheduler`
- `arxplore-airflow-dag-processor`

현재 Airflow DAG는 3개다.

- `arxplore_daily_collect` (매일 18:00 KST)
- `arxplore_maintenance` (3시간마다)
- `arxplore_langsmith_maintenance` (매일 03:00 KST, LangSmith trace 정리)

## 10. Airflow 운영 기준

현재 운영 모델은 다음과 같다.

- `arxplore_daily_collect`
  - 최신 HF Daily Papers raw 수집
  - `prepare_jobs` enqueue
- `arxplore_maintenance`
  - backfill
  - metadata enrichment
- `arxplore_langsmith_maintenance`
  - 오래된 LangSmith trace 정리
- 로컬 `prepare-worker`
  - prepare
  - embed

즉 서버 Airflow만 켜 둔다고 prepare와 embed가 자동으로 실행되는 구조가 아니다. 로컬 worker도 함께 떠 있어야 최신 수집분이 실제 `paper_fulltexts`, `paper_chunks`, `paper_embeddings`까지 이어진다.

## 11. 적재 상태 점검

적재 상태는 PostgreSQL에 직접 조회해 확인한다.

```sql
SELECT status, count(*) FROM prepare_jobs GROUP BY status;
SELECT source, count(*) FROM paper_fulltexts GROUP BY source;
SELECT count(*) FROM paper_chunks c
LEFT JOIN paper_embeddings e ON e.chunk_id = c.id
WHERE e.chunk_id IS NULL;
```

failed 잡은 `python scripts/requeue_failed_prepare_jobs.py --since YYYY-MM-DD`로 확인하고(기본 dry-run), `--apply`로 재등록한다.

운영상 가장 먼저 확인할 것은 아래 네 가지다.

1. raw가 MongoDB에 들어가는지
2. `prepare_jobs`가 생성되는지
3. `prepare-worker`가 이를 소비하는지
4. embedding backlog가 남아 있지 않은지

## 12. Windows 브라우저에서 서버 접근

WSL에만 Tailscale이 설치되어 있으면 Windows 브라우저에서 Airflow와 DB를 직접 보기 어렵다. 이 경우 `scripts/setup.sh`의 `forward` 서브커맨드로 포트 포워딩을 사용한다.

사전 준비:

```bash
sudo apt install openssh-server -y
sudo service ssh start
```

그 다음 실행:

```bash
bash scripts/setup.sh forward
```

제어 명령:

```bash
bash scripts/setup.sh forward status
bash scripts/setup.sh forward stop
bash scripts/setup.sh forward restart
```

포워딩 주소:

| 서비스 | Windows 브라우저 주소 |
|--------|----------------------|
| Airflow | `http://127.0.0.1:18080` |
| PostgreSQL | `127.0.0.1:15432` |
| MongoDB | `127.0.0.1:17017` |

## 13. 접속 정보 요약

### dev 컨테이너 / 코드 기준

| 서비스 | 주소 |
|--------|------|
| PostgreSQL | `<TAILSCALE_SERVER_IP>:15432` |
| MongoDB | `<TAILSCALE_SERVER_IP>:17017` |
| Airflow API | `http://<TAILSCALE_SERVER_IP>:18080` |
| Layout Parser | `http://layout-parser:5060` (prepare-worker와 같은 compose 네트워크) |

### Windows 브라우저 / DB 클라이언트 기준

| 서비스 | 주소 |
|--------|------|
| Airflow | `http://127.0.0.1:18080` |
| PostgreSQL | `127.0.0.1:15432` |
| MongoDB | `127.0.0.1:17017` |

## 14. LangSmith 설정

LangSmith는 별도 컨테이너 없이 Python 실행 환경에서 사용한다. 공용 프로젝트 이름은 유지하고, 각자 `LANGSMITH_TRACE_USER`만 개인 식별자로 맞춘다.

```env
LANGSMITH_TRACE_USER=홍길동
```
