# ArXplore 개발 및 운영 워크플로우

> SK네트웍스 AI 캠프 팀 프로젝트 당시 문서입니다. 역할·작업 방식은 그때 기준이며, 현재 코드 구조는 [README](../../README.md)와 [ARCHITECTURE](../architecture/ARCHITECTURE.md)를 따릅니다.

## 1. 문서 목적

이 문서는 ArXplore를 현재 코드 기준으로 어떻게 개발하고 운영할지 정리한 실행 문서다. 환경 준비 자체는 [TEAM_SETUP.md](./TEAM_SETUP.md)를 기준으로 하고, 본 문서는 "무엇이 이미 준비되어 있으며, 어떤 순서와 handoff로 작업해야 하는가"를 설명한다. 제품 기능과 현재 한계는 [README.md](../../README.md)를 따른다.

현재 기준에서 중요한 점은 다음과 같다.

- 데이터 적재 기반은 이미 구현되어 있다
- 남은 핵심 구현은 retrieval, answer chain, translation/summary, 논문 상세 문서, UI다
- 서버 수집 자동화와 로컬 prepare worker는 분리 운영된다
- 역할별 작업은 순차 대기보다 병렬+계약 정렬 방식으로 진행하는 것이 맞다

## 2. 기본 원칙

- 제품 기준은 [README.md](../../README.md)를 따른다
- 계층과 런타임 구조는 [ARCHITECTURE.md](../architecture/ARCHITECTURE.md)를 따른다
- 역할 경계는 [ROLES.md](./ROLES.md)를 따른다
- AI 작업 규칙은 [AGENTS.md](../architecture/AGENTS.md)를 따른다
- 공통 계약인 `PaperDetailDocument`, retrieval 결과 shape, answer payload는 쉽게 바꾸지 않는다
- DAG 파일은 가볍게 유지하고 실제 로직은 `src/pipeline`과 `src/integrations`에 둔다
- parser runtime은 서버가 아니라 로컬 개발용 PC에서 운영한다

## 3. 작업 시작 전 공통 확인

작업을 시작하기 전에 아래를 확인한다.

- `.env`가 최신인지
- Docker가 실행 중인지
- `arxplore-django`, `arxplore-nginx` 컨테이너가 정상인지
- 필요하면 parser 컨테이너가 올라와 있는지
- 서버 DB와 Tailscale 연결이 가능한지
- 현재 작업이 retrieval, answer, prompt, 논문 상세, UI 중 어디에 속하는지

상태 확인 명령:

```bash
docker compose ps
docker compose -p arxplore_server -f docker-compose.server.yml ps
```

`parser` 프로필 서비스(layout-parser, prepare-worker)는 위 첫 명령에 함께 표시된다.

## 4. 작업 모드

### 개발자 기본 모드

시연 검증은 단일 `docker-compose.yml`의 `arxplore-django`, `arxplore-nginx`, `arxplore-vite` 컨테이너를 기준으로 진행한다. 프론트엔드 dev server는 기본 실행에 포함된다.

```bash
bash scripts/setup.sh
docker compose ps django vite
```

새 DB를 쓰거나 스키마가 바뀐 뒤에는 스키마를 한 번 만든다. 리포지토리 생성자는 DDL을 실행하지 않는다.

```bash
docker compose exec django python /workspace/scripts/migrate_schema.py
```

원격 서버 없이 웹만 띄울 때는 `docker compose --profile local-db up -d`로 로컬 PostgreSQL을 함께 올린다(README Quick Start 참고).

이 모드에서 수행하는 작업:

- Python 코드 작성
- retrieval, prompt, chain 검증
- 단위 테스트(`pytest tests/unit`)
- Django API 실행
- React 프론트엔드 실행
- 간단한 데이터 점검

### 로컬 parser 모드

PDF 파싱 검증이나 실제 prepare를 돌릴 때는 parser 프로필을 함께 띄운다. parser와 prepare-worker가 같은 profile에 묶여 있어 한 번에 올라온다.

```bash
docker compose --profile parser up -d --build
docker logs -f arxplore-layout-parser
```

### 서버 통합 모드

서버 스택은 수집 자동화와 DB 운영을 담당한다.

```bash
bash scripts/setup-server.sh
docker compose -p arxplore_server -f docker-compose.server.yml ps
```

## 5. 현재 운영 흐름

현재 운영 흐름은 아래와 같다.

1. `arxplore_daily_collect`가 최신 raw를 MongoDB에 저장한다
2. 같은 날짜를 PostgreSQL `prepare_jobs`에 등록한다
3. 로컬 `prepare-worker`가 새 job을 기다린다
4. job을 claim하면 `prepare -> embed`를 수행한다
5. 결과는 PostgreSQL 정제층에 저장된다
6. `arxplore_maintenance`는 별도로 과거 raw 백필과 metadata enrichment를 수행한다
7. `arxplore_langsmith_maintenance`는 매일 03:00에 오래된 LangSmith trace를 정리한다

## 6. 현재 단계의 구현 순서

현재 단계에서는 이미 존재하는 적재 기반 위에서 아래 순서로 병렬 개발한다.

### 1단계: Retrieval 계층 정리

- lexical/vector/hybrid retrieval을 정리한다
- rerank와 section prior를 안정화한다
- 반환 shape를 고정한다

### 2단계: Answer 계층 정리

- retrieval 결과를 answer chain 입력으로 연결한다
- citation과 insufficient context 정책을 고정한다
- answer payload를 정의한다

### 3단계: Translation / Summary 계층 정리

- 한국어 번역 규칙과 상세 요약 구조를 정한다
- answer와 논문 상세 문서가 재사용할 수 있는 prompt 기준을 만든다

### 4단계: 논문 상세 문서 계층 정리

- `PaperDetailDocument` 생성 chain을 안정화한다
- overview / key findings 품질을 개선한다
- 평가 루프를 만든다

### 5단계: UI 소비 계층 정리

- retrieval 결과, answer payload, `PaperDetailDocument`를 React UI에 연결한다
- 논문 목록, 논문 상세, 검색 흐름을 통합한다

이 순서는 완전한 직렬 흐름이 아니라, 상위 계층 간 handoff를 명확히 하기 위한 기준이다.

## 7. 역할별 일상 작업 흐름

### Retrieval · 검색 품질 담당

1. 저장된 chunk와 embedding 상태를 확인한다
2. lexical/vector/hybrid 검색 결과를 비교한다
3. rerank와 score 규칙을 조정한다
4. 실패 사례를 샘플셋으로 문서화한다
5. 역할 2와 역할 5가 쓰는 반환 shape를 점검한다

### RAG 응답 · 근거 제어 담당

1. retrieval 결과 shape를 입력으로 받아 answer chain을 설계한다
2. citation과 evidence 노출 정책을 정리한다
3. 검색 부족 응답과 정상 응답을 분리한다
4. answer payload를 고정한다
5. UI 소비 계층과 필드를 맞춘다

### 한국어 번역 · 상세 요약 프롬프트 담당

1. 논문 단위와 chunk 단위 번역 전략을 비교한다
2. 상세 요약 구조를 설계한다
3. 용어, 문체, 길이 기준을 문서화한다
4. answer 계층과 논문 상세 문서 계층에 재사용 가능한 규칙을 넘긴다

### 논문 상세 · 프롬프트 평가 담당

1. 논문 입력을 기반으로 paper detail chain을 점검한다
2. overview와 key findings 역할을 분리한다
3. 샘플셋 또는 LangSmith로 평가 루프를 만든다
4. `PaperDetailDocument` 품질 기준을 문서화한다

### UI · 문서 소비 계층 담당

1. 논문 목록 화면, 논문 상세 화면, answer 영역, 검색 흐름을 먼저 고정한다
2. retrieval 결과와 answer payload를 화면에 연결한다
3. `PaperDetailDocument` 렌더링을 논문 상세 화면에 맞춘다
4. 빈 상태, 오류 상태, 로딩 상태를 정리한다

## 8. prepare와 적재 상태 확인

확인 대상:

- MongoDB raw 수집 상태
- `prepare_jobs` 상태
- `paper_fulltexts`, `paper_chunks`, `paper_embeddings` 적재 상태
- parser 컨테이너 health
- 로컬 `prepare-worker` 실행 상태

공식 점검 도구:

- `docker compose --profile parser up -d` (상시 worker)
- `docker compose --profile parser exec prepare-worker python3 -m src.pipeline.prepare_worker --mode auto --max-jobs-per-run 1` (1회 실행 점검)
- `python scripts/requeue_failed_prepare_jobs.py --since YYYY-MM-DD` (failed 잡 확인, 기본 dry-run. `--apply`로 재등록)
- PostgreSQL 직접 조회 (예: `SELECT status, count(*) FROM prepare_jobs GROUP BY status;`)

## 9. LangSmith 운영 방식

LangSmith는 공용 프로젝트 기준으로 trace를 축적한다. 현재 주로 보는 stage는 다음과 같다.

- `collect_papers`
- `backfill_collect_papers`
- `prepare_papers`
- `consume_prepare_queue`
- `embed_papers`
- `enrich_papers_metadata`
- `analyze_paper_detail`
- `paper_overview`
- `paper_key_findings`
- `summary`
- `rag_answer`

## 10. 통합 확인 순서

통합 검증은 아래 순서로 수행한다.

1. ingestion 상태 확인
   - raw가 MongoDB에 들어가는지
   - `prepare_jobs`가 생성되는지
2. prepare 상태 확인
   - `paper_fulltexts`와 `paper_chunks`가 늘어나는지
   - parser runtime이 응답하는지
3. embedding 상태 확인
   - `paper_embeddings`가 채워지는지
4. retrieval 결과 shape 확인
   - lexical/vector/hybrid가 공용 shape를 유지하는지
5. answer payload 확인
   - citation, 실패 상태, 응답 구조가 맞는지
6. 논문 상세 문서 생성 확인
   - `PaperDetailDocument` 구조가 안정적인지
7. UI 소비 확인
   - 검색, 답변, 논문 목록, 논문 상세가 같은 화면 흐름에 연결되는지

## 11. 정제층 재적재 원칙

파싱 기준이나 chunk 기준이 크게 바뀌면 MongoDB raw를 유지한 채 PostgreSQL 정제층을 다시 만드는 것이 더 안전하다.

원칙:

1. MongoDB raw는 source of truth다
2. PostgreSQL `papers`, `paper_fulltexts`, `paper_chunks`, `paper_embeddings`는 재생성 가능한 계층이다
3. parser 기준이 크게 달라지면 부분 덮어쓰기보다 재prepare와 재embed가 일관성 면에서 낫다

## 12. 현재 단계의 완료 기준

현재 워크플로우 기준으로 다음 상태에 도달하면 상위 제품 계층 구현을 진행할 준비가 된 것으로 본다.

- `daily_collect`와 `maintenance`가 자동으로 동작한다
- `prepare-worker`가 `prepare_jobs`를 소비한다
- parser runtime과 fallback 경로가 안정적이다
- `paper_chunks`와 `paper_embeddings`가 retrieval 가능한 상태로 유지된다
- retrieval 결과 shape와 answer payload shape가 고정된다
- 논문 상세 문서와 UI가 이 공용 계약 위에서 움직인다
