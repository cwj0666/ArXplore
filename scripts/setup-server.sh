#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
POSTGRES_VOLUME="arxplore_server_arxplore_postgres_data"
COMPOSE=(docker compose -f compose.server.yaml)

cd "${REPO_ROOT}"

if [[ ! -f .env ]]; then
  echo ".env 파일이 없습니다. 루트에 .env를 만든 뒤 다시 실행하세요."
  exit 1
fi

env_value() {
  local key="$1" default="${2:-}" value=""
  if [[ -n "${!key:-}" ]]; then
    printf '%s' "${!key}"
    return
  fi
  if [[ -f .env ]]; then
    value="$(grep -E "^${key}=" .env | tail -n 1 | cut -d= -f2- || true)"
    value="${value%$'\r'}"
    value="${value#\"}"; value="${value%\"}"
    value="${value#\'}"; value="${value%\'}"
  fi
  printf '%s' "${value:-${default}}"
}

for required in TAILSCALE_SERVER_IP POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DB; do
  if [[ -z "$(env_value "${required}")" ]]; then
    echo "${required}가 설정되지 않았습니다. .env에 값을 추가하세요."
    exit 1
  fi
done

if [[ "$(env_value POSTGRES_PASSWORD)" == change-me* ]]; then
  echo "POSTGRES_PASSWORD가 예시 값입니다. .env에 실제 비밀번호를 넣으세요."
  exit 1
fi

LEGACY_CONTAINERS="$(docker ps -a --format '{{.Names}}' | grep -E '^arxplore-(postgres|airflow-.+)$' || true)"
if [[ -n "${LEGACY_CONTAINERS}" ]]; then
  echo "이전 서버 스택(PostgreSQL + Airflow) 컨테이너가 남아 있습니다:"
  while read -r name; do echo "  ${name}"; done <<<"${LEGACY_CONTAINERS}"
  echo "같은 데이터 볼륨을 쓰므로 먼저 내리세요:"
  echo "  docker compose -f compose.server.yaml down --remove-orphans"
  exit 1
fi

SERVER_IP="$(env_value TAILSCALE_SERVER_IP)"

docker volume inspect "${POSTGRES_VOLUME}" >/dev/null 2>&1 || docker volume create "${POSTGRES_VOLUME}" >/dev/null

"${COMPOSE[@]}" build scheduler
"${COMPOSE[@]}" up -d --wait postgres

echo
echo "[server] 마이그레이션"
"${COMPOSE[@]}" run --rm -T scheduler migrate

"${COMPOSE[@]}" up -d scheduler

echo
echo "[server] 컨테이너 상태"
"${COMPOSE[@]}" ps

echo
echo "[server] 접속 정보"
echo "PostgreSQL: ${SERVER_IP}:$(env_value SERVER_POSTGRES_PORT 15432)"
echo
echo "스케줄러 로그:  docker compose -f compose.server.yaml logs -f scheduler"
echo "실행 이력:      docker compose -f compose.server.yaml exec scheduler arxplore scheduler history"
echo "수동 실행:      docker compose -f compose.server.yaml run --rm scheduler run collect --target-date YYYY-MM-DD"
