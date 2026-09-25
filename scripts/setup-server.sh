#!/usr/bin/env bash
set -euo pipefail

PROJECT_NAME="arxplore_server"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

for required in TAILSCALE_SERVER_IP AIRFLOW_FERNET_KEY AIRFLOW_ADMIN_USER; do
  if [[ -z "$(env_value "${required}")" ]]; then
    echo "${required}가 설정되지 않았습니다. .env에 값을 추가하세요."
    exit 1
  fi
done

SERVER_IP="$(env_value TAILSCALE_SERVER_IP)"

docker volume inspect arxplore_server_arxplore_postgres_data >/dev/null 2>&1 || docker volume create arxplore_server_arxplore_postgres_data >/dev/null
docker volume inspect arxplore_server_arxplore_mongo_data >/dev/null 2>&1 || docker volume create arxplore_server_arxplore_mongo_data >/dev/null
docker volume inspect arxplore_server_arxplore_airflow_logs >/dev/null 2>&1 || docker volume create arxplore_server_arxplore_airflow_logs >/dev/null

docker build -t arxplore-airflow -f docker/airflow/Dockerfile .
docker compose -p "${PROJECT_NAME}" -f docker-compose.server.yml up -d --build

echo
echo "[server] 컨테이너 상태"
docker compose -p "${PROJECT_NAME}" -f docker-compose.server.yml ps

echo
echo "[server] 접속 정보"
echo "Airflow: http://${SERVER_IP}:$(env_value SERVER_AIRFLOW_PORT 18080)"
echo "MongoDB: ${SERVER_IP}:$(env_value SERVER_MONGO_PORT 17017)"
echo "PostgreSQL: ${SERVER_IP}:$(env_value SERVER_POSTGRES_PORT 15432)"
