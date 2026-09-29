#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

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

if [[ "${1:-}" == "forward" ]]; then
  SERVER_IP="$(env_value TAILSCALE_SERVER_IP)"
  : "${SERVER_IP:?Set TAILSCALE_SERVER_IP in .env (server Tailscale IP)}"
  POSTGRES_PORT="$(env_value SERVER_POSTGRES_PORT 15432)"
  ACTION="${2:-start}"

  get_pids() {
    pgrep -f "ssh -N -L ${POSTGRES_PORT}:${SERVER_IP}:${POSTGRES_PORT} localhost" || true
  }

  is_running() { [[ -n "$(get_pids)" ]]; }

  stop_forward() {
    if is_running; then
      get_pids | xargs -r kill 2>/dev/null || true
      echo "[forward] 종료"
    else
      echo "[forward] 실행 중인 포워딩 없음"
    fi
  }

  case "${ACTION}" in
    stop)    stop_forward; exit 0 ;;
    status)
      if is_running; then echo "[forward] 실행 중 (PID: $(get_pids | tr '\n' ' '))"; else echo "[forward] 중지됨"; fi
      exit 0 ;;
    restart) stop_forward ;;
    start)   ;;
    *)
      echo "사용법: bash scripts/setup.sh forward [start|stop|status|restart]"
      exit 1 ;;
  esac

  if is_running; then
    echo "[forward] 이미 실행 중 — 재시작하려면: bash scripts/setup.sh forward restart"
    exit 0
  fi

  if ! python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',${POSTGRES_PORT})); s.close()" 2>/dev/null; then
    echo "[forward] 포트 ${POSTGRES_PORT}가 이미 사용 중입니다."
    exit 1
  fi

  echo "[forward] ${SERVER_IP} 으로 포트 포워딩"
  echo "  PostgreSQL: localhost:${POSTGRES_PORT}"

  ssh -N -L "${POSTGRES_PORT}:${SERVER_IP}:${POSTGRES_PORT}" localhost &

  SSH_PID=$!
  echo "[forward] 활성화 (PID: ${SSH_PID})"
  trap 'kill "${SSH_PID}" 2>/dev/null; echo "[forward] 종료"; exit 0' INT TERM
  wait "${SSH_PID}"
  exit 0
fi

if [[ -n "${1:-}" ]]; then
  echo "사용법: bash scripts/setup.sh [forward [start|stop|status|restart]]"
  exit 1
fi

if [[ ! -f .env ]]; then
  echo ".env 파일이 없습니다. cp .env.example .env 후 값을 채우세요."
  exit 1
fi

if [[ -z "$(env_value PROD_POSTGRES_HOST)" ]]; then
  echo "PROD_POSTGRES_HOST가 설정되지 않았습니다. .env에 컨테이너가 접속할 PostgreSQL host를 추가하세요."
  exit 1
fi

docker compose build

echo
echo "[arxplore] 스키마 확인"
if ! DB_STATUS="$(docker compose run --rm -T api db-status)"; then
  echo "PostgreSQL에 접속하지 못했습니다. PROD_POSTGRES_HOST와 POSTGRES_* 값을 확인하세요."
  exit 1
fi
printf '%s\n' "${DB_STATUS}"
if ! grep -qx 'ready: yes' <<<"${DB_STATUS}"; then
  echo
  echo "스키마가 준비되지 않았습니다. 마이그레이션을 적용한 뒤 다시 실행하세요:"
  echo "  docker compose run --rm api migrate"
  exit 1
fi

docker compose up -d --wait --wait-timeout 180

echo
echo "[arxplore] 컨테이너 상태"
docker compose ps

echo
echo "[arxplore] 접속 정보"
echo "Web:      http://localhost:$(env_value PROD_HTTP_PORT 80)   (nginx)"
echo "Vite:     docker compose --profile dev up -d vite 후 http://127.0.0.1:$(env_value FRONTEND_PORT 5173)"
echo
echo "내리기:              docker compose down"
echo "마이그레이션:        docker compose run --rm api migrate"
echo "parser + worker:    docker compose --profile parser up -d --build"
echo "포트 포워딩:         bash scripts/setup.sh forward"
