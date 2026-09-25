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

# ── forward 서브커맨드 ──────────────────────────────────────────────────────
if [[ "${1:-}" == "forward" ]]; then
  SERVER_IP="$(env_value TAILSCALE_SERVER_IP)"
  : "${SERVER_IP:?Set TAILSCALE_SERVER_IP in .env (server Tailscale IP)}"
  AIRFLOW_PORT="$(env_value SERVER_AIRFLOW_PORT 18080)"
  POSTGRES_PORT="$(env_value SERVER_POSTGRES_PORT 15432)"
  MONGO_PORT="$(env_value SERVER_MONGO_PORT 17017)"
  ACTION="${2:-start}"

  get_pids() {
    pgrep -f "ssh -N .*${AIRFLOW_PORT}:${SERVER_IP}.*${POSTGRES_PORT}:${SERVER_IP}.*${MONGO_PORT}:${SERVER_IP}" || true
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

  for port in "${AIRFLOW_PORT}" "${POSTGRES_PORT}" "${MONGO_PORT}"; do
    if ! python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',${port})); s.close()" 2>/dev/null; then
      echo "[forward] 포트 ${port}가 이미 사용 중입니다."
      exit 1
    fi
  done

  echo "[forward] ${SERVER_IP} 으로 포트 포워딩"
  echo "  Airflow:    localhost:${AIRFLOW_PORT}"
  echo "  PostgreSQL: localhost:${POSTGRES_PORT}"
  echo "  MongoDB:    localhost:${MONGO_PORT}"

  ssh -N \
    -L "${AIRFLOW_PORT}:${SERVER_IP}:${AIRFLOW_PORT}" \
    -L "${POSTGRES_PORT}:${SERVER_IP}:${POSTGRES_PORT}" \
    -L "${MONGO_PORT}:${SERVER_IP}:${MONGO_PORT}" \
    localhost &

  SSH_PID=$!
  echo "[forward] 활성화 (PID: ${SSH_PID})"
  trap "kill ${SSH_PID} 2>/dev/null; echo '[forward] 종료'; exit 0" INT TERM
  wait "${SSH_PID}"
  exit 0
fi

# ── 기본: 컨테이너 실행 ─────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
  echo ".env 파일이 없습니다."
  exit 1
fi

DJANGO_SECRET_VALUE="$(env_value DJANGO_SECRET_KEY)"
if [[ -z "${DJANGO_SECRET_VALUE}" || "${DJANGO_SECRET_VALUE}" == change-me* ]]; then
  echo "DJANGO_SECRET_KEY가 설정되지 않았습니다. .env에 실제 secret key를 추가하세요."
  exit 1
fi

PROD_POSTGRES_VALUE="$(env_value PROD_POSTGRES_HOST)"
if [[ -z "${PROD_POSTGRES_VALUE}" ]]; then
  echo "PROD_POSTGRES_HOST가 설정되지 않았습니다. .env에 메인 서버 PostgreSQL host를 추가하세요."
  exit 1
fi

docker compose up -d --build

echo
echo "[arxplore] 컨테이너 상태"
docker compose ps

echo
echo "[arxplore] 접속 정보"
echo "Web:      http://localhost:$(env_value PROD_HTTP_PORT 80)   (nginx)"
echo "Vite:     http://localhost:$(env_value FRONTEND_PORT 5173)  (프론트 수정 확인용)"
echo
echo "내리기:              docker compose down"
echo "parser + worker:    docker compose --profile parser up -d --build"
echo "포트 포워딩:         bash scripts/setup.sh forward"
