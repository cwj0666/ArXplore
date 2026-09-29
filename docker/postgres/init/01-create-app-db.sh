#!/usr/bin/env bash
set -euo pipefail

if [ -z "${APP_POSTGRES_DB:-}" ]; then
  echo "APP_POSTGRES_DB is not set; skipping extra database creation."
  exit 0
fi

psql -v ON_ERROR_STOP=1 -v app_db="${APP_POSTGRES_DB}" --username "${POSTGRES_USER}" --dbname "${POSTGRES_DB}" <<'EOSQL'
SELECT format('CREATE DATABASE %I', :'app_db')
WHERE NOT EXISTS (
  SELECT FROM pg_database WHERE datname = :'app_db'
)\gexec
EOSQL
