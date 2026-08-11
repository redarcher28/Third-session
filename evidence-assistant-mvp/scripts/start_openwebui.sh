#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BACKEND_ENV_PREFIX="${EVIDENCE_BACKEND_ENV:-/Users/quentincrane/conda_envs/evidence_mvp}"
OPENWEBUI_ENV_PREFIX="${OPENWEBUI_ENV:-/Users/quentincrane/conda_envs/open_webui}"
OPENWEBUI_DATA_DIR="${OPENWEBUI_DATA_DIR:-/Users/quentincrane/conda_envs/open_webui_data}"
BACKEND_HOST="${EVIDENCE_BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${EVIDENCE_BACKEND_PORT:-8000}"
OPENWEBUI_HOST="${OPENWEBUI_HOST:-127.0.0.1}"
OPENWEBUI_PORT="${OPENWEBUI_PORT:-8080}"

if [[ ! -x "$BACKEND_ENV_PREFIX/bin/uvicorn" ]]; then
  echo "缺少后端 Conda 环境或 uvicorn：$BACKEND_ENV_PREFIX" >&2
  exit 1
fi
if [[ ! -x "$OPENWEBUI_ENV_PREFIX/bin/open-webui" ]]; then
  echo "缺少 OpenWebUI Conda 环境：$OPENWEBUI_ENV_PREFIX" >&2
  echo "请先安装：conda run -p $OPENWEBUI_ENV_PREFIX python -m pip install open-webui" >&2
  exit 1
fi

mkdir -p "$OPENWEBUI_DATA_DIR"
SECRET_FILE="$OPENWEBUI_DATA_DIR/.webui_secret_key"
if [[ -s "$SECRET_FILE" ]]; then
  WEBUI_SECRET_KEY="$(<"$SECRET_FILE")"
else
  umask 077
  WEBUI_SECRET_KEY="$("$OPENWEBUI_ENV_PREFIX/bin/python" -c 'import secrets; print(secrets.token_urlsafe(48))')"
  printf '%s' "$WEBUI_SECRET_KEY" > "$SECRET_FILE"
fi

cleanup() {
  trap - INT TERM EXIT
  if [[ -n "${OPENWEBUI_PID:-}" ]] && kill -0 "$OPENWEBUI_PID" 2>/dev/null; then
    kill "$OPENWEBUI_PID" 2>/dev/null || true
  fi
  if [[ -n "${BACKEND_PID:-}" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup INT TERM EXIT

cd "$PROJECT_ROOT"

echo "启动证据助手后端：http://$BACKEND_HOST:$BACKEND_PORT"
conda run --no-capture-output -p "$BACKEND_ENV_PREFIX" \
  uvicorn src.app.api:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
BACKEND_PID=$!

for _ in {1..30}; do
  if curl -fsS "http://$BACKEND_HOST:$BACKEND_PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "启动 OpenWebUI 主前端：http://$OPENWEBUI_HOST:$OPENWEBUI_PORT"
OPENAI_API_BASE_URL="http://$BACKEND_HOST:$BACKEND_PORT/v1" \
OPENAI_API_BASE_URLS="http://$BACKEND_HOST:$BACKEND_PORT/v1" \
OPENAI_API_KEY="evidence-local" \
OPENAI_API_KEYS="evidence-local" \
DATA_DIR="$OPENWEBUI_DATA_DIR" \
WEBUI_SECRET_KEY="$WEBUI_SECRET_KEY" \
RAG_EMBEDDING_ENGINE="openai" \
RAG_EMBEDDING_MODEL="evidence-embedding" \
BYPASS_EMBEDDING_AND_RETRIEVAL="true" \
conda run --no-capture-output -p "$OPENWEBUI_ENV_PREFIX" \
  open-webui serve --host "$OPENWEBUI_HOST" --port "$OPENWEBUI_PORT" &
OPENWEBUI_PID=$!

echo "主界面：http://$OPENWEBUI_HOST:$OPENWEBUI_PORT/"
echo "备用界面：http://$BACKEND_HOST:$BACKEND_PORT/fallback"
wait "$BACKEND_PID" "$OPENWEBUI_PID"
