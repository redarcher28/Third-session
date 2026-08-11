#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BACKEND_ENV_SPEC="${EVIDENCE_BACKEND_ENV:-${EVIDENCE_BACKEND_ENV_NAME:-evidence_mvp}}"
OPENWEBUI_ENV_SPEC="${OPENWEBUI_ENV:-${OPENWEBUI_ENV_NAME:-open_webui}}"
if [[ -n "${OPENWEBUI_DATA_DIR:-}" ]]; then
  : "${OPENWEBUI_DATA_DIR}"
elif [[ -d "${HOME}/conda_envs/open_webui_data" ]]; then
  # 兼容本项目此前在 macOS 上使用的外置数据目录；新电脑走下面的系统默认目录。
  OPENWEBUI_DATA_DIR="${HOME}/conda_envs/open_webui_data"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  OPENWEBUI_DATA_DIR="${HOME}/Library/Application Support/evidence-assistant-mvp/openwebui"
else
  OPENWEBUI_DATA_DIR="${XDG_DATA_HOME:-${HOME}/.local/share}/evidence-assistant-mvp/openwebui"
fi
BACKEND_HOST="${EVIDENCE_BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${EVIDENCE_BACKEND_PORT:-8000}"
OPENWEBUI_HOST="${OPENWEBUI_HOST:-127.0.0.1}"
OPENWEBUI_PORT="${OPENWEBUI_PORT:-8080}"
OPENWEBUI_NAME="${OPENWEBUI_NAME:-证据台}"
EVIDENCE_SETTINGS_URL="${EVIDENCE_SETTINGS_URL:-http://$BACKEND_HOST:$BACKEND_PORT/settings}"

if [[ "$BACKEND_ENV_SPEC" == /* || "$BACKEND_ENV_SPEC" == ./* || "$BACKEND_ENV_SPEC" == ../* ]]; then
  BACKEND_CONDA_ARGS=(-p "$BACKEND_ENV_SPEC")
else
  BACKEND_CONDA_ARGS=(-n "$BACKEND_ENV_SPEC")
fi
if [[ "$OPENWEBUI_ENV_SPEC" == /* || "$OPENWEBUI_ENV_SPEC" == ./* || "$OPENWEBUI_ENV_SPEC" == ../* ]]; then
  OPENWEBUI_CONDA_ARGS=(-p "$OPENWEBUI_ENV_SPEC")
else
  OPENWEBUI_CONDA_ARGS=(-n "$OPENWEBUI_ENV_SPEC")
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "找不到 conda；请先安装 Miniconda/Anaconda 或 Miniforge。" >&2
  exit 1
fi
if ! conda run --no-capture-output "${BACKEND_CONDA_ARGS[@]}" \
    python -c 'import uvicorn' >/dev/null 2>&1; then
  echo "缺少后端 Conda 环境或 uvicorn：$BACKEND_ENV_SPEC" >&2
  echo "请先创建环境并安装 requirements.txt。" >&2
  exit 1
fi
if ! conda run --no-capture-output "${OPENWEBUI_CONDA_ARGS[@]}" \
    python -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("open_webui") else 1)' >/dev/null 2>&1; then
  echo "缺少 OpenWebUI Conda 环境或 open-webui：$OPENWEBUI_ENV_SPEC" >&2
  echo "请先安装：conda run ${OPENWEBUI_CONDA_ARGS[*]} python -m pip install open-webui" >&2
  exit 1
fi

mkdir -p "$OPENWEBUI_DATA_DIR"
SECRET_FILE="$OPENWEBUI_DATA_DIR/.webui_secret_key"
if [[ -s "$SECRET_FILE" ]]; then
  WEBUI_SECRET_KEY="$(<"$SECRET_FILE")"
else
  umask 077
  WEBUI_SECRET_KEY="$(conda run --no-capture-output "${OPENWEBUI_CONDA_ARGS[@]}" \
    python -c 'import secrets; print(secrets.token_urlsafe(48))')"
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
conda run --no-capture-output "${BACKEND_CONDA_ARGS[@]}" \
  uvicorn src.app.api:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" &
BACKEND_PID=$!

for _ in {1..30}; do
  if curl -fsS "http://$BACKEND_HOST:$BACKEND_PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "启动 OpenWebUI 主前端：http://$OPENWEBUI_HOST:$OPENWEBUI_PORT"
export WEBUI_NAME="$OPENWEBUI_NAME"
export DEFAULT_LOCALE="${DEFAULT_LOCALE:-zh-CN}"
export ENABLE_EVALUATION_ARENA_MODELS="false"
echo "安装 OpenWebUI 证据模型设置桥接"
conda run --no-capture-output "${OPENWEBUI_CONDA_ARGS[@]}" \
  python "$PROJECT_ROOT/scripts/install_openwebui_bridge.py" \
    --backend-url "http://$BACKEND_HOST:$BACKEND_PORT"
OPENAI_API_BASE_URL="http://$BACKEND_HOST:$BACKEND_PORT/v1" \
OPENAI_API_BASE_URLS="http://$BACKEND_HOST:$BACKEND_PORT/v1" \
OPENAI_API_KEY="evidence-local" \
OPENAI_API_KEYS="evidence-local" \
DATA_DIR="$OPENWEBUI_DATA_DIR" \
WEBUI_SECRET_KEY="$WEBUI_SECRET_KEY" \
RAG_EMBEDDING_ENGINE="openai" \
RAG_EMBEDDING_MODEL="evidence-embedding" \
BYPASS_EMBEDDING_AND_RETRIEVAL="true" \
conda run --no-capture-output "${OPENWEBUI_CONDA_ARGS[@]}" \
  open-webui serve --host "$OPENWEBUI_HOST" --port "$OPENWEBUI_PORT" &
OPENWEBUI_PID=$!

# Open WebUI 首次启动会先完成数据库迁移；迁移完成后合并证据台的
# banner、示例提问和 Arena 开关，不覆盖现有账号的其他配置。
for _ in {1..30}; do
  if curl -fsS "http://$OPENWEBUI_HOST:$OPENWEBUI_PORT/api/config" >/dev/null 2>&1; then
    conda run --no-capture-output "${OPENWEBUI_CONDA_ARGS[@]}" \
      python "$PROJECT_ROOT/scripts/configure_openwebui.py" \
        --data-dir "$OPENWEBUI_DATA_DIR" --settings-url "$EVIDENCE_SETTINGS_URL" \
      || echo "提示：Open WebUI 证据台配置合并失败，可稍后手动重跑 configure_openwebui.py" >&2
    break
  fi
  sleep 1
done

echo "主界面：http://$OPENWEBUI_HOST:$OPENWEBUI_PORT/"
echo "备用界面：http://$BACKEND_HOST:$BACKEND_PORT/fallback"
wait "$BACKEND_PID" "$OPENWEBUI_PID"
