REPO_ROOT=${REPO_ROOT:-/home/dell/mjwei/BEHAVIOR-1K}
B1K_DEMO_ROOT=${B1K_DEMO_ROOT:-/home/dell/mjwei/download_models/2025-challenge-demos}
TASK=${TASK:-turning_on_radio}
DEMO_EPISODE_INDEX=${DEMO_EPISODE_INDEX:-10}
SUBTASK_INDEX=${SUBTASK_INDEX:-}
MODEL_HOST=${MODEL_HOST:-0.0.0.0}
MODEL_PORT=${MODEL_PORT:-8007}
PYTHON_BIN=${PYTHON_BIN:-python}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Neither '${PYTHON_BIN}' nor 'python3' was found in PATH."
    exit 1
  fi
fi

SERVER_ARGS=()
if [ -n "${SUBTASK_INDEX}" ]; then
  SERVER_ARGS+=(--subtask-index "${SUBTASK_INDEX}")
fi

echo "=========================================="
echo "Serving demo expert for task: ${TASK}"
echo "Demo root: ${B1K_DEMO_ROOT}"
echo "Episode index: ${DEMO_EPISODE_INDEX}"
if [ -n "${SUBTASK_INDEX}" ]; then
  echo "Subtask index: ${SUBTASK_INDEX}"
fi
echo "Endpoint: ${MODEL_HOST}:${MODEL_PORT}"
echo "=========================================="

cd "${REPO_ROOT}" || exit 1
"${PYTHON_BIN}" OmniGibson/omnigibson/learning/demo_expert_policy_server.py \
  --demo-data-dir "${B1K_DEMO_ROOT}" \
  --task-name "${TASK}" \
  --episode-index "${DEMO_EPISODE_INDEX}" \
  "${SERVER_ARGS[@]}" \
  --host "${MODEL_HOST}" \
  --port "${MODEL_PORT}"
