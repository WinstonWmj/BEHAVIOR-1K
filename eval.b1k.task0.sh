export CUDA_VISIBLE_DEVICES=0

TASK=turning_on_radio # turning_on_radio hanging_pictures make_microwave_popcorn picking_up_trash set_up_a_coffee_station_in_your_kitchen
EVAL_LEVEL=${EVAL_LEVEL:-subtask}  # instance or subtask
POLICY_MODE=${POLICY_MODE:-websocket}  # websocket or demo_expert
MODEL_HOST=${MODEL_HOST:-localhost}
MODEL_PORT=${MODEL_PORT:-8007}
B1K_DEMO_ROOT=${B1K_DEMO_ROOT:-/home/dell/mjwei/download_models/2025-challenge-demos}
RUN_EPISODE_IDX=${RUN_EPISODE_IDX:-10}  # 10 340060 400070 10020 100010
# RUN_EPISODE_IDXS=${RUN_EPISODE_IDXS:-'[10020]'}  # Example: [10,100010]
RUN_EPISODE_IDXS=${RUN_EPISODE_IDXS:-'[10,20,30,60,70,100,110,140,160,170,180,190]'}  # Example: [10,100010]
SUBTASK_INDEX=${SUBTASK_INDEX:-0}
SUBTASK_END_INDEX=${SUBTASK_END_INDEX:-15}  # 3 3 7 11 15
SUBTASK_SKILL=${SUBTASK_SKILL:-'place on'}  # Example: press_radio / pickup_from_support / "press"
MAX_STEPS=${MAX_STEPS:-}
WRITE_VIDEO=${WRITE_VIDEO:-true}
WAITING_FOR_STAGE_COMPLETION=${WAITING_FOR_STAGE_COMPLETION:-true}
KEEP_RUNNING_AFTER_SUCCESS=${KEEP_RUNNING_AFTER_SUCCESS:-false}
HEADLESS=${HEADLESS:-true}
INSTANCE_IDS=${INSTANCE_IDS:-[0]}  # only used in instance mode; 0 4 6 1 0
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-${SCRIPT_DIR}}
OMNIGIBSON_ROOT=${OMNIGIBSON_ROOT:-${REPO_ROOT}/OmniGibson}
LOG_BASE=${LOG_BASE:-${REPO_ROOT}/logs/${EVAL_LEVEL}_eval/${TASK}-placeon-task_reward/warmup-${TASK}-placeon-warmupcontinue-371}
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH="${REPO_ROOT}/bddl:${OMNIGIBSON_ROOT}:${PYTHONPATH:-}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "Neither '${PYTHON_BIN}' nor 'python3' was found in PATH."
    exit 1
  fi
fi

echo "=========================================="
echo "Evaluating task: ${TASK} (${EVAL_LEVEL} mode)"
echo "Policy mode: ${POLICY_MODE}"
echo "Reward mode: task"
echo "Demo root: ${B1K_DEMO_ROOT}"
echo "headless: ${HEADLESS}"
echo "waiting_for_stage_completion: ${WAITING_FOR_STAGE_COMPLETION}"
echo "keep_running_after_success: ${KEEP_RUNNING_AFTER_SUCCESS}"
if [ -n "${SUBTASK_SKILL}" ]; then
  echo "Subtask skill filter: ${SUBTASK_SKILL}"
fi
if [ -z "${SUBTASK_SKILL}" ] && [ -n "${SUBTASK_INDEX}" ]; then
  if [ -n "${SUBTASK_END_INDEX}" ]; then
    echo "Subtask range: ${SUBTASK_INDEX} -> ${SUBTASK_END_INDEX}"
  else
    echo "Subtask index: ${SUBTASK_INDEX}"
  fi
fi
echo "=========================================="

COMMON_ARGS=(
  log_path=${LOG_BASE}/${TASK}
  task.name="${TASK}"
  eval_level="${EVAL_LEVEL}"
  headless=${HEADLESS}
  instance_reward_mode=task
  waiting_for_stage_completion=${WAITING_FOR_STAGE_COMPLETION}
  keep_running_after_success=${KEEP_RUNNING_AFTER_SUCCESS}
  env_wrapper._target_=omnigibson.learning.wrappers.rgb_wrapper.RGBWrapper
  demo_data_dir="${B1K_DEMO_ROOT}"
  run_episode_idx="${RUN_EPISODE_IDX}"
  write_video="${WRITE_VIDEO}"
)
if [ "${EVAL_LEVEL}" = "subtask" ]; then
  echo "Subtask episode indices: ${RUN_EPISODE_IDXS}"
  COMMON_ARGS+=(
    run_episode_indices=${RUN_EPISODE_IDXS}
  )

  # When a skill filter is provided, the evaluator will test every matching
  # occurrence inside each selected episode instead of requiring a manual subtask index.
  if [ -n "${SUBTASK_SKILL}" ]; then
    COMMON_ARGS+=(
      subtask_skill="${SUBTASK_SKILL}"
    )
  elif [ -n "${SUBTASK_INDEX}" ]; then
    COMMON_ARGS+=(
      subtask_index=${SUBTASK_INDEX}
    )
    if [ -n "${SUBTASK_END_INDEX}" ]; then
      COMMON_ARGS+=(
        subtask_end_index=${SUBTASK_END_INDEX}
      )
    fi
  fi
else
  echo "Eval instance ids: ${INSTANCE_IDS}"
fi


if [ "${EVAL_LEVEL}" = "subtask" ]; then
  echo "Demo expert parquet root: ${B1K_DEMO_ROOT}"
  echo "Demo expert episode index: ${RUN_EPISODE_IDX}"
  (
    cd "${OMNIGIBSON_ROOT}"
    "${PYTHON_BIN}" omnigibson/learning/eval.py \
      "${COMMON_ARGS[@]}" \
      policy=demo_expert \
      model.host=${MODEL_HOST} \
      model.port=${MODEL_PORT} \
      eval_on_train_instances=true \
      demo_expert_data_dir="${B1K_DEMO_ROOT}" \
      demo_expert_episode_index=${RUN_EPISODE_IDX} \
      eval_instance_ids="${INSTANCE_IDS}"
  )
else
  (
    cd "${OMNIGIBSON_ROOT}"
    "${PYTHON_BIN}" omnigibson/learning/eval.py \
      "${COMMON_ARGS[@]}" \
      policy=websocket \
      model.host=${MODEL_HOST} \
      model.port=${MODEL_PORT} \
      eval_on_train_instances=false \
      eval_instance_ids="${INSTANCE_IDS}" \
      max_steps=${MAX_STEPS}
  )
fi
