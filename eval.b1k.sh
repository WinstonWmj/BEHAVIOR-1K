export CUDA_VISIBLE_DEVICES=0

TASK=set_up_a_coffee_station_in_your_kitchen # turning_on_radio hanging_pictures make_microwave_popcorn picking_up_trash set_up_a_coffee_station_in_your_kitchen
EVAL_LEVEL=${EVAL_LEVEL:-instance}  # instance or subtask
POLICY_MODE=${POLICY_MODE:-demo_expert}  # websocket or demo_expert
MODEL_HOST=${MODEL_HOST:-localhost}
MODEL_PORT=${MODEL_PORT:-8007}
B1K_DEMO_ROOT=${B1K_DEMO_ROOT:-/home/dell/mjwei/download_models/2025-challenge-demos}
RUN_EPISODE_IDX=${RUN_EPISODE_IDX:-100010}  # 10 340060 400070 10020 100010
SUBTASK_INDEX=${SUBTASK_INDEX:-0}
SUBTASK_END_INDEX=${SUBTASK_END_INDEX:-15}  # 3 3 7 11 15
MAX_STEPS=${MAX_STEPS:-}
WAITING_FOR_STAGE_COMPLETION=${WAITING_FOR_STAGE_COMPLETION:-true}
KEEP_RUNNING_AFTER_SUCCESS=${KEEP_RUNNING_AFTER_SUCCESS:-true}
HEADLESS=${HEADLESS:-true}
INSTANCE_IDS=${INSTANCE_IDS:-[0]}  # only used in instance mode; 0 4 6 1 0
LOG_BASE=./logs/${EVAL_LEVEL}_eval/set_up_a_coffee_station_in_your_kitchen-task_reward/demoexpert-set_up_a_coffee_station_in_your_kitchen-hastext-debug-10-withreward-1
PYTHON_BIN=${PYTHON_BIN:-python}

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
if [ -n "${SUBTASK_INDEX}" ]; then
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
  env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper
  demo_data_dir="${B1K_DEMO_ROOT}"
  run_episode_idx="${RUN_EPISODE_IDX}"
)
if [ "${EVAL_LEVEL}" = "subtask" ]; then
  echo "Subtask episode indices: ${RUN_EPISODE_IDX}"
  
  if [ -n "${SUBTASK_INDEX}" ]; then
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


if [ "${POLICY_MODE}" = "demo_expert" ]; then
  echo "Demo expert parquet root: ${B1K_DEMO_ROOT}"
  echo "Demo expert episode index: ${RUN_EPISODE_IDX}"
  "${PYTHON_BIN}" OmniGibson/omnigibson/learning/eval.py \
    "${COMMON_ARGS[@]}" \
    policy=demo_expert \
    model.host=${MODEL_HOST} \
    model.port=${MODEL_PORT} \
    eval_on_train_instances=true \
    demo_expert_data_dir="${B1K_DEMO_ROOT}" \
    eval_instance_ids="${INSTANCE_IDS}"
else
  "${PYTHON_BIN}" OmniGibson/omnigibson/learning/eval.py \
    "${COMMON_ARGS[@]}" \
    policy=websocket \
    model.host=${MODEL_HOST} \
    model.port=${MODEL_PORT} \
    eval_on_train_instances=false \
    eval_instance_ids="${INSTANCE_IDS}" \
    max_steps=${MAX_STEPS}
fi
