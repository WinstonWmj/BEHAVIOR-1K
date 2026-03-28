export CUDA_VISIBLE_DEVICES=1

LOG_BASE=./logs/instance_eval/turning_on_radio_task_reward/pi05-turnonradio-sft-hastext-demoexpert
TASK=turning_on_radio
POLICY_MODE=${POLICY_MODE:-demo_expert}  # websocket or demo_expert
MODEL_HOST=${MODEL_HOST:-localhost}
MODEL_PORT=${MODEL_PORT:-8007}
DEMO_DATA_DIR=${DEMO_DATA_DIR:-/mnt/public/mjwei/download_models/2025-challenge-demos}
DEMO_EPISODE_INDEX=${DEMO_EPISODE_INDEX:-10}
DEMO_START_FRAME=${DEMO_START_FRAME:-0}

echo "=========================================="
echo "Evaluating task: ${TASK} (instance mode)"
echo "Policy mode: ${POLICY_MODE}"
echo "Reward mode: task"
echo "=========================================="

COMMON_ARGS=(
  log_path=${LOG_BASE}/${TASK}
  task.name="${TASK}"
  instance_reward_mode=task
  env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper
)

if [ "${POLICY_MODE}" = "demo_expert" ]; then
  echo "Demo expert parquet root: ${DEMO_DATA_DIR}"
  echo "Demo expert episode index: ${DEMO_EPISODE_INDEX}"
  python OmniGibson/omnigibson/learning/eval.py \
    "${COMMON_ARGS[@]}" \
    policy=demo_expert \
    model.host=${MODEL_HOST} \
    model.port=${MODEL_PORT} \
    eval_on_train_instances=true \
    demo_expert_data_dir="${DEMO_DATA_DIR}" \
    demo_expert_episode_index=${DEMO_EPISODE_INDEX} \
    demo_expert_start_frame=${DEMO_START_FRAME}
else
  EVAL_INSTANCE_IDS="[0]"
  echo "Eval split instance ids: ${EVAL_INSTANCE_IDS}"
  python OmniGibson/omnigibson/learning/eval.py \
    "${COMMON_ARGS[@]}" \
    policy=websocket \
    model.host=${MODEL_HOST} \
    model.port=${MODEL_PORT} \
    eval_on_train_instances=false \
    eval_instance_ids="${EVAL_INSTANCE_IDS}"
fi
