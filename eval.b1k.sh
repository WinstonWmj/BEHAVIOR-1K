export CUDA_VISIBLE_DEVICES=3

LOG_BASE=./logs/skill-comet/pi05_b1k-moveto-lr2.5e-step20k-200/10000-skill-simple
DEMO_DATA_DIR=/mnt/public/mjwei/download_models/2025-challenge-demos
TASKS_JSONL="${DEMO_DATA_DIR}/meta/tasks.jsonl"

# 用法:
#   ./eval.b1k.sh 13 0 7
#   ./eval.b1k.sh 3 4 5 6 7 8 9 10 11 12 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
# 若不传参，则默认评测 task_index=13 (loading_the_car)
if [ "$#" -gt 0 ]; then
  TASK_IDS=("$@")
else
  TASK_IDS=(13)
fi

TASKS=()
for TASK_ID in "${TASK_IDS[@]}"; do
  TASK_NAME=$(python - "$TASKS_JSONL" "$TASK_ID" <<'PY'
import json
import sys

jsonl_path = sys.argv[1]
target_id = int(sys.argv[2])

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if item.get("task_index") == target_id:
            print(item.get("task_name", ""))
            break
PY
)

  if [ -z "$TASK_NAME" ]; then
    echo "Error: task_index=${TASK_ID} not found in ${TASKS_JSONL}"
    exit 1
  fi

  TASKS+=("${TASK_NAME}")
done

for TASK in "${TASKS[@]}"; do
  echo "=========================================="
  echo "Evaluating task: $TASK (subtask mode)"
  echo "=========================================="
  python OmniGibson/omnigibson/learning/eval.py \
    log_path=${LOG_BASE}/${TASK} \
    policy=websocket \
    task.name="${TASK}" \
    model.host=localhost \
    model.port=8007 \
    demo_data_dir="${DEMO_DATA_DIR}" \
    subtask_skill_filter='["move to"]' \
    subtask_max_steps_multiplier=3 \
    eval_instance_ids="[0,1,2,3,4]" \
    subtask_eval_mode=simple \
    subtask_success_distance=1.2 \
    env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper
done

# ---------- Original instance-level eval (commented out) ----------
# for TASK in "${TASKS[@]}"; do
#   python OmniGibson/omnigibson/learning/eval.py \
#     log_path=${LOG_BASE}/${TASK} \
#     policy=websocket \
#     task.name="${TASK}" \
#     model.host=localhost \
#     model.port=8007 \
#     use_parallel_evaluator=false \
#     eval_instance_ids="[0,1,2,3,4,5,6,7,8,9]" \
#     env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper
# done
