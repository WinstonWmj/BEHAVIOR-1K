export CUDA_VISIBLE_DEVICES=7


TASKS=(
    "turning_on_radio"
    "picking_up_trash"
    "putting_away_Halloween_decorations"
)

LOG_BASE=./logs/pi_behavior_b1k_rft_1_5k/openpi-comet-rft-ft1.5k-50tasks-6000

for TASK in "${TASKS[@]}"; do
  echo "=========================================="
  echo "Evaluating task: $TASK"
  echo "=========================================="
  python OmniGibson/omnigibson/learning/eval.py \
    log_path=${LOG_BASE}/${TASK} \
    policy=websocket \
    task.name="${TASK}" \
    model.host=localhost \
    model.port=8001 \
    eval_instance_ids="[0,1,2,3,4,5,6,7,8,9]"
done
