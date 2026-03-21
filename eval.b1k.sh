export CUDA_VISIBLE_DEVICES=6


TASKS=(
    "loading_the_car"
)

LOG_BASE=./logs/pi_behavior_comet_rft_1_5k/pi05_b1k-loading_the_car-rft_pt50_step50k-task13/19999-reward

# eval_instance_ids="[0,1,2,3,4,5,6,7,8,9]" \

for TASK in "${TASKS[@]}"; do
  echo "=========================================="
  echo "Evaluating task: $TASK"
  echo "=========================================="
  python OmniGibson/omnigibson/learning/eval.py \
    log_path=${LOG_BASE}/${TASK} \
    policy=websocket \
    task.name="${TASK}" \
    model.host=localhost \
    model.port=8007 \
    use_parallel_evaluator=false \
    parallel_evaluator_start_idx=0 \
    parallel_evaluator_end_idx=10 \
    eval_instance_ids="[0,1,2,3,4,5,6,7,8,9]" \
    env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper
done
