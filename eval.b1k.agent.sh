export CUDA_VISIBLE_DEVICES=3
LOG_DIR="/mnt/public/mjwei/repo/behavior-1k-solution/BEHAVIOR-1K/logs/runs/pi05-b1kpt50-cs32/$(date +'%Y%m%d-%H:%M:%S')-eval-freeze_pies"
mkdir -p ${LOG_DIR}

# ── 多 policy server 启动说明 ──────────────────────────────────
# 终端 A (默认 policy, port 8000):
#   cd /home/dell/mjwei/openpi-comet && bash eval.policy.sh
#
# 终端 B (第二个 policy, port 8001):
#   uv run scripts/serve_b1k.py --task_name="freeze_pies" --port=8001 \
#     policy:checkpoint --policy.config=<另一个config> --policy.dir=<另一个ckpt路径>
# ────────────────────────────────────────────────────────────────

CMD="python OmniGibson/omnigibson/learning/eval_agent.py \
  log_path=${LOG_DIR} \
  policy=websocket \
  task.name=\"freeze_pies\" \
  model.host=localhost \
  model.port=8001 \
  env_wrapper._target_=omnigibson.learning.wrappers.RGBWrapper \
  +policy_endpoints.openvla.host=localhost \
  +policy_endpoints.openvla.port=8002"

echo ${CMD} > ${LOG_DIR}/eval_behavior.log

${CMD} 2>&1 | tee -a ${LOG_DIR}/eval_behavior.log
