# "turning_on_radio", "picking_up_trash", "putting_away_Halloween_decorations"
export CUDA_VISIBLE_DEVICES=7

python OmniGibson/omnigibson/learning/eval.py \
  log_path=./eval_logs \
  policy=websocket \
  task.name="turning_on_radio" \
  model.host=localhost \
  eval_instance_ids="[0,1,2,3,4,5,6,7,8,9]"