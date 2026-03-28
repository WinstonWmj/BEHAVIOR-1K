cd /mnt/public/mjwei/repo/behavior-1k-solution/BEHAVIOR-1K
python OmniGibson/omnigibson/learning/demo_expert_policy_server.py \
  --demo-data-dir /mnt/public/mjwei/download_models/2025-challenge-demos \
  --task-name turning_on_radio \
  --episode-index 10 \
  --start-frame 0 \
  --host 0.0.0.0 \
  --port 8007
