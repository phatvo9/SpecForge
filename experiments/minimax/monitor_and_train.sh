#!/bin/bash
# Monitor data regeneration and kick off training when done
set -e

DATA_DIR="/home/claudeuser/specforge_data/minimax2.7-spec-data"
TRAIN_RAW="$DATA_DIR/train_raw.jsonl"
LOG_FILE="$DATA_DIR/regen.log"

echo "Monitoring data regeneration..."
echo "Check progress: wc -l $TRAIN_RAW"

while true; do
    # Check if regeneration process is still running
    if ! pgrep -f "regenerate_data.py" > /dev/null 2>&1; then
        echo "$(date): Data regeneration process finished!"
        TOTAL=$(wc -l < "$TRAIN_RAW" 2>/dev/null || echo 0)
        echo "Total samples: $TOTAL"
        break
    fi

    CURRENT=$(wc -l < "$TRAIN_RAW" 2>/dev/null || echo 0)
    echo "$(date): $CURRENT samples generated..."
    sleep 300  # check every 5 minutes
done

echo "Data regeneration complete. Preparing training data..."

# Run the post-processing and training launch
source /root/miniconda3/etc/profile.d/conda.sh
conda activate phat

python3 << 'PYEOF'
import json
import random
import os

DATA_DIR = "/home/claudeuser/specforge_data/minimax2.7-spec-data"
TRAIN_RAW = f"{DATA_DIR}/train_raw.jsonl"

# Read all generated data
with open(TRAIN_RAW) as f:
    lines = f.readlines()
print(f"Total raw samples: {len(lines)}")

# Filter out failed/empty
valid = []
for line in lines:
    d = json.loads(line)
    if d.get("finish_reason") == "stop" and d.get("completion_tokens", 0) > 10:
        valid.append(d)
print(f"Valid samples (stop + >10 tokens): {len(valid)}")

# Remove source/metadata, keep only conversations
random.seed(42)
random.shuffle(valid)

# Split: last 500 for eval
eval_data = valid[:500]
train_data = valid[500:]

# Write train split (conversations only)
with open(f"{DATA_DIR}/train.jsonl", "w") as f:
    for item in train_data:
        f.write(json.dumps({"conversations": item["conversations"]}) + "\n")

# Write eval split
with open(f"{DATA_DIR}/eval.jsonl", "w") as f:
    for item in eval_data:
        f.write(json.dumps({"conversations": item["conversations"]}) + "\n")

print(f"Train: {len(train_data)}, Eval: {len(eval_data)}")
print(f"Saved to {DATA_DIR}/train.jsonl and {DATA_DIR}/eval.jsonl")

# Push to HF
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("phatv9/minimax2.7-spec-data", repo_type="dataset", exist_ok=True)
api.upload_file(
    path_or_fileobj=f"{DATA_DIR}/train.jsonl",
    path_in_repo="train.jsonl",
    repo_id="phatv9/minimax2.7-spec-data",
    repo_type="dataset",
    commit_message=f"Train split: {len(train_data)} MiniMax-M2.7 regenerated samples",
)
api.upload_file(
    path_or_fileobj=f"{DATA_DIR}/eval.jsonl",
    path_in_repo="eval.jsonl",
    repo_id="phatv9/minimax2.7-spec-data",
    repo_type="dataset",
    commit_message=f"Eval split: {len(eval_data)} MiniMax-M2.7 regenerated samples",
)
print("Pushed to phatv9/minimax2.7-spec-data")
PYEOF

echo "Data ready. Launching EAGLE3 training for MiniMax-M2.7..."

# Stop MiniMax inference server to free GPU 6 (training uses GPUs 0,3,4)
# Don't stop - training uses different GPUs

# Launch training
cd /root/phat/SpecForge/experiments/minimax/eagle3
docker compose -f docker-compose-train-minimax-eagle3.yaml up -d

echo "Training launched! Monitor with: docker logs minimax_eagle3_train -f"
