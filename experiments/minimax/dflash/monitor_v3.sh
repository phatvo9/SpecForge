#!/bin/bash
# Monitor DFlash v3 training until completion, then push best/latest checkpoint to HF
set -e

CKPT_BASE="/data3/phat/specforge/dflash/minimax-m2.7/dflash/minimax-m2.7-v3"
HF_REPO="phatv9/MiniMax-M2.7-DFlash-v3"
CONTAINER="minimax_dflash_train"
PYTHON="/root/miniconda3/envs/phat/bin/python"

echo "=== Monitoring DFlash v3 training ==="

while true; do
    STATUS=$(docker ps --filter "name=$CONTAINER" --format "{{.Status}}" 2>/dev/null)
    if [ -z "$STATUS" ]; then
        echo "$(date): Container stopped."
        break
    fi
    PROGRESS=$(docker logs "$CONTAINER" 2>&1 | grep "Training Epoch" | tail -1 | grep -oP 'Epoch \d+:\s+\d+%' | tail -1)
    EVAL=$(docker logs "$CONTAINER" 2>&1 | grep "INFO.*Eval" | tail -1 | grep -oP 'Step \d+.*' || echo "none")
    echo "$(date): $PROGRESS | Eval: $EVAL"
    sleep 300
done

# Get eval results
echo ""
echo "=== All Eval Results ==="
docker logs "$CONTAINER" 2>&1 | grep "INFO.*Eval" | grep -oP 'Step \d+.*'

# Find latest checkpoint
LATEST_CKPT=$(ls -d "$CKPT_BASE"/epoch_* 2>/dev/null | sort -t_ -k4 -n | tail -1)
echo ""
echo "Latest checkpoint: $LATEST_CKPT"

# Push latest
echo "=== Pushing latest checkpoint ==="
$PYTHON -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='$LATEST_CKPT',
    repo_id='$HF_REPO',
    commit_message='v3 final - $(basename $LATEST_CKPT)',
)
print('Pushed latest')
"

echo "=== Done ==="
