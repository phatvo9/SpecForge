#!/bin/bash
# Monitor DFlash bs10 training until completion, then push best checkpoint to HF
set -e

COMPOSE_DIR="/root/phat/SpecForge/experiments/minimax/dflash"
CKPT_BASE="/data3/phat/specforge/dflash/minimax-m2.7/dflash/minimax-m2.7-bs10"
HF_REPO="phatv9/MiniMax-M2.7-DFlash-bs10"
CONTAINER="minimax_dflash_train"

echo "=== Monitoring DFlash bs10 training ==="
echo "Checkpoints: $CKPT_BASE"
echo "HF repo: $HF_REPO"

# Wait for training to complete
while true; do
    STATUS=$(docker ps --filter "name=$CONTAINER" --format "{{.Status}}" 2>/dev/null)
    if [ -z "$STATUS" ]; then
        echo "$(date): Container stopped. Training complete or crashed."
        break
    fi

    # Get latest progress
    LATEST=$(docker logs "$CONTAINER" 2>&1 | grep "Training Epoch" | tail -1 | grep -oP 'Epoch \d+.*?\d+%' || echo "unknown")
    LATEST_EVAL=$(docker logs "$CONTAINER" 2>&1 | grep "INFO.*Eval" | tail -1 | grep -oP 'Step \d+.*' || echo "none")
    echo "$(date): Running - $LATEST | Last eval: $LATEST_EVAL"

    sleep 300  # Check every 5 minutes
done

# Check if training completed successfully (final checkpoint saved)
FINAL_CKPT=$(ls -d "$CKPT_BASE"/epoch_5_* 2>/dev/null | sort -t_ -k4 -n | tail -1)
if [ -z "$FINAL_CKPT" ]; then
    echo "WARNING: No epoch 5 (final) checkpoint found. Checking last checkpoint..."
fi

# Find best checkpoint by eval accuracy from logs
echo ""
echo "=== Eval Results ==="
docker logs "$CONTAINER" 2>&1 | grep "INFO.*Eval" | grep -oP 'Step \d+.*'

# Find the latest checkpoint (should be final)
LATEST_CKPT=$(ls -d "$CKPT_BASE"/epoch_* 2>/dev/null | sort -t_ -k4 -n | tail -1)
echo ""
echo "Latest checkpoint: $LATEST_CKPT"

# Also identify best eval checkpoint
BEST_STEP=$(docker logs "$CONTAINER" 2>&1 | grep "INFO.*Eval" | grep -oP 'Step (\d+).*Acc: ([\d.]+)' | awk -F'Step |, Acc: | \\[' '{print $2, $NF}' | sort -k2 -n | tail -1 | awk '{print $1}')
echo "Best eval step: $BEST_STEP"

# Find checkpoint closest to best step
if [ -n "$BEST_STEP" ]; then
    BEST_CKPT=$(ls -d "$CKPT_BASE"/epoch_*_step_* 2>/dev/null | while read d; do
        s=$(basename "$d" | grep -oP 'step_\K\d+')
        echo "$((s > BEST_STEP ? s - BEST_STEP : BEST_STEP - s)) $d"
    done | sort -n | head -1 | awk '{print $2}')
    echo "Best checkpoint: $BEST_CKPT"
fi

# Push latest checkpoint
echo ""
echo "=== Pushing latest checkpoint to HF ==="
cd "$LATEST_CKPT"

# Check if huggingface-cli is available
if ! command -v huggingface-cli &>/dev/null; then
    /root/miniconda3/envs/phat/bin/pip install -q huggingface_hub
fi

/root/miniconda3/envs/phat/bin/huggingface-cli upload "$HF_REPO" . . \
    --repo-type model \
    --private \
    --commit-message "MiniMax-M2.7 DFlash bs10 - latest checkpoint ($(basename $LATEST_CKPT))"

echo "Pushed latest: $LATEST_CKPT -> $HF_REPO"

# Push best checkpoint if different from latest
if [ -n "$BEST_CKPT" ] && [ "$BEST_CKPT" != "$LATEST_CKPT" ]; then
    BEST_HF_REPO="${HF_REPO}-best"
    echo ""
    echo "=== Pushing best checkpoint to HF ==="
    cd "$BEST_CKPT"
    /root/miniconda3/envs/phat/bin/huggingface-cli upload "$BEST_HF_REPO" . . \
        --repo-type model \
        --private \
        --commit-message "MiniMax-M2.7 DFlash bs10 - best eval checkpoint ($(basename $BEST_CKPT))"
    echo "Pushed best: $BEST_CKPT -> $BEST_HF_REPO"
fi

echo ""
echo "=== Done ==="
