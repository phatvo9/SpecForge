#!/bin/bash
# Monitor v3 and push checkpoint to HF every time eval acc improves
# - Intermediate pushes: skip training_state.pt (save HF storage)
# - Final push (done/crashed): include training_state.pt for resume
set -e

CKPT_BASE="/data3/phat/specforge/dflash/minimax-m2.7/dflash/minimax-m2.7-v3"
HF_REPO="phatv9/MiniMax-M2.7-DFlash-v3"
CONTAINER="minimax_dflash_train"
PYTHON="/root/miniconda3/envs/phat/bin/python"

echo "=== Monitor v3: push on eval improvement (skip training_state.pt until final) ==="

BEST_ACC="0.0"
LAST_PUSHED_STEP=""

while true; do
    STATUS=$(docker ps --filter "name=$CONTAINER" --format "{{.Status}}" 2>/dev/null)
    if [ -z "$STATUS" ]; then
        echo "$(date): Container stopped."
        break
    fi

    EVAL_LINE=$(docker logs "$CONTAINER" 2>&1 | grep "INFO.*Eval" | tail -1)
    CURR_ACC=$(echo "$EVAL_LINE" | grep -oP 'Acc: [\d.]+' | grep -oP '[\d.]+')
    CURR_STEP=$(echo "$EVAL_LINE" | grep -oP 'Step \d+' | grep -oP '\d+')
    EPOCH=$(docker logs "$CONTAINER" 2>&1 | grep "Training Epoch" | tail -1 | grep -oP 'Epoch \d+' | tail -1)
    DISK=$(df -h /data3 | tail -1 | awk '{print $5}')

    echo "$(date): $EPOCH | Step $CURR_STEP Acc=$CURR_ACC Best=$BEST_ACC | Disk=$DISK"

    if [ -n "$CURR_ACC" ] && [ -n "$CURR_STEP" ] && [ "$CURR_STEP" != "$LAST_PUSHED_STEP" ]; then
        IMPROVED=$($PYTHON -c "print(1 if float('${CURR_ACC:-0}') > float('${BEST_ACC:-0}') else 0)" 2>/dev/null || echo "0")
        if [ "$IMPROVED" = "1" ]; then
            BEST_ACC="$CURR_ACC"
            CKPT=$(ls -d "$CKPT_BASE"/epoch_* 2>/dev/null | sort -t_ -k4 -n | tail -1)
            if [ -n "$CKPT" ] && [ -f "$CKPT/model.safetensors" ]; then
                echo "  -> NEW BEST $CURR_ACC. Pushing $(basename $CKPT) (without training_state.pt)..."
                $PYTHON -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='$CKPT',
    repo_id='$HF_REPO',
    ignore_patterns=['training_state.pt'],
    commit_message='v3 $(basename $CKPT) eval_acc=${CURR_ACC}')
print('Pushed (no training_state)')
" 2>&1 | tail -3
                LAST_PUSHED_STEP="$CURR_STEP"
            fi
        fi
    fi

    sleep 300
done

# Final push WITH training_state.pt for resume
LATEST_CKPT=$(ls -d "$CKPT_BASE"/epoch_* 2>/dev/null | sort -t_ -k4 -n | tail -1)
if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT/model.safetensors" ]; then
    echo "=== FINAL push with training_state.pt: $(basename $LATEST_CKPT) ==="
    $PYTHON -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='$LATEST_CKPT',
    repo_id='$HF_REPO',
    commit_message='v3 FINAL $(basename $LATEST_CKPT) (with training_state)')
print('Pushed final with training_state')
" 2>&1 | tail -3
fi
echo "=== Done ==="
