#!/bin/bash
# Monitor v4 training → push best → start finetune on synthesis data
set -e

CKPT_BASE="/data3/phat/specforge/dflash/minimax-m2.7/dflash/minimax-m2.7-v4"
FT_CKPT_BASE="/data3/phat/specforge/dflash/minimax-m2.7/dflash/minimax-m2.7-v4-ft"
HF_REPO="phatv9/MiniMax-M2.7-DFlash-v4"
CONTAINER="minimax_dflash_train"
PYTHON="/root/miniconda3/envs/phat/bin/python"
COMPOSE_DIR="/root/phat/SpecForge/experiments/minimax/dflash"

echo "=== Monitor v4 → push → finetune ==="

BEST_ACC="0.0"

# Phase 1: Monitor v4 training
while true; do
    STATUS=$(docker ps --filter "name=$CONTAINER" --format "{{.Status}}" 2>/dev/null)
    if [ -z "$STATUS" ]; then
        echo "$(date): v4 container stopped."
        break
    fi

    EVAL_LINE=$(docker logs "$CONTAINER" 2>&1 | grep "INFO.*Eval" | tail -1)
    CURR_ACC=$(echo "$EVAL_LINE" | grep -oP 'Acc: [\d.]+' | grep -oP '[\d.]+')
    CURR_STEP=$(echo "$EVAL_LINE" | grep -oP 'Step \d+' | grep -oP '\d+')
    EPOCH=$(docker logs "$CONTAINER" 2>&1 | grep "Training Epoch" | tail -1 | grep -oP 'Epoch \d+' | tail -1)
    DISK=$(df -h /data3 | tail -1 | awk '{print $5}')

    echo "$(date): $EPOCH | Step $CURR_STEP Acc=$CURR_ACC Best=$BEST_ACC | Disk=$DISK"

    # Push if improved
    if [ -n "$CURR_ACC" ] && [ -n "$CURR_STEP" ]; then
        IMPROVED=$($PYTHON -c "print(1 if float('${CURR_ACC:-0}') > float('${BEST_ACC:-0}') else 0)" 2>/dev/null || echo "0")
        if [ "$IMPROVED" = "1" ]; then
            BEST_ACC="$CURR_ACC"
            CKPT=$(ls -d "$CKPT_BASE"/epoch_* 2>/dev/null | sort -t_ -k4 -n | tail -1)
            if [ -n "$CKPT" ] && [ -f "$CKPT/model.safetensors" ]; then
                echo "  -> NEW BEST $CURR_ACC. Pushing $(basename $CKPT)..."
                $PYTHON -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$HF_REPO', private=True, exist_ok=True)
api.upload_folder(folder_path='$CKPT', repo_id='$HF_REPO', ignore_patterns=['training_state.pt'], commit_message='v4 $(basename $CKPT) eval_acc=${CURR_ACC}')
print('Pushed')
" 2>&1 | tail -3
            fi
        fi
    fi

    sleep 300
done

# Phase 2: Push final v4 with training_state
LATEST_CKPT=$(ls -d "$CKPT_BASE"/epoch_* 2>/dev/null | sort -t_ -k4 -n | tail -1)
if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT/model.safetensors" ]; then
    echo "=== Pushing v4 FINAL with training_state: $(basename $LATEST_CKPT) ==="
    $PYTHON -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$HF_REPO', private=True, exist_ok=True)
api.upload_folder(folder_path='$LATEST_CKPT', repo_id='$HF_REPO', commit_message='v4 FINAL $(basename $LATEST_CKPT)')
print('Pushed final')
" 2>&1 | tail -3
fi

# Phase 3: Copy v4 weights to v4-ft checkpoint dir (no training_state)
echo "=== Setting up v4 finetune ==="
mkdir -p "$FT_CKPT_BASE"
docker run --rm \
    -v /data3/phat/specforge/dflash/minimax-m2.7/dflash:/dflash \
    lmsysorg/sglang:v0.5.10.post1-rocm700-mi35x bash -c "
mkdir -p /dflash/minimax-m2.7-v4-ft/epoch_0_step_0
cp /dflash/minimax-m2.7-v4/$(basename $LATEST_CKPT)/config.json /dflash/minimax-m2.7-v4-ft/epoch_0_step_0/
cp /dflash/minimax-m2.7-v4/$(basename $LATEST_CKPT)/dflash.py /dflash/minimax-m2.7-v4-ft/epoch_0_step_0/
cp /dflash/minimax-m2.7-v4/$(basename $LATEST_CKPT)/model.safetensors /dflash/minimax-m2.7-v4-ft/epoch_0_step_0/
ls -lh /dflash/minimax-m2.7-v4-ft/epoch_0_step_0/
"

# Phase 4: Start finetune
echo "=== Starting v4 finetune on 17K synthesis data ==="
chmod +x "$COMPOSE_DIR/train_minimax_dflash_v4_ft.sh"
sed -i "s|command:.*|command: bash /workspace/SpecForge/experiments/minimax/dflash/train_minimax_dflash_v4_ft.sh|" "$COMPOSE_DIR/docker-compose-train-dflash.yaml"
cd "$COMPOSE_DIR"
docker compose -f docker-compose-train-dflash.yaml up -d 2>&1

echo "=== v4 finetune launched. Monitoring... ==="

# Phase 5: Monitor finetune
FT_BEST_ACC="0.0"
while true; do
    STATUS=$(docker ps --filter "name=$CONTAINER" --format "{{.Status}}" 2>/dev/null)
    if [ -z "$STATUS" ]; then
        echo "$(date): v4-ft container stopped."
        break
    fi

    EVAL_LINE=$(docker logs "$CONTAINER" 2>&1 | grep "INFO.*Eval" | tail -1)
    CURR_ACC=$(echo "$EVAL_LINE" | grep -oP 'Acc: [\d.]+' | grep -oP '[\d.]+')
    EPOCH=$(docker logs "$CONTAINER" 2>&1 | grep "Training Epoch" | tail -1 | grep -oP 'Epoch \d+' | tail -1)

    echo "$(date): FT $EPOCH | Acc=$CURR_ACC Best=$FT_BEST_ACC"

    if [ -n "$CURR_ACC" ]; then
        IMPROVED=$($PYTHON -c "print(1 if float('${CURR_ACC:-0}') > float('${FT_BEST_ACC:-0}') else 0)" 2>/dev/null || echo "0")
        if [ "$IMPROVED" = "1" ]; then
            FT_BEST_ACC="$CURR_ACC"
            CKPT=$(ls -d "$FT_CKPT_BASE"/epoch_* 2>/dev/null | sort -t_ -k4 -n | tail -1)
            if [ -n "$CKPT" ] && [ -f "$CKPT/model.safetensors" ]; then
                echo "  -> NEW FT BEST $CURR_ACC. Pushing $(basename $CKPT)..."
                $PYTHON -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path='$CKPT', repo_id='$HF_REPO', ignore_patterns=['training_state.pt'], commit_message='v4-ft $(basename $CKPT) eval_acc=${CURR_ACC}')
print('Pushed')
" 2>&1 | tail -3
            fi
        fi
    fi

    sleep 300
done

# Push final finetune
FT_LATEST=$(ls -d "$FT_CKPT_BASE"/epoch_* 2>/dev/null | sort -t_ -k4 -n | tail -1)
if [ -n "$FT_LATEST" ] && [ -f "$FT_LATEST/model.safetensors" ]; then
    echo "=== Pushing v4-ft FINAL: $(basename $FT_LATEST) ==="
    $PYTHON -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path='$FT_LATEST', repo_id='$HF_REPO', commit_message='v4-ft FINAL $(basename $FT_LATEST)')
print('Pushed final ft')
" 2>&1 | tail -3
fi

echo "=== All done ==="
