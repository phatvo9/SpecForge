#!/bin/bash
# Smart monitor for DFlash v3:
# - If eval acc NOT improving after epoch 6 -> stop, push, start v4
# - If improving -> continue, push when done
set -e

CKPT_BASE="/data3/phat/specforge/dflash/minimax-m2.7/dflash/minimax-m2.7-v3"
HF_REPO="phatv9/MiniMax-M2.7-DFlash-v3"
CONTAINER="minimax_dflash_train"
PYTHON="/root/miniconda3/envs/phat/bin/python"
COMPOSE_DIR="/root/phat/SpecForge/experiments/minimax/dflash"

echo "=== Smart Monitor: DFlash v3 ==="
echo "Strategy: stop after epoch 6 if no improvement, else continue to epoch 12"

PREV_ACC=""
STALE_COUNT=0

while true; do
    STATUS=$(docker ps --filter "name=$CONTAINER" --format "{{.Status}}" 2>/dev/null)
    if [ -z "$STATUS" ]; then
        echo "$(date): Container stopped. Training complete or crashed."
        break
    fi

    # Get current epoch and eval
    EPOCH=$(docker logs "$CONTAINER" 2>&1 | grep "Training Epoch" | tail -1 | grep -oP 'Epoch \d+' | tail -1 | grep -oP '\d+')
    LATEST_EVAL=$(docker logs "$CONTAINER" 2>&1 | grep "INFO.*Eval" | tail -1 | grep -oP 'Acc: ([\d.]+)' | grep -oP '[\d.]+')
    LATEST_EVAL_LINE=$(docker logs "$CONTAINER" 2>&1 | grep "INFO.*Eval" | tail -1 | grep -oP 'Step \d+.*' || echo "none")

    echo "$(date): Epoch $EPOCH | Eval: $LATEST_EVAL_LINE"

    # Check plateau after epoch 6
    if [ -n "$EPOCH" ] && [ "$EPOCH" -ge 6 ] && [ -n "$LATEST_EVAL" ] && [ -n "$PREV_ACC" ]; then
        # Compare: if improvement < 0.5% over last 2 checks (10 min), consider stale
        IMPROVED=$($PYTHON -c "print(1 if float('$LATEST_EVAL') > float('$PREV_ACC') + 0.005 else 0)" 2>/dev/null || echo "0")
        if [ "$IMPROVED" = "0" ]; then
            STALE_COUNT=$((STALE_COUNT + 1))
            echo "  -> No significant improvement ($PREV_ACC -> $LATEST_EVAL). Stale count: $STALE_COUNT/3"
        else
            STALE_COUNT=0
            echo "  -> Improving ($PREV_ACC -> $LATEST_EVAL)"
        fi

        # If stale for 3 consecutive checks (15 min) after epoch 6, stop
        if [ "$STALE_COUNT" -ge 3 ]; then
            echo ""
            echo "=== PLATEAU DETECTED after epoch 6. Stopping v3, starting v4 ==="

            # Stop v3
            cd "$COMPOSE_DIR"
            docker compose -f docker-compose-train-dflash.yaml down 2>&1

            # Push v3 final
            LATEST_CKPT=$(ls -d "$CKPT_BASE"/epoch_* 2>/dev/null | sort -t_ -k4 -n | tail -1)
            echo "Pushing v3 final: $LATEST_CKPT"
            $PYTHON -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path='$LATEST_CKPT', repo_id='$HF_REPO', commit_message='v3 final (plateaued) - $(basename $LATEST_CKPT)')
print('Pushed v3')
"
            # Switch to v4
            echo ""
            echo "=== Starting v4 ==="
            chmod +x "$COMPOSE_DIR/train_minimax_dflash_v4.sh"
            mkdir -p /data3/phat/specforge/dflash/minimax-m2.7-v4
            sed -i 's|train_minimax_dflash_v3.sh|train_minimax_dflash_v4.sh|' "$COMPOSE_DIR/docker-compose-train-dflash.yaml"
            cd "$COMPOSE_DIR"
            docker compose -f docker-compose-train-dflash.yaml up -d 2>&1
            echo "v4 launched. Monitoring v4 until completion..."

            # Monitor v4 until done
            while true; do
                V4_STATUS=$(docker ps --filter "name=$CONTAINER" --format "{{.Status}}" 2>/dev/null)
                if [ -z "$V4_STATUS" ]; then
                    echo "$(date): v4 container stopped."
                    break
                fi
                V4_EVAL=$(docker logs "$CONTAINER" 2>&1 | grep "INFO.*Eval" | tail -1 | grep -oP 'Step \d+.*' || echo "none")
                V4_EPOCH=$(docker logs "$CONTAINER" 2>&1 | grep "Training Epoch" | tail -1 | grep -oP 'Epoch \d+' | tail -1)
                echo "$(date): v4 Epoch $V4_EPOCH | Eval: $V4_EVAL"
                sleep 300
            done

            # Push v4 final
            V4_CKPT_BASE="/data3/phat/specforge/dflash/minimax-m2.7/dflash/minimax-m2.7-v4"
            V4_LATEST=$(ls -d "$V4_CKPT_BASE"/epoch_* 2>/dev/null | sort -t_ -k4 -n | tail -1)
            echo "Pushing v4 final: $V4_LATEST"
            $PYTHON -c "
from huggingface_hub import HfApi
api = HfApi()
repo = 'phatv9/MiniMax-M2.7-DFlash-v4'
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(folder_path='$V4_LATEST', repo_id=repo, commit_message='v4 final 8layers bs8 gamma7 - $(basename $V4_LATEST)')
print('Pushed v4')
"
            echo "=== All done ==="
            exit 0
        fi
    fi

    PREV_ACC="$LATEST_EVAL"
    sleep 300
done

# If we get here, v3 finished naturally (12 epochs)
echo ""
echo "=== v3 completed all 12 epochs ==="
docker logs "$CONTAINER" 2>&1 | grep "INFO.*Eval" | grep -oP 'Step \d+.*'

LATEST_CKPT=$(ls -d "$CKPT_BASE"/epoch_* 2>/dev/null | sort -t_ -k4 -n | tail -1)
echo "Pushing v3 final: $LATEST_CKPT"
$PYTHON -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path='$LATEST_CKPT', repo_id='$HF_REPO', commit_message='v3 final (12 epochs) - $(basename $LATEST_CKPT)')
print('Pushed v3')
"
echo "=== Done ==="
