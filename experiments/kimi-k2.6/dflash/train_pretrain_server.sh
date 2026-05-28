#!/bin/bash
set -e

# ====================================================================
# Server-mode DFlash training for Kimi K2.6
#
# Architecture:
#   1. Hidden States Server (torchrun, 8 GPUs, TP=8) - loads target model
#   2. Training Process  (single process, GPU 0)     - trains draft model
#
# The server loads the full target model with TP and serves hidden states
# over TCP. The training process runs without torchrun.
# ====================================================================

# Disable core dumps (GPU + CPU) to prevent filling root disk
ulimit -c 0
export GPU_COREDUMP_ENABLE=0
export GPU_COREDUMP_DIR="/workspace/cache/gpucoredumps"
mkdir -p /workspace/cache/gpucoredumps

# All 8 GPUs (ROCR 0-7)
# Don't set ROCR_VISIBLE_DEVICES globally — set per-process below
export ROCR_VISIBLE_DEVICES=0,1,3,4,6
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TMPDIR="/workspace/cache/tmp"
export HF_HUB_TRUST_REMOTE_CODE=1
export TRUST_REMOTE_CODE=true
export TRANSFORMERS_TRUST_REMOTE_CODE=1

# All cache env vars set in docker-compose; just create dirs
mkdir -p /workspace/cache/torchinductor /workspace/cache/triton /workspace/cache/comgr /workspace/cache/tmp /workspace/cache/pip /workspace/cache/aiter_jit

# Install missing packages
python3 -c "import yunchang" 2>/dev/null || pip install --cache-dir /workspace/cache/pip yunchang 2>&1 | tail -3
python3 -c "import wandb" 2>/dev/null || pip install --cache-dir /workspace/cache/pip wandb==0.19.11 2>&1 | tail -3

# ── Configuration ─────────────────────────────────────────────────

NUM_GPUS=4
TP_SIZE=4
SERVER_PORT=29700

TARGET_MODEL=moonshotai/Kimi-K2.6
DRAFT_CONFIG=/workspace/SpecForge/configs/kimi-k2.6-dflash.json
TRAIN_DATA=/workspace/data/kimi-k2.6-spec-data/pretrain_800k.jsonl
EVAL_DATA=/workspace/data/kimi-k2.6-spec-data/eval_100.jsonl
OUTPUT_DIR=/workspace/checkpoints/dflash/kimi-k2.6-pretrain-server/

# ── Step 1: Launch Hidden States Server ───────────────────────────

echo "=== Starting Hidden States Server (TP=${TP_SIZE}, port=${SERVER_PORT}) ==="

ROCR_VISIBLE_DEVICES=0,3,4,6 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    -m specforge.modeling.target.sglang_hidden_server \
    --model-path $TARGET_MODEL \
    --port $SERVER_PORT \
    --tp-size $TP_SIZE \
    --mem-fraction-static 0.9 \
    --attention-backend aiter \
    --context-length 4096 \
    --max-total-tokens 4096 \
    --trust-remote-code \
    --dist-timeout 600 \
    &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready (the training script also waits,
# but this gives us early failure detection)
echo "Waiting for server to start..."
for i in $(seq 1 300); do
    if python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
try:
    s.connect(('127.0.0.1', $SERVER_PORT))
    s.close()
    sys.exit(0)
except:
    sys.exit(1)
" 2>/dev/null; then
        echo "Server is accepting connections after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server process died"
        exit 1
    fi
    sleep 1
done

# ── Step 2: Launch Training Process (single process) ──────────────

echo "=== Starting Training Process (single-process, sglang-server backend) ==="

# Training on GPU 1 (separate from server's GPUs 0,3,4,6)
ROCR_VISIBLE_DEVICES=1 python3 /workspace/SpecForge/experiments/kimi-k2.6/dflash/train_dflash_wrapper_kimi.py \
    --target-model-path $TARGET_MODEL \
    --draft-config-path $DRAFT_CONFIG \
    --train-data-path $TRAIN_DATA \
    --eval-data-path $EVAL_DATA \
    --build-dataset-num-proc 16 \
    --output-dir $OUTPUT_DIR \
    --target-model-backend sglang-server \
    --sglang-server-host 127.0.0.1 \
    --sglang-server-port $SERVER_PORT \
    --trust-remote-code \
    --num-epochs 5 \
    --batch-size 1 \
    --accumulation-steps 4 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.05 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template kimi-k2-instruct \
    --attention-backend sdpa \
    --block-size 8 \
    --num-anchors 512 \
    --loss-decay-gamma 4.0 \
    --embedding-key language_model.model.embed_tokens.weight \
    --lm-head-key language_model.lm_head.weight \
    --cache-dir /workspace/cache \
    --dist-timeout 600 \
    --resume \
    --save-interval 10000 \
    --eval-interval 2000 \
    --log-interval 100 \
    --report-to tensorboard

TRAIN_EXIT=$?

# ── Step 3: Cleanup ──────────────────────────────────────────────

echo "Training complete (exit code: $TRAIN_EXIT). Shutting down server..."

# The training script sends a shutdown command to the server.
# Give it a moment, then force kill if still running.
sleep 5
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server still running, sending SIGTERM..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null || true
fi

echo "=== Done ==="
exit $TRAIN_EXIT
