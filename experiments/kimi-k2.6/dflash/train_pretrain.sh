#!/bin/bash
set -e

# Disable core dumps (GPU + CPU) to prevent filling root disk
ulimit -c 0
export GPU_COREDUMP_ENABLE=0
export GPU_COREDUMP_DIR="/workspace/cache/gpucoredumps"
mkdir -p /workspace/cache/gpucoredumps

# All 8 GPUs (ROCR 0-7)
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TMPDIR="/workspace/cache/tmp"
export HF_HUB_TRUST_REMOTE_CODE=1
# All cache env vars set in docker-compose; just create dirs
mkdir -p /workspace/cache/torchinductor /workspace/cache/triton /workspace/cache/comgr /workspace/cache/tmp /workspace/cache/pip /workspace/cache/aiter_jit

# Only install missing packages, use /workspace/cache for pip cache to avoid filling root disk
pip install --cache-dir /workspace/cache/pip yunchang wandb==0.19.11 2>&1 | tail -3

# Kimi K2.6 DFlash pretrain
# Draft: 6 layers, hidden=7168, 64 heads, 8 KV heads (~3.5GB)
# Target: moonshotai/Kimi-K2.6 (61 layers, MoE, BF16 ~555GB)
# Init from z-lab/Kimi-K2.5-DFlash weights
# TP=8, DP=1

NUM_GPUS=8
TP_SIZE=4

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/kimi-k2.6/dflash/train_dflash_wrapper_kimi.py \
    --target-model-path moonshotai/Kimi-K2.6 \
    --draft-config-path /workspace/SpecForge/configs/kimi-k2.6-dflash.json \
    --train-data-path /workspace/data/kimi-k2.6-spec-data/pretrain_800k.jsonl \
    --eval-data-path /workspace/data/kimi-k2.6-spec-data/eval_100.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints/dflash/kimi-k2.6-pretrain/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.9 \
    --sglang-context-length 4096 \
    --sglang-max-total-tokens 4096 \
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
