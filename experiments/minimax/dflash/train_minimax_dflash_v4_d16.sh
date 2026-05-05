#!/bin/bash
set -e

# GPUs 0,1,2,3,6 (ROCR 1,2,3,0,6)
export ROCR_VISIBLE_DEVICES=1,2,3,0,6
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TORCHINDUCTOR_CACHE_DIR="/workspace/cache/compiled_kernels"

pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -5

# MiniMax-M2.7 DFlash v4-d16: training from scratch with block_size=16
# Draft: Qwen3-based, 8 layers, hidden=3072, 24 heads, 8 KV heads (~880M params)
# Target layers: [1, 12, 24, 36, 48, 59] from 62 total (6 features)
# block_size=16, gamma=7, lr=6e-4, 800K data

NUM_GPUS=5
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-config-path /workspace/SpecForge/configs/minimax-m2.7-dflash-v4-d16.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/pretrain_800k.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/eval_300k.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints/dflash/minimax-m2.7-v4-d16/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.9 \
    --sglang-context-length 8192 \
    --sglang-max-total-tokens 8192 \
    --trust-remote-code \
    --num-epochs 5 \
    --batch-size 1 \
    --accumulation-steps 3 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.05 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template minimax-m2 \
    --attention-backend sdpa \
    --block-size 16 \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --cache-dir /workspace/cache \
    --dist-timeout 180 \
    --save-interval 10000 \
    --eval-interval 1500 \
    --log-interval 50 \
    --report-to tensorboard
