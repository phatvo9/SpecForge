#!/bin/bash
set -e

# GPUs 0,3,4,6 (ROCR node 1=GPU0, 0=GPU3, 5=GPU4, 6=GPU6)
export ROCR_VISIBLE_DEVICES=1,0,5
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TORCHINDUCTOR_CACHE_DIR="/workspace/cache/compiled_kernels"

pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -10

# MiniMax-M2.7 DFlash v3 training
# Draft: Qwen3-based, 6 layers, hidden=3072, 24 heads, 8 KV heads
# Target layers: [1, 13, 25, 37, 49, 59] from 62 total (6 features)
# block_size=8, gamma=6, following z-lab Kimi-K2.5-DFlash recipe

NUM_GPUS=3
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-config-path /workspace/SpecForge/configs/minimax-m2.7-dflash-v3.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/pretrain_250k.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/eval_250k.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints/dflash/minimax-m2.7-v3/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.92 \
    --trust-remote-code \
    --num-epochs 12 \
    --batch-size 1 \
    --accumulation-steps 4 \
    --learning-rate 1e-5 \
    --warmup-ratio 0.04 \
    --max-grad-norm .7 \
    --max-length 4096 \
    --chat-template minimax-m2 \
    --attention-backend sdpa \
    --block-size 8 \
    --num-anchors 500 \
    --loss-decay-gamma 6.0 \
    \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --resume \
    --save-interval 5000 \
    --eval-interval 5000 \
    --log-interval 100 \
    --report-to tensorboard
