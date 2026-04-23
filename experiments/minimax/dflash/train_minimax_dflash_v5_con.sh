#!/bin/bash
set -e

# GPUs HIP-id 5,7
export ROCR_VISIBLE_DEVICES=5,7
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TORCHINDUCTOR_CACHE_DIR="/workspace/cache/compiled_kernels"

pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -5

# MiniMax-M2.7 DFlash v5-con
# 6 layers, auto target features (6), TP=2
# block_size=16, max_length=8192, anchors=1000, accum=6
# lr=5e-4 cosine, epoch=10, 800K data

NUM_GPUS=2
TP_SIZE=2

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-config-path /workspace/SpecForge/configs/minimax-m2.7-dflash-v5.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/pretrain_800k.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/eval_300k.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints/dflash/minimax-m2.7-v5-con/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.95 \
    --sglang-context-length 8192 \
    --sglang-max-total-tokens 8192 \
    --trust-remote-code \
    --num-epochs 10 \
    --batch-size 1 \
    --accumulation-steps 6 \
    --learning-rate 1e-6 \
    --warmup-ratio 0.1 \
    --max-grad-norm 0.5 \
    --max-length 8192 \
    --chat-template minimax-m2 \
    --attention-backend sdpa \
    --block-size 16 \
    --num-anchors 1000 \
    --loss-decay-gamma 7.0 \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --resume \
    --save-interval 10000 \
    --eval-interval 2000 \
    --log-interval 50 \
    --report-to tensorboard
