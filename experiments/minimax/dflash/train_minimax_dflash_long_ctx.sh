#!/bin/bash
set -e

# MiniMax-M2.7 DFlash long-ctx: pretrain with sliding_attention + long context
# New architecture: 7x sliding_attention + 1x full_attention, sliding_window=2048
# Train from scratch on 800K pretrain data

export ROCR_VISIBLE_DEVICES=7,1
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TORCHINDUCTOR_CACHE_DIR="/workspace/cache/compiled_kernels"

pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -5

NUM_GPUS=2
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper_v4_1.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-config-path /workspace/SpecForge/configs/minimax-m2.7-dflash-long-ctx.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/pretrain_400k.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/eval_500.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints/dflash/minimax-m2.7-long-ctx/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.9 \
    --sglang-context-length 14000 \
    --sglang-max-total-tokens 14000 \
    --trust-remote-code \
    --num-epochs 5 \
    --batch-size 2 \
    --accumulation-steps 4 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template minimax-m2 \
    --attention-backend sdpa \
    --block-size 8 \
    --num-anchors 512 \
    --loss-decay-gamma 5.0 \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --resume \
    --save-interval 10000 \
    --eval-interval 2000 \
    --log-interval 100 \
    --report-to tensorboard
