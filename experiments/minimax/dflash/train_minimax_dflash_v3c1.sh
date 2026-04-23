#!/bin/bash
set -e

# GPUs 4,6 (ROCR node 5=GPU4, 6=GPU6)
export ROCR_VISIBLE_DEVICES=5,6
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TORCHINDUCTOR_CACHE_DIR="/workspace/cache/compiled_kernels"

pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -5

# MiniMax-M2.7 DFlash v3c-1: continue finetuning on 17K reasoning data
# Init from v3c weights (eval 45.5%), constant lr=1e-5, 5 epochs

NUM_GPUS=2
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper_v3c1.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-config-path /workspace/SpecForge/configs/minimax-m2.7-dflash-v3.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/synthesis_17k.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/eval_reasoning.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints/dflash/minimax-m2.7-v3c1/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.95 \
    --trust-remote-code \
    --num-epochs 5 \
    --batch-size 1 \
    --accumulation-steps 1 \
    --learning-rate 1e-5 \
    --warmup-ratio 0.0 \
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
    --save-interval 5000 \
    --eval-interval 500 \
    --log-interval 100 \
    --report-to tensorboard
