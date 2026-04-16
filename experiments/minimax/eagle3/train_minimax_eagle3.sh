#!/bin/bash
set -e

# ROCR 0=Node2=GPU3, ROCR 1=Node3=GPU0, ROCR 5=Node7=GPU4
export ROCR_VISIBLE_DEVICES=0,1,5
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"

pip install openai-harmony accelerate datasets yunchang wandb tensorboard pydantic tqdm psutil numpy 2>&1 | tail -10

# MiniMax-M2.7 EAGLE3 training
# Following M2.5-Eagle3 training recipe:
# - LR: 2e-5, batch 1, max_length 2048, TTT=7
# - Draft: hidden=3072, 1 layer, 24 heads, 8 KV, intermediate=8192
# - Aux layers: [1, 30, 59] for M2.7's 62 layers

NUM_GPUS=3
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/data/train_eagle3_wrapper.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-model-config /workspace/SpecForge/configs/minimax-m2.7-eagle3.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/train.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/eval.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints/minimax-m2.7-eagle3/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.85 \
    --trust-remote-code \
    --num-epochs 9 \
    --batch-size 1 \
    --draft-accumulation-steps 4 \
    --learning-rate 2e-5 \
    --max-length 2048 \
    --ttt-length 7 \
    --chat-template minimax-m2 \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --save-interval 500 \
    --eval-interval 500 \
    --log-interval 50 \
    --report-to tensorboard
