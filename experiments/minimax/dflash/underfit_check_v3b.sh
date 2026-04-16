#!/bin/bash
set -e

# GPUs 0,3,4,6
export ROCR_VISIBLE_DEVICES=1,0,5,6
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TORCHINDUCTOR_CACHE_DIR="/workspace/cache/compiled_kernels"

pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -5

# Underfit check: v3 arch with new training params
NUM_GPUS=4
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-config-path /workspace/SpecForge/configs/minimax-m2.7-dflash-v3.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/underfit_train.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/underfit_eval.jsonl \
    --build-dataset-num-proc 4 \
    --output-dir /workspace/checkpoints/dflash/underfit-v3b/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.88 \
    --sglang-context-length 8192 \
    --trust-remote-code \
    --num-epochs 50 \
    --batch-size 1 \
    --accumulation-steps 1 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.05 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template minimax-m2 \
    --attention-backend sdpa \
    --block-size 8 \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --save-interval 99999 \
    --eval-interval 50 \
    --log-interval 10 \
    --report-to tensorboard
