#!/bin/bash
set -e

# GPUs 4,5 only (ROCR 4,5 — verify mapping)
export ROCR_VISIBLE_DEVICES=5,7
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TORCHINDUCTOR_CACHE_DIR="/workspace/cache/compiled_kernels"

pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -3

# TP=2 gradient accumulation test
# Using v4-d16 checkpoint + underfit data
# accum=4 to prove it works

NUM_GPUS=2
TP_SIZE=2

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-config-path /workspace/SpecForge/configs/minimax-m2.7-dflash-v4-d16.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/underfit_train.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/underfit_eval.jsonl \
    --build-dataset-num-proc 4 \
    --output-dir /workspace/checkpoints/dflash/tp2-accum-test/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.95 \
    --sglang-context-length 4096 \
    --sglang-max-total-tokens 4096 \
    --trust-remote-code \
    --num-epochs 10 \
    --batch-size 1 \
    --accumulation-steps 4 \
    --learning-rate 1e-8 \
    --warmup-ratio 0.0 \
    --max-grad-norm 1.0 \
    --max-length 4096 \
    --chat-template minimax-m2 \
    --attention-backend sdpa \
    --block-size 16 \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --resume \
    --save-interval 99999 \
    --eval-interval 25 \
    --log-interval 5 \
    --report-to tensorboard
