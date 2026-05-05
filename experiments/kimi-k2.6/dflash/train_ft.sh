#!/bin/bash
set -e

# GPUs 0,3,4,6 (ROCR 1,0,5,6)
export ROCR_VISIBLE_DEVICES=1,0,5,6
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TORCHINDUCTOR_CACHE_DIR="/workspace/cache/compiled_kernels"

pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -5

# Kimi K2.6 DFlash finetune on synthesis reasoning data
# Init from pretrain best checkpoint
# TP=2, DP=2, EP=2

NUM_GPUS=4
TP_SIZE=2

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/kimi-k2.6/dflash/train_dflash_wrapper_kimi.py \
    --target-model-path moonshotai/Kimi-K2.6 \
    --draft-config-path /workspace/SpecForge/configs/kimi-k2.6-dflash.json \
    --train-data-path /workspace/data/kimi-k2.6-spec-data/synthesis_25k_train.jsonl \
    --eval-data-path /workspace/data/kimi-k2.6-spec-data/synthesis_25k_eval.jsonl \
    --build-dataset-num-proc 32 \
    --output-dir /workspace/checkpoints/dflash/kimi-k2.6-ft/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.9 \
    --sglang-context-length 6144 \
    --sglang-max-total-tokens 6144 \
    --trust-remote-code \
    --num-epochs 15 \
    --batch-size 1 \
    --accumulation-steps 4 \
    --learning-rate 2e-5 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 6144 \
    --chat-template kimi-k2-instruct \
    --attention-backend sdpa \
    --block-size 8 \
    --num-anchors 400 \
    --loss-decay-gamma 4.0 \
    --cache-dir /workspace/cache \
    --dist-timeout 180 \
    --resume \
    --save-interval 5000 \
    --eval-interval 500 \
    --log-interval 100 \
    --report-to tensorboard
