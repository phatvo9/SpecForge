#!/bin/bash
set -e

# GPUs 0,1,2,3 (ROCR 1,2,3,0)
export ROCR_VISIBLE_DEVICES=1,2,3,0
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TORCHINDUCTOR_CACHE_DIR="/workspace/cache/compiled_kernels"

pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -5

# MiniMax-M2.7 DFlash v4-1-ft-con
# TP=4, max_length=16384, anchors=750
# lr=5e-7 constant, epoch=10, 25K synthesis data

NUM_GPUS=4
TP_SIZE=4

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper_v4_1_ft_con.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-config-path /workspace/SpecForge/configs/minimax-m2.7-dflash-v4.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/synthesis_25k_train.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/synthesis_25k_eval.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints/dflash/minimax-m2.7-v4-1-ft-con/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.95 \
    --sglang-context-length 20480 \
    --sglang-max-total-tokens 20480 \
    --trust-remote-code \
    --num-epochs 10 \
    --batch-size 1 \
    --accumulation-steps 2 \
    --learning-rate 5e-7 \
    --warmup-ratio 0.0 \
    --max-grad-norm 0.5 \
    --max-length 20480 \
    --chat-template minimax-m2 \
    --attention-backend sdpa \
    --block-size 16 \
    --num-anchors 1000 \
    --loss-decay-gamma 7.0 \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --resume \
    --save-interval 10000 \
    --eval-interval 1000 \
    --log-interval 50 \
    --report-to tensorboard
