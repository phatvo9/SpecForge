#!/bin/bash
set -e

# MiniMax-M2.7 DFlash long-ctx ft4
# Init from best ft2 step 47k, constant LR 1e-5
# Changes from ft3: max_length=30k, gamma=6, anchors=800

export ROCR_VISIBLE_DEVICES=1,2,5,7
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TORCHINDUCTOR_CACHE_DIR="/workspace/cache/compiled_kernels"

pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -5

NUM_GPUS=4
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper_v4_1.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-config-path /workspace/checkpoints/dflash/minimax-m2.7-long-ctx-ft4/epoch_0_step_0/config.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/synthesis_25k_train.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/synthesis_25k_eval.jsonl \
    --build-dataset-num-proc 96 \
    --output-dir /workspace/checkpoints/dflash/minimax-m2.7-long-ctx-ft4/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.96 \
    --sglang-context-length 32000 \
    --sglang-max-total-tokens 32000 \
    --trust-remote-code \
    --num-epochs 2 \
    --batch-size 1 \
    --accumulation-steps 5 \
    --learning-rate 1e-5 \
    --warmup-ratio 0.0 \
    --max-grad-norm 1.0 \
    --max-length 30000 \
    --chat-template minimax-m2 \
    --attention-backend sdpa \
    --block-size 8 \
    --num-anchors 800 \
    --loss-decay-gamma 4.0 \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --resume \
    --save-interval 1000 \
    --eval-interval 500 \
    --log-interval 50 \
    --report-to tensorboard
