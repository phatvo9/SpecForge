#!/bin/bash
set -e

# MiniMax-M2.7 DFlash long-ctx ft2
# Init from best pretrain step 326k, cosine LR 1e-4

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
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper_cosine.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-config-path /workspace/checkpoints/dflash/minimax-m2.7-long-ctx-ft2/epoch_0_step_0/config.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/synthesis_25k_train.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/synthesis_25k_eval.jsonl \
    --build-dataset-num-proc 96 \
    --output-dir /workspace/checkpoints/dflash/minimax-m2.7-long-ctx-ft2/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.9 \
    --sglang-context-length 20000 \
    --sglang-max-total-tokens 20000 \
    --trust-remote-code \
    --num-epochs 10 \
    --batch-size 1 \
    --accumulation-steps 4 \
    --learning-rate 1e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 18000 \
    --chat-template minimax-m2 \
    --attention-backend sdpa \
    --block-size 8 \
    --num-anchors 750 \
    --loss-decay-gamma 5.0 \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --resume \
    --save-interval 1000 \
    --eval-interval 500 \
    --log-interval 50 \
    --report-to tensorboard
