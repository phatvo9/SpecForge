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

# MiniMax-M2.7 DFlash v4-1-ft: finetune on 25K synthesis reasoning data
# Init from v4-1 step 370K weights (eval ~50.5%)
# lr=2e-5 cosine, 15 epochs, max_length=6144, anchors=500

NUM_GPUS=4
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-config-path /workspace/SpecForge/configs/minimax-m2.7-dflash-v4.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/synthesis_25k_train.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/synthesis_25k_eval.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints/dflash/minimax-m2.7-v4-1-ft/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.95 \
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
    --chat-template minimax-m2 \
    --attention-backend sdpa \
    --block-size 8 \
    --num-anchors 500 \
    --loss-decay-gamma 4.0 \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --resume \
    --save-interval 10000 \
    --eval-interval 500 \
    --log-interval 100 \
    --report-to tensorboard
