#!/bin/bash
set -e

# GPUs 0,4,6 (ROCR node 1=GPU0, 5=GPU4, 6=GPU6)
export ROCR_VISIBLE_DEVICES=0,5,6
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TORCHINDUCTOR_CACHE_DIR="/workspace/cache/compiled_kernels"

pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -5

# MiniMax-M2.7 DFlash v4 finetune on 17K synthesis reasoning data
# Init from v4 best checkpoint, lr=2e-5, gamma=4, 20 epochs

NUM_GPUS=3
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-config-path /workspace/SpecForge/configs/minimax-m2.7-dflash-v4.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/synthesis_17k.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/eval_reasoning.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints/dflash/minimax-m2.7-v4-ft/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.92 \
    --trust-remote-code \
    --num-epochs 20 \
    --batch-size 1 \
    --accumulation-steps 1 \
    --learning-rate 2e-5 \
    --warmup-ratio 0.0 \
    --max-grad-norm 1.0 \
    --max-length 2048 \
    --chat-template minimax-m2 \
    --attention-backend sdpa \
    --block-size 8 \
    --num-anchors 128 \
    --loss-decay-gamma 4.0 \
    --self-logit-distillation \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --resume \
    --save-interval 5000 \
    --eval-interval 1000 \
    --log-interval 100 \
    --report-to tensorboard
