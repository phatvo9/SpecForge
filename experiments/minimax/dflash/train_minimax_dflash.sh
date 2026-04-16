#!/bin/bash
set -e

# GPUs 0,3,4,6 (ROCR node 1=GPU0, 0=GPU3, 5=GPU4, 6=GPU6)
export ROCR_VISIBLE_DEVICES=1,0,5,6
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TORCHINDUCTOR_CACHE_DIR="/workspace/cache/compiled_kernels"

pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -10

# MiniMax-M2.7 DFlash training
# Draft: Qwen3-based, 5 layers, hidden=3072, 24 heads, 8 KV heads
# Target layers: [1, 16, 30, 44, 59] from 62 total
# block_size=10, mask_token_id=200054 (<|MASK|> added to tokenizer)

NUM_GPUS=4
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-config-path /workspace/SpecForge/configs/minimax-m2.7-dflash-bs10.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/pretrain_250k.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/eval_250k.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints/dflash/minimax-m2.7-bs10/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.92 \
    --trust-remote-code \
    --num-epochs 12 \
    --batch-size 2 \
    --accumulation-steps 4 \
    --learning-rate 1e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm .7 \
    --max-length 4096 \
    --chat-template minimax-m2 \
    --attention-backend sdpa \
    --block-size 10 \
    --num-anchors 400 \
    --loss-decay-gamma 5.0 \
    --mask-token-id 200054 \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --resume \
    --save-interval 5000 \
    --eval-interval 5000 \
    --log-interval 100 \
    --report-to tensorboard
