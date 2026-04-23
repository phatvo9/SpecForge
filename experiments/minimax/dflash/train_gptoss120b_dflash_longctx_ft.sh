#!/bin/bash
set -e

# 1 GPU (GPU 5, ROCR 5)
export ROCR_VISIBLE_DEVICES=5
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TORCHINDUCTOR_CACHE_DIR="/workspace/cache/compiled_kernels"

pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -5

# gpt-oss-120b DFlash long context finetune
# Pretrained: epoch_13_step_43277 (block_size=16, 8 layers, hidden=2880)
# Data: gptoss eagle3 training data (~20K)
# lr=1e-6, gamma=7, max_length=30720, anchors=1000

NUM_GPUS=1
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper.py \
    --target-model-path mshojaei77/gpt-oss-120b \
    --draft-config-path /workspace/checkpoints/gptoss120b-dflash-longctx-ft/epoch_0_step_0/config.json \
    --train-data-path /workspace/data/gptoss-eagle3-train-all4-train.jsonl \
    --eval-data-path /workspace/data/gptoss-eagle3-train-all4-eval.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints/gptoss120b-dflash-longctx-ft/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.95 \
    --sglang-context-length 30720 \
    --sglang-max-total-tokens 30720 \
    --trust-remote-code \
    --num-epochs 10 \
    --batch-size 2 \
    --accumulation-steps 2 \
    --learning-rate 1e-6 \
    --warmup-ratio 0.05 \
    --max-grad-norm 0.7 \
    --max-length 30720 \
    --chat-template gpt-oss \
    --attention-backend sdpa \
    --block-size 16 \
    --num-anchors 1000 \
    --loss-decay-gamma 7.0 \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --resume \
    --save-interval 5000 \
    --eval-interval 1000 \
    --log-interval 50 \
    --report-to tensorboard
