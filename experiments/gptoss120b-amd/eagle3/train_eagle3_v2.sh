#!/bin/bash
set -e

# Restrict to GPUs 0,3,4 BEFORE any torch import
# ROCR uses KFD node order, not rocm-smi GPU IDs
# ROCR 0=Node2=GPU3(free), ROCR 1=Node3=GPU0(free), ROCR 5=Node7=GPU4(free)
export ROCR_VISIBLE_DEVICES=0,1,5
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

# Add SpecForge to PYTHONPATH (use image's native torch/transformers/sglang)
export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"

# Install only missing dependencies (skip sglang/torch/transformers - use image's versions)
pip install openai-harmony accelerate datasets yunchang wandb tensorboard pydantic tqdm psutil numpy 2>&1 | tail -10

# Finetune from v1 best checkpoint (step 14000)
CKPT_DIR="/workspace/checkpoints-v1/eagle3/gpt-oss-120b/epoch_7_step_14000"

NUM_GPUS=3
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/data/train_eagle3_wrapper.py \
    --target-model-path mshojaei77/gpt-oss-120b \
    --ckpt-dir "$CKPT_DIR" \
    --train-data-path /workspace/data/gptoss-eagle3-train-v2.jsonl \
    --train-only-last-turn \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints-v2/eagle3/gpt-oss-120b/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.85 \
    --sglang-context-length 16000 \
    --num-epochs 10 \
    --batch-size 2 \
    --learning-rate 1e-5 \
    --max-length 16000 \
    --chat-template gpt-oss \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --log-interval 50 \
    --report-to tensorboard
