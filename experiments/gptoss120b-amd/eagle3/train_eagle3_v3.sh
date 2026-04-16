#!/bin/bash
set -e

# Restrict to GPUs 0,3,4
# ROCR 0=Node2=GPU3, ROCR 1=Node3=GPU0, ROCR 5=Node7=GPU4
export ROCR_VISIBLE_DEVICES=0,1,5
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"

pip install openai-harmony accelerate datasets yunchang wandb tensorboard pydantic tqdm psutil numpy 2>&1 | tail -10

# v2c best checkpoint (epoch_6_step_23000, tensorboard step ~5750)
BEST_CKPT="/workspace/checkpoints-v2c/eagle3/gpt-oss-120b/epoch_6_step_23000"

echo "Using checkpoint: $BEST_CKPT"

NUM_GPUS=3
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/data/train_eagle3_wrapper.py \
    --target-model-path mshojaei77/gpt-oss-120b \
    --resume \
    --train-data-path /workspace/data/gptoss-eagle3-train-v3-aabench-converted.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints-v3/eagle3/gpt-oss-120b/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.85 \
    --num-epochs 18 \
    --batch-size 2 \
    --learning-rate 1e-6 \
    --max-length 8192 \
    --chat-template gpt-oss \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --save-interval 250 \
    --eval-interval 250 \
    --log-interval 50 \
    --report-to tensorboard
