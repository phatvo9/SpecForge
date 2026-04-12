#!/bin/bash
set -e

# ROCR 0=Node2=GPU3, ROCR 1=Node3=GPU0, ROCR 5=Node7=GPU4
export ROCR_VISIBLE_DEVICES=0,1,5
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"

pip install openai-harmony accelerate datasets yunchang wandb tensorboard pydantic tqdm psutil numpy 2>&1 | tail -10

# Best v2 checkpoint (lowest loss at step 10000)
# v1 best checkpoint (accept_len 1.50)
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
    --eval-data-path /workspace/data/gptoss-eagle3-train-v3-aabench.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints-v2b/eagle3/gpt-oss-120b/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.85 \
    --sglang-context-length 16000 \
    --num-epochs 10 \
    --batch-size 2 \
    --draft-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --max-length 16000 \
    --chat-template gpt-oss \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --log-interval 50 \
    --report-to tensorboard
