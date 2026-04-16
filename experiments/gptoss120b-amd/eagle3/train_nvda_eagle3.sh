#!/bin/bash
set -e

# ROCR 0=Node2=GPU3, ROCR 1=Node3=GPU0, ROCR 5=Node7=GPU4
export ROCR_VISIBLE_DEVICES=0,1,5
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"

pip install openai-harmony accelerate datasets yunchang wandb tensorboard pydantic tqdm psutil numpy 2>&1 | tail -10

# NVIDIA pretrained eagle3 checkpoint
CKPT_DIR="/workspace/data/nvda-eagle3-patched"

NUM_GPUS=3
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/data/train_eagle3_wrapper.py \
    --target-model-path mshojaei77/gpt-oss-120b \
    --ckpt-dir "$CKPT_DIR" \
    --freeze-lm-head \
    --train-data-path /workspace/data/gptoss-eagle3-train-all4.jsonl \
    --eval-data-path /workspace/data/eagle3-gptoss-120b-eval/eval.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints/nvda-eagle3-gptoss120b/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.85 \
    --num-epochs 10 \
    --batch-size 2 \
    --draft-accumulation-steps 4 \
    --learning-rate 1e-4 \
    --max-length 8192 \
    --chat-template gpt-oss \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --save-interval 250 \
    --eval-interval 250 \
    --log-interval 50 \
    --report-to tensorboard
