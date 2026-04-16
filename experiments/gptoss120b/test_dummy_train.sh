#!/bin/bash
set -e
export ROCR_VISIBLE_DEVICES=0
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"

pip install yunchang accelerate datasets wandb tensorboard pydantic tqdm psutil numpy openai-harmony -q 2>&1 | tail -5

torchrun \
    --standalone \
    --nproc_per_node 1 \
    /workspace/data/train_eagle3_wrapper.py \
    --target-model-path mshojaei77/gpt-oss-120b \
    --train-data-path /tmp/regen_test_5_clean_output.jsonl \
    --build-dataset-num-proc 1 \
    --output-dir /tmp/dummy_train_output/ \
    --tp-size 1 \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.85 \
    --num-epochs 1 \
    --batch-size 1 \
    --learning-rate 2e-5 \
    --max-length 8192 \
    --chat-template gpt-oss \
    --cache-dir /tmp/dummy_cache \
    --dist-timeout 120 \
    --save-interval 9999 \
    --log-interval 1 \
    --max-num-steps 3 \
    --report-to tensorboard
