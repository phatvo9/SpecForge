#!/bin/bash
set -e

# GPU 3=ROCR0, GPU 4=ROCR5
export ROCR_VISIBLE_DEVICES=0,5
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"

pip install openai-harmony accelerate datasets yunchang wandb tensorboard pydantic tqdm psutil numpy 2>&1 | tail -10

CKPT_DIR="/workspace/checkpoints-v2c/eagle3/gpt-oss-120b/epoch_7_step_24000"

NUM_GPUS=2
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
    --output-dir /workspace/checkpoints-v2d/eagle3/gpt-oss-120b/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.85 \
    --num-epochs 15 \
    --batch-size 2 \
    --draft-accumulation-steps 4 \
    --learning-rate 1e-5 \
    --max-length 8192 \
    --chat-template gpt-oss \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --save-interval 500 \
    --eval-interval 500 \
    --log-interval 50 \
    --report-to tensorboard
