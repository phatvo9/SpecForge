#!/bin/bash
set -e

# GPU 3=ROCR0, GPU 4=ROCR5
export ROCR_VISIBLE_DEVICES=0,5
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"

pip install openai-harmony accelerate datasets yunchang tensorboard pydantic tqdm psutil numpy 2>&1 | tail -10

# Best v2c checkpoint
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
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints-v4/eagle3/gpt-oss-120b/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.85 \
    --num-epochs 9 \
    --batch-size 1 \
    --draft-accumulation-steps 4 \
    --learning-rate 2e-5 \
    --max-length 2048 \
    --ttt-length 7 \
    --chat-template gpt-oss \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --log-interval 50 \
    --report-to tensorboard
