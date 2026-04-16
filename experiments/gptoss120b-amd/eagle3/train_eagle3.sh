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

# Pretrained EAGLE3 model path (mounted from host HF cache)
EAGLE3_CKPT="/root/.cache/huggingface/hub/models--lmsys--EAGLE3-gpt-oss-120b-bf16/snapshots/21e624039fdac60defaaa74344e03da5627a8678"

# Verify GPU mapping
python3 -c "
import torch
n = torch.cuda.device_count()
print(f'Visible GPUs: {n}')
for i in range(n):
    props = torch.cuda.get_device_properties(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f'  GPU {i}: {props.name}, {total/1e9:.1f}GB total, {free/1e9:.1f}GB free')
"

NUM_GPUS=3
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/data/train_eagle3_wrapper.py \
    --target-model-path mshojaei77/gpt-oss-120b \
    --ckpt-dir "$EAGLE3_CKPT" \
    --train-data-path /workspace/data/gptoss-eagle3-train.jsonl \
    --build-dataset-num-proc 16 \
    --output-dir /workspace/checkpoints/eagle3/gpt-oss-120b/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.85 \
    --num-epochs 10 \
    --batch-size 2 \
    --resume \
    --learning-rate 5e-5 \
    --max-length 8192 \
    --chat-template gpt-oss \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --log-interval 50 \
    --report-to tensorboard
