#!/bin/bash
set -e

# GPU 6 = ROCR index 6
export ROCR_VISIBLE_DEVICES=6
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

export PYTHONPATH="/workspace/SpecForge:/workspace/SpecForge/benchmarks:$PYTHONPATH"

# Install missing deps
pip install openai-harmony accelerate datasets yunchang wandb tensorboard pydantic tqdm psutil numpy 2>&1 | tail -5

# Verify GPU
python3 -c "
import torch
n = torch.cuda.device_count()
for i in range(n):
    free, total = torch.cuda.mem_get_info(i)
    print(f'GPU {i}: {total/1e9:.1f}GB total, {free/1e9:.1f}GB free')
"

CKPT="/workspace/checkpoints-v3/eagle3/gpt-oss-120b/epoch_17_step_23748"

cd /workspace/SpecForge/benchmarks

python bench_eagle3.py \
    --model mshojaei77/gpt-oss-120b \
    --speculative-draft-model-path "$CKPT" \
    --port 30100 \
    --config-list 1,3,1,4 \
    --benchmark-list mtbench:50 \
    --attention-backend aiter \
    --mem-fraction-static 0.85 \
    --tp 1 \
    --output-dir /workspace/checkpoints-v3/eagle3/gpt-oss-120b/bench_results
