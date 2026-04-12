#!/bin/bash
set -e

# GPU 6 = ROCR index 6
export ROCR_VISIBLE_DEVICES=6
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_DFLASH_SPEC_V2=1

export PYTHONPATH="/workspace/SpecForge:/workspace/SpecForge/benchmarks:$PYTHONPATH"

# Verify GPU
python3 -c "
import torch
n = torch.cuda.device_count()
for i in range(n):
    free, total = torch.cuda.mem_get_info(i)
    print(f'GPU {i}: {total/1e9:.1f}GB total, {free/1e9:.1f}GB free')
"

CKPT="/workspace/checkpoints/dflash/gpt-oss-120b/epoch_9_step_34000"

cd /app/sglang/python

python -m sglang.launch_server \
    --model-path mshojaei77/gpt-oss-120b \
    --speculative-algorithm DFLASH \
    --speculative-draft-model-path "$CKPT" \
    --speculative-num-draft-tokens 16 \
    --tp-size 1 \
    --dtype bfloat16 \
    --attention-backend aiter \
    --speculative-draft-attention-backend aiter \
    --mem-fraction-static 0.75 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30100
