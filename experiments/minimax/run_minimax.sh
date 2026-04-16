#!/bin/bash
set -e

# GPUs 0,3,4,6 = ROCR 1,0,5,6
export ROCR_VISIBLE_DEVICES=1,0,5,6
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

# Verify GPU
python3 -c "
import torch
n = torch.cuda.device_count()
for i in range(n):
    free, total = torch.cuda.mem_get_info(i)
    print(f'GPU {i}: {total/1e9:.1f}GB total, {free/1e9:.1f}GB free')
"

cd /app/sglang/python

python3 -m sglang.launch_server \
    --model MiniMaxAI/MiniMax-M2.7 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --reasoning-parser minimax-append-think \
    --tool-call-parser minimax-m2 \
    --attention-backend aiter \
    --mem-fraction-static 0.85 \
    --host 0.0.0.0 \
    --port 30200
