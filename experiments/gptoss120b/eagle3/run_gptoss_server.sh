#!/bin/bash
set -e

export ROCR_VISIBLE_DEVICES=1
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

python3 -m sglang.launch_server \
    --model-path mshojaei77/gpt-oss-120b \
    --tp-size 1 \
    --attention-backend aiter \
    --mem-fraction-static 0.85 \
    --reasoning-parser gpt-oss \
    --tool-call-parser gpt-oss \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30300
