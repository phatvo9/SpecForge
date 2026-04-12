#!/bin/bash
set -e

# ROCR 0=Node2=GPU3, ROCR 1=Node3=GPU0, ROCR 5=Node7=GPU4
export ROCR_VISIBLE_DEVICES=0,1,5
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TORCHINDUCTOR_CACHE_DIR="/workspace/cache/compiled_kernels"

pip install openai-harmony accelerate datasets yunchang wandb tensorboard pydantic tqdm psutil numpy 2>&1 | tail -10

# z-lab pretrained DFlash for gpt-oss-120b (0.8B params, 8 layers, Qwen3-based)
# Original block_size=10, we finetune with block_size=16
# Config in output dir has block_size=16 override
# mask_token_id=200000 from z-lab config

NUM_GPUS=3
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/gptoss120b/dflash/train_dflash_wrapper.py \
    --target-model-path mshojaei77/gpt-oss-120b \
    --draft-config-path /workspace/checkpoints/dflash/gpt-oss-120b/zlab_pretrained/config.json \
    --resume \
    --train-data-path /workspace/data/gptoss-eagle3-train-all4-train.jsonl \
    --eval-data-path /workspace/data/gptoss-eagle3-train-all4-eval.jsonl \
    --output-dir /workspace/checkpoints/dflash/gpt-oss-120b/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.85 \
    --sglang-context-length 16000 \
    --num-epochs 13 \
    --batch-size 2 \
    --accumulation-steps 4 \
    --learning-rate 6e-4 \
    --warmup-ratio 0.04 \
    --max-grad-norm 1.0 \
    --max-length 16000 \
    --chat-template gpt-oss \
    --attention-backend sdpa \
    --block-size 16 \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --mask-token-id 200000 \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --save-interval 250 \
    --eval-interval 200 \
    --log-interval 50 \
    --report-to tensorboard
