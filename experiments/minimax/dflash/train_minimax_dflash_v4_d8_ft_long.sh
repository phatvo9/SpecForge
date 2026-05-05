#!/bin/bash
set -e

# GPUs 0,3,6 (ROCR 1,0,6)
export ROCR_VISIBLE_DEVICES=1,0,6
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
export TORCHINDUCTOR_CACHE_DIR="/workspace/cache/compiled_kernels"

pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -5

# MiniMax-M2.7 DFlash v4-d8-ft-long: continue v4-1-ft with long context
# Init from v4-1-ft final checkpoint (block_size=8, epoch_15_step_77850)
# Long context ft with block_size=8, gamma=5.0, 3 GPUs DP=3

NUM_GPUS=3
TP_SIZE=1

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper_v4_1_ft_con.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-config-path /workspace/SpecForge/configs/minimax-m2.7-dflash-v4.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/synthesis_25k_train.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/synthesis_25k_eval.jsonl \
    --build-dataset-num-proc 40 \
    --output-dir /workspace/checkpoints/dflash/minimax-m2.7-v4-d8-ft-long/ \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.9 \
    --sglang-context-length 14000 \
    --sglang-max-total-tokens 14000 \
    --trust-remote-code \
    --num-epochs 10 \
    --batch-size 1 \
    --accumulation-steps 4 \
    --learning-rate 5e-6 \
    --warmup-ratio 0.0 \
    --max-grad-norm 1.0 \
    --max-length 14000 \
    --chat-template minimax-m2 \
    --attention-backend sdpa \
    --block-size 8 \
    --num-anchors 800 \
    --loss-decay-gamma 5.0 \
    --cache-dir /workspace/cache \
    --dist-timeout 120 \
    --resume \
    --save-interval 5000 \
    --eval-interval 500 \
    --log-interval 100 \
    --report-to tensorboard
