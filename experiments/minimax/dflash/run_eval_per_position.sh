#!/bin/bash
set -e

export ROCR_VISIBLE_DEVICES=6
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"

pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -3

export DFLASH_CKPT="/workspace/checkpoints/dflash/minimax-m2.7-v3c/epoch_9_step_60000"
export EVAL_DATA="/workspace/data/minimax2.7-spec-data/eval_reasoning.jsonl"
export TARGET_MODEL="MiniMaxAI/MiniMax-M2.7"
export MAX_SAMPLES=50

torchrun --standalone --nproc_per_node 1 /workspace/SpecForge/experiments/minimax/dflash/eval_wrapper.py
