#!/bin/bash
set -e
export ROCR_VISIBLE_DEVICES=1,2,3,0
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"
pip install openai-harmony accelerate datasets yunchang wandb==0.19.11 tensorboard pydantic tqdm psutil numpy 2>&1 | tail -3
NUM_GPUS=4
torchrun --standalone --nproc_per_node $NUM_GPUS \
    /workspace/SpecForge/experiments/minimax/dflash/train_dflash_wrapper.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-config-path /workspace/SpecForge/configs/minimax-m2.7-dflash-v4.json \
    --train-data-path /workspace/data/minimax2.7-spec-data/underfit_train.jsonl \
    --eval-data-path /workspace/data/minimax2.7-spec-data/underfit_eval.jsonl \
    --build-dataset-num-proc 4 \
    --output-dir /workspace/checkpoints/dflash/mem-test-ft/ \
    --tp-size 1 --target-model-backend sglang --sglang-attention-backend aiter \
    --sglang-mem-fraction-static 0.95 --sglang-context-length 6144 --sglang-max-total-tokens 6144 \
    --trust-remote-code --num-epochs 3 --batch-size 1 --accumulation-steps 4 \
    --learning-rate 2e-5 --warmup-ratio 0.0 --max-grad-norm 1.0 --max-length 6144 \
    --chat-template minimax-m2 --attention-backend sdpa --block-size 8 --num-anchors 500 \
    --loss-decay-gamma 4.0 --cache-dir /workspace/cache --dist-timeout 120 \
    --save-interval 99999 --eval-interval 50 --log-interval 10 --report-to tensorboard
