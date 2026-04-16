#!/bin/bash
set -e

# GPU 6 only for benchmark (ROCR node 6)
export ROCR_VISIBLE_DEVICES=6
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"

pip install openai-harmony accelerate datasets yunchang pydantic tqdm psutil numpy typing_extensions 2>&1 | tail -5

# Patch MiniMax model to use typing_extensions.Unpack for Python 3.10
python3 -c "
import glob, re
files = glob.glob('/root/.cache/huggingface/**/modeling_minimax_m2.py', recursive=True)
for f in files:
    text = open(f).read()
    if 'from typing import Optional, Union, Unpack' in text:
        text = text.replace(
            'from typing import Optional, Union, Unpack',
            'from typing import Optional, Union\ntry:\n    from typing import Unpack\nexcept ImportError:\n    from typing_extensions import Unpack'
        )
        open(f, 'w').write(text)
        print(f'Patched {f}')
"


# Find latest checkpoint
CKPT_DIR="/workspace/dflash_checkpoint"
LATEST_CKPT=$(ls -d ${CKPT_DIR}/epoch_* 2>/dev/null | sort -t_ -k4 -n | tail -1)
echo "=== Using DFlash checkpoint: $LATEST_CKPT ==="

export DFLASH_CKPT="$LATEST_CKPT"
python3 /workspace/SpecForge/experiments/minimax/dflash/bench_dflash_offline.py
