#!/bin/bash
# Test both data formats through SpecForge pipeline
# Run inside gptoss_server container:
#   docker exec -it gptoss_server bash /workspace/SpecForge/experiments/gptoss120b/test_data_format.sh

set -e
export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"

pip install yunchang -q 2>&1 | tail -1

python3 << 'PYEOF'
import sys

# Compat shims for sglang docker
try:
    from transformers.utils.generic import check_model_inputs
except ImportError:
    import transformers.utils.generic as _tug
    def check_model_inputs(func): return func
    _tug.check_model_inputs = check_model_inputs
try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    try:
        from transformers.modeling_utils import GradientCheckpointingLayer
        import types
        mod = types.ModuleType("transformers.modeling_layers")
        mod.GradientCheckpointingLayer = GradientCheckpointingLayer
        sys.modules["transformers.modeling_layers"] = mod
    except: pass
try:
    from transformers.masking_utils import create_causal_mask
except ImportError:
    import types
    mod = types.ModuleType("transformers.masking_utils")
    mod.create_causal_mask = lambda *a, **kw: None
    mod.create_sliding_window_causal_mask = lambda *a, **kw: None
    sys.modules["transformers.masking_utils"] = mod

from specforge.data.preprocessing import build_eagle3_dataset
from transformers import AutoTokenizer
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained("mshojaei77/gpt-oss-120b")

##############################################
# Test 1: SpecForge regenerated format
##############################################
print("=" * 60)
print("TEST 1: SpecForge regenerated format (user/assistant + reasoning_content)")
print("=" * 60)
try:
    ds1 = load_dataset("json", data_files="/tmp/regen_test_5_clean_output.jsonl")["train"]
    r1 = build_eagle3_dataset(dataset=ds1, tokenizer=tok, chat_template="gpt-oss", max_length=512, num_proc=1)
    print(f"SUCCESS: {len(r1)} samples")
    print(f"Tokens: {len(r1[0]['input_ids'])}, Loss tokens: {r1[0]['loss_mask'].sum().item()}")
    print(f"Decoded:\n{tok.decode(r1[0]['input_ids'])[:500]}")
except Exception as e:
    print(f"FAILED: {e}")

##############################################
# Test 2: Our training data format
##############################################
print()
print("=" * 60)
print("TEST 2: Our format (assistant_reasoning_effort/assistant_analysis/assistant_final)")
print("=" * 60)
try:
    ds2 = load_dataset("json", data_files="/tmp/our_train.jsonl")["train"].select(range(1))
    r2 = build_eagle3_dataset(dataset=ds2, tokenizer=tok, chat_template="gpt-oss", max_length=512, num_proc=1)
    print(f"SUCCESS: {len(r2)} samples")
    print(f"Tokens: {len(r2[0]['input_ids'])}, Loss tokens: {r2[0]['loss_mask'].sum().item()}")
    print(f"Decoded:\n{tok.decode(r2[0]['input_ids'])[:500]}")
except Exception as e:
    print(f"FAILED: {e}")
PYEOF
