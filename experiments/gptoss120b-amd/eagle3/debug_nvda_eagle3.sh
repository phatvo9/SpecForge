#!/bin/bash
set -e

export ROCR_VISIBLE_DEVICES=0
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

export PYTHONPATH="/workspace/SpecForge:$PYTHONPATH"

pip install openai-harmony accelerate datasets yunchang wandb tensorboard pydantic tqdm psutil numpy 2>&1 | tail -5

python3 << 'PYEOF'
import torch
import json
import sys
sys.path.insert(0, '/workspace/SpecForge')

# Apply compat patches
exec(open('/workspace/data/train_eagle3_wrapper.py').read().split('# Run the actual training script')[0])

from specforge.modeling.auto import AutoEagle3DraftModel, AutoDraftModelConfig

ckpt = "/workspace/data/nvda-eagle3-patched"

# Load config
config = AutoDraftModelConfig.from_file(f"{ckpt}/config.json")
print(f"Config: hidden_size={config.hidden_size}, draft_vocab_size={config.draft_vocab_size}")
print(f"Intermediate: {config.intermediate_size}, heads: {config.num_attention_heads}, kv_heads: {config.num_key_value_heads}")

# Load model
model = AutoEagle3DraftModel.from_pretrained(ckpt, torch_dtype=torch.bfloat16).cuda()

# Check weights
print("\nModel parameters:")
for name, param in model.named_parameters():
    has_nan = torch.isnan(param).any().item()
    has_inf = torch.isinf(param).any().item()
    print(f"  {name}: shape={list(param.shape)}, dtype={param.dtype}, nan={has_nan}, inf={has_inf}, mean={param.float().mean():.6f}, std={param.float().std():.6f}")

for name, buf in model.named_buffers():
    has_nan = torch.isnan(buf.float()).any().item()
    print(f"  [buffer] {name}: shape={list(buf.shape)}, dtype={buf.dtype}, nan={has_nan}")

# Quick forward test
print("\nForward test...")
seq_len = 10
# hidden_states: concat of 3 aux layers = hidden_size * 3
dummy_hidden = torch.randn(1, seq_len, 2880 * 3, dtype=torch.bfloat16).cuda()
dummy_input_ids = torch.randint(0, 1000, (1, seq_len)).cuda()
dummy_embeds = model.embed_tokens(dummy_input_ids)

with torch.no_grad():
    try:
        # Test forward
        out = model(hidden_states=dummy_hidden, inputs_embeds=dummy_embeds)
        has_nan = torch.isnan(out).any().item()
        print(f"Forward output: shape={out.shape}, nan={has_nan}, mean={out.float().mean():.6f}, std={out.float().std():.6f}")

        # Test compute_logits
        logits = model.compute_logits(out)
        has_nan = torch.isnan(logits).any().item()
        print(f"Logits: shape={logits.shape}, nan={has_nan}, mean={logits.float().mean():.6f}, std={logits.float().std():.6f}")

        # Test with actual-scale hidden states (not random noise)
        dummy_hidden2 = torch.randn(1, seq_len, 2880 * 3, dtype=torch.bfloat16).cuda() * 0.01
        out2 = model(hidden_states=dummy_hidden2, inputs_embeds=dummy_embeds)
        logits2 = model.compute_logits(out2)
        print(f"Small input logits: nan={torch.isnan(logits2).any().item()}, mean={logits2.float().mean():.6f}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Forward failed: {e}")

PYEOF
