#!/usr/bin/env python3
"""Offline DFlash benchmark: measure acceptance length/rate for MiniMax-M2.7."""

import sys
import types

# Compat shims for older transformers in Docker image
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
        mod = types.ModuleType("transformers.modeling_layers")
        mod.GradientCheckpointingLayer = GradientCheckpointingLayer
        sys.modules["transformers.modeling_layers"] = mod
    except ImportError:
        pass
try:
    from transformers.masking_utils import create_causal_mask
except ImportError:
    mod = types.ModuleType("transformers.masking_utils")
    mod.create_causal_mask = lambda *a, **kw: None
    mod.create_sliding_window_causal_mask = lambda *a, **kw: None
    sys.modules["transformers.masking_utils"] = mod
from transformers import PreTrainedModel
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = set()
print("[compat] shims applied")

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.path.insert(0, "/workspace/SpecForge")
from specforge.modeling.draft.dflash import DFlashDraftModel, extract_context_feature

# Config
CKPT_DIR = os.environ.get("DFLASH_CKPT", "/workspace/dflash_checkpoint/epoch_4_step_250000")
TARGET_MODEL = "MiniMaxAI/MiniMax-M2.7"
MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0  # greedy
NUM_PROMPTS = 10

PROMPTS = [
    "Explain the concept of speculative decoding in large language models.",
    "Write a Python function to compute the Fibonacci sequence using dynamic programming.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the architecture of a transformer model step by step.",
    "Write a bash script that monitors disk usage and sends an alert when it exceeds 90%.",
    "Explain how mixture of experts (MoE) models work and their advantages.",
    "What is the difference between supervised and unsupervised learning?",
    "Write a SQL query to find the top 10 customers by total order value.",
    "Explain the CAP theorem in distributed systems.",
    "How does gradient descent work in training neural networks?",
]

def main():
    print(f"=== DFlash Offline Benchmark ===")
    print(f"Checkpoint: {CKPT_DIR}")
    print(f"Target: {TARGET_MODEL}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Temperature: {TEMPERATURE}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)

    # Load draft model
    print("Loading draft model...")
    draft_config = AutoConfig.from_pretrained(CKPT_DIR)
    draft_model = DFlashDraftModel.from_pretrained(
        CKPT_DIR, torch_dtype=torch.bfloat16
    ).cuda().eval()
    print(f"Draft model: {sum(p.numel() for p in draft_model.parameters()):,} params")
    print(f"Block size: {draft_model.block_size}")
    print(f"Target layer IDs: {draft_model.target_layer_ids}")
    print(f"Mask token ID: {draft_model.mask_token_id}")

    # Load target model
    print("Loading target model (this takes a while)...")
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        output_hidden_states=True,
    ).cuda().eval()
    print(f"Target model loaded")
    print()

    # Get stop token IDs
    eos_id = tokenizer.eos_token_id
    stop_ids = [eos_id] if eos_id is not None else []

    # Benchmark
    all_acceptance_lengths = []
    total_tokens = 0
    total_steps = 0
    total_time = 0

    for i, prompt in enumerate(PROMPTS[:NUM_PROMPTS]):
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()

        print(f"[{i+1}/{NUM_PROMPTS}] Prompt: {prompt[:60]}...")
        print(f"  Input tokens: {input_ids.shape[1]}")

        t0 = time.time()
        output_ids = draft_model.spec_generate(
            target=target_model,
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            stop_token_ids=stop_ids,
            temperature=TEMPERATURE,
        )
        elapsed = time.time() - t0

        new_tokens = output_ids.shape[1] - input_ids.shape[1]
        # Reconstruct acceptance lengths from the generate internals
        # We need to re-run to get them, but spec_generate doesn't return them
        # Instead, estimate from tokens generated vs steps taken
        # Each step produces at least 1 token (the verified one)
        # Total steps ≈ tokens / avg_acceptance_length

        output_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0

        print(f"  Generated: {new_tokens} tokens in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
        print(f"  Output: {output_text[:100]}...")
        print()

        total_tokens += new_tokens
        total_time += elapsed

    # We need acceptance lengths - let me patch spec_generate to return them
    print("=" * 60)
    print(f"=== Running with acceptance length tracking ===")
    print()

    # Monkey-patch to capture acceptance lengths
    acceptance_data = []

    orig_spec_generate = DFlashDraftModel.spec_generate

    @torch.inference_mode()
    def patched_spec_generate(self, target, input_ids, max_new_tokens, stop_token_ids, temperature):
        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        block_size = self.block_size

        output_ids = torch.full(
            (1, max_length + block_size), self.mask_token_id,
            dtype=torch.long, device=target.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)

        from transformers import DynamicCache
        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        output = target(
            input_ids, position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target, use_cache=True,
            logits_to_keep=1, output_hidden_states=True,
        )
        output_ids[:, :num_input_tokens] = input_ids

        from specforge.modeling.draft.dflash import sample
        output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature)
        target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)

        acceptance_lengths = []
        start = num_input_tokens
        while start < max_length:
            block_output_ids = output_ids[:, start:start + block_size].clone()
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                self(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[:, past_key_values_draft.get_seq_length():start + block_size],
                    past_key_values=past_key_values_draft,
                    use_cache=True, is_causal=False,
                )[:, -block_size + 1:, :]
            )
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)

            output = target(
                block_output_ids,
                position_ids=position_ids[:, start:start + block_size],
                past_key_values=past_key_values_target,
                use_cache=True, output_hidden_states=True,
            )
            posterior = sample(output.logits, temperature)
            acc_len = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1).sum(dim=1)[0].item()
            )
            output_ids[:, start:start + acc_len + 1] = block_output_ids[:, :acc_len + 1]
            output_ids[:, start + acc_len + 1] = posterior[:, acc_len]
            start += acc_len + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )[:, :acc_len + 1, :]
            acceptance_lengths.append(acc_len + 1)

            if stop_token_ids is not None and any(
                sid in output_ids[:, num_input_tokens:] for sid in stop_token_ids
            ):
                break

        acceptance_data.append(acceptance_lengths)
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        return output_ids

    DFlashDraftModel.spec_generate = patched_spec_generate

    for i, prompt in enumerate(PROMPTS[:NUM_PROMPTS]):
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()

        output_ids = draft_model.spec_generate(
            target=target_model, input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS, stop_token_ids=stop_ids,
            temperature=TEMPERATURE,
        )

        acc = acceptance_data[i]
        avg_acc = sum(acc) / len(acc) if acc else 0
        new_tokens = output_ids.shape[1] - input_ids.shape[1]
        print(f"[{i+1}] {prompt[:50]}...")
        print(f"  Tokens: {new_tokens} | Steps: {len(acc)} | Avg acceptance: {avg_acc:.2f} | Per-step: {acc[:20]}")
        print()

    # Summary
    print("=" * 60)
    all_acc = [a for seq in acceptance_data for a in seq]
    if all_acc:
        avg = sum(all_acc) / len(all_acc)
        total_tokens = sum(sum(seq) for seq in acceptance_data)
        total_steps = sum(len(seq) for seq in acceptance_data)
        print(f"SUMMARY:")
        print(f"  Total tokens generated: {total_tokens}")
        print(f"  Total verification steps: {total_steps}")
        print(f"  Avg acceptance length: {avg:.2f} tokens/step")
        print(f"  Max block size: {draft_model.block_size}")
        print(f"  Acceptance rate: {avg/draft_model.block_size*100:.1f}%")
        print(f"  Speedup estimate: {avg:.2f}x (vs 1 token/step baseline)")

        # Per-position acceptance
        max_bs = draft_model.block_size
        pos_counts = [0] * max_bs
        pos_totals = [0] * max_bs
        for seq in acceptance_data:
            for acc_len in seq:
                for p in range(max_bs):
                    pos_totals[p] += 1
                    if acc_len > p:
                        pos_counts[p] += 1
        print(f"\n  Per-position acceptance rate:")
        for p in range(max_bs):
            rate = pos_counts[p] / pos_totals[p] * 100 if pos_totals[p] > 0 else 0
            print(f"    Position {p+1}: {rate:.1f}%")


if __name__ == "__main__":
    main()
