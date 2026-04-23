#!/usr/bin/env python3
"""Per-position accuracy eval — runs inside train_dflash_wrapper.py context."""

import os
import sys
import math
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer
from datasets import load_dataset

from specforge.core.dflash import OnlineDFlashModel, create_dflash_sdpa_mask
from specforge.data import build_eagle3_dataset
from specforge.distributed import init_distributed, get_tp_group, get_dp_group
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.dflash_target_model import get_dflash_target_model
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.args import SGLangBackendArgs
from specforge.utils import print_on_rank0

CKPT = os.environ.get("DFLASH_CKPT")
EVAL_DATA = os.environ.get("EVAL_DATA")
TARGET_MODEL = os.environ.get("TARGET_MODEL", "MiniMaxAI/MiniMax-M2.7")
BLOCK_SIZE = 8
NUM_ANCHORS = 64
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "30"))


def main():
    init_distributed(timeout=120, tp_size=1)

    print_on_rank0(f"Checkpoint: {CKPT}")
    print_on_rank0(f"Eval data: {EVAL_DATA}")

    # Load draft
    draft_config = AutoConfig.from_pretrained(CKPT)
    draft_config._attn_implementation = "sdpa"
    draft_model = DFlashDraftModel(draft_config).cuda().to(torch.bfloat16)
    loaded = DFlashDraftModel.from_pretrained(CKPT, torch_dtype=torch.bfloat16)
    draft_model.load_state_dict(loaded.state_dict())
    del loaded
    draft_model.eval()
    print_on_rank0(f"Draft: {sum(p.numel() for p in draft_model.parameters()):,} params, bs={draft_model.block_size}")

    # Load target via sglang
    target_model = get_dflash_target_model(
        pretrained_model_name_or_path=TARGET_MODEL,
        backend="sglang",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    target_model.set_capture_layers(draft_model.target_layer_ids)

    # Load embeddings/head
    target_components = TargetEmbeddingsAndHead.from_pretrained(
        TARGET_MODEL, device="cuda", trust_remote_code=True
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, trust_remote_code=True)
    mask_token_id = draft_model.mask_token_id or 200054

    # Eval data
    eval_dataset = load_dataset("json", data_files=EVAL_DATA)["train"]
    eval_processed = build_eagle3_dataset(
        dataset=eval_dataset, tokenizer=tokenizer,
        chat_template="minimax-m2", max_length=4096,
    )
    min_loss_tokens = 2 * BLOCK_SIZE
    eval_processed = eval_processed.filter(lambda x: x["loss_mask"].sum() >= min_loss_tokens)
    print_on_rank0(f"Eval samples: {len(eval_processed)} (using {MAX_SAMPLES})")

    bs = BLOCK_SIZE
    pos_correct = [0] * bs
    pos_total = [0] * bs

    with torch.no_grad():
        for idx in range(min(len(eval_processed), MAX_SAMPLES)):
          try:
            data = eval_processed[idx]
            input_ids = torch.tensor(data["input_ids"]).reshape(1, -1).cuda()
            attention_mask = torch.tensor(data["attention_mask"]).reshape(1, -1).cuda()
            loss_mask = torch.tensor(data["loss_mask"]).reshape(1, -1).float().cuda()

            target_output = target_model.generate_dflash_data(input_ids, attention_mask, loss_mask)
            hidden_states = target_output.hidden_states.cuda()

            bsz, seq_len = input_ids.shape
            device = input_ids.device

            max_anchor = max(seq_len - bs, 0)
            valid = loss_mask[:, :max_anchor + 1] > 0.5
            valid_counts = valid.sum(dim=1)
            max_n = min(NUM_ANCHORS, int(valid_counts.max().item()) - 1)
            if max_n <= 0:
                continue

            indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0)
            masked_indices = torch.where(valid, indices, torch.tensor(seq_len + 1, device=device))
            random_vals = torch.rand(bsz, max_anchor + 1, device=device)
            random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))
            _, sorted_idx = random_vals.sort(dim=1)
            gathered = torch.gather(masked_indices, 1, sorted_idx)
            anchors = gathered[:, :max_n].sort(dim=1).values
            keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(max=max_n)
            anchors = torch.where(keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device))

            n = anchors.shape[1]
            noise_ids = torch.full((bsz, n * bs), mask_token_id, dtype=torch.long, device=device)
            block_starts = (torch.arange(n, device=device) * bs).unsqueeze(0).expand(bsz, -1)
            valid_anchors = anchors.clamp(0, seq_len - 1)
            anchor_tokens = torch.gather(input_ids, 1, valid_anchors)
            batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
            noise_ids[batch_idx, block_starts] = torch.where(keep_mask, anchor_tokens,
                torch.tensor(mask_token_id, dtype=torch.long, device=device))
            noise_embedding = target_components.embed_tokens(noise_ids)

            context_pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            offsets = torch.arange(bs, device=device).view(1, 1, -1)
            draft_pos = (anchors.unsqueeze(-1) + offsets).view(bsz, -1)
            full_pos = torch.cat([context_pos, draft_pos], dim=1)

            attn_mask = create_dflash_sdpa_mask(anchors, keep_mask, seq_len, bs, device)

            output_hidden = draft_model(
                position_ids=full_pos, noise_embedding=noise_embedding,
                target_hidden=hidden_states, attention_mask=attn_mask,
            )
            logits = target_components.lm_head(output_hidden)

            label_offsets = torch.arange(0, bs, device=device).view(1, 1, -1)
            label_indices = anchors.unsqueeze(-1) + label_offsets
            valid_label_mask = label_indices < seq_len
            safe_label_indices = label_indices.clamp(max=seq_len - 1)
            target_ids = torch.gather(input_ids.unsqueeze(1).expand(-1, n, -1), 2, safe_label_indices)

            weight_mask = keep_mask.unsqueeze(-1).expand(-1, -1, bs).float()
            weight_mask = weight_mask * valid_label_mask.float()
            pos_in_block = torch.arange(bs, device=device).view(1, 1, -1)
            weight_mask = weight_mask * (pos_in_block > 0).float()
            orig_loss_mask = torch.gather(loss_mask.unsqueeze(1).expand(-1, n, -1), 2, safe_label_indices)
            weight_mask = weight_mask * orig_loss_mask

            # logits is [bsz, n*bs, vocab] — reshape to [bsz, n, bs, vocab]
            logits_reshaped = logits.view(bsz, n, bs, -1)
            pred_ids = torch.argmax(logits_reshaped, dim=-1)  # [bsz, n, bs]
            for p in range(1, bs):
                mask_p = weight_mask[:, :, p]
                correct_p = (pred_ids[:, :, p] == target_ids[:, :, p]).float() * mask_p
                pos_correct[p] += correct_p.sum().item()
                pos_total[p] += mask_p.sum().item()

            if (idx + 1) % 10 == 0:
                print_on_rank0(f"  Processed {idx + 1}/{min(len(eval_processed), MAX_SAMPLES)}")
          except Exception as e:
            print_on_rank0(f"  Skipped sample {idx}: {e}")
            torch.cuda.empty_cache()
            continue

    print_on_rank0(f"\n{'='*50}")
    print_on_rank0(f"Per-position accuracy (block_size={bs}):")
    print_on_rank0(f"{'Position':<10} {'Accuracy':<10}")
    print_on_rank0("-" * 20)
    for p in range(1, bs):
        acc = pos_correct[p] / pos_total[p] * 100 if pos_total[p] > 0 else 0
        print_on_rank0(f"  {p:<10} {acc:.1f}%")
    total_c = sum(pos_correct[1:])
    total_t = sum(pos_total[1:])
    overall = total_c / total_t * 100 if total_t > 0 else 0
    print_on_rank0(f"\n  Overall: {overall:.1f}%")


if __name__ == "__main__":
    main()
