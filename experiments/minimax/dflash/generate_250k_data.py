#!/usr/bin/env python3
"""Generate 250K training dataset for MiniMax-M2.7 DFlash from multiple sources."""

import hashlib
import json
import os
import random
from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = "/home/claudeuser/specforge_data/minimax2.7-spec-data"
EXISTING_PRETRAIN = os.path.join(OUTPUT_DIR, "pretrain.jsonl")
SEED = 42
random.seed(SEED)

# Target composition (~270K before dedup, ~250K after):
# 5x existing sources:
#   sharegpt_gpt4:      ~121K  (45% of 270K)
#   ultrachat_200k:     ~94K   (35% of 270K)
#   open-perfectblend:  ~54K   (20% of 270K)
# New source:
#   Magpie-Pro-300K:    ~20K
SOURCES = {
    "sharegpt_gpt4": {
        "hf_id": "shibing624/sharegpt_gpt4",
        "split": "train",
        "n_samples": 121_000,
        "conv_key": "conversations",  # already has role/content format
        "role_map": {"human": "user", "gpt": "assistant", "system": "system"},
    },
    "ultrachat": {
        "hf_id": "HuggingFaceH4/ultrachat_200k",
        "split": "train_sft",
        "n_samples": 94_000,
        "conv_key": "messages",
        "role_map": {"user": "user", "assistant": "assistant", "system": "system"},
    },
    "perfectblend": {
        "hf_id": "mlabonne/open-perfectblend",
        "split": "train",
        "n_samples": 54_000,
        "conv_key": "conversations",
        "role_map": {"human": "user", "gpt": "assistant", "system": "system"},
    },
    "magpie": {
        "hf_id": "Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered",
        "split": "train",
        "n_samples": 20_000,
        "conv_key": "conversations",
        "role_map": {"human": "user", "gpt": "assistant", "system": "system"},
    },
}


def normalize_conversation(conv_list, role_map):
    """Normalize conversation to [{"role": ..., "content": ...}] format."""
    normalized = []
    for turn in conv_list:
        if isinstance(turn, dict):
            # Handle {"role": ..., "content": ...} format
            if "role" in turn and "content" in turn:
                role = role_map.get(turn["role"], turn["role"])
                normalized.append({"role": role, "content": turn["content"]})
            # Handle {"from": ..., "value": ...} format (sharegpt style)
            elif "from" in turn and "value" in turn:
                role = role_map.get(turn["from"], turn["from"])
                normalized.append({"role": role, "content": turn["value"]})
            else:
                return None  # Unknown format
        else:
            return None
    return normalized


def conversation_hash(conv):
    """Create a hash of the first user message for deduplication."""
    for turn in conv:
        if turn["role"] == "user":
            text = turn["content"].strip()[:500]  # First 500 chars of first user msg
            return hashlib.md5(text.encode()).hexdigest()
    return None


def load_existing_hashes():
    """Load hashes from existing pretrain.jsonl to avoid duplicates."""
    hashes = set()
    if os.path.exists(EXISTING_PRETRAIN):
        print(f"Loading existing data hashes from {EXISTING_PRETRAIN}...")
        with open(EXISTING_PRETRAIN) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    h = conversation_hash(data["conversations"])
                    if h:
                        hashes.add(h)
                except:
                    continue
        print(f"  Found {len(hashes)} existing conversation hashes")
    return hashes


def sample_dataset(source_name, config, existing_hashes):
    """Sample and normalize conversations from a HF dataset."""
    print(f"\n=== Loading {source_name}: {config['hf_id']} ===")
    ds = load_dataset(config["hf_id"], split=config["split"])
    print(f"  Total samples in dataset: {len(ds)}")

    n_target = min(config["n_samples"], len(ds))
    indices = list(range(len(ds)))
    random.shuffle(indices)

    conversations = []
    seen_hashes = set()
    skipped_dup = 0
    skipped_format = 0
    skipped_short = 0

    for idx in indices:
        if len(conversations) >= n_target:
            break

        sample = ds[idx]
        conv_key = config["conv_key"]
        if conv_key not in sample or not sample[conv_key]:
            skipped_format += 1
            continue

        conv = normalize_conversation(sample[conv_key], config["role_map"])
        if conv is None or len(conv) < 2:
            skipped_format += 1
            continue

        # Skip very short conversations
        total_len = sum(len(t["content"]) for t in conv)
        if total_len < 100:
            skipped_short += 1
            continue

        # Dedup
        h = conversation_hash(conv)
        if h is None:
            skipped_format += 1
            continue
        if h in existing_hashes or h in seen_hashes:
            skipped_dup += 1
            continue

        seen_hashes.add(h)
        conversations.append(conv)

    print(f"  Sampled: {len(conversations)} | Skipped: dup={skipped_dup}, format={skipped_format}, short={skipped_short}")
    return conversations, seen_hashes


def main():
    existing_hashes = load_existing_hashes()
    all_hashes = set(existing_hashes)
    all_conversations = []

    for name, config in SOURCES.items():
        convs, new_hashes = sample_dataset(name, config, all_hashes)
        all_conversations.extend(convs)
        all_hashes.update(new_hashes)

    # Shuffle
    random.shuffle(all_conversations)
    print(f"\n=== Total conversations: {len(all_conversations)} ===")

    # Split: 99% train, 1% eval (but keep existing eval.jsonl separate)
    n_eval = 1000
    eval_convs = all_conversations[:n_eval]
    train_convs = all_conversations[n_eval:]

    # Write train
    train_path = os.path.join(OUTPUT_DIR, "pretrain_250k.jsonl")
    with open(train_path, "w") as f:
        for conv in train_convs:
            f.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")
    print(f"Written train: {len(train_convs)} samples -> {train_path}")

    # Write eval
    eval_path = os.path.join(OUTPUT_DIR, "eval_250k.jsonl")
    with open(eval_path, "w") as f:
        for conv in eval_convs:
            f.write(json.dumps({"conversations": conv}, ensure_ascii=False) + "\n")
    print(f"Written eval: {len(eval_convs)} samples -> {eval_path}")

    # Stats
    print(f"\nDataset size: {os.path.getsize(train_path) / 1e9:.2f} GB (train)")
    print("Done!")


if __name__ == "__main__":
    main()
