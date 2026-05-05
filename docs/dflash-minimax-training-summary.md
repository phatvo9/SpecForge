# DFlash Training Summary for MiniMax-M2.7

## Overview

DFlash is a block diffusion speculative decoding method that trains a small draft model to predict blocks of consecutive tokens in parallel. Unlike autoregressive draft models (EAGLE3), DFlash predicts all tokens in a block simultaneously using hidden states captured from the target model's intermediate layers.

We trained DFlash draft models for MiniMax-M2.7 (62-layer MoE, ~10B active params, FP8) using SpecForge on AMD Instinct MI355X GPUs.

## Architecture Design

### Draft Model

The draft model is a Qwen3-based transformer with these key components:

- **Backbone**: Stack of Qwen3DFlashDecoderLayer modules with full attention
- **Feature Fusion (fc layer)**: Linear projection that combines hidden states from multiple target model layers
  - Input: concatenation of selected target hidden states `[batch, seq_len, num_target_layers * hidden_size]`
  - Output: `[batch, seq_len, hidden_size]`
- **Dual KV Attention**: Each attention layer attends to both context (target hidden states) and noise (draft embeddings)
- **Mask Token**: Special `<|MASK|>` token (ID 200054) for positions to be predicted

### How to Choose Architecture

| Parameter | Guidance | Our Choices |
|-----------|----------|-------------|
| **num_hidden_layers** | Paper recommends 1:1 ratio with num_target_layers (features). More layers = higher acc but more memory | 5 (v1-v2), 6 (v3,v5), 8 (v4,v4-d16) |
| **hidden_size** | Must match target model's hidden_size (uses target's embed_tokens + lm_head) | 3072 (MiniMax-M2.7) |
| **num_attention_heads / num_key_value_heads** | Match target or use smaller. GQA ratio matters for efficiency | 24 heads / 8 KV heads |
| **target_layer_ids** | Evenly spaced across target's layers. More features = richer signal but larger fc projection | [1,12,24,36,48,59] (6 from 62 layers) |
| **block_size** | 8 or 16. Larger = more tokens per step but harder to predict. Can't adapt post-training | 8 (v3,v4,v5), 16 (v1,v4-d16) |
| **model_type** | "qwen3" — standard for all DFlash models in SpecForge, SGLang inference is arch-agnostic | qwen3 |

**Key constraint**: `hidden_size` must match target because DFlash shares the target's embedding table and LM head. The draft model only trains the transformer layers + fc projection.

### Target Layer Selection

Target layer IDs determine which intermediate hidden states are captured during the target model's forward pass and fed to the draft model via the fc layer.

```
MiniMax-M2.7: 62 layers
6 features: [1, 12, 24, 36, 48, 59]  (evenly spaced)
5 features: [1, 16, 30, 44, 59]      (evenly spaced)
```

The `dflash_config.target_layer_ids` in the config can be set explicitly or left empty for auto-generation via `build_target_layer_ids(num_total_layers, num_features)`.

## Training Process

### How DFlash Training Works

1. **Target Forward Pass**: Run input through target model (MiniMax-M2.7 via SGLang), capture hidden states from selected layers
2. **Anchor Sampling**: Randomly sample `num_anchors` positions in the sequence where blocks start
3. **Block Construction**: For each anchor, create a block of `block_size` tokens: position 0 = real anchor token, positions 1-N = MASK tokens
4. **Draft Forward**: Draft model processes all blocks, predicting tokens at MASK positions using fused target hidden states as context
5. **Loss**: Weighted cross-entropy between draft predictions and actual next tokens
6. **Loss Decay**: Positions weighted by `exp(-(k-1)/gamma)` — earlier positions in block weighted more

### Loss Decay Gamma

The gamma parameter controls how quickly the loss weight decays across positions in a block:

| block_size | Recommended gamma | Rationale |
|-----------|-------------------|-----------|
| 8 | 4.0 - 5.0 | Smaller blocks, moderate decay |
| 16 | 7.0 - 10.0 | Larger blocks need slower decay to train later positions |

### Key Training Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `--num-anchors` | Random anchor positions per sequence | 500-800 (memory-dependent) |
| `--block-size` | Tokens per block | 8 or 16 |
| `--loss-decay-gamma` | Position weight decay | 4-10 (see above) |
| `--learning-rate` | Peak LR | 6e-4 (pretrain), 1e-5 to 5e-5 (finetune) |
| `--accumulation-steps` | Gradient accumulation | 4 |
| `--max-length` | Sequence truncation | 4096 (pretrain), 14000-15500 (long ctx ft) |
| `--sglang-mem-fraction-static` | GPU memory for target model + KV | 0.9 (note: aiter multiplies by 0.85 when context > 8192) |

### SGLang Backend Notes

- MiniMax-M2.7 is served via SGLang for target hidden state extraction
- FP8 quantization: model takes ~214GB at TP=1 per GPU (288GB MI355X)
- `aiter` attention backend multiplies `mem_fraction_static` by 0.85 when `context_length > 8192`
- `sglang-context-length` and `sglang-max-total-tokens` should match `max-length`
- TP=2 has NaN instability on certain samples; TP=1 is stable but memory-limited

### Memory Budget (TP=1, MI355X 288GB)

| Component | Size |
|-----------|------|
| MiniMax-M2.7 FP8 | ~214 GB |
| KV Cache (15K tokens) | ~3.5 GB |
| **Free for training** | **~68 GB** |
| Draft model (880M params BF16) | ~1.7 GB |
| Optimizer states (Adam, FSDP sharded) | ~3.5 GB |
| Activations (anchors-dependent) | remainder |

Anchors are the main memory variable. With 68GB free: ~600-800 anchors at 14-15K context works. Pretrain with 4K context can use 512 anchors comfortably.

## Experiments Summary

### Phase 1: Pretraining (800K public data)

| Version | Layers | block_size | Features | Data | LR | Best Eval Acc |
|---------|--------|-----------|----------|------|-----|---------------|
| v1 | 5 | 16 | 5 | 53K | 6e-4 | 18.6% |
| v2 (bs10) | 5 | 10 | 5 | 228K | 6e-4 | 43.0% |
| v3 | 6 | 8 | 6 | 278K | 6e-4 | 43.0% |
| v4 | 8 | 8 | 6 | 278K | 6e-4 | 46.0% |
| v4-1 | 8 | 8 | 6 | 800K | constant 5e-5 | **50.5%** |
| v5 | 6 | 8 | 6 (auto) | 800K | 8e-4 | 45.9% |
| v4-d16 | 8 | 16 | 6 | 800K | 6e-4 cosine | **38.5%** |

### Phase 2: Finetuning (25K synthesis reasoning data)

| Version | Base | block_size | gamma | LR | max_length | Best Eval Acc |
|---------|------|-----------|-------|-----|-----------|---------------|
| v4-1-ft | v4-1 (50.5%) | 8 | 4.0 | 2e-5 cosine | 6144 | 47.5% |
| v4-1-ft-con | v4-1 | 16* | 7.0 | constant 1e-8 | 20480 | failed (NaN, block mismatch) |
| v4-d16-ft | v4-d16 (38.5%) | 16 | 10.0 | 5e-5 cosine | 14000 | **34.3%** |
| v4-d8-ft-long | v4-1-ft (47.5%) | 8 | 5.0 | 5e-5 cosine + 5e-6 const | 14000-15500 | **47.8%** |

*Cannot adapt block_size=8 weights to block_size=16 — produces NaN (untrained positions).

### Key Learnings

1. **More layers help**: 8 layers (880M) > 6 layers (660M) > 5 layers (550M)
2. **More data helps**: 800K > 278K > 228K > 53K
3. **block_size=8 consistently outperforms block_size=16** on accuracy metrics (50.5% vs 38.5%)
4. **block_size can't be changed post-training** — positions 9-15 are untrained in bs=8 models, causing NaN if switched to bs=16
5. **1:1 ratio recommended**: Match draft layers to number of target features (paper recommendation)
6. **Synthesis data helps for long context** but eval acc on synthesis data is lower than general eval (harder distribution)
7. **Constant LR can squeeze extra gains** after cosine LR plateaus (47.3% -> 47.8%)
8. **TP=2 has NaN issues** on certain samples — TP=1 with DP is more stable
9. **Self-logit distillation**: Attempted but reverted due to FP8 weight loading issues in Docker image. Patch saved for future work.

## What We Customized

### Code Changes

1. **`scripts/train_dflash.py`**: Added NaN skip for training (zero grad on NaN loss) and NaN filtering for eval loss averaging
2. **`specforge/args.py`**: Added `--sglang-max-total-tokens` and `--sglang-max-running-requests` CLI args
3. **`train_dflash_wrapper.py`**: Compatibility wrapper for SGLang Docker image (transformers shims, SGLang kwargs filtering)
4. **`train_dflash_wrapper_v4_1_ft_con.py`**: Constant LR wrapper (patches BF16Optimizer with WarmupConstantLRBF16Optimizer)
5. **MiniMax SGLang model patch**: Added `set_dflash_layers_to_capture()` method to `/root/phat/official_sglang/python/sglang/srt/models/minimax_m2.py`

### Infrastructure

- Docker image: `lmsysorg/sglang:v0.5.10.post1-rocm700-mi35x`
- Separate docker-compose files per experiment (unique container names)
- Checkpoint management: copy weights without training_state.pt for fresh optimizer on finetune
- TensorBoard logging preserved across runs when requested

## HuggingFace Repos (Private)

| Repo | Description | Best Acc |
|------|-------------|----------|
| `phatv9/MiniMax-M2.7-DFlash-v4` | v4/v4-1 best (block_size=8) | 50.5% |
| `phatv9/MiniMax-M2.7-DFlash-v4-d16-new` | v4-d16 pretrained (block_size=16) | 38.5% |
| `phatv9/MiniMax-M2.7-DFlash-v4-d16-ft` | v4-d16 finetuned | 34.3% |
| `phatv9/MiniMax-M2.7-DFlash-v4-d8-ft-long` | v4-d8 long context finetuned | 47.8% |
| `phatv9/MiniMax-M2.7-DFlash-v5` | v5 (6 layers, auto features) | 45.9% |
| `phatv9/minimax2.7-spec-data-800k` | 800K training data | - |
| `phatv9/minimax2.7-17k-synthesis` | 17K synthesis + eval | - |

## Data

### Pretraining Data (800K)
- **Sources**: ShareGPT, Ultrachat, Open-PerfectBlend, Magpie-Pro
- **Statistics**: 99.8% under 4K tokens (very short sequences)
- **Path**: `/home/claudeuser/specforge_data/minimax2.7-spec-data/pretrain_800k.jsonl`

### Synthesis Data (25K)
- **Source**: MiniMax-M2.7 generated reasoning data
- **Statistics**: Median 5.2K tokens, 39.3% exceeds 8K, 11.3% exceeds 16K
- **Split**: 24,876 train / 500 eval (>1K tokens, no overlap)
- **Path**: `/home/claudeuser/specforge_data/minimax2.7-spec-data/synthesis_25k_train.jsonl`

## Checkpoint Paths

```
/data3/phat/specforge/dflash/minimax-m2.7/dflash/
  minimax-m2.7-v4-d16/          # v4-d16 pretrain checkpoints
  minimax-m2.7-v4-d16-ft/       # v4-d16 finetune checkpoints
  minimax-m2.7-v4-d8-ft-long/   # v4-d8 long context finetune
  minimax-m2.7-v4-1-ft/         # v4-1-ft checkpoints
```
