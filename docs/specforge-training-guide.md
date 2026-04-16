# SpecForge Training Guide

## Overview

SpecForge trains **speculative decoding draft models** (EAGLE3, DFlash) that predict what a large target model will output. At inference time, the draft model generates multiple tokens in parallel, and the target model validates them — accepted tokens are committed, rejected ones discarded. This achieves 1.5-4x inference speedup.

---

## 1. Data Generation

### What the data looks like

Training data consists of **conversations** in ShareGPT-like format:

```json
{
  "conversations": [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "The answer is 4."}
  ]
}
```

Some models use additional roles:
- `system` — system prompt
- `assistant_reasoning_effort` — reasoning level (gpt-oss specific)
- `assistant_analysis` / `assistant_final` — chain-of-thought channels (gpt-oss)
- `reasoning_content` — thinking blocks (MiniMax, Qwen3)

### How conversations become training samples

```
Raw conversation
    │
    ▼
Chat Template (e.g., "qwen", "gpt-oss", "minimax-m2")
    │  Converts roles → special tokens
    │  e.g., "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>"
    ▼
Tokenizer
    │  Text → token IDs
    ▼
Loss Mask Generator
    │  Marks which tokens the draft model should learn to predict
    │  loss_mask = 1 for assistant tokens, 0 for user/system tokens
    ▼
Training Sample: { input_ids, loss_mask, attention_mask }
```

### Data for pretraining vs fine-tuning

| Stage | Data Source | Responses By | Temperature |
|-------|-----------|-------------|-------------|
| **Pretraining** | Public datasets (ShareGPT, UltraChat, PerfectBlend) | Original (ChatGPT, Llama, etc.) | N/A |
| **Fine-tuning** | Same prompts, regenerated responses | Target model | 0 or 0.8 |

**Why regenerate?** The draft model learns to predict the *target model's* token distribution. Training on the target model's own responses teaches it the target's specific patterns.

### Regeneration process

1. Extract user/system messages from source datasets
2. Send to target model server (e.g., sglang with `temperature=0`)
3. Collect responses
4. Format as conversations
5. Split into train/eval

---

## 2. EAGLE3 Training

### Architecture

```
Target Model (frozen, e.g., gpt-oss-120b)
    │
    ├── Layer 1 hidden states  ──┐
    ├── Layer N/2 hidden states ──┼── Concatenate → (batch, seq_len, 3 × hidden_size)
    ├── Layer N-3 hidden states ──┘
    │
    └── Final logits → (batch, seq_len, vocab_size)

Draft Model (trainable, ~500M params)
    │
    ├── FC projection: 3H → H
    ├── 1 Transformer layer (attention + MLP)
    ├── RMSNorm
    └── LM Head → (batch, seq_len, draft_vocab_size)
```

### Key components

| Component | What it does |
|-----------|-------------|
| **Aux hidden states** | 3 layers from target model (early, mid, late) concatenated into 3×H vector |
| **FC layer** | Projects 3×H → H to feed into transformer |
| **Embeddings** | From target model, frozen during training |
| **LM Head** | Maps hidden → draft_vocab logits; frozen or trainable |
| **Vocab mapping (t2d/d2t)** | Maps between full vocab (e.g., 200K) and draft vocab (e.g., 32K) |

### Online training loop

```
For each batch:
  ┌─────────────────────────────────────────────┐
  │ 1. TARGET MODEL (frozen)                    │
  │    Input: input_ids, attention_mask          │
  │    Output:                                   │
  │      • aux_hidden_states (3 layers concat)   │
  │      • target_logits (full vocab)            │
  │      • loss_mask                             │
  │    Settings: temperature=0, max_new_tokens=1 │
  └─────────────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────┐
  │ 2. DRAFT MODEL (trainable)                  │
  │    TTT Loop (ttt_length=7 iterations):      │
  │      a. Embed current input tokens          │
  │      b. Backbone(embed + projected_hidden)  │
  │      c. Compute draft logits                │
  │      d. Loss = -Σ target_p × log(draft_p)  │
  │      e. Accuracy = argmax match             │
  │      f. Shift sequence left by 1 token      │
  │                                             │
  │    Returns: 7 losses + 7 accuracies         │
  └─────────────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────┐
  │ 3. BACKWARD + UPDATE                        │
  │    Weighted loss = Σ 0.8^i × loss[i]        │
  │    (Earlier positions weighted higher)       │
  │    Gradient accumulation → optimizer step    │
  └─────────────────────────────────────────────┘
```

### Loss function: LogSoftmaxLoss

The draft model learns to match the **target model's output distribution**, not just the correct token:

```
loss = -Σ[ position_mask × target_p × log_softmax(draft_logits) ]
```

Where:
- `target_p = softmax(target_logits)` — target model's probability distribution
- `draft_logits` — draft model's raw logits
- `position_mask` — 1 for tokens that should be predicted (from loss_mask)

This is a **KL divergence** objective: the draft learns the full distribution, not just the argmax.

### Metrics

| Metric | Description | Good value |
|--------|------------|------------|
| `train/acc_0` | Accuracy at position 0 (next token) | > 0.80 |
| `train/acc_1` | Accuracy at position 1 (2 tokens ahead) | > 0.78 |
| ... | ... | ... |
| `train/acc_6` | Accuracy at position 6 (7 tokens ahead) | > 0.70 |
| `train/ploss_0` | Loss at position 0 | < 1.0 |
| `eval/acc_0` | Eval accuracy (should track train) | close to train |

**TTT (Test-Time Training) length**: Number of unrolled positions. `ttt_length=7` means the draft predicts 7 tokens ahead. Each position gets its own loss and accuracy metric.

### Offline vs Online

| Aspect | Online | Offline |
|--------|--------|---------|
| Target model | Loaded in GPU memory during training | Pre-computed, stored on disk |
| Data format | Raw conversations | Pre-extracted hidden states + logits |
| GPU usage | Target + Draft model both in memory | Draft model only |
| Flexibility | Can change data on the fly | Fixed dataset |
| Speed | Slower (target forward pass each step) | Faster (no target inference) |
| Typical use | SGLang backend, 1 target + N draft GPUs | Large-scale distributed |

---

## 3. DFlash Training

### Architecture

DFlash uses a **block diffusion** approach — instead of predicting tokens one-by-one, it predicts a **block of tokens in parallel**.

```
Target Model (frozen)
    │
    ├── Layer 1 hidden states  ──┐
    ├── Layer N/4 hidden states ──┤
    ├── Layer N/2 hidden states ──┼── Concatenate → context features
    ├── Layer 3N/4 hidden states ──┤
    └── Layer N-3 hidden states ──┘

Draft Model (trainable, Qwen3-based, ~0.8B)
    │
    ├── FC: N_layers × H → H (project context)
    ├── Hidden Norm
    ├── K Decoder Layers (cross-attend to context + self-attend within block)
    ├── RMSNorm
    └── Target model's LM Head (frozen) → logits
```

### Key differences from EAGLE3

| Aspect | EAGLE3 | DFlash |
|--------|--------|--------|
| **Prediction** | Token-by-token, unrolled T steps | Block of B tokens in parallel |
| **Draft layers** | 1 transformer layer | 5-8 layers (deeper) |
| **Attention** | Causal (standard) | Block-wise: context + bidirectional intra-block |
| **LM Head** | Own head (draft_vocab) | Uses target's head (full vocab) |
| **Aux layers** | 3 (early, mid, late) | 5+ (evenly spaced) |
| **Loss** | KL divergence per position | Cross-entropy with exponential decay |
| **Mask token** | Not used | MASK token for unfilled block positions |
| **Architecture** | LLaMA-based | Qwen3-based |

### Online training loop

```
For each batch:
  ┌─────────────────────────────────────────────┐
  │ 1. TARGET MODEL                             │
  │    Extract hidden states from 5 layers      │
  │    Concatenate → context features           │
  └─────────────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────┐
  │ 2. ANCHOR SAMPLING                          │
  │    Randomly pick N anchor positions         │
  │    (where loss_mask > 0)                    │
  │    Each anchor starts a prediction block    │
  └─────────────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────┐
  │ 3. PREPARE NOISE INPUT                      │
  │    For each anchor at position p:           │
  │    Block = [token_p, MASK, MASK, ..., MASK] │
  │    Size = block_size (e.g., 16)             │
  │    Draft must fill in the MASKs             │
  └─────────────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────┐
  │ 4. DRAFT MODEL (all blocks in parallel)     │
  │    Attention: each block sees               │
  │      • Context tokens (up to anchor pos)    │
  │      • Other tokens in same block (bidir)   │
  │      • NOT other blocks                     │
  │                                             │
  │    Output: logits for each block position   │
  └─────────────────────────────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────────┐
  │ 5. LOSS + ACCURACY                          │
  │    Labels = actual tokens at pos p+1..p+B   │
  │    Weight = decay × loss_mask × valid_mask  │
  │    decay = exp(-(k-1)/gamma) per position   │
  │    Loss = weighted cross-entropy            │
  └─────────────────────────────────────────────┘
```

### Loss decay

Positions further from the anchor are harder to predict, so they're weighted less:

```
weight[k] = exp(-(k-1) / gamma)

gamma=7 (block_size=16):  pos1=1.00, pos2=0.87, pos5=0.57, pos10=0.28, pos15=0.13
gamma=5 (block_size=10):  pos1=1.00, pos2=0.82, pos5=0.45, pos9=0.20
```

### Metrics

| Metric | Description |
|--------|------------|
| `train/loss` | Weighted cross-entropy over all blocks |
| `train/accuracy` | Token-match rate on valid block positions |
| `eval/loss` | Same on held-out data |
| `eval/accuracy` | Same on held-out data |

---

## 4. Training Configurations

### Typical EAGLE3 config

```bash
torchrun --nproc_per_node 3 scripts/train_eagle3.py \
    --target-model-path MiniMaxAI/MiniMax-M2.7 \
    --draft-model-config configs/minimax-m2.7-eagle3.json \
    --train-data-path data/pretrain.jsonl \
    --target-model-backend sglang \
    --num-epochs 9 \
    --batch-size 1 \
    --draft-accumulation-steps 4 \
    --learning-rate 2e-5 \
    --max-length 2048 \
    --ttt-length 7 \
    --chat-template minimax-m2 \
    --save-interval 1000 \
    --report-to tensorboard
```

### Typical DFlash config

```bash
torchrun --nproc_per_node 3 scripts/train_dflash.py \
    --target-model-path mshojaei77/gpt-oss-120b \
    --draft-config-path configs/gpt-oss-120b-dflash.json \
    --train-data-path data/train.jsonl \
    --target-model-backend sglang \
    --num-epochs 10 \
    --batch-size 2 \
    --accumulation-steps 4 \
    --learning-rate 6e-4 \
    --max-length 16000 \
    --block-size 16 \
    --num-anchors 512 \
    --loss-decay-gamma 7.0 \
    --attention-backend sdpa \
    --save-interval 250 \
    --report-to tensorboard
```

### Two-stage training recipe (recommended)

1. **Pretrain** (9 epochs): Mixed public data (ShareGPT 45%, UltraChat 35%, PerfectBlend 20%). Teaches general language patterns.

2. **Fine-tune** (6 epochs): Regenerated data from target model at temp=0. Specializes draft to match target's specific distribution.

---

## 5. Key Concepts

### Draft vocab mapping

The draft model uses a **reduced vocabulary** (e.g., 32K instead of 200K) for efficiency. The mapping is built from token frequencies in the training data:

- `t2d[i]` — target token index for draft token i
- `d2t[i]` — boolean: is target token i in the draft vocab?

During loss computation, only draft-vocab logits are compared. At inference, the mapping converts back to full vocab.

### Aux hidden state layer selection

EAGLE3 concatenates hidden states from 3 target model layers:

```
eagle_aux_hidden_state_layer_ids = [early, middle, late]
```

Typical pattern: `[1, num_layers//2, num_layers-4]`

| Model | Layers | Aux IDs |
|-------|--------|---------|
| gpt-oss-120b (36L) | [1, 17, 33] | early, mid, late |
| MiniMax-M2.7 (62L) | [1, 30, 58] | early, mid, late |

### Acceptance length → speedup

At inference, higher acceptance length = more tokens accepted per verification step = higher speedup:

| Accept Length | Approximate Speedup |
|--------------|-------------------|
| 1.0 | 1.0x (no benefit) |
| 2.0 | 1.5-1.8x |
| 3.0 | 2.0-2.5x |
| 4.0 | 2.5-3.0x |
| 5.0+ | 3.0x+ |

Training accuracy correlates with acceptance length:
- `acc_0 > 0.80` → good first-token prediction
- `acc_0..acc_6` all > 0.70 → strong multi-token prediction → high acceptance length
