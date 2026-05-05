# MiniMax-M2.7 DFlash Experiment Results

## Model Architecture

| Config | Layers | block_size | Features (target_layer_ids) | Params |
|--------|--------|-----------|---------------------------|--------|
| v4 (d8) | 8 | 8 | [1,12,24,36,48,59] (6) | ~880M |
| v4-d16 | 8 | 16 | [1,12,24,36,48,59] (6) | ~880M |

Both use: hidden_size=3072, 24 heads, 8 KV heads, intermediate_size=8192, model_type=qwen3

---

## block_size=8 (d8) Pipeline

### Phase 1: Pretrain (v4-1)

| Setting | Value |
|---------|-------|
| Data | 800K public (ShareGPT, Ultrachat, PerfectBlend, Magpie) |
| Config | minimax-m2.7-dflash-v4.json |
| LR | constant 5e-5 (via wrapper) |
| Epochs | ~2 (stopped at best) |
| max_length | 4096 |
| anchors | 512 |
| gamma | 4.0 |
| GPUs | 5 (TP=1, DP=5) |
| block_size | 8 |

| Metric | Train | Eval |
|--------|-------|------|
| Loss | ~1.5-2.5 | ~2.1 |
| **Acc** | ~45-55% | **50.5%** |

HF: `phatv9/MiniMax-M2.7-DFlash-v4` (epoch_1_step_290000)

### Phase 2: Finetune (v4-1-ft)

| Setting | Value |
|---------|-------|
| Init from | v4-1 step 370K (eval acc 50.5%) |
| Data | 25K synthesis reasoning (24.8K train / 500 eval) |
| LR | 2e-5 cosine |
| Epochs | 15 |
| max_length | 6144 |
| anchors | 500 |
| gamma | 4.0 |
| GPUs | 4 (TP=1, DP=4) |

| Metric | Train | Eval |
|--------|-------|------|
| Loss | ~2.0-3.0 | ~2.5 |
| **Acc** | ~35-50% | **47.5%** |

HF: `phatv9/MiniMax-M2.7-DFlash-v4` (v4-1-ft FINAL epoch_15_step_77850)

Note: Eval acc dropped from 50.5% (general data) to 47.5% (synthesis data) because synthesis data is harder (longer reasoning sequences).

### Phase 3: Long-Context Finetune (v4-d8-ft-long)

#### Run 1: Cosine LR

| Setting | Value |
|---------|-------|
| Init from | v4-1-ft final (eval acc 47.5%) |
| Data | 25K synthesis |
| LR | 1e-5 cosine → 5e-5 cosine |
| Epochs | 25 + 10 |
| max_length | 15500 → 14000 |
| anchors | 630 → 800 |
| gamma | 5.0 |
| GPUs | 3 (TP=1, DP=3) |

| Metric | Train | Eval |
|--------|-------|------|
| Loss | ~1.6-2.5 | ~2.32 |
| **Acc** | ~44-60% | **47.7%** |

#### Run 2: Constant LR (continuation)

| Setting | Value |
|---------|-------|
| Init from | d8-ft-long cosine final (eval acc 47.7%) |
| LR | constant 5e-6 |
| Epochs | 10 |
| max_length | 14000 |
| anchors | 800 |
| gamma | 5.0 |

| Metric | Train | Eval |
|--------|-------|------|
| Loss | ~1.6-2.5 | ~2.31 |
| **Acc** | ~44-60% | **47.8%** |

HF: `phatv9/MiniMax-M2.7-DFlash-v4-d8-ft-long` (FINAL epoch_10_step_82910)

---

## block_size=16 (d16) Pipeline

### Phase 1: Pretrain (v4-d16)

| Setting | Value |
|---------|-------|
| Data | 800K public |
| Config | minimax-m2.7-dflash-v4-d16.json |
| LR | 6e-4 cosine |
| Epochs | 5 (stopped at epoch 4, 33%) |
| max_length | 4096 |
| anchors | 512 |
| gamma | 7.0 |
| GPUs | 5 (TP=1, DP=5) |
| block_size | 16 |

| Metric | Train | Eval |
|--------|-------|------|
| Loss | ~1.5-2.5 | ~2.49 |
| **Acc** | ~37-56% | **38.5%** |

HF: `phatv9/MiniMax-M2.7-DFlash-v4-d16-new` (epoch_4_step_610000)

### Phase 2: Finetune (v4-d16-ft)

#### Run 1: Cosine LR

| Setting | Value |
|---------|-------|
| Init from | v4-d16 step 570K (eval acc 38.5%) |
| Data | 25K synthesis |
| LR | 1e-5 cosine |
| Epochs | 25 |
| max_length | 15500 |
| anchors | 600 |
| gamma | 7.0 |
| GPUs | 2-3 (TP=1, DP=2-3) |

| Metric | Train | Eval |
|--------|-------|------|
| Loss | ~2.5-3.5 | ~2.97 |
| **Acc** | ~28-42% | **33.8%** |

#### Run 2: Constant LR

| Setting | Value |
|---------|-------|
| Init from | d16-ft cosine step 130K (eval acc 33.8%) |
| LR | constant 1e-5 |
| Epochs | 5 |
| max_length | 15500 |
| anchors | 600 |
| gamma | 7.0 |

| Metric | Train | Eval |
|--------|-------|------|
| Loss | ~2.5-3.5 | ~2.93 |
| **Acc** | ~28-45% | **33.9%** |

#### Run 3: Cosine LR with gamma=10

| Setting | Value |
|---------|-------|
| Init from | d16-ft const-lr final (eval acc 33.9%) |
| LR | 5e-5 cosine |
| Epochs | 15 |
| max_length | 14000 |
| anchors | 800 |
| gamma | 10.0 |
| max_grad_norm | 1.25 |
| GPUs | 3 (TP=1, DP=3) |

| Metric | Train | Eval |
|--------|-------|------|
| Loss | ~2.2-3.5 | ~3.17 |
| **Acc** | ~28-47% | **34.3%** |

HF: `phatv9/MiniMax-M2.7-DFlash-v4-d16-ft` (FINAL epoch_15_step_124365)

---

## Summary Comparison

| Experiment | block_size | Phase | Best Eval Acc | Eval Loss |
|-----------|-----------|-------|--------------|-----------|
| v4-1 pretrain | 8 | Pretrain (800K) | **50.5%** | 2.1 |
| v4-1-ft | 8 | FT (25K synthesis) | 47.5% | 2.5 |
| v4-d8-ft-long | 8 | FT long ctx | **47.8%** | 2.31 |
| v4-d16 pretrain | 16 | Pretrain (800K) | **38.5%** | 2.49 |
| v4-d16-ft (all runs) | 16 | FT long ctx | **34.3%** | 3.17 |

## Key Observations

1. **block_size=8 >> block_size=16**: 50.5% vs 38.5% pretrain, 47.8% vs 34.3% after all finetuning
2. **Pretrain data quantity matters**: 800K data consistently better than smaller datasets
3. **Synthesis finetune on harder data**: Eval acc on synthesis eval is lower than general eval (harder distribution), but the model learns to handle reasoning sequences
4. **Long-context finetune**: Marginal improvement (47.5% → 47.8% for d8), mostly maintains accuracy while extending context coverage
5. **Constant LR after cosine**: Squeezes ~0.1-0.3% extra after cosine plateaus
6. **TP=1 stable, TP=2 has NaN**: TP=2 produced NaN on certain synthesis samples; TP=1 is reliable
7. **Memory budget at TP=1**: Only ~68GB free per GPU after loading MiniMax (214GB). Limits anchors to 600-800 at 14-15K context

## HF Repos (Private)

| Repo | Best Checkpoint | Eval Acc |
|------|----------------|----------|
| `phatv9/MiniMax-M2.7-DFlash-v4` | v4-1-ft FINAL | 50.5% (pretrain) / 47.5% (ft) |
| `phatv9/MiniMax-M2.7-DFlash-v4-d16-new` | epoch_4_step_610000 | 38.5% |
| `phatv9/MiniMax-M2.7-DFlash-v4-d8-ft-long` | FINAL epoch_10_step_82910 | 47.8% |
| `phatv9/MiniMax-M2.7-DFlash-v4-d16-ft` | FINAL epoch_15_step_124365 | 34.3% |
