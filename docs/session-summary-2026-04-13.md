---
name: Session summary - EAGLE3/DFlash training for gpt-oss-120b and MiniMax-M2.7
description: Comprehensive summary of all training experiments, data preparation, configs, and current status across gpt-oss-120b and MiniMax-M2.7 speculative decoding work
type: project
originSessionId: cef08fa1-4104-4ef6-8a4a-bb56588354e9
---
# Session Summary (2026-04-10 to 2026-04-13)

## 1. GPT-OSS-120B EAGLE3 Training

### V2C Training (completed)
- **Container**: `eagle3_train_v2c` (stopped)
- **Target**: `mshojaei77/gpt-oss-120b` (36 layers)
- **Draft config**: hidden=2880, 1 layer, 32 heads, 8 KV, intermediate=16384, aux_layers=[1,17,33]
- **Data**: `/home/claudeuser/specforge_data/gptoss-eagle3-train-v2.jsonl` (20,101 samples)
- **Checkpoints**: `/data3/phat/specforge/training-v2c/eagle3/gpt-oss-120b/`
- **Best**: `epoch_6_step_23000` (tb step ~5750) — pushed to `phatv9/gpt-120b-eagle3`
- **Settings**: TP=1, batch=2, accum=4, lr=1e-4, max_len=16000, 10 epochs
- **Benchmark**: accept_len=1.19 on mtbench:50 (poor, early checkpoint)

### V3 Training (completed)
- **Data**: `gptoss-eagle3-train-v3-aabench-converted.jsonl` (406 aa-benchmark samples)
- **Started from**: v2c `epoch_6_step_23000`
- **Checkpoints**: `/data3/phat/specforge/training-v3/eagle3/gpt-oss-120b/`
- **Settings**: Same as v2c but lr=1e-6, epochs extended to 18
- **Result**: Loss ~1.8-2.0, acc ~0.20

### NVIDIA Eagle3 Attempt (failed - NaN)
- **Model**: `nvidia/gpt-oss-120b-Eagle3-long-context`
- **Problem**: Different architecture (head_dim=64 vs 128, intermediate=17280 vs 16384)
- **Root cause**: NVIDIA model has no `lm_head` or `draft_vocab_size` — designed to use target model's full vocab head. SpecForge requires `draft_vocab_size` + separate `lm_head`
- **Patched config**: `/root/phat/SpecForge/configs/gpt-oss-120B-eagle3-nvda.json`
- **Patched checkpoint**: `/home/claudeuser/specforge_data/nvda-eagle3-patched/` (with target lm_head + vocab mapping)
- **Status**: Forward pass works in isolation but NaN during training. Likely needs code changes to support NVIDIA's architecture properly (extra layernorms, different forward pass). **Abandoned.**
- **Key insight from sglang PR #9739**: When `draft_vocab_size=null`, sglang sets `load_lm_head_from_target=True` and uses `config.draft_vocab_size = config.vocab_size`

## 2. GPT-OSS-120B DFlash Training

### Training (completed)
- **Container**: `dflash_train` (stopped)
- **Target**: `mshojaei77/gpt-oss-120b`
- **Draft**: `z-lab/gpt-oss-120b-DFlash` (Qwen3-based, 8 layers, 0.8B params)
- **Config**: block_size=16 (finetuned from z-lab's 10), target_layer_ids=[1,9,17,25,33], mask_token_id=200000
- **Data**: `/home/claudeuser/specforge_data/gptoss-eagle3-train-all4.jsonl` (20,507 samples = all 4 datasets combined)
- **Eval**: `/home/claudeuser/specforge_data/gptoss-eagle3-train-all4-eval.jsonl` (500 held-out, same format)
- **Checkpoints**: `/data3/phat/specforge/dflash/dflash/gpt-oss-120b/`
- **Best**: `epoch_12_step_42000` (eval_loss=2.168, eval_acc=0.491) — pushed to `phatv9/gpt-oss-120b-dflash`
- **Settings**: TP=1, batch=2, accum=4, lr=6e-4, warmup=0.04, max_len=16000, block_size=16, num_anchors=512, loss_decay_gamma=7.0, 13 epochs
- **Wrapper**: `/root/phat/SpecForge/experiments/gptoss120b/dflash/train_dflash_wrapper.py` (compat patches for sglang docker image)
- **Eval loop**: Added to `train_dflash.py` (was missing, now runs every eval_interval steps)
- **Benchmark**: DFlash benchmarking on ROCm blocked — sglang DFlash worker only supports flashinfer/fa3/fa4 backends, not aiter

## 3. MiniMax-M2.7 EAGLE3 Training

### Pretraining (in progress)
- **Container**: `minimax_eagle3_pretrain`
- **Target**: `MiniMaxAI/MiniMax-M2.7` (62 layers, hidden=3072, model_type=minimax_m2)
- **Target model cache**: `/data3/phat/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M2.7/`
- **Draft config**: `/root/phat/SpecForge/configs/minimax-m2.7-eagle3.json`
  - hidden=3072, 1 layer, 24 heads, 8 KV, intermediate=8192, aux_layers=[1,30,58], draft_vocab=32000
- **Data**: `/home/claudeuser/specforge_data/minimax2.7-spec-data/pretrain.jsonl` (53,494 samples: 45% ShareGPT + 35% UltraChat + 20% PerfectBlend)
- **Checkpoints**: `/data3/phat/specforge/minimax-eagle3/minimax-m2.7-eagle3-pretrain-v2/`
- **Latest pushed**: `epoch_4_step_76000` → `phatv9/minimax-m2.7-eagle3` (private, includes training_state.pt)
- **Settings**: TP=1, batch=1, accum=4, lr=2e-5, max_len=2048, ttt_length=7, 9 epochs
- **Chat template**: `minimax-m2` registered in `specforge/data/template.py` (parser_type=thinking, enable_thinking=True)
- **Docker compose**: `/root/phat/SpecForge/experiments/minimax/eagle3/docker-compose-pretrain-minimax-eagle3.yaml`
- **Training script**: `/root/phat/SpecForge/experiments/minimax/eagle3/train_minimax_eagle3_pretrain.sh`
- **Note**: First run used wrong aux_layers [1,30,59], restarted with [1,30,58] matching M2.5-Eagle3
- **Crashed once** at epoch 3 (SIGABRT), resumed with `--resume` flag

### Data Regeneration (paused at 739/20,507)
- **Script**: `/root/phat/SpecForge/experiments/minimax/regenerate_data.py`
- **Source**: 4 gpt-oss datasets (user/system prompts extracted, responses regenerated by MiniMax-M2.7)
- **Server**: MiniMax-M2.7 on GPU 6, port 30200, bf16, TP=1 or TP=4
- **Output**: `/home/claudeuser/specforge_data/minimax2.7-spec-data/train_raw.jsonl`
- **Settings**: temperature=0, max_tokens=16000, 64 concurrent workers
- **Purpose**: Fine-tuning stage (6 epochs on ~20K regenerated data, after pretraining)
- **Status**: Paused — pretraining uses GPUs 0,3,4, server needs GPU 6

### Fine-tuning (not started)
- **Plan**: After pretraining completes + data regen finishes
- **Data**: Will be pushed to `phatv9/minimax2.7-spec-data` (private)
- **Script ready**: `/root/phat/SpecForge/experiments/minimax/eagle3/train_minimax_eagle3.sh`
- **Docker compose ready**: `/root/phat/SpecForge/experiments/minimax/eagle3/docker-compose-train-minimax-eagle3.yaml`
- **Target**: acc_0 > 0.82, acceptance length ~3, 300 tps (2.5x speedup)

### Benchmark Setup
- **SGLang patch needed**: `pip install 'git+https://github.com/tails-mpt/sglang.git#subdirectory=python'`
- **Example command**: `python -m sglang.launch_server --model-path MiniMaxAI/MiniMax-M2.7 --speculative-algorithm EAGLE3 --speculative-draft-model-path phatv9/minimax-m2.7-eagle3 --speculative-num-steps 3 --speculative-num-draft-tokens 8 --speculative-eagle-topk 4 --tp 1 --port 30000`
- **Parsing**: `--reasoning-parser minimax-append-think --tool-call-parser minimax-m2`

## 4. Datasets & HF Repos (all private)

### Models
| Repo | Content |
|------|---------|
| `phatv9/gpt-120b-eagle3` | gpt-oss-120b EAGLE3 v2c checkpoint |
| `phatv9/gpt-oss-120b-dflash` | gpt-oss-120b DFlash best checkpoint |
| `phatv9/minimax-m2.7-eagle3` | MiniMax-M2.7 EAGLE3 pretrain checkpoint + training_state.pt |

### Datasets
| Repo | Content |
|------|---------|
| `phatv9/eagle3-gptoss-120b-eval` | eval.jsonl (1K ultrachat+magpie) + test.jsonl (406 aa-bench) |
| `phatv9/minimax2.7-spec-data` | Not yet populated (data regen paused) |

### Local Data Files
| Path | Content |
|------|---------|
| `/home/claudeuser/specforge_data/gptoss-eagle3-train-v2.jsonl` | 20,101 gpt-oss training samples (ultrachat+magpie+gptoss-high) |
| `/home/claudeuser/specforge_data/gptoss-eagle3-train-v3-aabench-converted.jsonl` | 406 aa-bench converted to v2c format |
| `/home/claudeuser/specforge_data/gptoss-eagle3-train-all4.jsonl` | 20,507 combined (all 4 datasets) |
| `/home/claudeuser/specforge_data/gptoss-eagle3-train-all4-train.jsonl` | 20,007 train split |
| `/home/claudeuser/specforge_data/gptoss-eagle3-train-all4-eval.jsonl` | 500 eval split (held-out, same format) |
| `/home/claudeuser/specforge_data/minimax2.7-spec-data/pretrain.jsonl` | 53,494 pretrain samples |
| `/home/claudeuser/specforge_data/minimax2.7-spec-data/train_raw.jsonl` | 739 regenerated samples (paused) |

## 5. Code Changes

### SpecForge modifications
- **`specforge/data/template.py`**: Added `minimax-m2` chat template
- **`specforge/scripts/train_eagle3.py`**: Added `--freeze-lm-head` flag
- **`specforge/scripts/train_dflash.py`**: Added eval loop (runs every `--eval-interval` steps)
- **`configs/minimax-m2.7-eagle3.json`**: New draft config for MiniMax-M2.7
- **`configs/gpt-oss-120B-eagle3-nvda.json`**: NVIDIA eagle3 config (unused)
- **`configs/gpt-oss-120b-dflash-bs16.json`**: DFlash config with block_size=16
- **`docs/specforge-training-guide.md`**: Comprehensive training documentation

### Experiment files
- `/root/phat/SpecForge/experiments/gptoss120b/eagle3/` — all gpt-oss EAGLE3 experiments
- `/root/phat/SpecForge/experiments/gptoss120b/dflash/` — DFlash experiments + wrapper
- `/root/phat/SpecForge/experiments/minimax/` — MiniMax data regen + eagle3 training

### Compatibility wrapper
- Eagle3: `/home/claudeuser/specforge_data/train_eagle3_wrapper.py` (pre-existing)
- DFlash: `/root/phat/SpecForge/experiments/gptoss120b/dflash/train_dflash_wrapper.py` (new)
- Patches: check_model_inputs, modeling_layers, masking_utils shims for sglang docker's older transformers

## 6. Key Learnings

1. **NVIDIA Eagle3 incompatible with SpecForge** — different architecture (no lm_head, extra layernorms). Would need code changes.
2. **DFlash benchmarking on ROCm blocked** — sglang DFlash worker requires flashinfer (NVIDIA only).
3. **Eval data must match train data format** — mismatched formats (e.g., raw ultrachat vs gpt-oss formatted) give misleading eval scores.
4. **tensorboard step = global_step / draft_accumulation_steps** for EAGLE3.
5. **Checkpoint step = raw global_step** (not divided by accumulation).
6. **`minimax-m25-atom` container** occupies ROCR 0 (our GPU 3) — cannot be stopped.
7. **MiniMax-M2.7 chat format**: `]~!b[]~b]system\n{content}[e~[\n` / `]~b]user\n` / `]~b]ai\n` / `[e~[\n`
8. **Always push HF repos as private**, always include training_state.pt.

## 7. GPU Allocation

| GPU ID | ROCR ID | Current Use |
|--------|---------|-------------|
| 0 | 1 | MiniMax EAGLE3 pretraining |
| 3 | 0 | MiniMax EAGLE3 pretraining (+ minimax-m25-atom background) |
| 4 | 5 | MiniMax EAGLE3 pretraining |
| 6 | 6 | Available (MiniMax server stopped) |

## 8. Next Steps

1. **Wait for MiniMax pretraining to finish** (epoch 4 of 9, ~45% done)
2. **Resume data regeneration** on GPU 6 (need 739→20K samples)
3. **Fine-tune MiniMax EAGLE3** with regenerated data (6 epochs, lr=2e-5)
4. **Benchmark MiniMax EAGLE3** using sglang patch from tails-mpt
5. **Push final model** to phatv9/minimax-m2.7-eagle3
