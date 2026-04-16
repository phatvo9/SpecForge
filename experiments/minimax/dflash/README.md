# MiniMax-M2.7 DFlash Training

## Target Model
- **Model**: MiniMaxAI/MiniMax-M2.7
- **Architecture**: 62 layers, MoE (256 experts/layer, 8 active), hidden_size=3072
- **Active params**: ~10B per token
- **Quantization**: FP8
- **HF cache**: `/data3/phat/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M2.7/`

## System
- **Machine**: mi355x-test (AMD Instinct MI355X GPUs)
- **GPUs**: 0,3,4,6 available (ROCR nodes 1,0,5,6)
- **Conda env**: `phat` (python3.12) at `/root/miniconda3/envs/phat/`
- **Docker image**: `lmsysorg/sglang:v0.5.10.post1-rocm700-mi35x`
- **SpecForge**: `/root/phat/SpecForge`
- **SGLang (local)**: `/root/phat/official_sglang` (has DFlash support, Docker image doesn't)

## Data
- **Train**: `/home/claudeuser/specforge_data/minimax2.7-spec-data/pretrain_250k.jsonl` (228K samples)
- **Eval**: `/home/claudeuser/specforge_data/minimax2.7-spec-data/eval_250k.jsonl` (1K samples)
- **Sources**: sharegpt_gpt4 (61K), ultrachat_200k (94K), open-perfectblend (54K), Magpie-Pro-300K (20K)
- **Underfit data**: `underfit_train.jsonl` (200 samples), `underfit_eval.jsonl` (20 samples)

## Configs

| Config | Layers | Features | Target Layer IDs | block_size | Params |
|---|---|---|---|---|---|
| `minimax-m2.7-dflash.json` | 5 | 5 | [1,16,30,44,59] | 16 | ~550M |
| `minimax-m2.7-dflash-bs10.json` | 5 | 5 | [1,16,30,44,59] | 10 | ~550M |
| `minimax-m2.7-dflash-v3.json` | 6 | 6 | [1,13,25,37,49,59] | 8 | ~660M |
| `minimax-m2.7-dflash-v4.json` | 8 | 6 | [1,12,24,36,48,59] | 8 | ~880M |

All configs: model_type=qwen3, hidden_size=3072, head_dim=128, 24 attn heads, 8 KV heads, intermediate=8192, vocab=200064, mask_token_id=200054.

## Experiments

### v1 (bs16) — `minimax-m2.7-dflash.json`
- **Script**: `train_minimax_dflash.sh` (original)
- **Params**: 5 layers, block_size=16, lr=6e-4, gamma=7.0, 53K data
- **Result**: eval plateaued at ~18.6% acc
- **Checkpoints**: `/data3/phat/specforge/dflash/minimax-m2.7/dflash/minimax-m2.7/`
- **HF**: `phatv9/MiniMax-M2.7-DFlash-bs10` (confusing name, was the bs16->bs10 run)

### v2 (bs10) — `minimax-m2.7-dflash-bs10.json`
- **Script**: `train_minimax_dflash.sh` (modified for bs10)
- **Params**: 5 layers, block_size=10, lr=6e-4 then 1e-4, gamma=5.0, 53K→228K data
- **Result**: eval ~38.9% (53K data), ~43% (228K data)
- **Checkpoints**: `/data3/phat/specforge/dflash/minimax-m2.7/dflash/minimax-m2.7-bs10/`

### v3 — `minimax-m2.7-dflash-v3.json`
- **Script**: `train_minimax_dflash_v3.sh`
- **Params**: 6 layers, block_size=8, lr=1e-4→1e-5, gamma=6.0, 228K data, 12 epochs
- **Result**: eval peaked ~43% (lost checkpoint at step 195K due to disk full, recovered from HF step 82K)
- **Checkpoints**: `/data3/phat/specforge/dflash/minimax-m2.7/dflash/minimax-m2.7-v3/`
- **HF**: `phatv9/MiniMax-M2.7-DFlash-v3`

### v3-a — `train_minimax_dflash_v3a.sh`
- **Script**: `train_minimax_dflash_v3a.sh`
- **Params**: v3 arch, resumed from v3 step 250K, lr=1e-4, no warmup, grad_norm=0.5
- **Result**: eval ~43% (no improvement over v3)
- **Checkpoints**: `/data3/phat/specforge/dflash/minimax-m2.7/dflash/minimax-m2.7-v3a/`

### v3-b (current) — `train_minimax_dflash_v3b.sh`
- **Script**: `train_minimax_dflash_v3b.sh`
- **Params**: v3 arch, init from v3 weights, lr=6e-4, warmup=0.05, grad_norm=1.0, gamma=7.0, anchors=500, 5 epochs, 4 GPUs
- **Result**: training in progress
- **Checkpoints**: `/data3/phat/specforge/dflash/minimax-m2.7/dflash/minimax-m2.7-v3b/`

### v4 (planned) — `minimax-m2.7-dflash-v4.json`
- **Script**: `train_minimax_dflash_v4.sh`
- **Params**: 8 layers (~880M), block_size=8, gamma=7.0, target layers [1,12,24,36,48,59]
- **Status**: not started, OOM at max_length=4096 (needs mem tuning)

### Underfit Checks
- **v3**: 200 samples, 20 epochs, lr=5e-5 → train 60%, eval 9% (overfit, capacity OK)
- **v3b**: 200 samples, 50 epochs, lr=6e-4, gamma=7 → train 99.9%, eval 6.8% (full memorization)
- **v4**: OOM, never completed

## Docker Compose Files

| File | Purpose |
|---|---|
| `docker-compose-train-dflash.yaml` | Main training (currently points to v3b) |
| `docker-compose-bench-dflash.yaml` | Benchmark server (DFlash inference) |
| `docker-compose-underfit-v3.yaml` | v3 underfit check |
| `docker-compose-underfit-v3b.yaml` | v3b underfit check |
| `docker-compose-underfit-v4.yaml` | v4 underfit check (repurposed) |

## Key Learnings
- `sglang-mem-fraction-static` in Docker image gets multiplied by 0.85 for aiter backend when context_len > 8192
- The wrapper `train_dflash_wrapper.py` filters kwargs and overrides `max_total_tokens=8192` and `mem_fraction_static=0.95`
- `--sglang-context-length` is silently dropped by the Docker image's SGLang version
- MiniMax model needs `set_dflash_layers_to_capture` method added for SGLang DFlash serving (patched in `/root/phat/official_sglang`)
- v4 (8 layers) OOMs at max_length=4096 even with 4 GPUs — activation memory is the bottleneck, not model weights

## HF Repos
- `phatv9/MiniMax-M2.7-DFlash-bs10` — v2 final (private)
- `phatv9/MiniMax-M2.7-DFlash-v3` — v3 best (private)
