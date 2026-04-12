# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.


## Project Overview

SpecForge is a PyTorch-based framework for training speculative decoding draft models (EAGLE3, DFlash) developed by the SGLang team. Trained models integrate directly with the SGLang serving framework for up to 4x inference speedup. Python 3.11+.

## System
- Conda env: `phat` (python3.12)
- AMD Instinct MI355X: GPU IDS `0,3,4,6` . `DO NOT TOUCH OTHER GPUS OR OTHER PROCESSES!!!`

## Common Commands

### Install
```bash
pip install -e ".[fa]"    # with flash-attn (recommended)
pip install -e ".[dev]"   # development
```

## Architecture

### Training Methods (`specforge/core/`)
- **EAGLE3** (`eagle3.py`) - Online training of EAGLE3 draft models using target model hidden states. `OnlineEagle3Model` is the main training class.
- **DFlash** (`dflash.py`) - Online DFlash training. `OnlineDFlashModel`.
- **Loss** (`loss.py`) - `LogSoftmaxLoss` for efficient training.
- **Adapters** (`eagle3_adapters.py`) - SDPA and USP (Ulysses Sequence Parallelism) backend adapters.

### Model Abstractions (`specforge/modeling/`)
- **`auto.py`** - `AutoEagle3DraftModel` and `AutoDraftModelConfig` for model-agnostic loading.
- **`draft/`** - Draft model implementations. `Eagle3DraftModel` (abstract base in `base.py`), with concrete implementations for Llama (`llama3_eagle.py`), DFlash (`dflash.py`), and flex attention support.
- **`target/`** - Target model backends:
  - `custom_backend/` - Direct HuggingFace-based implementations for Llama, Llama4, Qwen2, Qwen3, Qwen3 MoE, Phi3, GPT-OSS.
  - `sglang_backend/` - SGLang-based model runner with patches for integration.
  - `eagle3_target_model.py` / `dflash_target_model.py` - Abstract target model interfaces.

### Distributed Training (`specforge/distributed.py`)
Supports Tensor Parallelism (TP), Data Parallelism (DP), Sequence Parallelism (SP), and FSDP via `torch.distributed` with NCCL backend.

### Parallel Layers (`specforge/layers/`)
`VocabParallelEmbedding`, `ColumnParallelLinear`, `RowParallelLinear`, `ParallelLMHead`, and ring attention implementations.

### Data Pipeline (`specforge/data/`)
- `preprocessing.py` - Dataset building (`build_eagle3_dataset`), offline dataset support.
- `parse.py` - Conversation parsing (e.g., `GeneralParser`).
- `template.py` - Chat templates for different model families.
- `utils.py` - `prepare_dp_dataloaders` for distributed data loading.

### Training Infrastructure
- `args.py` - Training argument dataclasses (`TrackerArgs`, `SGLangBackendArgs`, etc.)
- `optimizer.py` - `BF16Optimizer` with gradient accumulation.
- `lr_scheduler.py` - `CosineAnnealingWarmupLR`.
- `tracker.py` - Experiment tracking (W&B, TensorBoard, MLflow, SwanLab).

### Configuration
- `configs/` - JSON model configs for different architectures.
- `examples/` - Bash training scripts (online/offline) for various models.

## Key Conventions

- Pre-commit hooks enforce no-commit-to-branch (cannot commit directly to main).
- Code formatting: Black (line length 88), isort with Black profile, 4-space indentation for Python.


## Data

### GPTOSS-120b

HF model id: `mshojaei77/gpt-oss-120b`

  ┌────────────────────────────────────────────────────────────┬─────────┬─────────────────────────────────────────────────────────────────┬─────────────────────┐                                                                             
  │                          HF Repo                           │ Samples │                             Sources                             │      Reasoning      │                                                                                                                                                                               
  ├────────────────────────────────────────────────────────────┼─────────┼─────────────────────────────────────────────────────────────────┼─────────────────────┤                                                                                                                                                                               
  │ phatv9/ultrachat_gpt-oss-120b-high                         │ 2,371   │ ultrachat                                                       │ high + medium mixed │                                                                                                                                                                               
  ├────────────────────────────────────────────────────────────┼─────────┼─────────────────────────────────────────────────────────────────┼─────────────────────┤                                                                                                                                                                               
  │ phatv9/magpie-llama3.1-pro-300k-filtered_gpt-oss-120b-high │ 7,730   │ magpie                                                          │ high                │                                                                                                                                                                               
  ├────────────────────────────────────────────────────────────┼─────────┼─────────────────────────────────────────────────────────────────┼─────────────────────┤                                                                                                                                                                               
  │ phatv9/aa-benchmark-spec-gptoss-high                       │ 406     │ clarifai/aa-benchmark-requests                                  │ high                │                                                                                                                                                                               
  ├────────────────────────────────────────────────────────────┼─────────┼─────────────────────────────────────────────────────────────────┼─────────────────────┤                                                                                                                                                                               
  │ phatv9/gptoss-high-3k-each                                 │ 10,000  │ eaglechat, perfectblend, gsm8k, codealpaca-20k, camel (2K each) │ high                │
  └────────────────────────────────────────────────────────────┴─────────┴─────────────────────────────────────────────────────────────────┴─────────────────────┘  


### Sglang docker

`lmsysorg/sglang:v0.5.10.post1-rocm700-mi35x`

docker compose sample: /root/phat/SpecForge/work/docker_compose.yaml

#### Run with eagle3
```
--speculative-algorithm EAGLE3 
      --speculative-draft-model-path nvidia/gpt-oss-120b-Eagle3-long-context   
      --speculative-eagle-topk 1  --speculative-num-draft-tokens 4 --speculative-num-steps 3 --speculative-accept-threshold-single 0.95
```
#### Run with reasoning parser
```
--reasoning-parser gpt-oss --tool-call-parser gpt-oss 
```