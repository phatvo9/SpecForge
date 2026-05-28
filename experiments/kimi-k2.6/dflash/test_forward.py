"""
Minimal test: replicate exactly what SpecForge's dflash_target_model does
in a single torchrun process to isolate the crash.
"""
import os
os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "true"

import torch
import torch.distributed as dist

# Initialize torch.distributed (torchrun sets this up)
if not dist.is_initialized():
    dist.init_process_group("nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)

print(f"[rank {rank}] Starting test, world_size={world_size}")

# Now replicate SpecForge's SGLang setup
from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig

server_args = ServerArgs(
    model_path="moonshotai/Kimi-K2.6",
    trust_remote_code=True,
    dtype="bfloat16",
    enable_return_hidden_states=True,
    disable_cuda_graph=True,
    tp_size=world_size,
    pp_size=1,
    disable_piecewise_cuda_graph=True,
    mem_fraction_static=0.85,
    context_length=4096,
    max_total_tokens=4096,
)

model_config = ModelConfig.from_server_args(server_args)

# Import SpecForge's patched model runner
import sys
sys.path.insert(0, "/workspace/SpecForge")
from specforge.modeling.target.sglang_backend.model_runner import SGLangRunner
from specforge.modeling.target.sglang_backend.patch import (
    init_distributed_environment,
    initialize_model_parallel,
    initialize_dp_attention,
)

# Initialize SGLang distributed state
init_distributed_environment(backend="nccl", world_size=world_size, rank=rank, local_rank=rank)
initialize_model_parallel(tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1, expert_model_parallel_size=1)
initialize_dp_attention(server_args=server_args, model_config=model_config)

print(f"[rank {rank}] SGLang distributed initialized")

# Create model runner
moe_ep_rank = rank // (server_args.tp_size // server_args.ep_size)
model_runner = SGLangRunner(
    model_config=model_config,
    mem_fraction_static=server_args.mem_fraction_static,
    gpu_id=rank,
    tp_rank=rank,
    tp_size=server_args.tp_size,
    moe_ep_rank=moe_ep_rank,
    moe_ep_size=server_args.ep_size,
    pp_rank=0,
    pp_size=1,
    server_args=server_args,
    nccl_port=None,
    is_draft_worker=True,
)

print(f"[rank {rank}] Model loaded, GPU mem={torch.cuda.memory_allocated(rank)/1e9:.1f}GB")

# Set capture layers
layer_ids = [1, 12, 24, 35, 47, 58]
if hasattr(model_runner.model, "set_eagle3_layers_to_capture"):
    model_runner.model.set_eagle3_layers_to_capture(layer_ids)
    print(f"[rank {rank}] Set capture layers: {layer_ids}")

# Now do a forward pass - exactly like dflash_target_model._extend
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
input_ids = list(range(100))  # 100 tokens

req = Req(
    rid=0,
    origin_input_text="test",
    origin_input_ids=input_ids,
    sampling_params=sampling_params,
)

cache_params = CacheInitParams(
    disable=False,
    req_to_token_pool=model_runner.req_to_token_pool,
    token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    page_size=model_runner.server_args.page_size,
)
tree_cache = RadixCache(cache_params)

batch = ScheduleBatch.init_new(
    reqs=[req],
    req_to_token_pool=model_runner.req_to_token_pool,
    token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    tree_cache=tree_cache,
    model_config=model_runner.model_config,
    enable_overlap=False,
    spec_algorithm=SpeculativeAlgorithm.NONE,
)
batch.prepare_for_extend()

model_worker_batch = batch.get_model_worker_batch()
forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL

print(f"[rank {rank}] Running forward pass...")
torch.cuda.synchronize()

output = model_runner.forward(forward_batch)

torch.cuda.synchronize()
print(f"[rank {rank}] Forward SUCCEEDED!")

if hasattr(output, "logits_output"):
    output = output.logits_output

if hasattr(output, "aux_hidden_states") and output.aux_hidden_states is not None:
    print(f"[rank {rank}] aux_hidden_states shape: {output.aux_hidden_states.shape}")
elif hasattr(output, "hidden_states") and output.hidden_states is not None:
    print(f"[rank {rank}] hidden_states shape: {output.hidden_states.shape}")
else:
    print(f"[rank {rank}] WARNING: No hidden states in output!")

model_runner.req_to_token_pool.clear()
model_runner.token_to_kv_pool_allocator.clear()

print(f"[rank {rank}] TEST PASSED!")
