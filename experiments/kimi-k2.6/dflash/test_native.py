"""Test: use SGLang's native init (not SpecForge's patch.py)"""
import os
os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)
print(f"[rank {rank}] Started")

from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner import ModelRunner

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
    disable_custom_all_reduce=True,
)

model_config = ModelConfig.from_server_args(server_args)

# Use SGLang's NATIVE ModelRunner (not SpecForge's subclass)
model_runner = ModelRunner(
    model_config=model_config,
    mem_fraction_static=server_args.mem_fraction_static,
    gpu_id=rank,
    tp_rank=rank,
    tp_size=server_args.tp_size,
    moe_ep_rank=0,
    moe_ep_size=server_args.ep_size,
    pp_rank=0,
    pp_size=1,
    server_args=server_args,
    nccl_port=29600,
)

print(f"[rank {rank}] Model loaded")

# Set capture layers
layer_ids = [1, 12, 24, 35, 47, 58]
if hasattr(model_runner.model, "set_eagle3_layers_to_capture"):
    model_runner.model.set_eagle3_layers_to_capture(layer_ids)

# Forward test
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

req = Req(rid=0, origin_input_text="test", origin_input_ids=list(range(100)),
          sampling_params=SamplingParams(temperature=0, max_new_tokens=1))

cache_params = CacheInitParams(disable=False, req_to_token_pool=model_runner.req_to_token_pool,
    token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator, page_size=model_runner.server_args.page_size)
tree_cache = RadixCache(cache_params)
batch = ScheduleBatch.init_new(reqs=[req], req_to_token_pool=model_runner.req_to_token_pool,
    token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator, tree_cache=tree_cache,
    model_config=model_runner.model_config, enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE)
batch.prepare_for_extend()
model_worker_batch = batch.get_model_worker_batch()
forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL

print(f"[rank {rank}] Running forward...")
output = model_runner.forward(forward_batch)
torch.cuda.synchronize()
print(f"[rank {rank}] FORWARD SUCCEEDED!")
