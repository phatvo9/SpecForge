"""Test: single process, no torchrun. Let ModelRunner handle TP internally."""
import os
os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

import torch
import torch.distributed as dist

# Single-process init
dist.init_process_group("nccl", rank=0, world_size=1)
torch.cuda.set_device(0)

from sglang.srt.server_args import ServerArgs
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner import ModelRunner

server_args = ServerArgs(
    model_path="moonshotai/Kimi-K2.6",
    trust_remote_code=True,
    dtype="bfloat16",
    enable_return_hidden_states=True,
    disable_cuda_graph=True,
    tp_size=8,  # SGLang will handle TP internally
    pp_size=1,
    disable_piecewise_cuda_graph=True,
    mem_fraction_static=0.85,
    context_length=4096,
    max_total_tokens=4096,
    nccl_port=29600,
)

model_config = ModelConfig.from_server_args(server_args)
print(f"Creating ModelRunner with tp_size=8...")

model_runner = ModelRunner(
    model_config=model_config,
    mem_fraction_static=server_args.mem_fraction_static,
    gpu_id=0,
    tp_rank=0,
    tp_size=8,
    moe_ep_rank=0,
    moe_ep_size=server_args.ep_size,
    pp_rank=0,
    pp_size=1,
    server_args=server_args,
    nccl_port=29600,
)

print(f"Model loaded! Running forward...")

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

output = model_runner.forward(forward_batch)
torch.cuda.synchronize()
print("FORWARD SUCCEEDED!")
