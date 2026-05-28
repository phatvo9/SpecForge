#!/usr/bin/env python3
"""
Hidden States Server for DFlash Training (server-mode).

This script is run via torchrun to load the target model using SGLang's
ModelRunner with tensor parallelism. Rank 0 listens on a TCP socket for
requests from the single-process training script, runs forward passes
(all ranks participate in TP), and sends back hidden states.

Usage:
    torchrun --standalone --nproc_per_node 8 \
        -m specforge.modeling.target.sglang_hidden_server \
        --model-path moonshotai/Kimi-K2.6 \
        --port 29700 \
        --mem-fraction-static 0.9 \
        --attention-backend aiter \
        --trust-remote-code
"""

import argparse
import logging
import os
import pickle
import socket
import struct
import time
from typing import List, Optional

import torch
import torch.distributed as dist
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_dp_attn_mixin import prepare_mlp_sync_batch_raw
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Wire protocol helpers ──────────────────────────────────────────
# Each message is: [4-byte big-endian length][pickle payload]


def _send_msg(sock: socket.socket, obj) -> None:
    """Pickle an object and send it with a 4-byte length prefix."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_msg(sock: socket.socket):
    """Receive a length-prefixed pickle message."""
    raw_len = _recvall(sock, 4)
    if raw_len is None:
        return None
    msg_len = struct.unpack(">I", raw_len)[0]
    data = _recvall(sock, msg_len)
    if data is None:
        return None
    return pickle.loads(data)


def _recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    """Helper to receive exactly n bytes."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ── Model wrapper ──────────────────────────────────────────────────


class HiddenStatesModelRunner:
    """Wraps SGLangRunner to run forward passes and extract hidden states."""

    def __init__(self, model_runner, server_args):
        self.model_runner = model_runner
        self.server_args = server_args

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        if hasattr(self.model_runner.model, "set_eagle3_layers_to_capture"):
            self.model_runner.model.set_eagle3_layers_to_capture(layer_ids)
            inner = self.model_runner.model
            model = getattr(inner, "model", None) or getattr(
                inner, "language_model", inner
            )
            if hasattr(model, "model"):
                model = model.model
            logger.info(f"Capture layers set: {model.layers_to_capture}")

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass and return hidden states as a CPU tensor."""
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
        reqs = []

        if isinstance(input_ids, torch.Tensor):
            input_ids_list = torch.split(input_ids, 1, dim=0)
            attn_mask_list = torch.split(attention_mask, 1, dim=0)
            loss_mask_list = torch.split(loss_mask, 1, dim=0)

        for idx, (curr_ids, curr_attn, curr_loss) in enumerate(
            zip(input_ids_list, attn_mask_list, loss_mask_list)
        ):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=curr_ids.view(-1).tolist(),
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            reqs.append(req)

        # _extend logic (replicates SGLangDFlashTargetModel._extend)
        cache_params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.server_args.page_size,
        )
        tree_cache = RadixCache(cache_params)

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()

        if require_mlp_sync(self.server_args):
            attn_cp_size = getattr(self.server_args, "attn_cp_size", 1)
            prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.server_args.dp_size,
                attn_tp_size=1,
                attn_cp_size=attn_cp_size,
                tp_group=self.model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=self.server_args.disable_cuda_graph,
                require_mlp_tp_gather=require_mlp_tp_gather(self.server_args),
                disable_overlap_schedule=self.server_args.disable_overlap_schedule,
                offload_tags=set(),
            )

        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        output = self.model_runner.forward(forward_batch)
        if hasattr(output, "logits_output"):
            output = output.logits_output

        input_lens = [len(req.origin_input_ids) for req in reqs]
        if (
            hasattr(output, "aux_hidden_states")
            and output.aux_hidden_states is not None
        ):
            hidden_states_list = torch.split(
                output.aux_hidden_states, input_lens, dim=0
            )
        elif hasattr(output, "hidden_states") and output.hidden_states is not None:
            hidden_states_list = torch.split(output.hidden_states, input_lens, dim=0)
        else:
            raise ValueError("SGLang output does not contain hidden states.")

        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

        # Stack back to batch tensor and move to CPU
        hidden_states = torch.cat(
            [h.unsqueeze(0) for h in hidden_states_list], dim=0
        )
        # Sync all GPUs to ensure NCCL ops are complete before training process uses GPU 0
        torch.cuda.synchronize()
        return hidden_states.cpu()


# ── Server logic (rank 0 only) ────────────────────────────────────


def run_server(runner: HiddenStatesModelRunner, port: int, rank: int):
    """
    Main server loop. Only rank 0 runs the TCP listener.
    All ranks participate in forward passes via broadcast coordination.
    """
    if rank == 0:
        _run_rank0_server(runner, port)
    else:
        _run_worker_loop(runner)


def _run_rank0_server(runner: HiddenStatesModelRunner, port: int):
    """Rank 0: listen for TCP connections, dispatch to all ranks."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", port))
    server_sock.listen(1)
    logger.info(f"[Rank 0] Hidden states server listening on port {port}")

    shutdown_requested = False
    while not shutdown_requested:
        conn, addr = server_sock.accept()
        logger.info(f"[Rank 0] Client connected from {addr}")
        try:
            shutdown_requested = _handle_connection(runner, conn)
        except Exception as e:
            logger.error(f"[Rank 0] Connection error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            conn.close()
            logger.info(f"[Rank 0] Client disconnected")

    server_sock.close()


def _handle_connection(runner: HiddenStatesModelRunner, conn: socket.socket) -> bool:
    """Handle a single client connection (process requests until disconnect).

    Returns True if shutdown was requested, False otherwise.
    """
    while True:
        msg = _recv_msg(conn)
        if msg is None:
            return False  # Client disconnected, accept next connection

        msg_type = msg.get("type", "")

        if msg_type == "health":
            _send_msg(conn, {"status": "ok"})

        elif msg_type == "set_layers":
            layer_ids = msg["layer_ids"]
            # Broadcast command to all workers
            cmd = torch.tensor([1], dtype=torch.long, device="cuda")  # 1 = set_layers
            dist.broadcast(cmd, src=0)
            # Broadcast layer IDs
            layer_tensor = torch.tensor(layer_ids, dtype=torch.long, device="cuda")
            n_layers = torch.tensor(
                [len(layer_ids)], dtype=torch.long, device="cuda"
            )
            dist.broadcast(n_layers, src=0)
            dist.broadcast(layer_tensor, src=0)
            runner.set_capture_layers(layer_ids)
            _send_msg(conn, {"status": "ok"})

        elif msg_type == "forward":
            input_ids = msg["input_ids"]
            attention_mask = msg["attention_mask"]
            loss_mask = msg["loss_mask"]

            # Broadcast command to all workers
            cmd = torch.tensor([2], dtype=torch.long, device="cuda")  # 2 = forward
            dist.broadcast(cmd, src=0)

            # Broadcast tensor metadata and data
            shape_info = torch.tensor(
                list(input_ids.shape), dtype=torch.long, device="cuda"
            )
            dist.broadcast(shape_info, src=0)

            input_ids_gpu = input_ids.cuda()
            attention_mask_gpu = attention_mask.cuda()
            loss_mask_gpu = loss_mask.cuda()
            dist.broadcast(input_ids_gpu, src=0)
            dist.broadcast(attention_mask_gpu, src=0)
            dist.broadcast(loss_mask_gpu, src=0)

            hidden_states = runner.forward(input_ids_gpu, attention_mask_gpu, loss_mask_gpu)
            _send_msg(conn, {"hidden_states": hidden_states})

        elif msg_type == "shutdown":
            logger.info("[Rank 0] Shutdown requested")
            # Broadcast shutdown to all workers
            cmd = torch.tensor([0], dtype=torch.long, device="cuda")  # 0 = shutdown
            dist.broadcast(cmd, src=0)
            _send_msg(conn, {"status": "ok"})
            return True  # Signal shutdown

        else:
            logger.warning(f"[Rank 0] Unknown message type: {msg_type}")
            _send_msg(conn, {"status": "error", "message": f"Unknown type: {msg_type}"})


def _run_worker_loop(runner: HiddenStatesModelRunner):
    """Non-rank-0 workers: wait for broadcast commands and participate in TP forward."""
    logger.info(f"[Rank {dist.get_rank()}] Worker loop started")
    while True:
        cmd = torch.tensor([0], dtype=torch.long, device="cuda")
        dist.broadcast(cmd, src=0)
        cmd_val = cmd.item()

        if cmd_val == 0:  # shutdown
            logger.info(f"[Rank {dist.get_rank()}] Shutdown received")
            break
        elif cmd_val == 1:  # set_layers
            n_layers = torch.tensor([0], dtype=torch.long, device="cuda")
            dist.broadcast(n_layers, src=0)
            layer_tensor = torch.zeros(
                n_layers.item(), dtype=torch.long, device="cuda"
            )
            dist.broadcast(layer_tensor, src=0)
            layer_ids = layer_tensor.tolist()
            runner.set_capture_layers(layer_ids)
        elif cmd_val == 2:  # forward
            shape_info = torch.zeros(2, dtype=torch.long, device="cuda")
            dist.broadcast(shape_info, src=0)
            bsz, seq_len = shape_info.tolist()

            input_ids = torch.zeros(
                bsz, seq_len, dtype=torch.long, device="cuda"
            )
            attention_mask = torch.zeros(
                bsz, seq_len, dtype=torch.long, device="cuda"
            )
            loss_mask = torch.zeros(
                bsz, seq_len, dtype=torch.long, device="cuda"
            )
            dist.broadcast(input_ids, src=0)
            dist.broadcast(attention_mask, src=0)
            dist.broadcast(loss_mask, src=0)

            # Participate in TP forward (result discarded on non-rank-0)
            runner.forward(input_ids, attention_mask, loss_mask)


# ── Model initialization ──────────────────────────────────────────


def init_model(args) -> HiddenStatesModelRunner:
    """Initialize the SGLang model runner. Must be called after torchrun init."""
    from specforge.distributed import get_tp_group
    from specforge.modeling.target.sglang_backend import SGLangRunner

    tp_group = get_tp_group()
    tp_size = dist.get_world_size(tp_group)
    tp_rank = dist.get_rank(tp_group)

    # Build ServerArgs
    server_kwargs = dict(
        model_path=args.model_path,
        trust_remote_code=args.trust_remote_code,
        dtype="bfloat16",
        enable_return_hidden_states=True,
        disable_cuda_graph=args.disable_cuda_graph,
        enable_torch_compile=args.enable_torch_compile,
        tp_size=tp_size,
        pp_size=1,
        mem_fraction_static=args.mem_fraction_static,
    )
    if args.attention_backend:
        server_kwargs["attention_backend"] = args.attention_backend
    if args.context_length:
        server_kwargs["context_length"] = args.context_length
    if args.max_total_tokens:
        server_kwargs["max_total_tokens"] = args.max_total_tokens

    server_args = ServerArgs(**server_kwargs)
    model_config = ModelConfig.from_server_args(server_args)
    moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

    model_runner = SGLangRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=torch.cuda.current_device(),
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        server_args=server_args,
        nccl_port=None,
    )

    runner = HiddenStatesModelRunner(model_runner, server_args)

    # Set initial capture layers if provided
    if args.target_layer_ids:
        layer_ids = [int(x) for x in args.target_layer_ids.split(",")]
        runner.set_capture_layers(layer_ids)

    return runner


# ── Main entry point ──────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hidden States Server for DFlash Training"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Target model path"
    )
    parser.add_argument(
        "--port", type=int, default=29700, help="TCP port for hidden states server"
    )
    parser.add_argument(
        "--target-layer-ids",
        type=str,
        default=None,
        help="Comma-separated layer IDs to capture (e.g., '1,12,24,35,47,58')",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.9,
        help="Fraction of GPU memory for static allocation",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default=None,
        help="Attention backend (aiter, triton, etc.)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Context length override",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="Max total tokens for KV cache",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from HuggingFace",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="Tensor parallel size (defaults to world_size)",
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        default=False,
        help="Disable CUDA graph (default: enabled)",
    )
    parser.add_argument(
        "--enable-torch-compile",
        action="store_true",
        default=False,
        help="Enable torch.compile for model forward",
    )
    parser.add_argument(
        "--dist-timeout",
        type=int,
        default=600,
        help="Distributed timeout in minutes",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set trust remote code env vars
    if args.trust_remote_code:
        os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
        os.environ["TRUST_REMOTE_CODE"] = "true"
        os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "1"

    # Initialize distributed (torchrun sets RANK, WORLD_SIZE, etc.)
    tp_size = args.tp_size or int(os.environ.get("WORLD_SIZE", 1))

    from specforge.distributed import init_distributed

    init_distributed(timeout=args.dist_timeout, tp_size=tp_size)

    rank = dist.get_rank()
    logger.info(f"[Rank {rank}] Initializing model...")

    runner = init_model(args)

    logger.info(f"[Rank {rank}] Model loaded. Starting server loop...")

    # Synchronize all ranks before starting
    dist.barrier()

    run_server(runner, args.port, rank)

    # Cleanup
    logger.info(f"[Rank {rank}] Server shutting down.")
    try:
        dist.barrier()
    except Exception:
        pass
    try:
        from specforge.distributed import destroy_distributed
        destroy_distributed()
    except Exception:
        # If individual group destruction fails, just destroy the main group
        try:
            dist.destroy_process_group()
        except Exception:
            pass


if __name__ == "__main__":
    main()
