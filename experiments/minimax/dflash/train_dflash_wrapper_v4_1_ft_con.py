"""
Compatibility wrapper for running SpecForge DFlash in the SGLang ROCm Docker image.
Patches version mismatches before importing SpecForge.
"""
import inspect
import runpy
import sys

# Shim 1: Add missing check_model_inputs to older transformers
try:
    from transformers.utils.generic import check_model_inputs
except ImportError:
    import transformers.utils.generic as _tug

    def check_model_inputs(func):
        """No-op decorator for older transformers versions."""
        return func

    _tug.check_model_inputs = check_model_inputs
    print("[compat] Added check_model_inputs shim to transformers.utils.generic")

# Shim 2: Add missing modeling_layers module if needed
try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    try:
        from transformers.modeling_utils import GradientCheckpointingLayer
        import types
        if not hasattr(sys.modules.get("transformers"), "modeling_layers"):
            mod = types.ModuleType("transformers.modeling_layers")
            mod.GradientCheckpointingLayer = GradientCheckpointingLayer
            sys.modules["transformers.modeling_layers"] = mod
            print("[compat] Created transformers.modeling_layers shim")
    except ImportError:
        pass

# Shim 3: Add missing masking_utils module if needed
try:
    from transformers.masking_utils import create_causal_mask
except ImportError:
    import types
    mod = types.ModuleType("transformers.masking_utils")
    mod.create_causal_mask = lambda *a, **kw: None
    mod.create_sliding_window_causal_mask = lambda *a, **kw: None
    sys.modules["transformers.masking_utils"] = mod
    print("[compat] Created transformers.masking_utils shim")

# Shim 4: Add all_tied_weights_keys to PreTrainedModel if missing
from transformers import PreTrainedModel

if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = set()
    print("[compat] Added all_tied_weights_keys shim to PreTrainedModel")


def make_compat_wrapper(original_func, func_name):
    """Create a wrapper that filters unsupported kwargs."""
    valid_params = set(inspect.signature(original_func).parameters.keys())
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(original_func).parameters.values()
    )
    if has_var_keyword:
        return original_func

    def wrapper(*args, **kwargs):
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
        removed = set(kwargs.keys()) - set(filtered.keys())
        if removed:
            print(f"[compat] {func_name}: filtered kwargs: {removed}")
        return original_func(*args, **filtered)

    wrapper.__name__ = original_func.__name__
    wrapper.__qualname__ = original_func.__qualname__
    return wrapper


# Patch SGLangBackendArgs.to_kwargs
from sglang.srt.server_args import ServerArgs

server_args_params = set(inspect.signature(ServerArgs.__init__).parameters.keys())

import specforge.args as sa

_orig_to_kwargs = sa.SGLangBackendArgs.to_kwargs


def _patched_to_kwargs(self):
    kwargs = _orig_to_kwargs(self)
    filtered = {k: v for k, v in kwargs.items() if k in server_args_params}
    removed = set(kwargs.keys()) - set(filtered.keys())
    if removed:
        print(f"[compat] SGLangBackendArgs.to_kwargs: filtered {removed}")
    # No more hardcoded overrides — values come from CLI args
    print(f"[compat] SGLangBackendArgs final kwargs: {filtered}")
    return filtered


sa.SGLangBackendArgs.to_kwargs = _patched_to_kwargs

# Patch init_model_parallel_group
import sglang.srt.distributed
import sglang.srt.distributed.parallel_state as ps

_orig_init_mpg = sglang.srt.distributed.init_model_parallel_group
patched_init_mpg = make_compat_wrapper(_orig_init_mpg, "init_model_parallel_group")
sglang.srt.distributed.init_model_parallel_group = patched_init_mpg

import specforge.modeling.target.sglang_backend.patch as sf_patch

sf_patch.init_model_parallel_group = patched_init_mpg

# Patch GroupCoordinator.__init__
_orig_gc_init = ps.GroupCoordinator.__init__
patched_gc_init = make_compat_wrapper(_orig_gc_init, "GroupCoordinator.__init__")
ps.GroupCoordinator.__init__ = patched_gc_init

# Patch compute_dp_attention_world_info
from sglang.srt.layers import dp_attention as dp_attn

_orig_cdawi = dp_attn.compute_dp_attention_world_info
patched_cdawi = make_compat_wrapper(_orig_cdawi, "compute_dp_attention_world_info")
dp_attn.compute_dp_attention_world_info = patched_cdawi
sf_patch.compute_dp_attention_world_info = patched_cdawi

print("[compat] All patches applied successfully")

# Patch scheduler to warmup then constant LR
import specforge.optimizer as _opt
import torch
_OrigBF16Optimizer = _opt.BF16Optimizer
WARMUP_STEPS = 100
class WarmupConstantLRBF16Optimizer(_OrigBF16Optimizer):
    def __init__(self, *args, **kwargs):
        target_lr = kwargs.get('lr', args[1] if len(args) > 1 else 6e-4)
        super().__init__(*args, **kwargs)
        # Reset optimizer LR (parent's cosine scheduler may have modified it)
        for pg in self.optimizer.param_groups:
            pg['lr'] = target_lr
            pg['initial_lr'] = target_lr
        def lr_lambda(step):
            if step < WARMUP_STEPS:
                return float(step + 1) / float(WARMUP_STEPS)
            return 1.0
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        print(f"[compat] Using warmup({WARMUP_STEPS}) + constant LR = {target_lr}")
_opt.BF16Optimizer = WarmupConstantLRBF16Optimizer
print("[compat] Patched BF16Optimizer for warmup + constant LR")

# Run the actual training script
sys.argv[0] = "/workspace/SpecForge/scripts/train_dflash.py"
runpy.run_path("/workspace/SpecForge/scripts/train_dflash.py", run_name="__main__")
