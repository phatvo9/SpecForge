"""
Minimal wrapper: forces trust_remote_code for Kimi K2.6.
Only patches SGLangBackendArgs if sglang backend is used (not sglang-server).
"""
import os
import runpy
import sys

os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "true"
os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "1"

import transformers
_orig_tok = transformers.AutoTokenizer.from_pretrained.__func__
@classmethod
def _patched_tok(cls, *a, **kw):
    kw.setdefault("trust_remote_code", True)
    return _orig_tok(cls, *a, **kw)
transformers.AutoTokenizer.from_pretrained = _patched_tok

_orig_cfg = transformers.AutoConfig.from_pretrained.__func__
@classmethod
def _patched_cfg(cls, *a, **kw):
    kw.setdefault("trust_remote_code", True)
    return _orig_cfg(cls, *a, **kw)
transformers.AutoConfig.from_pretrained = _patched_cfg

# Only patch SGLangBackendArgs if using in-process sglang backend
if "--target-model-backend" in sys.argv and "sglang-server" not in sys.argv:
    try:
        import inspect
        from sglang.srt.server_args import ServerArgs
        import specforge.args as sa

        server_args_params = set(inspect.signature(ServerArgs.__init__).parameters.keys())
        _orig_to_kwargs = sa.SGLangBackendArgs.to_kwargs

        def _patched_to_kwargs(self):
            kwargs = _orig_to_kwargs(self)
            filtered = {k: v for k, v in kwargs.items() if k in server_args_params}
            removed = set(kwargs.keys()) - set(filtered.keys())
            if removed:
                print(f"[compat] SGLangBackendArgs: filtered {removed}")
            return filtered

        sa.SGLangBackendArgs.to_kwargs = _patched_to_kwargs
    except ImportError:
        pass

sys.argv[0] = "/workspace/SpecForge/scripts/train_dflash.py"
runpy.run_path("/workspace/SpecForge/scripts/train_dflash.py", run_name="__main__")
