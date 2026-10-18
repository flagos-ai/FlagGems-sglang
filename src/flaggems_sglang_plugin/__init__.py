# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional SGLang adapter; importing this module never initializes CUDA."""

import functools
import hashlib
import importlib
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
_handle = None
_configure_original = None
_SOURCE_PINS = {
    "sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer": "a367425a9c89fb087650d857423e3df75c5ab102b68baf9821135c4f21712fb3",  # noqa: E501 (source identity)
    "sglang.kernels.ops.quantization.fp8_kernel": "5422b04361bc1d7cde2aba2e1077e150d477dedab255e1c9628a49ddcb9e9d24",
    "sglang.kernels.ops.layernorm.mhc": "a2377a7880c98b8559718021841af9e2e3497cb19a93e613e9efa13fdaae00b6",
}


class Installation:
    """Own only bindings replaced by this adapter; restore them by identity."""

    def __init__(self, bindings, replacements):
        self.bindings = bindings
        self.replacements = replacements

    def uninstall(self):
        global _handle
        for module, name, old, new in reversed(self.bindings):
            if getattr(module, name, None) is new:
                setattr(module, name, old)
        if _handle is self:
            _handle = None


def install():
    """Install the source-pinned SM90 adapter in the current scheduler process.

    The full-model adapter is deliberately narrower than the public numerical
    APIs. Other SGLang source revisions must be reviewed explicitly before
    extending these pins. No model weights, routing, communication or caches
    are replaced. Invoke before model construction / CUDA Graph capture.
    """
    global _handle
    if _handle is not None:
        return _handle
    import torch

    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("DSV4.1 adapter requires SM90")
    modules = {}
    for name, expected in _SOURCE_PINS.items():
        module = importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError("unreviewed SGLang operator source: " + name)
        modules[name] = module

    import flaggems_sglang as gems
    from flag_gems.fused.dsv41 import mhc as model_mhc
    from flag_gems.fused.dsv41.fp4_indexer import (
        FP4_INDEX_SCORE_ABI,
        fp4_index_logits_decode,
    )
    from flag_gems.ops._w8a8_block_fp8_configs import (
        _H100_MIDRANGE_FP8_CONFIGS,
        BLOCK_FP8_SM90_ABI,
    )
    from flag_gems.ops.w8a8_block_fp8_matmul import w8a8_block_fp8_matmul

    if (BLOCK_FP8_SM90_ABI, FP4_INDEX_SCORE_ABI, model_mhc.MHC_EXACT_ABI) != (1, 1, 1):
        raise RuntimeError(
            "incompatible FlagGems SM90 operator ABI; install the pinned dsv41 extra"
        )

    fp4 = modules["sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer"]
    mhc = modules["sglang.kernels.ops.layernorm.mhc"]
    fp8 = modules["sglang.kernels.ops.quantization.fp8_kernel"]
    replacements = {}
    replacements[fp4.fp4_index_logits_decode] = fp4_index_logits_decode
    for name in [
        "fp4_index_logits_req_to_token",
        "fp4_index_logits_candidate_blocks",
    ]:
        replacements[getattr(fp4, name)] = getattr(gems, name)
    for name in ["hc_mix_stats", "hc_mix_stats_sinkhorn_exact", "hc_combine"]:
        replacements[getattr(mhc, name)] = getattr(model_mhc, name)
    original = fp8.w8a8_block_fp8_matmul_triton

    @functools.wraps(original)
    def block_fp8(A, B, As, Bs, block_size, output_dtype=torch.float16):
        supported = (
            not torch._dynamo.is_compiling()
            and A.ndim == B.ndim == 2
            and tuple(block_size) == (32, 32)
            and A.dtype == B.dtype == torch.float8_e4m3fn
            and As.dtype == Bs.dtype == torch.float32
            and output_dtype in (torch.bfloat16, torch.float32)
            and A.is_contiguous()
            and B.is_contiguous()
            and (A.shape[0], B.shape[0], A.shape[1]) in _H100_MIDRANGE_FP8_CONFIGS
        )
        if supported:
            return w8a8_block_fp8_matmul(A, B, As, Bs, [32, 32], output_dtype)
        return original(A, B, As, Bs, block_size, output_dtype)

    replacements[original] = block_fp8
    bindings = []
    # Rebind already imported aliases as well as the defining modules. Match
    # function objects, never broad attribute names or unrelated libraries.
    for name, module in list(sys.modules.items()):
        if not name.startswith("sglang.") or module is None:
            continue
        for attr, value in list(vars(module).items()):
            if not callable(value):
                continue
            try:
                replacement = replacements.get(value)
            except TypeError:
                continue
            if replacement is not None:
                setattr(module, attr, replacement)
                bindings.append((module, attr, value, replacement))
    _handle = Installation(bindings, replacements)
    logger.info(
        "FlagGems DSV4.1 SM90 operator adapter installed: device=%d, bindings=%d",
        torch.cuda.current_device(),
        len(bindings),
    )
    return _handle


def load_plugin():
    """SGLang entry point; default-off and defer GPU imports to scheduler setup."""
    global _configure_original
    if os.getenv("FLAGGEMS_SGLANG_DSV41", "0") != "1":
        return
    scheduler = sys.modules.get("sglang.srt.managers.scheduler")
    if scheduler is None or _configure_original is not None:
        return
    _configure_original = scheduler.configure_scheduler_process

    @functools.wraps(_configure_original)
    def configure(*args, **kwargs):
        result = _configure_original(*args, **kwargs)
        import torch

        gpu_id = kwargs.get("gpu_id", args[1] if len(args) > 1 else None)
        if gpu_id is None:
            raise RuntimeError("scheduler GPU identity missing")
        torch.cuda.set_device(gpu_id)
        install()
        return result

    scheduler.configure_scheduler_process = configure


__all__ = ["install", "load_plugin"]
