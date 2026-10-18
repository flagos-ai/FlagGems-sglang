# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Explicit, default-off adapter for the frozen historical R9 W4A8 fork.

Call install() before model construction / Graph capture after explicitly
enabling FLAGGEMS_SGLANG_W4A8_MOE=1 and the four existing experiment flags.
This module is not registered by the current-main plugin entry point. The
source pins intentionally reject a current-main or collector-modified runner.
Uninstall restores owned Python bindings; captured Graphs require recapture.
"""

import contextvars
import copy
import functools
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path

from ._w4a8_native_tactics import BASELINE_TACTICS

ENV = "FLAGGEMS_SGLANG_W4A8_MOE"
REQUIRED_FLAGS = (
    "SGLANG_EXPERIMENTAL_HUMMING_CF32_NOSTREAMK_R10",
    "SGLANG_EXPERIMENTAL_HUMMING_W4A8_BF16WIRE_R18",
    "SGLANG_EXPERIMENTAL_HUMMING_W4A8_W2_STORE_R24",
    "SGLANG_EXPERIMENTAL_HUMMING_W4A8_W13_GEMM_R24",
)
SOURCE_PINS = {
    "humming.py": "bafde29ff520355908778585ac0b49751e8c3f88ca28a30f6c818559a3c6a15a",
    "r18_masked_group0.py": "5b68900d27c7417276a47b311b5b55846447f860efb4a50c4876ada1dcf5b971",
    "r18_w4a8_contract.py": "0d277ec7e525a50adfc0d10ef2aa691df34dc069ae84a7f68d66cd40482015e6",
    "r19_masked_swiglu.py": "eece18dc9012ebfa40ef76e5194fd4f9b7e1347246639e088e338262c9b13a9c",
    "r10_humming_numerics.py": "8fa998f5fdcd8beeb1b2cf66d5affc4948fcde2320e1908f68c8721f0badc4dd",
    "r24_native_a8_w13_tuning.py": "4e63c3b41142816d25a0f1c98368a15ad98c529e661fc8fcf4af4a521c2d6678",
    "r24_native_a8_w2_tuning.py": "5f4eee8f2bf240b3f62630ee340467bea8e3425192f6ee84fa608a7a2335738e",
}
MATH_PINS = {
    "calc_scale": "b53a57f1529ff453206930a478af9dbff286c038ea255b38073bba4bfcd8baf1",
    "finalize_scale": "b4b5d86a8e16c8958ff9161e8b96628a673b711ed49c898736dbbe084c1fd163",
    "quant_tensor": "7e33e8d6e93291b8d14179f9e5496073d632d9643d6d6406b90d3719f2afd464",
}
_context = contextvars.ContextVar("flaggems_w4a8_moe", default=None)
_handle = None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _points(e):
    return (
        (48, 96, 144, 192, 240, 288, 336, 384)
        if e == 48
        else (16, 32, 48, 64, 80, 96, 112, 128, 160)
    )


def tune_table(table, e, stage):
    """Split native (lower,upper] intervals at measured integer hint points."""
    if e not in (16, 48) or stage not in ("w13", "w2"):
        raise ValueError("Unsupported expert geometry")
    out = []
    for lower, upper, original in table:
        cursor = lower
        for point in _points(e):
            if not lower < point <= (upper if upper > 0 else 1 << 62):
                continue
            expected = BASELINE_TACTICS[f"{e}/{stage}/{point}"]
            # Humming's live Python tables contain tuple shapes; the captured
            # native JSON ABI (and frozen bank) represents those as lists.
            if json.loads(json.dumps(original)) != expected:
                raise ValueError(
                    f"Unreviewed native W4A8 tactic at E={e} {stage} hint={point}"
                )
            if cursor < point - 1:
                out.append((cursor, point - 1, copy.deepcopy(original)))
            candidate = copy.deepcopy(original)
            candidate["use_tma_c"] = False
            out.append((point - 1, point, candidate))
            cursor = point
        if upper <= 0 or cursor < upper:
            out.append((cursor, upper, copy.deepcopy(original)))
    # Exact point completeness also prevents a partially changed table.
    for point in _points(e):
        matched = [t for lo, hi, t in out if lo < point <= (hi if hi > 0 else 1 << 62)]
        if len(matched) != 1 or matched[0].get("use_tma_c") is not False:
            raise ValueError("Native tactic table has gaps or overlapping intervals")
    return out


def _eligible(runner, hidden, ids, counts, expected_m, input_scale):
    import torch

    e = getattr(runner, "num_experts", None)
    if (getattr(runner, "global_num_experts", None), e) not in ((384, 48), (128, 16)):
        return None
    layer = runner.layer
    if (
        getattr(layer, "_dsv4_mxfp4_backend", None) != "humming"
        or getattr(layer, "_humming_uses_deepep_fp8_dispatch", None) is not False
    ):
        return None
    if (
        runner.activation != "silu"
        or runner.swiglu_limit != 10.0
        or input_scale is not None
    ):
        return None
    if tuple(hidden.shape) != (e, 1024, 5120) or hidden.dtype != torch.bfloat16:
        return None
    topk = 6 if e == 48 else 3
    if (
        counts.dtype != torch.int32
        or tuple(counts.shape) != (e,)
        or ids.ndim != 2
        or ids.shape[1] != topk
    ):
        return None
    if (
        not hidden.is_cuda
        or not counts.is_cuda
        or hidden.device != counts.device
        or not hidden.is_contiguous()
        or not counts.is_contiguous()
    ):
        return None
    hint = runner.estimate_local_valid_shape_m(ids, expected_m)
    if type(hint) is not int or hint not in _points(e):
        return None
    for stage, n, k in (("w13", 4608, 5120), ("w2", 5120, 2304)):
        m = getattr(layer, "humming_metas", {}).get(stage)
        if m is None:
            return None
        expected = {
            "num_experts": e,
            "shape_n": n,
            "shape_k": k,
            "pad_shape_n": 0,
            "pad_shape_k": 0,
            "input_scale_group_size": 0,
            "weight_scale_group_size": 32,
            "weight_scale_group_size_n": 0,
            "has_zero_point": False,
            "has_bias": False,
            "use_int_weight_scale": False,
            "use_fused_e8m0_scale": True,
            "use_packed_k_layout": False,
        }
        if any(getattr(m, key, None) != value for key, value in expected.items()):
            return None
        if any(
            str(getattr(m, key, None)) != value
            for key, value in {
                "a_dtype": "float8e4m3",
                "b_dtype": "float4e2m1",
                "c_dtype": "bfloat16",
                "as_dtype": "float32",
                "bs_dtype": "float8e8m0",
            }.items()
        ):
            return None
        if any(
            getattr(getattr(m, key, None), "value", getattr(m, key, None)) != value
            for key, value in {
                "mma_type": "wgmma",
                "weight_scale_type": "group",
                "weight_scale_2_type": "tensor",
            }.items()
        ):
            return None
    return (runner, e, hint)


class Installation:
    def __init__(self, bindings):
        self.bindings = bindings

    def uninstall(self):
        global _handle
        for module, name, old, new in reversed(self.bindings):
            if getattr(module, name, None) is new:
                setattr(module, name, old)
        if _handle is self:
            _handle = None


def install():
    """Install only the exact experimental source and compiler/runtime tuple."""
    global _handle
    if os.environ.get(ENV) != "1":
        return None
    if _handle is not None:
        return _handle
    if any(os.environ.get(key) != "1" for key in REQUIRED_FLAGS):
        raise ValueError(
            "W4A8 adapter requires the four existing R10/R18/R24 experiment flags"
        )
    import torch
    import triton

    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (
        torch.cuda.get_device_capability() != (9, 0)
        or props.multi_processor_count != 132
        or "H100" not in props.name
    ):
        raise RuntimeError("W4A8 adapter is qualified only on 132-SM H100")
    if torch.__version__ != "2.13.0+cu130" or triton.__version__ != "3.6.0":
        raise RuntimeError(
            "W4A8 adapter requires the qualified Torch2.13/cu130 and FlagTree Triton3.6 runtime"
        )
    if (
        importlib.metadata.version("flagtree") != "0.7.0+gite86aa7cf"
        or importlib.metadata.version("humming-kernels") != "0.1.12"
    ):
        raise RuntimeError("Unreviewed FlagTree/Humming version")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Install before Graph capture")
    pkg = "sglang.srt.layers.moe.moe_runner"
    runner_module = importlib.import_module(pkg + ".humming")
    root = Path(runner_module.__file__).parent
    for name, expected in SOURCE_PINS.items():
        if _sha(root / name) != expected:
            raise RuntimeError("Unreviewed historical W4A8 source: " + name)
    import humming

    if (
        _sha(Path(humming.__file__).parent / "_native/x86_64/libhumming_launcher.so")
        != "97623b4157123f59bd8506727f7b52a9d8f934ebdc9329bbd5db164af5b088ff"
    ):
        raise RuntimeError("Unreviewed native Humming launcher")
    from humming.ops.input import calc_scale, finalize_scale, quant_tensor

    math = (calc_scale, finalize_scale, quant_tensor)
    for fn in math:
        if hashlib.sha256(fn.src.encode()).hexdigest() != MATH_PINS[fn.__name__]:
            raise RuntimeError("Unreviewed native quant math: " + fn.__name__)
    from flag_gems.fused.dsv41.masked_row_quant import (
        MASKED_ROW_QUANT_ABI,
        masked_row_quant,
    )

    if MASKED_ROW_QUANT_ABI != 1:
        raise RuntimeError("Incompatible FlagGems masked row quant ABI")
    contract = importlib.import_module(pkg + ".r18_w4a8_contract")
    contract.validate_configuration()
    if not contract.ENABLED:
        raise RuntimeError("R18 must be enabled before importing the runner")
    quant_module = importlib.import_module(pkg + ".r18_masked_group0")
    cls = runner_module.HummingRunnerCore
    original_run = cls._run_grouped_masked_gemm
    original_configs = cls.get_humming_gemm_configs
    original_quant = quant_module.masked_quant

    @functools.wraps(original_run)
    def run(
        self,
        hidden_states,
        topk_ids,
        topk_weights,
        expert_num_tokens,
        expected_m,
        hidden_states_scale=None,
    ):
        ctx = _eligible(
            self,
            hidden_states,
            topk_ids,
            expert_num_tokens,
            expected_m,
            hidden_states_scale,
        )
        token = _context.set(ctx)
        try:
            return original_run(
                self,
                hidden_states,
                topk_ids,
                topk_weights,
                expert_num_tokens,
                expected_m,
                hidden_states_scale,
            )
        finally:
            _context.reset(token)

    @functools.wraps(original_configs)
    def configs(self, gemm_type):
        native = original_configs(self, gemm_type)
        ctx = _context.get()
        if (
            ctx is None
            or ctx[0] is not self
            or getattr(gemm_type, "value", gemm_type) != "grouped_masked"
        ):
            return native
        if native["compute_config"] != {
            "gemm_type": "grouped_masked",
            "use_f16_accum": False,
        }:
            raise ValueError(
                "W4A8 adapter requires original grouped-masked FP32 compute"
            )
        cache = getattr(self, "_flaggems_w4a8_configs", None)
        if cache is not None and cache[0] is native:
            return cache[1]
        changed = copy.deepcopy(native)
        for stage in ("w13", "w2"):
            table = tune_table(native[stage + "_tuning_config"], ctx[1], stage)
            changed[stage + "_tuning_config"] = table
            changed[stage + "_tuning_config_str"] = json.dumps(table)
        self._flaggems_w4a8_configs = (native, changed)
        return changed

    @functools.wraps(original_quant)
    def quant(inputs, quanted, scales, masked_m, *, config=None):
        ctx = _context.get()
        if ctx is None or ctx[2] == (48 if ctx[1] == 48 else 16):
            return original_quant(inputs, quanted, scales, masked_m, config=config)
        if config is None or (config.kind, config.row_splits, config.num_warps) != (
            "compact",
            16,
            8,
        ):
            raise ValueError("Original compact16w8 launch contract changed")
        if tuple(inputs.shape[:2]) != (ctx[1], 1024) or inputs.shape[-1] not in (
            5120,
            2304,
        ):
            raise ValueError("Unreviewed live expert row geometry")
        return masked_row_quant(
            inputs,
            quanted,
            scales,
            masked_m,
            calc_scale=math[0],
            finalize_scale=math[1],
            quant_tensor=math[2],
            workers=132,
            num_warps=4,
        )

    bindings = [
        (cls, "_run_grouped_masked_gemm", original_run, run),
        (cls, "get_humming_gemm_configs", original_configs, configs),
        (quant_module, "masked_quant", original_quant, quant),
    ]
    for module, name, old, new in bindings:
        setattr(module, name, new)
    _handle = Installation(bindings)
    return _handle
