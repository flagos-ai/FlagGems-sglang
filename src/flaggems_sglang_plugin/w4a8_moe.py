# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Source-qualified W4A8 masked MoE scheduling for SM90.

Install before model construction / Graph capture. Actual layer metadata
selects W4A8; the adapter does not change the model's quantization settings.
"""

import copy
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

from ._w4a8_native_tactics import BASELINE_TACTICS

SOURCE_PINS = {
    "humming.py": (
        "bafde29ff520355908778585ac0b49751e8c3f88ca28a30f6c818559a3c6a15a",
        "57bd948627238de937ad0394f59e40a8b49e6eca91e62c74880df19e0ff847b6",
    )
}
MATH_PINS = {
    "calc_scale": "b53a57f1529ff453206930a478af9dbff286c038ea255b38073bba4bfcd8baf1",
    "finalize_scale": "b4b5d86a8e16c8958ff9161e8b96628a673b711ed49c898736dbbe084c1fd163",
    "quant_tensor": "7e33e8d6e93291b8d14179f9e5496073d632d9643d6d6406b90d3719f2afd464",
}
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
    if (getattr(runner, "global_num_experts", None), e) not in (
        (384, 48),
        (128, 16),
    ):
        return None
    layer = runner.layer
    if (
        getattr(layer, "_dsv4_mxfp4_backend", None) != "humming"
        or getattr(layer, "_humming_uses_deepep_fp8_dispatch", False) is not False
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


def install_if_supported():
    """Use the historical adapter when its runner interface is present."""
    name = "sglang.srt.layers.moe.moe_runner.humming"
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError as exc:
        if name == exc.name or name.startswith(exc.name + "."):
            return None
        raise
    if spec is None or spec.origin is None:
        return None
    root = Path(spec.origin).parent
    if (
        not (root / "humming.py").is_file()
        or _sha(root / "humming.py") not in SOURCE_PINS["humming.py"]
    ):
        return None
    return install()


def _configs(runner, experts):
    cached = getattr(runner, "_flaggems_w4a8_configs", None)
    if cached is not None:
        return cached
    compute = {"gemm_type": "grouped_masked", "use_f16_accum": False}
    result = {
        "compute_config": compute,
        "compute_config_str": json.dumps(compute),
    }
    points = _points(experts)
    for stage in ("w13", "w2"):
        table = [
            (
                0 if i == 0 else points[i - 1],
                point,
                copy.deepcopy(BASELINE_TACTICS[f"{experts}/{stage}/{point}"]),
            )
            for i, point in enumerate(points)
        ]
        tuned = tune_table(table, experts, stage)
        result[stage + "_tuning_config"] = tuned
        result[stage + "_tuning_config_str"] = json.dumps(tuned)
    runner._flaggems_w4a8_configs = result
    return result


def _run_w4a8(
    runner,
    hidden,
    ids,
    counts,
    ctx,
    *,
    torch,
    method,
    gemm_type,
    compact_quant,
    activate,
    quant,
):
    # Preserve native workspace allocation and its two separate activation /
    # quantization writes: their aliased storage cannot be fused in place.
    _, experts, hint = ctx
    configs = _configs(runner, experts)
    flat = hidden.view(-1, hidden.shape[-1])
    buffers = runner.prepare_buffers(
        hidden_states=flat,
        topk_ids=ids,
        gemm_type=gemm_type,
    )
    rows = flat.shape[0]
    for key in ("r18_fc1_scale", "r18_fc2_scale"):
        if key not in buffers:
            buffers[key] = torch.empty(
                (rows, 1), dtype=torch.float32, device=hidden.device
            )

    def quantize(inputs, output, scales):
        if hint == _points(experts)[0]:
            return compact_quant(inputs, output, scales, counts)
        return quant(inputs, output, scales, counts)

    inputs, scales = quantize(
        hidden,
        buffers["quanted_gate_up_input"],
        buffers["r18_fc1_scale"],
    )
    method.forward_layer(
        layer=runner.layer,
        inputs=inputs,
        input_scale=scales,
        outputs=buffers["gate_up_output"],
        valid_shape_m=hint,
        expert_layout=counts,
        compute_config=configs["compute_config_str"],
        tuning_config=configs["w13_tuning_config_str"],
        sublayer_name="w13",
    )
    gate_up = buffers["gate_up_output"].view(experts, hidden.shape[1], -1)
    activate(gate_up, buffers["activation_output"], counts)
    inputs, scales = quantize(
        buffers["activation_output"].view(experts, hidden.shape[1], -1),
        buffers["quanted_down_input"],
        buffers["r18_fc2_scale"],
    )
    method.forward_layer(
        layer=runner.layer,
        inputs=inputs,
        input_scale=scales,
        outputs=buffers["down_output"].view(-1, hidden.shape[-1]),
        valid_shape_m=hint,
        expert_layout=counts,
        compute_config=configs["compute_config_str"],
        tuning_config=configs["w2_tuning_config_str"],
        sublayer_name="w2",
    )
    return buffers["down_output"]


def install():
    """Install the measured schedules on the qualified runner/runtime."""
    global _handle
    if _handle is not None:
        return _handle
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
        if _sha(root / name) not in expected:
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
    from flaggems_sglang_plugin.compat.dsv41.masked_row_quant import (
        MASKED_ROW_QUANT_ABI,
        masked_row_quant,
    )

    if MASKED_ROW_QUANT_ABI != 1:
        raise RuntimeError("Incompatible FlagGems masked row quant ABI")
    from flaggems_sglang_plugin.compat.dsv41.masked_swiglu import masked_swiglu

    activation_module = importlib.import_module(
        "sglang.kernels.ops.moe.fused_moe_triton_kernels"
    )

    def compact_quant(inputs, output, scales, counts, *, config=None):
        return masked_row_quant(
            inputs,
            output,
            scales,
            counts,
            calc_scale=math[0],
            finalize_scale=math[1],
            quant_tensor=math[2],
            row_splits=16,
            num_warps=8,
        )

    def activate(inputs, output, counts, **kwargs):
        return masked_swiglu(
            inputs,
            output,
            counts,
            activate=activation_module._apply_activation,
        )

    cls = runner_module.HummingRunnerCore
    original = cls._run_grouped_masked_gemm

    def quant(inputs, quanted, scales, counts):
        return masked_row_quant(
            inputs,
            quanted,
            scales,
            counts,
            calc_scale=math[0],
            finalize_scale=math[1],
            quant_tensor=math[2],
            workers=132,
            num_warps=4,
        )

    @functools.wraps(original)
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
        if ctx is None:
            args = (
                self,
                hidden_states,
                topk_ids,
                topk_weights,
                expert_num_tokens,
                expected_m,
            )
            if hidden_states_scale is None:
                return original(*args)
            return original(*args, hidden_states_scale=hidden_states_scale)
        return _run_w4a8(
            self,
            hidden_states,
            topk_ids,
            expert_num_tokens,
            ctx,
            torch=torch,
            method=runner_module.HummingMethod,
            gemm_type=runner_module.HummingGemmType.GROUPED_MASKED,
            compact_quant=compact_quant,
            activate=activate,
            quant=quant,
        )

    cls._run_grouped_masked_gemm = run
    _handle = Installation([(cls, "_run_grouped_masked_gemm", original, run)])
    return _handle
