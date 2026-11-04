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

"""fla_layernorm_gated baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.attention.fla.fused_norm_gate.layer_norm_gated_fwd
    调用形态复刻 problems/fla/layernorm_gated/baseline.py：
        layer_norm_gated_fwd(x, g, weight, bias,
            activation=activation, eps=eps, out_dtype=x.dtype,
            is_rms_norm=is_rms_norm)
    返回 (y, mean, rstd, residual_out)，另分配输出，不原地写回。

输入构造复刻 problems/fla/layernorm_gated/cases.py 的 BENCH_CASES：
    _case(t, 2048) for t in (1, 8, 64, 512, 4096)
    x/g = randn(t, d).to(dtype); weight = randn(d).to(dtype);
    bias = randn(d).to(dtype)（has_bias 默认 True）；
    activation="swish", eps=1e-5, is_rms_norm=True。
dtype 取 benchmark 的 [bfloat16]。
"""

import torch

OP_NAME = "fla_layernorm_gated"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_EPS = 1e-5
_TS = [1, 8, 64, 512, 4096]
_D = 2048


def native():
    """解析 sglang.kernels.ops.attention.fla.fused_norm_gate.layer_norm_gated_fwd。"""
    try:
        from sglang.kernels.ops.attention.fla.fused_norm_gate import (
            layer_norm_gated_fwd as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"t": t, "d": _D} for t in _TS]


def build_inputs(binding, dtype, device):
    t, d = binding["t"], binding["d"]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(t, d, generator=g, device=device, dtype=torch.float32).to(dtype)
    gate = torch.randn(t, d, generator=g, device=device, dtype=torch.float32).to(dtype)
    weight = torch.randn(d, generator=g, device=device, dtype=torch.float32).to(dtype)
    bias = torch.randn(d, generator=g, device=device, dtype=torch.float32).to(dtype)
    return (x, gate, weight, bias), {
        "activation": "swish",
        "eps": _EPS,
        "out_dtype": x.dtype,
        "is_rms_norm": True,
    }


def key_shape(binding):
    return [binding["t"], binding["d"]]


def config(binding, dtype):
    t, d = binding["t"], binding["d"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [t, d], "dtype": dt},
            "g": {"shape": [t, d], "dtype": dt},
            "weight": {"shape": [d], "dtype": dt},
            "bias": {"shape": [d], "dtype": dt},
            "activation": {"scalar": "swish"},
            "eps": {"scalar": _EPS},
            "out_dtype": {"scalar": dt},
            "is_rms_norm": {"scalar": True},
        },
        "outputs": {
            "y": {"shape": [t, d], "dtype": dt},
            "mean": {"note": "RMSNorm 下为 None"},
            "rstd": {"shape": [t], "dtype": "torch.float32"},
            "residual_out": {"note": "无 residual 融合时为 None"},
        },
        "dims": {"t": t, "d": d},
    }
