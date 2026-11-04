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

"""mamba_layernorm_gated baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.attention.fla.layernorm_gated._layer_norm_fwd
    调用形态复刻 problems/mamba/layernorm_gated/baseline.py：
        out, mean, rstd = _layer_norm_fwd(x, weight, bias, eps, z=z,
            group_size=group_size, norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm)
    另分配输出（out/mean/rstd），不原地写回输入。本题只记 out。

输入构造复刻 problems/mamba/layernorm_gated/cases.py 的 BENCH_CASES
（均为 group_size=128、norm_before_gate=True、is_rms_norm=True、has_z=True）：
    x = randn(m, n).to(dtype); weight = randn(n).to(dtype); bias = randn(n).to(dtype)
    z = randn(m, n).to(dtype); eps=1e-5
dtype 取 benchmark/test_mamba_layernorm_gated.py 的 [bfloat16]。
"""

import torch

OP_NAME = "mamba_layernorm_gated"
DTYPES = [torch.bfloat16]
IS_INPLACE = False  # 另分配输出

_EPS = 1e-5
_GROUP_SIZE = 128
# (m, n)
_SHAPES = [(m, 4096) for m in (1, 8, 64, 512, 4096)]


def native():
    """解析 sglang.kernels.ops.attention.fla.layernorm_gated._layer_norm_fwd。"""
    try:
        from sglang.kernels.ops.attention.fla.layernorm_gated import (
            _layer_norm_fwd as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "n": n} for (m, n) in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["m"], binding["n"]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(m, n, generator=g, device=device, dtype=torch.float32).to(dtype)
    weight = torch.randn(n, generator=g, device=device, dtype=torch.float32).to(dtype)
    bias = torch.randn(n, generator=g, device=device, dtype=torch.float32).to(dtype)
    z = torch.randn(m, n, generator=g, device=device, dtype=torch.float32).to(dtype)
    return (x, weight, bias, _EPS), {
        "z": z,
        "group_size": _GROUP_SIZE,
        "norm_before_gate": True,
        "is_rms_norm": True,
    }


def key_shape(binding):
    return [binding["m"], binding["n"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    ngroups = n // _GROUP_SIZE
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, n], "dtype": dt},
            "weight": {"shape": [n], "dtype": dt},
            "bias": {"shape": [n], "dtype": dt},
            "eps": {"scalar": _EPS},
            "z": {"shape": [m, n], "dtype": dt},
            "group_size": {"scalar": _GROUP_SIZE},
            "norm_before_gate": {"scalar": True},
            "is_rms_norm": {"scalar": True},
        },
        "outputs": {
            "out": {"shape": [m, n], "dtype": dt},
            "mean": {"shape": [m * ngroups], "dtype": "torch.float32",
                     "note": "本题只记 out"},
            "rstd": {"shape": [m * ngroups], "dtype": "torch.float32",
                     "note": "本题只记 out"},
        },
        "dims": {
            "m": m, "n": n, "group_size": _GROUP_SIZE, "ngroups": ngroups,
        },
    }
