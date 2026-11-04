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

"""per_token_group_quant_fp8 baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.fp8_kernel.per_token_group_quant_fp8
    调用形态复刻 problems/quantization/per_token_group_quant_fp8/baseline.py：
        per_token_group_quant_fp8(x, group_size)
    返回 (q, scale) 新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES（group_size=128）：
    x=(rand(m,k)*2-1).to(bf16).contiguous()。
BENCH: _x(m,k) for m in (1,8,64,512,4096) for k in (2048,4096,8192)。
"""

import torch

OP_NAME = "per_token_group_quant_fp8"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_GROUP = 128
_SHAPES = [
    {"m": m, "k": k}
    for m in (1, 8, 64, 512, 4096)
    for k in (2048, 4096, 8192)
]


def native():
    """解析 per_token_group_quant_fp8。"""
    try:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            per_token_group_quant_fp8 as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    m, k = binding["m"], binding["k"]
    g = torch.Generator(device=device).manual_seed(0)
    x = (
        (torch.rand(m, k, generator=g, device=device, dtype=torch.float32) * 2 - 1)
        .to(dtype)
        .contiguous()
    )
    return (x, _GROUP), {}


def key_shape(binding):
    return [binding["m"], binding["k"]]


def config(binding, dtype):
    m, k = binding["m"], binding["k"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, k], "dtype": dt},
            "group_size": {"scalar": _GROUP},
        },
        "outputs": {
            "q": {"shape": [m, k], "dtype": "torch.float8_e4m3fn"},
            "scale": {"shape": [m, k // _GROUP], "dtype": "torch.float32"},
        },
        "dims": {"m": m, "k": k, "group_size": _GROUP},
    }
