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

"""per_token_group_quant_8bit_v2 baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.per_token_group_quant_8bit_v2.
        per_token_group_quant_8bit_v2
    调用形态复刻 baseline.py：
        q, s = output_q.clone(), output_s.clone()
        per_token_group_quant_8bit_v2(input, q, s, group_size, eps, min_8bit, max_8bit)
        return q, s
    kernel 原地写入 q/s（baseline 先 clone）。IS_INPLACE=True：采样前重建实参。

输入构造复刻 cases.py 的 BENCH_CASES（group_size=128, eps=1e-10,
min/max=∓fp8_max）：
    input=randn(m,k).to(bf16)；output_q=zeros(m,k) fp8_e4m3；
    output_s=zeros(m,k//group) fp32。
BENCH: _case(m,k,seed=9) for m in (1,8,64,512,4096) for k in (2048,4096,7168)。
"""

import torch

OP_NAME = "per_token_group_quant_8bit_v2"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 原地写入 output_q / output_s

_GROUP = 128
_EPS = 1e-10
_SEED = 9
_SHAPES = [
    {"m": m, "k": k}
    for m in (1, 8, 64, 512, 4096)
    for k in (2048, 4096, 7168)
]


def native():
    """解析 per_token_group_quant_8bit_v2。"""
    try:
        from sglang.kernels.ops.quantization.per_token_group_quant_8bit_v2 import (
            per_token_group_quant_8bit_v2 as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    m, k = binding["m"], binding["k"]
    g = torch.Generator(device=device).manual_seed(_SEED)
    x = torch.randn(m, k, device=device, generator=g).to(dtype).contiguous()
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    output_q = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device=device)
    output_s = torch.zeros(m, k // _GROUP, dtype=torch.float32, device=device)
    return (x, output_q, output_s, _GROUP, _EPS, -fp8_max, fp8_max), {}


def key_shape(binding):
    return [binding["m"], binding["k"]]


def config(binding, dtype):
    m, k = binding["m"], binding["k"]
    dt = str(dtype)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    return {
        "inputs": {
            "input": {"shape": [m, k], "dtype": dt},
            "output_q": {"shape": [m, k], "dtype": "torch.float8_e4m3fn",
                         "note": "原地写回"},
            "output_s": {"shape": [m, k // _GROUP], "dtype": "torch.float32",
                         "note": "原地写回"},
            "group_size": {"scalar": _GROUP},
            "eps": {"scalar": _EPS},
            "min_8bit": {"scalar": -fp8_max},
            "max_8bit": {"scalar": fp8_max},
        },
        "outputs": {
            "output_q": {"shape": [m, k], "dtype": "torch.float8_e4m3fn"},
            "output_s": {"shape": [m, k // _GROUP], "dtype": "torch.float32"},
        },
        "dims": {"m": m, "k": k, "group_size": _GROUP},
    }
