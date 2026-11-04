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

"""per_tensor_quant_fp8 baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.per_tensor_quant_fp8.per_tensor_quant_fp8
    调用形态复刻 problems/quantization/per_tensor_quant_fp8/baseline.py：
        q, s = output_q.clone(), output_s.clone()
        per_tensor_quant_fp8(input, q, s, is_static); return q, s
    kernel 原地写入 q/s（baseline 先 clone）。IS_INPLACE=True：采样前重建实参。

输入构造复刻 cases.py 的 BENCH_CASES（is_static=False）：
    input=randn(m,k).to(bf16)；output_q=zeros(m,k) fp8_e4m3；output_s=zeros(1) fp32。
BENCH: _case(m,k,seed=9) for m in (1,8,64,512,4096) for k in (2048,4096,8192)。
"""

import torch

OP_NAME = "per_tensor_quant_fp8"
DTYPES = [torch.bfloat16]
IS_INPLACE = True  # kernel 原地写入 output_q / output_s

_SEED = 9
_IS_STATIC = False
_SHAPES = [
    {"m": m, "k": k}
    for m in (1, 8, 64, 512, 4096)
    for k in (2048, 4096, 8192)
]


def native():
    """解析 per_tensor_quant_fp8。"""
    try:
        from sglang.kernels.ops.quantization.per_tensor_quant_fp8 import (
            per_tensor_quant_fp8 as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    m, k = binding["m"], binding["k"]
    g = torch.Generator(device=device).manual_seed(_SEED)
    x = torch.randn(m, k, device=device, generator=g).to(dtype)
    output_q = torch.zeros(m, k, dtype=torch.float8_e4m3fn, device=device)
    output_s = torch.ones(1, dtype=torch.float32, device=device) * (
        0.01 if _IS_STATIC else 0.0
    )
    return (x, output_q, output_s, _IS_STATIC), {}


def key_shape(binding):
    return [binding["m"], binding["k"]]


def config(binding, dtype):
    m, k = binding["m"], binding["k"]
    dt = str(dtype)
    return {
        "inputs": {
            "input": {"shape": [m, k], "dtype": dt},
            "output_q": {"shape": [m, k], "dtype": "torch.float8_e4m3fn",
                         "note": "原地写回"},
            "output_s": {"shape": [1], "dtype": "torch.float32", "note": "原地写回"},
            "is_static": {"scalar": _IS_STATIC},
        },
        "outputs": {
            "output_q": {"shape": [m, k], "dtype": "torch.float8_e4m3fn"},
            "output_s": {"shape": [1], "dtype": "torch.float32"},
        },
        "dims": {"m": m, "k": k, "is_static": _IS_STATIC},
    }
