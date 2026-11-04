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

"""per_token_quant_int8_sum baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.int8_kernel.per_token_quant_int8
    （与 per_token_quant_int8 同一 kernel，区别在 cal_sum=True 额外返回按行求和）
    调用形态复刻 baseline.py：
        per_token_quant_int8(x, scale_dtype=scale_dtype, cal_sum=cal_sum)
    返回 (q, scale, row_sum) 新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES（cal_sum=True, scale_dtype=fp32）：
    x=randn(m,n).to(bf16).contiguous()。
BENCH: _case(m,n,seed=9) for m in (1,8,64,512,4096) for n in (2048,4096,8192)。
"""

import torch

OP_NAME = "per_token_quant_int8_sum"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SEED = 9
_CAL_SUM = True
_SHAPES = [
    {"m": m, "n": n}
    for m in (1, 8, 64, 512, 4096)
    for n in (2048, 4096, 8192)
]


def native():
    """解析 per_token_quant_int8（cal_sum 路径）。"""
    try:
        from sglang.kernels.ops.quantization.int8_kernel import (
            per_token_quant_int8 as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["m"], binding["n"]
    g = torch.Generator(device=device).manual_seed(_SEED)
    x = torch.randn(m, n, device=device, generator=g).to(dtype).contiguous()
    return (x,), {"scale_dtype": torch.float32, "cal_sum": _CAL_SUM}


def key_shape(binding):
    return [binding["m"], binding["n"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, n], "dtype": dt},
            "scale_dtype": {"scalar": "torch.float32"},
            "cal_sum": {"scalar": _CAL_SUM},
        },
        "outputs": {
            "q": {"shape": [m, n], "dtype": "torch.int8"},
            "scale": {"shape": [m, 1], "dtype": "torch.float32"},
            "row_sum": {"shape": [m, 1], "dtype": "torch.float32",
                        "note": "cal_sum=True 额外返回的按行和"},
        },
        "dims": {"m": m, "n": n, "cal_sum": _CAL_SUM},
    }
