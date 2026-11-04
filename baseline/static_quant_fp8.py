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

"""static_quant_fp8 baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.fp8_kernel.static_quant_fp8
    调用形态复刻 baseline.py：
        static_quant_fp8(x, x_s, repeat_scale=False)
    返回 (q, scale) 新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    x=(randn(m,n)*3).to(bf16)；x_s=(|x|.max()/fp8_max) 形如 (1,) fp32。
BENCH: _case(m,n) for m in (1,8,64,512,4096) for n in (2048,4096,8192)。
"""

import torch

OP_NAME = "static_quant_fp8"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SHAPES = [
    {"m": m, "n": n}
    for m in (1, 8, 64, 512, 4096)
    for n in (2048, 4096, 8192)
]


def native():
    """解析 static_quant_fp8。"""
    try:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            static_quant_fp8 as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["m"], binding["n"]
    g = torch.Generator(device=device).manual_seed(0)
    x = (torch.randn(m, n, generator=g, device=device, dtype=torch.float32) * 3).to(dtype)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = (x.abs().float().max() / fp8_max).reshape(1).contiguous()
    return (x, scale), {}


def key_shape(binding):
    return [binding["m"], binding["n"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, n], "dtype": dt},
            "x_s": {"shape": [1], "dtype": "torch.float32"},
        },
        "outputs": {
            "q": {"shape": [m, n], "dtype": "torch.float8_e4m3fn"},
            "scale": {"shape": [1], "dtype": "torch.float32"},
        },
        "dims": {"m": m, "n": n, "repeat_scale": False},
    }
