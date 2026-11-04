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

"""fp8_quantize baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.fp8_quantize.fp8_quantize
    调用形态复刻 problems/quantization/fp8_quantize/baseline.py：
        fp8_quantize(x, scale_inv=scale_inv, fp8_dtype=fp8_dtype)
    返回新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 problems/quantization/fp8_quantize/cases.py 的 BENCH_CASES：
    x=randn(m,n).to(bf16)；scale_inv=1.0；fp8_dtype=float8_e4m3fn。
BENCH: _case(_x((m,n))) for m in (1,8,64,512,4096) for n in (1024,4096,8192)。
dtype 取 cases 的输入 dtype [bfloat16]。
"""

import torch

OP_NAME = "fp8_quantize"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SCALE_INV = 1.0
_SHAPES = [
    {"m": m, "n": n}
    for m in (1, 8, 64, 512, 4096)
    for n in (1024, 4096, 8192)
]


def native():
    """解析 fp8_quantize。"""
    try:
        from sglang.kernels.ops.quantization.fp8_quantize import fp8_quantize as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["m"], binding["n"]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(m, n, generator=g, device=device, dtype=torch.float32).to(dtype)
    return (x,), {"scale_inv": _SCALE_INV, "fp8_dtype": torch.float8_e4m3fn}


def key_shape(binding):
    return [binding["m"], binding["n"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, n], "dtype": dt},
            "scale_inv": {"scalar": _SCALE_INV},
            "fp8_dtype": {"scalar": "torch.float8_e4m3fn"},
        },
        "outputs": {
            "out": {"shape": [m, n], "dtype": "torch.float8_e4m3fn"},
        },
        "dims": {"m": m, "n": n},
    }
