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

"""to_mxfp8 baseline（方案 B，sglang）。

native：sglang.kernels.ops.quantization.mxfp8_quant.to_mxfp8
    调用形态复刻 baseline.py：
        t = to_mxfp8(x); return t.data, t.scale
    返回一个带 .data/.scale 的量化张量对象，不原地写回；故 IS_INPLACE=False。
    native 本身即 to_mxfp8(x)；解包 .data/.scale 是 baseline 的后处理，不影响
    被采样的 kernel 调用。

输入构造复刻 cases.py 的 BENCH_CASES：
    x=randn(m,k).to(bf16)。
BENCH: _case(m,k,seed=9) for m in (1,8,64,512,4096) for k in (1024,4096,8192)。
"""

import torch

OP_NAME = "to_mxfp8"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_SEED = 9
_SHAPES = [
    {"m": m, "k": k}
    for m in (1, 8, 64, 512, 4096)
    for k in (1024, 4096, 8192)
]


def native():
    """解析 to_mxfp8。"""
    try:
        from sglang.kernels.ops.quantization.mxfp8_quant import to_mxfp8 as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    m, k = binding["m"], binding["k"]
    g = torch.Generator(device=device).manual_seed(_SEED)
    x = torch.randn(m, k, device=device, generator=g).to(dtype)
    return (x,), {}


def key_shape(binding):
    return [binding["m"], binding["k"]]


def config(binding, dtype):
    m, k = binding["m"], binding["k"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, k], "dtype": dt},
        },
        "outputs": {
            "data": {"shape": [m, k], "dtype": "torch.float8_e4m3fn"},
            "scale": {"shape": [m, k // 32], "dtype": "torch.uint8",
                      "note": "UE8M0 group scale"},
        },
        "dims": {"m": m, "k": k, "mx_group": 32},
    }
