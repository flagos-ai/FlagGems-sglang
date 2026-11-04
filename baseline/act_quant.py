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

"""act_quant baseline（方案 B，sglang）。

native：sglang.kernels.ops.attention.dsa.triton_kernel.act_quant
    调用形态复刻 problems/dsa/act_quant/baseline.py：
        act_quant(x, block_size=block_size, scale_fmt=scale_fmt)
    返回 (q, scale) 新张量，不原地写回；故 IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES（block_size=128, scale_fmt=None）：
    x=randn(m,n).to(bf16)。
BENCH: _case(m,4096) for m in (1,8,64,512,4096)。
"""

import torch

OP_NAME = "act_quant"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_BLOCK = 128
_N = 4096
_SHAPES = [{"m": m, "n": _N} for m in (1, 8, 64, 512, 4096)]


def native():
    """解析 act_quant。"""
    try:
        from sglang.kernels.ops.attention.dsa.triton_kernel import act_quant as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [dict(b) for b in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["m"], binding["n"]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(m, n, generator=g, device=device, dtype=torch.float32).to(dtype)
    return (x,), {"block_size": _BLOCK, "scale_fmt": None}


def key_shape(binding):
    return [binding["m"], binding["n"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, n], "dtype": dt},
            "block_size": {"scalar": _BLOCK},
            "scale_fmt": {"scalar": None},
        },
        "outputs": {
            "q": {"shape": [m, n], "dtype": "torch.float8_e4m3fn"},
            "scale": {"shape": [m, n // _BLOCK], "dtype": "torch.float32"},
        },
        "dims": {"m": m, "n": n, "block_size": _BLOCK},
    }
