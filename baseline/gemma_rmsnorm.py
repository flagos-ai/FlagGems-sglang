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

"""gemma_rmsnorm baseline（方案 B，sglang）。

native：
    sglang.kernels.ops.layernorm.minimax_m3_rmsnorm.gemma_rmsnorm
    调用形态复刻 problems/activation_norm/gemma_rmsnorm/baseline.py：
        gemma_rmsnorm(x, weight, eps)
    返回新张量，不原地写回 -> IS_INPLACE=False。

输入构造复刻该 problem 的 cases.py 的 BENCH_CASES（均为 2D (m, n)，非 strided）：
    x = randn(m, n).to(dtype)；weight = (0.5 * randn(n)).to(dtype)，eps=1e-6。
    dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "gemma_rmsnorm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_EPS = 1e-6
_SHAPES = [(m, n) for m in (1, 8, 64, 512, 4096) for n in (1024, 4096, 8192)]


def native():
    """解析 minimax_m3_rmsnorm.gemma_rmsnorm。"""
    try:
        from sglang.kernels.ops.layernorm.minimax_m3_rmsnorm import gemma_rmsnorm as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"m": m, "n": n} for (m, n) in _SHAPES]


def build_inputs(binding, dtype, device):
    m, n = binding["m"], binding["n"]
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(m, n, dtype=torch.float32, device=device, generator=g).to(dtype)
    weight = (
        0.5 * torch.randn(n, dtype=torch.float32, device=device, generator=g)
    ).to(dtype)
    return (x, weight, _EPS), {}


def key_shape(binding):
    return [binding["m"], binding["n"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, n], "dtype": dt},
            "weight": {"shape": [n], "dtype": dt},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [m, n], "dtype": dt},
        },
        "dims": {"m": m, "n": n},
    }
