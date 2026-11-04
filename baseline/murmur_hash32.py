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

"""murmur_hash32 baseline（方案 B，sglang）。

native（problem 属 utility 组，符号在 sampling 下）：
    sglang.kernels.ops.sampling.murmur_hash.murmur_hash32
    调用形态复刻 problems/utility/murmur_hash32/baseline.py：
        murmur_hash32(seed, positions, col_indices)
    产出新张量返回 -> IS_INPLACE=False。

输入构造复刻 problems/utility/murmur_hash32/cases.py 的 BENCH_CASES：
    seed = randint(0, 1<<30, (n,)) int32
    positions = randint(0, 1<<20, (n,)) int32
    col_indices = arange(m) int32
    其中 m=1024，n in (1,8,64,512,4096)。
全整型输入，DTYPES 记 int32。
"""

import torch

OP_NAME = "murmur_hash32"
DTYPES = [torch.int32]
IS_INPLACE = False

_M = 1024
_N = [1, 8, 64, 512, 4096]


def native():
    """解析 sglang.kernels.ops.sampling.murmur_hash.murmur_hash32。"""
    try:
        from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32 as op
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [{"n": n, "m": _M} for n in _N]


def build_inputs(binding, dtype, device):
    n, m = binding["n"], binding["m"]
    g = torch.Generator(device=device).manual_seed(9)
    s = torch.randint(0, 1 << 30, (n,), dtype=torch.int32, device=device, generator=g)
    p = torch.randint(0, 1 << 20, (n,), dtype=torch.int32, device=device, generator=g)
    c = torch.arange(m, dtype=torch.int32, device=device)
    return (s, p, c), {}


def key_shape(binding):
    return [binding["n"], binding["m"]]


def config(binding, dtype):
    n, m = binding["n"], binding["m"]
    return {
        "inputs": {
            "seed": {"shape": [n], "dtype": "torch.int32"},
            "positions": {"shape": [n], "dtype": "torch.int32"},
            "col_indices": {"shape": [m], "dtype": "torch.int32"},
        },
        "outputs": {
            "out": {"shape": [n, m], "dtype": "torch.int32"},
        },
        "dims": {"n": n, "m": m},
    }
