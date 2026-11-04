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

"""chunk_local_cumsum_scalar baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.attention.fla.cumsum.chunk_local_cumsum_scalar
    调用形态复刻 problems/fla/chunk_local_cumsum_scalar/baseline.py：
        chunk_local_cumsum_scalar(g, chunk_size, reverse=reverse, scale=scale)
    返回新张量，不原地写回。

输入构造复刻 problems/fla/chunk_local_cumsum_scalar/cases.py 的 BENCH_CASES：
    g = randn(batch, nchunks*chunk_size, nheads).to(dtype)
    reverse=False, scale=None。dtype 取 benchmark 的 [bfloat16]。
"""

import torch

OP_NAME = "chunk_local_cumsum_scalar"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

# (batch, nchunks, chunk_size, nheads)
_CASES = [
    (8, 16, 64, 32),
    (32, 4, 64, 64),
]


def native():
    """解析 sglang.kernels.ops.attention.fla.cumsum.chunk_local_cumsum_scalar。"""
    try:
        from sglang.kernels.ops.attention.fla.cumsum import (
            chunk_local_cumsum_scalar as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {"batch": b, "nchunks": nc, "chunk_size": cs, "nheads": h}
        for (b, nc, cs, h) in _CASES
    ]


def build_inputs(binding, dtype, device):
    b, nc, cs, h = (
        binding["batch"],
        binding["nchunks"],
        binding["chunk_size"],
        binding["nheads"],
    )
    t = nc * cs
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(b, t, h, generator=g, device=device, dtype=torch.float32).to(dtype)
    return (x, cs), {"reverse": False, "scale": None}


def key_shape(binding):
    t = binding["nchunks"] * binding["chunk_size"]
    return [binding["batch"], t, binding["nheads"]]


def config(binding, dtype):
    b, nc, cs, h = (
        binding["batch"],
        binding["nchunks"],
        binding["chunk_size"],
        binding["nheads"],
    )
    t = nc * cs
    dt = str(dtype)
    return {
        "inputs": {
            "g": {"shape": [b, t, h], "dtype": dt},
            "chunk_size": {"scalar": cs},
            "reverse": {"scalar": False},
            "scale": {"scalar": None},
        },
        "outputs": {
            "out": {"shape": [b, t, h], "dtype": dt},
        },
        "dims": {"batch": b, "nchunks": nc, "chunk_size": cs, "nheads": h, "t": t},
    }
