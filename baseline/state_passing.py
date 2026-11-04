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

"""state_passing baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.mamba.triton_ops.ssd_state_passing._state_passing_fwd
    调用形态复刻 problems/mamba/state_passing/baseline.py：
        _state_passing_fwd(states, dA_cumsum, initial_states=None)
    另分配输出，不原地写回输入。

输入构造复刻 problems/mamba/state_passing/cases.py 的 BENCH_CASES。dA_cumsum 由
chunk_cumsum 参考实现产出，这里内联该前处理（_chunk_cumsum_reference），保持模块
自包含：
    states = randn(batch, nchunks, nheads, dim).to(dtype)
    raw_dt = rand(batch, nchunks*chunk_size, nheads); A = -(rand(nheads)+0.1)
    _, dA_cumsum = chunk_cumsum(raw_dt, A, chunk_size)
    两条 BENCH 都 initial_states=None。
dtype 取 benchmark/test_state_passing.py 的 [bfloat16]。
"""

import torch

OP_NAME = "state_passing"
DTYPES = [torch.bfloat16]
IS_INPLACE = False  # 另分配输出

# (batch, nchunks, chunk_size, nheads, dim)
_SHAPES = [
    (8, 16, 256, 32, 8192),
    (32, 4, 256, 64, 8192),
]


def _chunk_cumsum_reference(dt, A, chunk_size):
    """复刻 problems/mamba/chunk_cumsum/reference_torch.py（dt_bias=None,
    dt_softplus=False），产出 (dt_out, dA_cumsum)。state_passing 只用 dA_cumsum。"""
    import math

    batch, seqlen, nheads = dt.shape
    nchunks = math.ceil(seqlen / chunk_size)
    dt_f = dt.float().clamp(min=0.0)
    dt_out = (
        dt_f.reshape(batch, nchunks, chunk_size, nheads)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    dA = dt_out * A.float().view(1, nheads, 1, 1)
    dA_cumsum = dA.cumsum(dim=-1)
    return dt_out, dA_cumsum


def native():
    """解析 sglang.kernels.ops.mamba.triton_ops.ssd_state_passing._state_passing_fwd。"""
    try:
        from sglang.kernels.ops.mamba.triton_ops.ssd_state_passing import (
            _state_passing_fwd as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "batch": b,
            "nchunks": nc,
            "chunk_size": cs,
            "nheads": nh,
            "dim": dim,
        }
        for (b, nc, cs, nh, dim) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    batch = binding["batch"]
    nchunks = binding["nchunks"]
    chunk_size = binding["chunk_size"]
    nheads = binding["nheads"]
    dim = binding["dim"]
    g = torch.Generator(device=device).manual_seed(0)
    states = torch.randn(
        batch, nchunks, nheads, dim, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    raw_dt = torch.rand(
        batch, nchunks * chunk_size, nheads, generator=g, device=device, dtype=torch.float32
    )
    a = -torch.rand(nheads, generator=g, device=device, dtype=torch.float32) - 0.1
    _, dA_cumsum = _chunk_cumsum_reference(raw_dt, a, chunk_size)
    return (states, dA_cumsum), {"initial_states": None}


def key_shape(binding):
    return (
        f"b{binding['batch']}_nc{binding['nchunks']}_cs{binding['chunk_size']}"
        f"_nh{binding['nheads']}_dim{binding['dim']}"
    )


def config(binding, dtype):
    batch = binding["batch"]
    nchunks = binding["nchunks"]
    chunk_size = binding["chunk_size"]
    nheads = binding["nheads"]
    dim = binding["dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "states": {"shape": [batch, nchunks, nheads, dim], "dtype": dt},
            "dA_cumsum": {"shape": [batch, nheads, nchunks, chunk_size],
                          "dtype": "torch.float32"},
            "initial_states": {"scalar": None},
        },
        "outputs": {
            "out": {"shape": [batch, nchunks, nheads, dim], "dtype": dt},
            "final_states": {"shape": [batch, nheads, dim], "dtype": "torch.float32"},
        },
        "dims": {
            "batch": batch, "nchunks": nchunks, "chunk_size": chunk_size,
            "nheads": nheads, "dim": dim,
        },
    }
