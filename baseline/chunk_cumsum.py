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

"""chunk_cumsum baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state._chunk_cumsum_fwd
    调用形态复刻 problems/mamba/chunk_cumsum/baseline.py：
        _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    另分配输出，不原地写回输入。

输入构造复刻 problems/mamba/chunk_cumsum/cases.py 的 BENCH_CASES：
    seqlen = nchunks * chunk_size
    dt = randn(batch, seqlen, nheads).to(dtype)
    A = -(rand(nheads) + 0.1)      # fp32
    dt_bias = randn(nheads)        # fp32（两条 BENCH 都 dt_bias=True）
    两条 BENCH 都 dt_softplus=True。
dtype 取 benchmark/test_chunk_cumsum.py 的 [bfloat16]。
"""

import torch

OP_NAME = "chunk_cumsum"
DTYPES = [torch.bfloat16]
IS_INPLACE = False  # 另分配输出

# (batch, nchunks, chunk_size, nheads, dt_bias, dt_softplus)
_SHAPES = [
    (8, 16, 256, 32, True, True),
    (32, 4, 256, 64, True, True),
]


def native():
    """解析 sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state._chunk_cumsum_fwd。"""
    try:
        from sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state import (
            _chunk_cumsum_fwd as op,
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
            "dt_bias": dtb,
            "dt_softplus": dts,
        }
        for (b, nc, cs, nh, dtb, dts) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    batch = binding["batch"]
    nchunks = binding["nchunks"]
    chunk_size = binding["chunk_size"]
    nheads = binding["nheads"]
    use_bias = binding["dt_bias"]
    dt_softplus = binding["dt_softplus"]
    seqlen = nchunks * chunk_size
    g = torch.Generator(device=device).manual_seed(0)
    dt = torch.randn(
        batch, seqlen, nheads, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    a = -torch.rand(nheads, generator=g, device=device, dtype=torch.float32) - 0.1
    bias = None
    if use_bias:
        bias = torch.randn(nheads, generator=g, device=device, dtype=torch.float32)
    return (dt, a, chunk_size), {"dt_bias": bias, "dt_softplus": dt_softplus}


def key_shape(binding):
    return (
        f"b{binding['batch']}_nc{binding['nchunks']}_cs{binding['chunk_size']}"
        f"_nh{binding['nheads']}"
    )


def config(binding, dtype):
    batch = binding["batch"]
    nchunks = binding["nchunks"]
    chunk_size = binding["chunk_size"]
    nheads = binding["nheads"]
    use_bias = binding["dt_bias"]
    dt_softplus = binding["dt_softplus"]
    seqlen = nchunks * chunk_size
    dt = str(dtype)
    inputs = {
        "dt": {"shape": [batch, seqlen, nheads], "dtype": dt},
        "A": {"shape": [nheads], "dtype": "torch.float32"},
        "chunk_size": {"scalar": chunk_size},
        "dt_bias": ({"shape": [nheads], "dtype": "torch.float32"}
                    if use_bias else {"scalar": None}),
        "dt_softplus": {"scalar": dt_softplus},
    }
    return {
        "inputs": inputs,
        "outputs": {
            "dA_cumsum": {"shape": [batch, nheads, nchunks, chunk_size],
                          "dtype": "torch.float32"},
            "dt_out": {"shape": [batch, nheads, nchunks, chunk_size],
                       "dtype": "torch.float32"},
        },
        "dims": {
            "batch": batch, "nchunks": nchunks, "chunk_size": chunk_size,
            "seqlen": seqlen, "nheads": nheads,
            "dt_bias": use_bias, "dt_softplus": dt_softplus,
        },
    }
