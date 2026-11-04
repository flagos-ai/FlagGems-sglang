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

"""chunk_state baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state._chunk_state_fwd
    调用形态复刻 problems/mamba/chunk_state/baseline.py：
        _chunk_state_fwd(B, x, dt, dA_cumsum)
    另分配输出，不原地写回输入。

输入构造复刻 problems/mamba/chunk_state/cases.py 的 BENCH_CASES。dt/dA_cumsum 由
chunk_cumsum 参考实现产出，这里内联该前处理（见 _chunk_cumsum_reference），保持
模块自包含（NCU 子进程只 import 本模块）：
    seqlen = nchunks * chunk_size
    x = randn(batch, seqlen, nheads, headdim).to(dtype)
    B = randn(batch, seqlen, ngroups, dstate).to(dtype)
    raw_dt = rand(batch, seqlen, nheads)   # fp32
    A = -(rand(nheads) + 0.1)              # fp32
    dt_out, dA_cumsum = chunk_cumsum(raw_dt, A, chunk_size)  # dt_softplus=False
dtype 取 benchmark/test_chunk_state.py 的 [bfloat16]。
"""

import torch

OP_NAME = "chunk_state"
DTYPES = [torch.bfloat16]
IS_INPLACE = False  # 另分配输出

# (batch, nchunks, chunk_size, nheads, ngroups, headdim, dstate)
_SHAPES = [
    (8, 16, 256, 32, 8, 64, 128),
    (32, 4, 256, 64, 8, 64, 128),
]


def _chunk_cumsum_reference(dt, A, chunk_size):
    """复刻 problems/mamba/chunk_cumsum/reference_torch.py（dt_bias=None,
    dt_softplus=False），产出 (dt_out, dA_cumsum)。"""
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
    """解析 sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state._chunk_state_fwd。"""
    try:
        from sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state import (
            _chunk_state_fwd as op,
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
            "ngroups": ng,
            "headdim": hd,
            "dstate": ds,
        }
        for (b, nc, cs, nh, ng, hd, ds) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    batch = binding["batch"]
    nchunks = binding["nchunks"]
    chunk_size = binding["chunk_size"]
    nheads = binding["nheads"]
    ngroups = binding["ngroups"]
    headdim = binding["headdim"]
    dstate = binding["dstate"]
    seqlen = nchunks * chunk_size
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        batch, seqlen, nheads, headdim, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    b = torch.randn(
        batch, seqlen, ngroups, dstate, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    raw_dt = torch.rand(
        batch, seqlen, nheads, generator=g, device=device, dtype=torch.float32
    )
    a = -torch.rand(nheads, generator=g, device=device, dtype=torch.float32) - 0.1
    dt_out, dA_cumsum = _chunk_cumsum_reference(raw_dt, a, chunk_size)
    return (b, x, dt_out, dA_cumsum), {}


def key_shape(binding):
    return (
        f"b{binding['batch']}_nc{binding['nchunks']}_cs{binding['chunk_size']}"
        f"_nh{binding['nheads']}_ng{binding['ngroups']}_hd{binding['headdim']}"
        f"_ds{binding['dstate']}"
    )


def config(binding, dtype):
    batch = binding["batch"]
    nchunks = binding["nchunks"]
    chunk_size = binding["chunk_size"]
    nheads = binding["nheads"]
    ngroups = binding["ngroups"]
    headdim = binding["headdim"]
    dstate = binding["dstate"]
    seqlen = nchunks * chunk_size
    dt = str(dtype)
    return {
        "inputs": {
            "B": {"shape": [batch, seqlen, ngroups, dstate], "dtype": dt},
            "x": {"shape": [batch, seqlen, nheads, headdim], "dtype": dt},
            "dt": {"shape": [batch, nheads, nchunks, chunk_size],
                   "dtype": "torch.float32"},
            "dA_cumsum": {"shape": [batch, nheads, nchunks, chunk_size],
                          "dtype": "torch.float32"},
        },
        "outputs": {
            "states": {"shape": [batch, nchunks, nheads, headdim, dstate],
                       "dtype": "torch.float32"},
        },
        "dims": {
            "batch": batch, "nchunks": nchunks, "chunk_size": chunk_size,
            "seqlen": seqlen, "nheads": nheads, "ngroups": ngroups,
            "headdim": headdim, "dstate": dstate,
        },
    }
