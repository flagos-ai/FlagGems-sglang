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

"""chunk_state_varlen baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state.chunk_state_varlen
    调用形态复刻 problems/mamba/chunk_state_varlen/baseline.py：
        chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states)
    另分配输出，不原地写回输入。

输入构造复刻 problems/mamba/chunk_state_varlen/cases.py 的 BENCH_CASES。dt/dA_cumsum
由 chunk_cumsum 参考实现产出，这里内联该前处理（_chunk_cumsum_reference），保持
模块自包含：
    total_seqlen = sum(seq_lens)（选得每序列不跨 chunk 边界）
    cu_seqlens = cumsum([0, *seq_lens])
    x = randn(total_seqlen, nheads, headdim).to(dtype)
    B = randn(total_seqlen, ngroups, dstate).to(dtype)
    raw_dt = rand(1, total_seqlen, nheads); A = -(rand(nheads)+0.1)
    dt_out, dA_cumsum = chunk_cumsum(raw_dt, A, chunk_size)，各 squeeze(0)
    chunk_states = randn(nchunks, nheads, headdim, dstate).to(dtype)
dtype 取 benchmark/test_chunk_state_varlen.py 的 [bfloat16]。
"""

import torch

OP_NAME = "chunk_state_varlen"
DTYPES = [torch.bfloat16]
IS_INPLACE = False  # 另分配输出

# (seq_lens, chunk_size, nheads, ngroups, headdim, dstate)
_SHAPES = [
    ([256] * 8, 256, 32, 8, 64, 128),
    ([256] * 32, 256, 64, 8, 64, 128),
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
    """解析 sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state.chunk_state_varlen。"""
    try:
        from sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state import (
            chunk_state_varlen as op,
        )
    except Exception:
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "seq_lens": sl,
            "chunk_size": cs,
            "nheads": nh,
            "ngroups": ng,
            "headdim": hd,
            "dstate": ds,
        }
        for (sl, cs, nh, ng, hd, ds) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    seq_lens = list(binding["seq_lens"])
    chunk_size = binding["chunk_size"]
    nheads = binding["nheads"]
    ngroups = binding["ngroups"]
    headdim = binding["headdim"]
    dstate = binding["dstate"]

    cu_list = [0]
    for length in seq_lens:
        cu_list.append(cu_list[-1] + length)
    total_seqlen = cu_list[-1]
    assert total_seqlen % chunk_size == 0
    nchunks = total_seqlen // chunk_size
    cu_seqlens = torch.tensor(cu_list, dtype=torch.int32, device=device)

    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        total_seqlen, nheads, headdim, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    b = torch.randn(
        total_seqlen, ngroups, dstate, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    raw_dt = torch.rand(
        1, total_seqlen, nheads, generator=g, device=device, dtype=torch.float32
    )
    a = -torch.rand(nheads, generator=g, device=device, dtype=torch.float32) - 0.1
    dt_out, dA_cumsum = _chunk_cumsum_reference(raw_dt, a, chunk_size)
    dt_out = dt_out.squeeze(0)
    dA_cumsum = dA_cumsum.squeeze(0)

    chunk_states = torch.randn(
        nchunks, nheads, headdim, dstate, generator=g, device=device, dtype=torch.float32
    ).to(dtype)

    return (b, x, dt_out, dA_cumsum, cu_seqlens, chunk_states), {}


def key_shape(binding):
    seq_lens = list(binding["seq_lens"])
    return (
        f"nseq{len(seq_lens)}_len{seq_lens[0]}_cs{binding['chunk_size']}"
        f"_nh{binding['nheads']}_ng{binding['ngroups']}_hd{binding['headdim']}"
        f"_ds{binding['dstate']}"
    )


def config(binding, dtype):
    seq_lens = list(binding["seq_lens"])
    chunk_size = binding["chunk_size"]
    nheads = binding["nheads"]
    ngroups = binding["ngroups"]
    headdim = binding["headdim"]
    dstate = binding["dstate"]
    nseq = len(seq_lens)
    total_seqlen = sum(seq_lens)
    nchunks = total_seqlen // chunk_size
    dt = str(dtype)
    return {
        "inputs": {
            "B": {"shape": [total_seqlen, ngroups, dstate], "dtype": dt},
            "x": {"shape": [total_seqlen, nheads, headdim], "dtype": dt},
            "dt": {"shape": [nheads, nchunks, chunk_size], "dtype": "torch.float32"},
            "dA_cumsum": {"shape": [nheads, nchunks, chunk_size],
                          "dtype": "torch.float32"},
            "cu_seqlens": {"shape": [nseq + 1], "dtype": "torch.int32"},
            "chunk_states": {"shape": [nchunks, nheads, headdim, dstate], "dtype": dt},
        },
        "outputs": {
            "states": {"shape": [nseq, nheads, headdim, dstate],
                       "dtype": "torch.float32"},
        },
        "dims": {
            "seq_lens": seq_lens, "nseq": nseq, "total_seqlen": total_seqlen,
            "chunk_size": chunk_size, "nchunks": nchunks, "nheads": nheads,
            "ngroups": ngroups, "headdim": headdim, "dstate": dstate,
        },
    }
