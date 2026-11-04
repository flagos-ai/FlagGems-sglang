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

"""bmm_chunk baseline（方案 B，sglang）。

native（spec「SGLang baseline」字段）：
    sglang.kernels.ops.mamba.triton_ops.ssd_bmm._bmm_chunk_fwd
    调用形态复刻 problems/mamba/bmm_chunk/baseline.py：
        _bmm_chunk_fwd(a, b, chunk_size, causal=causal)
    另分配输出，不原地写回输入。

输入构造复刻 problems/mamba/bmm_chunk/cases.py 的 BENCH_CASES：
    seqlen = nchunks * chunk_size
    a = randn(batch, seqlen, ngroups, k).to(dtype)
    b = randn(batch, seqlen, ngroups, k).to(dtype)
    两条 BENCH 都用 causal=True。
dtype 取 benchmark/test_bmm_chunk.py 的 [bfloat16]。
"""

import torch

OP_NAME = "bmm_chunk"
DTYPES = [torch.bfloat16]
IS_INPLACE = False  # 另分配输出

# (batch, nchunks, chunk_size, ngroups, k, causal)
_SHAPES = [
    (8, 16, 256, 8, 64, True),
    (32, 4, 256, 8, 64, True),
]


def native():
    """解析 sglang.kernels.ops.mamba.triton_ops.ssd_bmm._bmm_chunk_fwd。"""
    try:
        from sglang.kernels.ops.mamba.triton_ops.ssd_bmm import (
            _bmm_chunk_fwd as op,
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
            "ngroups": ng,
            "k": k,
            "causal": causal,
        }
        for (b, nc, cs, ng, k, causal) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    batch = binding["batch"]
    nchunks = binding["nchunks"]
    chunk_size = binding["chunk_size"]
    ngroups = binding["ngroups"]
    k = binding["k"]
    causal = binding["causal"]
    seqlen = nchunks * chunk_size
    g = torch.Generator(device=device).manual_seed(0)
    a = torch.randn(
        batch, seqlen, ngroups, k, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    b = torch.randn(
        batch, seqlen, ngroups, k, generator=g, device=device, dtype=torch.float32
    ).to(dtype)
    return (a, b, chunk_size), {"causal": causal}


def key_shape(binding):
    return (
        f"b{binding['batch']}_nc{binding['nchunks']}_cs{binding['chunk_size']}"
        f"_ng{binding['ngroups']}_k{binding['k']}_causal{int(binding['causal'])}"
    )


def config(binding, dtype):
    batch = binding["batch"]
    nchunks = binding["nchunks"]
    chunk_size = binding["chunk_size"]
    ngroups = binding["ngroups"]
    k = binding["k"]
    causal = binding["causal"]
    seqlen = nchunks * chunk_size
    dt = str(dtype)
    return {
        "inputs": {
            "a": {"shape": [batch, seqlen, ngroups, k], "dtype": dt},
            "b": {"shape": [batch, seqlen, ngroups, k], "dtype": dt},
            "chunk_size": {"scalar": chunk_size},
            "causal": {"scalar": causal},
        },
        "outputs": {
            "out": {"shape": [batch, nchunks, ngroups, chunk_size, chunk_size],
                    "dtype": dt},
        },
        "dims": {
            "batch": batch, "nchunks": nchunks, "chunk_size": chunk_size,
            "seqlen": seqlen, "ngroups": ngroups, "k": k, "causal": causal,
        },
    }
