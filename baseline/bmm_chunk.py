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

"""Native SGLang collection workload for bmm_chunk."""

import torch

OP_NAME = "bmm_chunk"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

CAUSAL = True
SHAPES = [(8, 16, 256, 8, 64), (32, 4, 256, 8, 64)]
MORE_SHAPES = [(1, 1, 8, 2, 16), (2, 3, 16, 2, 32), (3, 2, 32, 4, 64)]


def native():
    """解析 sglang.kernels.ops.mamba.triton_ops.ssd_bmm._bmm_chunk_fwd。"""
    try:
        from sglang.kernels.ops.mamba.triton_ops.ssd_bmm import (
            _bmm_chunk_fwd as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "batch": batch,
            "nchunks": nchunks,
            "chunk_size": chunk_size,
            "ngroups": ngroups,
            "k": k,
            **{"causal": CAUSAL},
        }
        for (batch, nchunks, chunk_size, ngroups, k) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    batch, nchunks, chunk_size, ngroups, k = shape
    g = torch.Generator(device=device).manual_seed(0)
    seqlen = nchunks * chunk_size
    a = torch.randn(
        batch,
        seqlen,
        ngroups,
        k,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    b = torch.randn(
        batch,
        seqlen,
        ngroups,
        k,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    return a, b, chunk_size, CAUSAL


def build_inputs(binding, dtype, device):
    shape = (
        binding["batch"],
        binding["nchunks"],
        binding["chunk_size"],
        binding["ngroups"],
        binding["k"],
    )
    a, b, chunk_size, causal = build_case(shape, dtype, device)
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
            "out": {
                "shape": [batch, nchunks, ngroups, chunk_size, chunk_size],
                "dtype": dt,
            },
        },
        "dims": {
            "batch": batch,
            "nchunks": nchunks,
            "chunk_size": chunk_size,
            "seqlen": seqlen,
            "ngroups": ngroups,
            "k": k,
            "causal": causal,
        },
    }
