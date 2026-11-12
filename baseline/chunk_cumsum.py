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

"""Native SGLang collection workload for chunk_cumsum."""

import torch

OP_NAME = "chunk_cumsum"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(8, 16, 256, 32), (32, 4, 256, 64)]
MORE_SHAPES = [(1, 1, 8, 4), (2, 3, 16, 8), (3, 2, 32, 16)]


def native():
    """解析 sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state._chunk_cumsum_fwd。"""
    try:
        from sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state import (
            _chunk_cumsum_fwd as op,
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
            "nheads": nheads,
            **{"dt_bias": True, "dt_softplus": True},
        }
        for (batch, nchunks, chunk_size, nheads) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    batch, nchunks, chunk_size, nheads = shape
    g = torch.Generator(device=device).manual_seed(0)
    seqlen = nchunks * chunk_size
    dt = torch.randn(
        batch,
        seqlen,
        nheads,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    # A must stay negative for the decay to be stable.
    a = (
        -torch.rand(nheads, generator=g, device=device, dtype=torch.float32)
        - 0.1
    )
    dt_bias = torch.randn(
        nheads, generator=g, device=device, dtype=torch.float32
    )
    return dt, a, chunk_size, dict(dt_bias=dt_bias, dt_softplus=True)


def build_inputs(binding, dtype, device):
    shape = (
        binding["batch"],
        binding["nchunks"],
        binding["chunk_size"],
        binding["nheads"],
    )
    *args, kwargs = build_case(shape, dtype, device)
    return tuple(args), kwargs


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
        "dt_bias": (
            {"shape": [nheads], "dtype": "torch.float32"}
            if use_bias
            else {"scalar": None}
        ),
        "dt_softplus": {"scalar": dt_softplus},
    }
    return {
        "inputs": inputs,
        "outputs": {
            "dA_cumsum": {
                "shape": [batch, nheads, nchunks, chunk_size],
                "dtype": "torch.float32",
            },
            "dt_out": {
                "shape": [batch, nheads, nchunks, chunk_size],
                "dtype": "torch.float32",
            },
        },
        "dims": {
            "batch": batch,
            "nchunks": nchunks,
            "chunk_size": chunk_size,
            "seqlen": seqlen,
            "nheads": nheads,
            "dt_bias": use_bias,
            "dt_softplus": dt_softplus,
        },
    }
