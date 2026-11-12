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

"""Native SGLang collection workload for chunk_state."""

import torch

if __package__:
    from .__chunk_inputs import chunk_cumsum_inputs
else:
    from __chunk_inputs import chunk_cumsum_inputs

OP_NAME = "chunk_state"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(8, 16, 256, 32, 8, 64, 128), (32, 4, 256, 64, 8, 64, 128)]
MORE_SHAPES = [
    (1, 1, 8, 4, 2, 16, 8),
    (2, 3, 16, 8, 2, 32, 16),
    (3, 2, 32, 16, 4, 64, 32),
]


def native():
    """解析 sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state._chunk_state_fwd。"""
    try:
        from sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state import (
            _chunk_state_fwd as op,
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
            "ngroups": ngroups,
            "headdim": headdim,
            "dstate": dstate,
        }
        for (batch, nchunks, chunk_size, nheads, ngroups, headdim, dstate) in (
            *SHAPES,
            *MORE_SHAPES,
        )
    ]


def build_case(shape, cur_dtype, device):
    batch, nchunks, chunk_size, nheads, ngroups, headdim, dstate = shape
    g = torch.Generator(device=device).manual_seed(0)
    seqlen = nchunks * chunk_size
    x = torch.randn(
        batch,
        seqlen,
        nheads,
        headdim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    b = torch.randn(
        batch,
        seqlen,
        ngroups,
        dstate,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    raw_dt = torch.rand(
        batch,
        seqlen,
        nheads,
        generator=g,
        device=device,
        dtype=torch.float32,
    )
    a = (
        -torch.rand(nheads, generator=g, device=device, dtype=torch.float32)
        - 0.1
    )
    # dt/dA_cumsum come from the chunk_cumsum stage that precedes this op.
    dt_out, dA_cumsum = chunk_cumsum_inputs(raw_dt, a, chunk_size)
    return b, x, dt_out, dA_cumsum


def build_inputs(binding, dtype, device):
    shape = (
        binding["batch"],
        binding["nchunks"],
        binding["chunk_size"],
        binding["nheads"],
        binding["ngroups"],
        binding["headdim"],
        binding["dstate"],
    )
    return build_case(shape, dtype, device), {}


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
            "dt": {
                "shape": [batch, nheads, nchunks, chunk_size],
                "dtype": "torch.float32",
            },
            "dA_cumsum": {
                "shape": [batch, nheads, nchunks, chunk_size],
                "dtype": "torch.float32",
            },
        },
        "outputs": {
            "states": {
                "shape": [batch, nchunks, nheads, headdim, dstate],
                "dtype": "torch.float32",
            },
        },
        "dims": {
            "batch": batch,
            "nchunks": nchunks,
            "chunk_size": chunk_size,
            "seqlen": seqlen,
            "nheads": nheads,
            "ngroups": ngroups,
            "headdim": headdim,
            "dstate": dstate,
        },
    }
