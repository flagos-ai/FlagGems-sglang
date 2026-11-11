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

"""Native SGLang collection workload for state_passing."""

import torch

if __package__:
    from .__chunk_inputs import chunk_cumsum_inputs
else:
    from __chunk_inputs import chunk_cumsum_inputs

OP_NAME = "state_passing"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [
    (8, 16, 256, 32, 8192, False),
    (32, 4, 256, 64, 8192, False),
]
MORE_SHAPES = [
    (1, 1, 8, 4, 16, False),
    (2, 3, 16, 8, 32, True),
    (3, 5, 32, 16, 64, False),
]


def native():
    """解析 sglang.kernels.ops.mamba.triton_ops.ssd_state_passing._state_passing_fwd。"""
    try:
        from sglang.kernels.ops.mamba.triton_ops.ssd_state_passing import (
            _state_passing_fwd as op,
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
            "dim": dim,
            "has_init": has_init,
        }
        for (batch, nchunks, chunk_size, nheads, dim, has_init) in (
            *SHAPES,
            *MORE_SHAPES,
        )
    ]


def build_case(shape, cur_dtype, device):
    batch, nchunks, chunk_size, nheads, dim, has_init = shape
    g = torch.Generator(device=device).manual_seed(0)
    states = torch.randn(
        batch,
        nchunks,
        nheads,
        dim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    raw_dt = torch.rand(
        batch,
        nchunks * chunk_size,
        nheads,
        generator=g,
        device=device,
        dtype=torch.float32,
    )
    # A must stay negative for the decay to be stable.
    a = (
        -torch.rand(nheads, generator=g, device=device, dtype=torch.float32)
        - 0.1
    )
    # dA_cumsum comes from the chunk_cumsum stage that precedes this op.
    _, dA_cumsum = chunk_cumsum_inputs(raw_dt, a, chunk_size)
    initial_states = None
    if has_init:
        initial_states = torch.randn(
            batch, nheads, dim, generator=g, device=device, dtype=torch.float32
        ).to(cur_dtype)
    return states, dA_cumsum, initial_states


def build_inputs(binding, dtype, device):
    shape = (
        binding["batch"],
        binding["nchunks"],
        binding["chunk_size"],
        binding["nheads"],
        binding["dim"],
        binding["has_init"],
    )
    *args, initial = build_case(shape, dtype, device)
    return tuple(args), {"initial_states": initial}


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
            "dA_cumsum": {
                "shape": [batch, nheads, nchunks, chunk_size],
                "dtype": "torch.float32",
            },
            "initial_states": (
                {"shape": [batch, nheads, dim], "dtype": dt}
                if binding["has_init"]
                else {"scalar": None}
            ),
        },
        "outputs": {
            "out": {"shape": [batch, nchunks, nheads, dim], "dtype": dt},
            "final_states": {
                "shape": [batch, nheads, dim],
                "dtype": "torch.float32",
            },
        },
        "dims": {
            "batch": batch,
            "nchunks": nchunks,
            "chunk_size": chunk_size,
            "nheads": nheads,
            "dim": dim,
        },
    }
