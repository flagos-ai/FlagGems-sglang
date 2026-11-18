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

"""Native SGLang collection workload for chunk_state_varlen."""

import torch

if __package__:
    from .__chunk_inputs import chunk_cumsum_inputs
else:
    from __chunk_inputs import chunk_cumsum_inputs

OP_NAME = "chunk_state_varlen"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [
    ((256,) * 8, 256, 32, 8, 64, 128),
    ((256,) * 32, 256, 64, 8, 64, 128),
]
MORE_SHAPES = [
    ((3, 5), 8, 4, 2, 16, 8),
    ((10, 6, 4, 12), 16, 8, 2, 32, 16),
    ((20, 12, 5, 27, 32), 32, 16, 4, 64, 32),
]


def native():
    """解析 sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state.chunk_state_varlen。"""
    try:
        from sglang.kernels.ops.mamba.triton_ops.ssd_chunk_state import (
            chunk_state_varlen as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "seq_lens": seq_lens,
            "chunk_size": chunk_size,
            "nheads": nheads,
            "ngroups": ngroups,
            "headdim": headdim,
            "dstate": dstate,
        }
        for (seq_lens, chunk_size, nheads, ngroups, headdim, dstate) in (
            *SHAPES,
            *MORE_SHAPES,
        )
    ]


def build_case(shape, cur_dtype, device):
    seq_lens, chunk_size, nheads, ngroups, headdim, dstate = shape
    g = torch.Generator(device=device).manual_seed(0)

    cu_list = [0]
    for length in seq_lens:
        cu_list.append(cu_list[-1] + length)
    total_seqlen = cu_list[-1]
    nchunks = total_seqlen // chunk_size
    cu_seqlens = torch.tensor(cu_list, dtype=torch.int32, device=device)

    x = torch.randn(
        total_seqlen,
        nheads,
        headdim,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    b = torch.randn(
        total_seqlen,
        ngroups,
        dstate,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    raw_dt = torch.rand(
        1,
        total_seqlen,
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
    chunk_states = torch.randn(
        nchunks,
        nheads,
        headdim,
        dstate,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    return (
        b,
        x,
        dt_out.squeeze(0),
        dA_cumsum.squeeze(0),
        cu_seqlens,
        chunk_states,
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["seq_lens"],
        binding["chunk_size"],
        binding["nheads"],
        binding["ngroups"],
        binding["headdim"],
        binding["dstate"],
    )
    return build_case(shape, dtype, device), {}


def key_shape(binding):
    seq_lens = list(binding["seq_lens"])
    return (
        f"nseq{len(seq_lens)}_len{'x'.join(map(str, seq_lens))}_cs{binding['chunk_size']}"
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
            "dt": {
                "shape": [nheads, nchunks, chunk_size],
                "dtype": "torch.float32",
            },
            "dA_cumsum": {
                "shape": [nheads, nchunks, chunk_size],
                "dtype": "torch.float32",
            },
            "cu_seqlens": {"shape": [nseq + 1], "dtype": "torch.int32"},
            "chunk_states": {
                "shape": [nchunks, nheads, headdim, dstate],
                "dtype": dt,
            },
        },
        "outputs": {
            "states": {
                "shape": [nseq, nheads, headdim, dstate],
                "dtype": "torch.float32",
            },
        },
        "dims": {
            "seq_lens": seq_lens,
            "nseq": nseq,
            "total_seqlen": total_seqlen,
            "chunk_size": chunk_size,
            "nchunks": nchunks,
            "nheads": nheads,
            "ngroups": ngroups,
            "headdim": headdim,
            "dstate": dstate,
        },
    }
