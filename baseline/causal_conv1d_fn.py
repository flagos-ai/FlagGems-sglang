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

"""Native SGLang collection workload for causal_conv1d_fn."""

import torch

OP_NAME = "causal_conv1d_fn"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

MUTATED_INPUTS = ("conv_states",)

SHAPES = [(8, 2048, 4096, 4), (32, 512, 2048, 4)]
MORE_SHAPES = [(1, 7, 16, 4), (4, 1, 8, 3)]


def native():
    """解析 sglang.kernels.ops.mamba.causal_conv1d_triton.causal_conv1d_fn。"""
    try:
        from sglang.kernels.ops.mamba.causal_conv1d_triton import (
            causal_conv1d_fn as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "num_seqs": num_seqs,
            "seq_len": seq_len,
            "dim": dim,
            "width": width,
            **{"seq_lens": [seq_len] * num_seqs},
        }
        for (num_seqs, seq_len, dim, width) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    num_seqs, seq_len, dim, width = shape
    g = torch.Generator(device=device).manual_seed(0)
    seq_lens = [seq_len] * num_seqs
    x = torch.randn(
        dim,
        sum(seq_lens),
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    weight = torch.randn(
        dim, width, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    bias = torch.randn(
        dim, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    query_start_loc = torch.zeros(
        num_seqs + 1, dtype=torch.int32, device=device
    )
    query_start_loc[1:] = torch.cumsum(
        torch.tensor(seq_lens, dtype=torch.int32, device=device), dim=0
    )
    # ``activation`` keeps its "silu" default: the vendored
    # unpack_to_args_kwargs drops bare str positionals.
    return x, weight, bias, query_start_loc, seq_lens


def build_inputs(binding, dtype, device):
    shape = (
        binding["num_seqs"],
        binding["seq_len"],
        binding["dim"],
        binding["width"],
    )
    x, weight, bias, starts, lengths = build_case(shape, dtype, device)
    batch = len(lengths)
    states = x.new_zeros((batch, weight.shape[0], weight.shape[1] - 1))
    return (x, weight, bias), {
        "conv_states": states,
        "query_start_loc": starts,
        "seq_lens_cpu": lengths,
        "cache_indices": torch.arange(batch, dtype=torch.int32, device=device),
        "has_initial_state": torch.zeros(
            batch, dtype=torch.bool, device=device
        ),
        "activation": "silu",
    }


def key_shape(binding):
    seq_lens = list(binding["seq_lens"])
    return (
        f"nseq{len(seq_lens)}_len{seq_lens[0]}_dim{binding['dim']}"
        f"_w{binding['width']}"
    )


def config(binding, dtype):
    seq_lens = list(binding["seq_lens"])
    dim = binding["dim"]
    width = binding["width"]
    total = sum(seq_lens)
    batch = len(seq_lens)
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [dim, total], "dtype": dt},
            "weight": {"shape": [dim, width], "dtype": dt},
            "bias": {"shape": [dim], "dtype": dt},
            "conv_states": {
                "shape": [batch, dim, width - 1],
                "dtype": dt,
                "note": "原地更新（kernel 把尾窗写回）",
            },
            "query_start_loc": {"shape": [batch + 1], "dtype": "torch.int32"},
            "seq_lens_cpu": {"note": "seq_lens python list，长度 batch"},
            "cache_indices": {"shape": [batch], "dtype": "torch.int32"},
            "has_initial_state": {"shape": [batch], "dtype": "torch.bool"},
            "activation": {"scalar": "silu"},
        },
        "outputs": {
            "out": {"shape": [dim, total], "dtype": dt},
        },
        "dims": {
            "seq_lens": seq_lens,
            "batch": batch,
            "total": total,
            "dim": dim,
            "width": width,
        },
    }
