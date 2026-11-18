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

"""Native SGLang collection workload for causal_conv1d_update."""

import torch

OP_NAME = "causal_conv1d_update"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

MUTATED_INPUTS = (1,)

SHAPES = [(64, 4096, 4, 1), (256, 2048, 4, 1)]
MORE_SHAPES = [(2, 16, 4, 1), (5, 64, 4, 1), (3, 32, 3, 4)]


def native():
    """解析 sglang.kernels.ops.mamba.causal_conv1d_triton.causal_conv1d_update。"""
    try:
        from sglang.kernels.ops.mamba.causal_conv1d_triton import (
            causal_conv1d_update as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"batch": batch, "dim": dim, "width": width, "seqlen": seqlen}
        for (batch, dim, width, seqlen) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    batch, dim, width, seqlen = shape
    g = torch.Generator(device=device).manual_seed(0)
    state_len = width - 1
    x = torch.randn(
        batch,
        dim,
        seqlen,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    if seqlen == 1:
        x = x.squeeze(-1)
    conv_state = torch.randn(
        batch,
        dim,
        state_len,
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
    # ``activation`` keeps its "silu" default: unpack_to_args_kwargs drops
    # bare str positionals.
    return x, conv_state, weight, bias


def build_inputs(binding, dtype, device):
    shape = (
        binding["batch"],
        binding["dim"],
        binding["width"],
        binding["seqlen"],
    )
    return (*build_case(shape, dtype, device), "silu"), {}


def key_shape(binding):
    return (
        f"b{binding['batch']}_dim{binding['dim']}_w{binding['width']}"
        f"_sq{binding['seqlen']}"
    )


def config(binding, dtype):
    batch = binding["batch"]
    dim = binding["dim"]
    width = binding["width"]
    seqlen = binding["seqlen"]
    state_len = width - 1
    dt = str(dtype)
    # seqlen==1 时 x 被 squeeze 成 (batch, dim)，否则 (batch, dim, seqlen)。
    x_shape = [batch, dim] if seqlen == 1 else [batch, dim, seqlen]
    return {
        "inputs": {
            "x": {"shape": x_shape, "dtype": dt},
            "conv_state": {
                "shape": [batch, dim, state_len],
                "dtype": dt,
                "note": "原地更新",
            },
            "weight": {"shape": [dim, width], "dtype": dt},
            "bias": {"shape": [dim], "dtype": dt},
            "activation": {"scalar": "silu"},
        },
        "outputs": {
            "out": {"shape": x_shape, "dtype": dt},
        },
        "dims": {
            "batch": batch,
            "dim": dim,
            "width": width,
            "seqlen": seqlen,
            "state_len": state_len,
        },
    }
