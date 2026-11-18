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

"""Native SGLang v0.5.21 collection workload for chunk_gated_delta_rule_fwd."""

import torch

OP_NAME = "chunk_gated_delta_rule_fwd"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [
    (1, 512, 16, 128, 128),
    (1, 2048, 16, 128, 128),
    (2, 2048, 16, 128, 128),
    (1, 4096, 16, 128, 128),
]
MORE_SHAPES = [(4, 4096, 16, 128, 128)]


def native():
    try:
        from sglang.kernels.ops.attention.fla.chunk import (
            chunk_gated_delta_rule as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op


def binding_for_case(shape):
    return dict(zip(("B", "T", "H", "K", "V"), shape))


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    B, T, H, K, V = shape
    g = torch.Generator(device=device).manual_seed(9)
    q = torch.randn(B, T, H, K, device=device, generator=g).to(cur_dtype)
    k = torch.randn(B, T, H, K, device=device, generator=g).to(cur_dtype)
    v = torch.randn(B, T, H, V, device=device, generator=g).to(cur_dtype)
    # g = logsigmoid(u), matching the operator's own data generation.
    u = torch.rand(B, T, H, device=device, generator=g)
    gate = torch.nn.functional.logsigmoid(u)
    beta = torch.rand(B, T, H, device=device, generator=g)
    scale = K**-0.5
    return (
        q,
        k,
        v,
        gate,
        beta,
        scale,
        None,
        False,
        None,
        {"use_qk_l2norm_in_kernel": True},
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["B"],
        binding["T"],
        binding["H"],
        binding["K"],
        binding["V"],
    )
    *args, initial_state, _, cu_seqlens, kwargs = build_case(
        shape, dtype, device
    )
    # The native kernel reads a slot index even when there is no state pool.
    state_indices = torch.full(
        (shape[0],), -1, device=device, dtype=torch.int32
    )
    return tuple(args), dict(
        kwargs,
        initial_state=initial_state,
        initial_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        inplace_update=False,
    )


def key_shape(binding):
    return [
        binding["B"],
        binding["T"],
        binding["H"],
        binding["K"],
        binding["V"],
    ]


def config(binding, dtype):
    B, T, H, K, V = (
        binding["B"],
        binding["T"],
        binding["H"],
        binding["K"],
        binding["V"],
    )

    def tensor(shape, tensor_dtype=str(dtype)):
        return {"shape": shape, "dtype": tensor_dtype}

    return {
        "inputs": {
            "q": tensor([B, T, H, K]),
            "k": tensor([B, T, H, K]),
            "v": tensor([B, T, H, V]),
            "g": tensor([B, T, H], "torch.float32"),
            "beta": tensor([B, T, H], "torch.float32"),
            "scale": {"scalar": K**-0.5},
            "initial_state": {"scalar": None},
            "initial_state_indices": tensor([B], "torch.int32"),
            "cu_seqlens": {"scalar": None},
            "use_qk_l2norm_in_kernel": {"scalar": True},
            "inplace_update": {"scalar": False},
        },
        "outputs": {"out": tensor([B, T, H, V])},
        "dims": dict(binding),
    }
