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

"""Native SGLang v0.5.21 collection workload for fused_recurrent_gated_delta_rule."""

import torch

OP_NAME = "fused_recurrent_gated_delta_rule"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [
    (1, 1, 16, 32, 128, 128),
    (8, 1, 16, 32, 128, 128),
    (64, 1, 16, 32, 128, 128),
    (1, 2048, 16, 32, 128, 128),
]
MORE_SHAPES = [(256, 1, 16, 32, 128, 128)]


def native():
    try:
        from sglang.kernels.ops.attention.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op


def binding_for_case(shape):
    return dict(zip(("B", "T", "H", "HV", "K", "V"), shape))


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    B, T, H, HV, K, V = shape
    g = torch.Generator(device=device).manual_seed(9)
    q = torch.randn(B, T, H, K, device=device, generator=g).to(cur_dtype)
    k = torch.randn(B, T, H, K, device=device, generator=g).to(cur_dtype)
    v = torch.randn(B, T, HV, V, device=device, generator=g).to(cur_dtype)
    gate = -torch.rand(B, T, HV, device=device, generator=g) * 0.5
    beta = torch.rand(B, T, HV, device=device, generator=g)
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
        {"use_qk_l2norm_in_kernel": True},
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["B"],
        binding["T"],
        binding["H"],
        binding["HV"],
        binding["K"],
        binding["V"],
    )
    *args, kwargs = build_case(shape, dtype, device)
    return tuple(args), kwargs


def key_shape(binding):
    return [
        binding["B"],
        binding["T"],
        binding["H"],
        binding["HV"],
        binding["K"],
        binding["V"],
    ]


def config(binding, dtype):
    B, T, H, HV, K, V = (
        binding["B"],
        binding["T"],
        binding["H"],
        binding["HV"],
        binding["K"],
        binding["V"],
    )

    def tensor(shape, tensor_dtype=str(dtype)):
        return {"shape": shape, "dtype": tensor_dtype}

    return {
        "inputs": {
            "q": tensor([B, T, H, K]),
            "k": tensor([B, T, H, K]),
            "v": tensor([B, T, HV, V]),
            "g": tensor([B, T, HV], "torch.float32"),
            "beta": tensor([B, T, HV], "torch.float32"),
            "scale": {"scalar": K**-0.5},
            "initial_state": {"scalar": None},
            "output_final_state": {"scalar": False},
            "use_qk_l2norm_in_kernel": {"scalar": True},
        },
        "outputs": {"out": tensor([B, T, HV, V])},
        "dims": dict(binding),
    }
