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

"""Native SGLang v0.5.21 collection workload for fused_recurrent_gated_delta_rule_packed_decode."""

import torch

OP_NAME = "fused_recurrent_gated_delta_rule_packed_decode"
DTYPES = [torch.bfloat16]
IS_INPLACE = True
MUTATED_INPUTS = (6, 7)

SHAPES = [
    (1, 32, 16, 128, 128),
    (8, 32, 16, 128, 128),
    (64, 32, 16, 128, 128),
    (256, 32, 16, 128, 128),
]
MORE_SHAPES = [(512, 32, 16, 128, 128)]


def native():
    try:
        from sglang.kernels.ops.attention.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op


def binding_for_case(shape):
    return dict(zip(("B", "HV", "H", "K", "V"), shape))


def grid():
    return [binding_for_case(shape) for shape in (*SHAPES, *MORE_SHAPES)]


def build_case(shape, cur_dtype, device):
    B, HV, H, K, V = shape
    num_slots = B + 1
    qk = H * K
    g = torch.Generator(device=device).manual_seed(9)
    mixed_qkv = (
        torch.randn(B, 2 * qk + HV * V, device=device, generator=g) * 0.3
    ).to(cur_dtype)
    a = torch.randn(B, HV, device=device, generator=g).to(cur_dtype)
    b = torch.randn(B, HV, device=device, generator=g).to(cur_dtype)
    A_log = torch.randn(HV, device=device, generator=g)
    dt_bias = torch.randn(HV, device=device, generator=g)
    initial_state = (
        torch.randn(num_slots, HV, V, K, device=device, generator=g) * 0.1
    )
    out = torch.zeros(B, 1, HV, V, device=device, dtype=cur_dtype)
    ssm_state_indices = torch.arange(
        1, B + 1, device=device, dtype=torch.int32
    )
    scale = K**-0.5
    return (
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        (ssm_state_indices),
        {"use_qk_l2norm_in_kernel": False},
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["B"],
        binding["HV"],
        binding["H"],
        binding["K"],
        binding["V"],
    )
    *args, kwargs = build_case(shape, dtype, device)
    return tuple(args), kwargs


def key_shape(binding):
    return [
        binding["B"],
        binding["HV"],
        binding["H"],
        binding["K"],
        binding["V"],
    ]


def config(binding, dtype):
    B, HV, H, K, V = (
        binding["B"],
        binding["HV"],
        binding["H"],
        binding["K"],
        binding["V"],
    )

    def tensor(shape, tensor_dtype=str(dtype)):
        return {"shape": shape, "dtype": tensor_dtype}

    return {
        "inputs": {
            "mixed_qkv": tensor([B, 2 * H * K + HV * V]),
            "a": tensor([B, HV]),
            "b": tensor([B, HV]),
            "A_log": tensor([HV], "torch.float32"),
            "dt_bias": tensor([HV], "torch.float32"),
            "scale": {"scalar": K**-0.5},
            "initial_state": tensor([B + 1, HV, V, K], "torch.float32"),
            "out": tensor([B, 1, HV, V]),
            "ssm_state_indices": tensor([B], "torch.int32"),
            "use_qk_l2norm_in_kernel": {"scalar": False},
        },
        "outputs": {
            "out": tensor([B, 1, HV, V]),
            "initial_state": tensor([B + 1, HV, V, K], "torch.float32"),
        },
        "dims": dict(binding),
    }
