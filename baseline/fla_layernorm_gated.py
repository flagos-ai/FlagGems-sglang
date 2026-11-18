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

"""Native SGLang collection workload for fla_layernorm_gated."""

import torch

OP_NAME = "fla_layernorm_gated"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(t, 2048, "swish", 1, 1) for t in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [
    (1, 64, "swish", 1, 1),
    (37, 256, "sigmoid", 1, 1),
    (83, 512, "swish", 0, 0),
]
_EPS = 1e-5


def native():
    """解析 sglang.kernels.ops.attention.fla.fused_norm_gate.layer_norm_gated_fwd。"""
    try:
        from sglang.kernels.ops.attention.fla.fused_norm_gate import (
            layer_norm_gated_fwd as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "t": t,
            "d": d,
            "activation": activation,
            "is_rms_norm": is_rms_norm,
            "has_bias": has_bias,
        }
        for (t, d, activation, is_rms_norm, has_bias) in (
            *SHAPES,
            *MORE_SHAPES,
        )
    ]


def build_case(shape, cur_dtype, device):
    t, d, activation, is_rms_norm, has_bias = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(t, d, generator=g, device=device, dtype=torch.float32).to(
        cur_dtype
    )
    gate = torch.randn(
        t, d, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    weight = torch.randn(
        d, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    bias = None
    if has_bias:
        bias = torch.randn(
            d, generator=g, device=device, dtype=torch.float32
        ).to(cur_dtype)
    # ``activation`` is a str, so it rides in a trailing dict along with the
    # remaining keyword-only style options.
    return (
        x,
        gate,
        weight,
        bias,
        {
            "activation": activation,
            "eps": _EPS,
            "is_rms_norm": bool(is_rms_norm),
        },
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["t"],
        binding["d"],
        binding["activation"],
        binding["is_rms_norm"],
        binding["has_bias"],
    )
    *args, kwargs = build_case(shape, dtype, device)
    return tuple(args), dict(kwargs, out_dtype=dtype)


def key_shape(binding):
    return "t{t}_d{d}_{activation}_rms{is_rms_norm}_bias{has_bias}".format(
        **binding
    )


def config(binding, dtype):
    t, d = binding["t"], binding["d"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [t, d], "dtype": dt},
            "g": {"shape": [t, d], "dtype": dt},
            "weight": {"shape": [d], "dtype": dt},
            "bias": (
                {"shape": [d], "dtype": dt}
                if binding["has_bias"]
                else {"scalar": None}
            ),
            "activation": {"scalar": binding["activation"]},
            "eps": {"scalar": _EPS},
            "out_dtype": {"scalar": dt},
            "is_rms_norm": {"scalar": bool(binding["is_rms_norm"])},
        },
        "outputs": {
            "y": {"shape": [t, d], "dtype": dt},
            "mean": {"note": "RMSNorm 下为 None"},
            "rstd": {"shape": [t], "dtype": "torch.float32"},
            "residual_out": {"note": "无 residual 融合时为 None"},
        },
        "dims": {"t": t, "d": d},
    }
