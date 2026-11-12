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

"""Native SGLang collection workload for mamba_layernorm_gated."""

import torch

OP_NAME = "mamba_layernorm_gated"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

EPS = 1e-5
SHAPES = [(m, 4096, 128) for m in (1, 8, 64, 512, 4096)]
MORE_SHAPES = [(1, 64, 64), (37, 256, 64), (83, 512, 128), (4, 1024, 1024)]


def native():
    """解析 sglang.kernels.ops.attention.fla.layernorm_gated._layer_norm_fwd。"""
    try:
        from sglang.kernels.ops.attention.fla.layernorm_gated import (
            _layer_norm_fwd as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {"m": m, "n": n, "group_size": group_size}
        for (m, n, group_size) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    m, n, group_size = shape
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(m, n, generator=g, device=device, dtype=torch.float32).to(
        cur_dtype
    )
    weight = torch.randn(
        n, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    bias = torch.randn(n, generator=g, device=device, dtype=torch.float32).to(
        cur_dtype
    )
    z = torch.randn(m, n, generator=g, device=device, dtype=torch.float32).to(
        cur_dtype
    )
    # ``norm_before_gate``/``is_rms_norm`` keep their defaults; the vendored
    # unpack_to_args_kwargs routes the dict into kwargs.
    return x, weight, bias, EPS, dict(z=z, group_size=group_size)


def build_inputs(binding, dtype, device):
    shape = (binding["m"], binding["n"], binding["group_size"])
    *args, kwargs = build_case(shape, dtype, device)
    return tuple(args), dict(kwargs, norm_before_gate=True, is_rms_norm=True)


def key_shape(binding):
    return [binding["m"], binding["n"], binding["group_size"]]


def config(binding, dtype):
    m, n = binding["m"], binding["n"]
    ngroups = n // binding["group_size"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [m, n], "dtype": dt},
            "weight": {"shape": [n], "dtype": dt},
            "bias": {"shape": [n], "dtype": dt},
            "eps": {"scalar": EPS},
            "z": {"shape": [m, n], "dtype": dt},
            "group_size": {"scalar": binding["group_size"]},
            "norm_before_gate": {"scalar": True},
            "is_rms_norm": {"scalar": True},
        },
        "outputs": {
            "out": {"shape": [m, n], "dtype": dt},
            "mean": {
                "shape": [m * ngroups],
                "dtype": "torch.float32",
                "note": "本题只记 out",
            },
            "rstd": {
                "shape": [m * ngroups],
                "dtype": "torch.float32",
                "note": "本题只记 out",
            },
        },
        "dims": {
            "m": m,
            "n": n,
            "group_size": binding["group_size"],
            "ngroups": ngroups,
        },
    }
