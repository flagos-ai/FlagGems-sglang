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

"""Native SGLang collection workload for selective_state_update."""

import torch

OP_NAME = "selective_state_update"
DTYPES = [torch.bfloat16]
IS_INPLACE = True

MUTATED_INPUTS = (0, "out")

SHAPES = [
    (64, 32, 128, 128, 8, True, True, True, True),
    (256, 64, 64, 128, 8, True, True, True, True),
]
MORE_SHAPES = [
    (1, 4, 16, 8, 1, True, True, True, True),
    (5, 8, 64, 16, 2, True, True, True, True),
    (3, 16, 128, 32, 4, False, True, True, False),
    # D / z / dt_bias all absent, exercising the kernel's None guards.
    (2, 8, 32, 16, 2, False, False, False, False),
]


def native():
    """解析 sglang.kernels.ops.mamba.triton_ops.mamba_ssm.selective_state_update。"""
    try:
        from sglang.kernels.ops.mamba.triton_ops.mamba_ssm import (
            selective_state_update as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        return None
    return op if callable(op) else None


def grid():
    return [
        {
            "batch": batch,
            "nheads": nheads,
            "dim": dim,
            "dstate": dstate,
            "ngroups": ngroups,
            "has_z": has_z,
            "has_d": has_d,
            "has_dt_bias": has_dt_bias,
            "dt_softplus": dt_softplus,
        }
        for (
            batch,
            nheads,
            dim,
            dstate,
            ngroups,
            has_z,
            has_d,
            has_dt_bias,
            dt_softplus,
        ) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    (
        batch,
        nheads,
        dim,
        dstate,
        ngroups,
        has_z,
        has_d,
        has_dt_bias,
        dt_softplus,
    ) = shape
    g = torch.Generator(device=device).manual_seed(0)
    state = torch.randn(
        batch,
        nheads,
        dim,
        dstate,
        generator=g,
        device=device,
        dtype=torch.float32,
    ).to(cur_dtype)
    x = torch.randn(
        batch, nheads, dim, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    dt = torch.randn(
        batch, nheads, dim, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    # A must stay negative for the decay to be stable.
    a = (
        -torch.rand(
            nheads,
            dim,
            dstate,
            generator=g,
            device=device,
            dtype=torch.float32,
        )
        - 0.1
    )
    b = torch.randn(
        batch, ngroups, dstate, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    c = torch.randn(
        batch, ngroups, dstate, generator=g, device=device, dtype=torch.float32
    ).to(cur_dtype)
    # D / z / dt_bias are all optional per the signature; the kernel guards
    # them with USE_D / USE_Z / USE_DT_BIAS.
    d = None
    if has_d:
        d = torch.randn(
            nheads, dim, generator=g, device=device, dtype=torch.float32
        )
    z = None
    if has_z:
        z = torch.randn(
            batch, nheads, dim, generator=g, device=device, dtype=torch.float32
        ).to(cur_dtype)
    dt_bias = None
    if has_dt_bias:
        dt_bias = torch.randn(
            nheads, dim, generator=g, device=device, dtype=torch.float32
        )
    return (
        state,
        x,
        dt,
        a,
        b,
        c,
        dict(D=d, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus),
    )


def build_inputs(binding, dtype, device):
    shape = (
        binding["batch"],
        binding["nheads"],
        binding["dim"],
        binding["dstate"],
        binding["ngroups"],
        binding["has_z"],
        binding["has_d"],
        binding["has_dt_bias"],
        binding["dt_softplus"],
    )
    *args, kwargs = build_case(shape, dtype, device)
    return tuple(args), dict(kwargs, out=torch.zeros_like(args[1]))


def key_shape(binding):
    return (
        f"b{binding['batch']}_nh{binding['nheads']}_dim{binding['dim']}"
        f"_ds{binding['dstate']}_ng{binding['ngroups']}"
        f"_z{int(binding['has_z'])}_d{int(binding['has_d'])}"
        f"_bias{int(binding['has_dt_bias'])}_sp{int(binding['dt_softplus'])}"
    )


def config(binding, dtype):
    batch = binding["batch"]
    nheads = binding["nheads"]
    dim = binding["dim"]
    dstate = binding["dstate"]
    ngroups = binding["ngroups"]
    dt = str(dtype)
    return {
        "inputs": {
            "state": {
                "shape": [batch, nheads, dim, dstate],
                "dtype": dt,
                "note": "原地更新",
            },
            "x": {"shape": [batch, nheads, dim], "dtype": dt},
            "dt": {"shape": [batch, nheads, dim], "dtype": dt},
            "A": {"shape": [nheads, dim, dstate], "dtype": "torch.float32"},
            "B": {"shape": [batch, ngroups, dstate], "dtype": dt},
            "C": {"shape": [batch, ngroups, dstate], "dtype": dt},
            "D": (
                {"shape": [nheads, dim], "dtype": "torch.float32"}
                if binding["has_d"]
                else {"scalar": None}
            ),
            "z": (
                {"shape": [batch, nheads, dim], "dtype": dt}
                if binding["has_z"]
                else {"scalar": None}
            ),
            "dt_bias": (
                {"shape": [nheads, dim], "dtype": "torch.float32"}
                if binding["has_dt_bias"]
                else {"scalar": None}
            ),
            "dt_softplus": {"scalar": binding["dt_softplus"]},
            "out": {
                "shape": [batch, nheads, dim],
                "dtype": dt,
                "note": "原地写回",
            },
        },
        "outputs": {
            "out": {"shape": [batch, nheads, dim], "dtype": dt},
            "state": {
                "shape": [batch, nheads, dim, dstate],
                "dtype": dt,
                "note": "原地更新后的 state",
            },
        },
        "dims": {
            "batch": batch,
            "nheads": nheads,
            "dim": dim,
            "dstate": dstate,
            "ngroups": ngroups,
        },
    }
