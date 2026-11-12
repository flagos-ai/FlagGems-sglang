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

"""Native SGLang collection workload for group_norm_silu."""

import torch

OP_NAME = "group_norm_silu"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

SHAPES = [(n, c, s, 32) for n in (1, 2) for c, s in ((128, 64), (256, 32))]
MORE_SHAPES = [(n, c, s, 32) for n in (1, 2) for c, s in ((512, 16),)]


def native():
    """解析 triton_group_norm_silu。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.norm.group_norm_silu_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.group_norm_silu import (
            triton_group_norm_silu as op,
        )
    except (ImportError, AttributeError, OSError, RuntimeError):
        try:
            from sglang.kernels.ops.diffusion.norm.group_norm_silu_triton import (
                triton_group_norm_silu as op,
            )
        except (ImportError, AttributeError, OSError, RuntimeError):
            return None
    return op if callable(op) else None


def grid():
    return [
        {"n": n, "c": c, "spatial": spatial, "num_groups": num_groups}
        for (n, c, spatial, num_groups) in (*SHAPES, *MORE_SHAPES)
    ]


def build_case(shape, cur_dtype, device):
    n, c, spatial, num_groups = shape
    eps = 1e-5
    g = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        n, c, spatial, spatial, dtype=torch.float32, device=device, generator=g
    ).to(cur_dtype)
    w = torch.randn(c, dtype=torch.float32, device=device, generator=g).to(
        cur_dtype
    )
    b = torch.randn(c, dtype=torch.float32, device=device, generator=g).to(
        cur_dtype
    )
    return x, w, b, {"num_groups": num_groups, "eps": eps}


def build_inputs(binding, dtype, device):
    shape = (
        binding["n"],
        binding["c"],
        binding["spatial"],
        binding["num_groups"],
    )
    x, weight, bias, kwargs = build_case(shape, dtype, device)
    return (x, weight, bias, kwargs["num_groups"], kwargs["eps"]), {}


def key_shape(binding):
    return [binding["n"], binding["c"], binding["spatial"], binding["spatial"]]


def config(binding, dtype):
    n, c, spatial = binding["n"], binding["c"], binding["spatial"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [n, c, spatial, spatial], "dtype": dt},
            "weight": {"shape": [c], "dtype": dt},
            "bias": {"shape": [c], "dtype": dt},
            "num_groups": {"scalar": binding["num_groups"]},
            "eps": {"scalar": 1e-5},
        },
        "outputs": {
            "out": {"shape": [n, c, spatial, spatial], "dtype": dt},
        },
        "dims": {
            "n": n,
            "c": c,
            "spatial": spatial,
            "num_groups": binding["num_groups"],
        },
    }
