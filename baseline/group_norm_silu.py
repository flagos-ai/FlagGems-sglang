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

"""group_norm_silu baseline（方案 B，sglang）。

native：复刻 problems/diffusion/group_norm_silu/baseline.py：
    sglang.kernels.ops.diffusion.triton.group_norm_silu.triton_group_norm_silu
        triton_group_norm_silu(x, weight, bias, num_groups, eps)
    返回新张量 -> IS_INPLACE=False。

输入构造复刻 cases.py 的 BENCH_CASES：
    x = randn(n, c, spatial, spatial).bf16; weight/bias = randn(c).bf16;
    num_groups=32; eps=1e-5。BENCH 遍历 n∈{1,2}，(c, spatial)∈{(128,64),(256,32),(512,16)}。
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "group_norm_silu"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_NUM_GROUPS = 32
_EPS = 1e-5
_SHAPES = [(n, c, s) for n in (1, 2) for (c, s) in ((128, 64), (256, 32), (512, 16))]


def native():
    """解析 triton_group_norm_silu。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.norm.group_norm_silu_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.group_norm_silu import (
            triton_group_norm_silu as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.diffusion.norm.group_norm_silu_triton import (
                triton_group_norm_silu as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [{"n": n, "c": c, "spatial": s} for (n, c, s) in _SHAPES]


def build_inputs(binding, dtype, device):
    n, c, spatial = binding["n"], binding["c"], binding["spatial"]
    g = torch.Generator(device=device).manual_seed(9)
    x = torch.randn(n, c, spatial, spatial, device=device, generator=g).to(dtype)
    w = torch.randn(c, device=device, generator=g).to(dtype)
    b = torch.randn(c, device=device, generator=g).to(dtype)
    return (x, w, b, _NUM_GROUPS, _EPS), {}


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
            "num_groups": {"scalar": _NUM_GROUPS},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [n, c, spatial, spatial], "dtype": dt},
        },
        "dims": {"n": n, "c": c, "spatial": spatial, "num_groups": _NUM_GROUPS},
    }
