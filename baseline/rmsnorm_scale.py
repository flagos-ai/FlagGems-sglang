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

"""rmsnorm_scale baseline（方案 B，sglang）。

native：复刻 problems/diffusion/rmsnorm_scale/baseline.py：
    sglang.kernels.ops.diffusion.triton.native_bf16_rmsnorm.rmsnorm_scale
        rmsnorm_scale(x, weight, scale, eps)
    返回新张量 -> IS_INPLACE=False。

输入构造复刻 problems/diffusion/rmsnorm_scale/cases.py 的 BENCH_CASES：
    x = randn(rows, dim).bf16; weight = randn(dim).bf16;
    scale = randn(rows, dim).bf16（BENCH 中 scale_rows == rows）; eps=1e-6
dtype 取 cases.py 的 [bfloat16]。
"""

import torch

OP_NAME = "rmsnorm_scale"
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_EPS = 1e-6
_SHAPES = [(r, d) for r in (8, 64, 1024, 8192) for d in (2048, 4096)]


def native():
    """解析 rmsnorm_scale。

    sglang 0.5.21 把该符号挪到了 sglang.kernels.ops.diffusion.norm.native_bf16_rmsnorm_triton，故先按 spec 路径再回退已安装路径。
    """
    try:
        from sglang.kernels.ops.diffusion.triton.native_bf16_rmsnorm import (
            rmsnorm_scale as op,
        )
    except Exception:
        try:
            from sglang.kernels.ops.diffusion.norm.native_bf16_rmsnorm_triton import (
                rmsnorm_scale as op,
            )
        except Exception:
            return None
    return op if callable(op) else None


def grid():
    return [{"rows": r, "dim": d} for (r, d) in _SHAPES]


def build_inputs(binding, dtype, device):
    rows, dim = binding["rows"], binding["dim"]
    g = torch.Generator(device=device).manual_seed(0)

    def _t(*shape):
        return torch.randn(*shape, dtype=torch.float32, device=device, generator=g).to(dtype)

    x = _t(rows, dim).contiguous()
    weight = _t(dim).contiguous()
    scale = _t(rows, dim).contiguous()
    return (x, weight, scale, _EPS), {}


def key_shape(binding):
    return [binding["rows"], binding["dim"]]


def config(binding, dtype):
    rows, dim = binding["rows"], binding["dim"]
    dt = str(dtype)
    return {
        "inputs": {
            "x": {"shape": [rows, dim], "dtype": dt},
            "weight": {"shape": [dim], "dtype": dt},
            "scale": {"shape": [rows, dim], "dtype": dt},
            "eps": {"scalar": _EPS},
        },
        "outputs": {
            "out": {"shape": [rows, dim], "dtype": dt},
        },
        "dims": {"rows": rows, "dim": dim},
    }
